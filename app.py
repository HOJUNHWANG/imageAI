# venv311\Scripts\activate
# app.py
import os
import time
import re
import numpy as np
from PIL import Image, ImageFilter

# opencv 있으면 더 좋고, 없으면 PIL로만 동작
try:
    import cv2
except Exception:
    cv2 = None

def normalize_space(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def comma_join_unique(items):
    out = []
    seen = set()
    for x in items:
        x = normalize_space(x).strip(",")
        if not x:
            continue
        key = x.lower()
        if key not in seen:
            out.append(x)
            seen.add(key)
    return ", ".join(out)

def build_default_negative(mode: str) -> str:
    base = [
        "low quality", "blurry", "jpeg artifacts", "bad anatomy", "deformed",
        "extra arms", "extra hands", "extra fingers", "missing fingers",
        "plastic skin", "over-smoothed skin", "uncanny", "text", "watermark", "logo"
    ]
    wear_block = ["bare chest", "nipples", "shirtless"]  # 옷입히기 모드에서 상반신 노출 방지
    remove_block = ["shirt", "t-shirt", "top", "clothing", "fabric", "textile", "sweater", "hoodie", "jacket", "vest"]

    if mode == "Wear / Change Clothes":
        return comma_join_unique(base + wear_block)
    if mode == "Remove Clothes":
        return comma_join_unique(base + remove_block)
    return comma_join_unique(base)

def enrich_prompt(prompt: str, mode: str) -> str:
    """
    자연어 -> diffusion 토큰 느낌으로 확장
    핵심: remove 모드에서는 fabric/cloth 관련 긍정 토큰을 넣지 말 것.
    """
    p = normalize_space(prompt)
    if not p:
        return ""

    common_quality = [
        "photorealistic", "high detail", "realistic lighting", "sharp focus",
        "natural skin texture", "consistent skin tone", "clean edges"
    ]

    # 옷 입히기(변경)일 때: 섬유/주름/재질 토큰이 도움됨
    wear_tokens = [
        "realistic fabric texture", "natural folds", "subtle wrinkles",
        "proper garment seam", "correct neckline", "clean shoulder line"
    ]

    # 옷 제거일 때: fabric 토큰은 오히려 방해(‘옷처럼’ 칠해버리는 원인)
    remove_tokens = [
        "natural shoulders and arms", "realistic skin texture",
        "consistent shading", "anatomically plausible torso",
        "skin pores", "soft natural shadows"
    ]

    # 사용자 프롬프트 자체에 "remove cloth" 같은 게 있으면,
    # remove 모드에서 더 강하게 “no clothing”을 명시해도 됨.
    if mode == "Remove Clothes":
        remove_goal = [
            "shirtless", "no shirt", "no clothing on upper body",
            "natural male chest", "natural body hair"
        ]
        return comma_join_unique([p] + remove_goal + remove_tokens + common_quality)

    if mode == "Wear / Change Clothes":
        # 사용자 프롬프트가 “black tank top” 같은 의상 지시를 포함하므로,
        # 의상 품질 토큰을 덧붙이되 과도한 바디 변경을 막는 토큰도 같이.
        keep_identity = [
            "keep face unchanged", "keep hairstyle unchanged", "keep background unchanged",
            "preserve original pose"
        ]
        return comma_join_unique([p] + wear_tokens + common_quality + keep_identity)

    # General
    return comma_join_unique([p] + common_quality)

import cv2
from PIL import Image
import gradio as gr
import torch

print("[HW] torch version:", torch.__version__)
print("[HW] cuda available:", torch.cuda.is_available())
print("[HW] cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("[HW] cuda name:", torch.cuda.get_device_name(0))
print("[HW] torch cuda build:", torch.version.cuda)

import traceback

from diffusers import AutoPipelineForInpainting

from sam_utils import SamMaskerManager
from mp_tasks_utils import MPTasksHelper, build_sleeve_mask_v5_tasks, build_top_mask_v5_tasks
from ui_text import APP_TITLE, APP_SUBTITLE, HOW_TO_MD, SLIDER_HINTS_MD

from prompt_enricher import enrich_positive, enrich_negative
from mp_tasks_utils import (
    build_sleeve_mask_v5_tasks,
    build_top_mask_v5_tasks,
    build_pants_mask_v5_tasks,
    build_hair_mask_v5_tasks,
    build_background_mask_v5_tasks,
)

# -----------------------------------------------------------------------------
# Runtime configuration
# -----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

MODEL_ID = os.getenv("MODEL_ID", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
MOCK_INPAINT = os.getenv("MOCK_INPAINT", "0") == "1"

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = pick_device()

# Speed knobs (safe defaults)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

print(f"DEVICE={DEVICE}, MOCK_INPAINT={MOCK_INPAINT}, MODEL_ID={MODEL_ID}")

# -----------------------------------------------------------------------------
# Global state (simple, explicit, easy to reason about)
# -----------------------------------------------------------------------------

STATE = {
    "working_pil": None,      # PIL image (resized working)
    "working_np": None,       # RGB np.uint8
    "orig_pil": None,         # original PIL (for optional restore)
    "mask_u8": None,          # current edit mask (uint8 0/255, working size)
    "auto_mask_candidates": []# list of masks uint8
}

# -----------------------------------------------------------------------------
# Helpers: image resize, mask post processing
# -----------------------------------------------------------------------------

def resize_to_long_side(pil: Image.Image, long_side: int) -> Image.Image:
    """
    Resizes while preserving aspect ratio.
    Avoids the “stretched” look caused by naive width/height assignment.
    """
    w, h = pil.size
    if max(w, h) == long_side:
        return pil
    scale = long_side / float(max(w, h))
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    return pil.resize((nw, nh), Image.LANCZOS)

def to_rgb_np(pil: Image.Image) -> np.ndarray:
    return np.array(pil.convert("RGB"), dtype=np.uint8)

def overlay_mask(image_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """
    Red overlay visualization for debugging/UX.
    """
    vis = image_rgb.copy()
    red = np.zeros_like(vis)
    red[..., 0] = 255
    alpha = 0.45
    m = (mask_u8 > 0)
    vis[m] = (vis[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)
    return vis

def postprocess_mask(mask_u8: np.ndarray, expand_px: int, blur_px: int) -> np.ndarray:
    """
    Applies user-controlled expand/blur.
    Expand:
      - morphological dilation; grows the mask outward
    Blur:
      - gaussian blur; produces softer edges
    """
    m = (mask_u8 > 0).astype(np.uint8) * 255

    if expand_px > 0:
        k = expand_px * 2 + 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8), iterations=1)

    if blur_px > 0:
        k = blur_px * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)

    # Ensure uint8 [0..255]
    m = np.clip(m, 0, 255).astype(np.uint8)
    return m

def refine_mask_soft(mask_u8: np.ndarray, expand_px: int, blur_px: int) -> np.ndarray:
    """
    mask_u8: uint8, 0/255
    expand_px: dilation radius 느낌
    blur_px: feather 정도
    return: uint8 0~255 (soft mask)
    """
    if mask_u8 is None:
        return None

    m = mask_u8
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 127).astype(np.uint8) * 255

    # 1) Expand (dilation)
    if expand_px and expand_px > 0:
        if cv2 is not None:
            k = max(1, int(expand_px))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
            m = cv2.dilate(m, kernel, iterations=1)
        else:
            # PIL fallback: MaxFilter로 간단한 dilation 비슷하게
            img = Image.fromarray(m)
            # size는 홀수여야 함
            size = int(expand_px) * 2 + 1
            img = img.filter(ImageFilter.MaxFilter(size=size))
            m = np.array(img, dtype=np.uint8)

    # 2) Blur / Feather (soft edge)
    if blur_px and blur_px > 0:
        if cv2 is not None:
            # sigma는 blur_px 기반으로
            sigma = max(0.1, blur_px / 2.0)
            m = cv2.GaussianBlur(m, (0, 0), sigmaX=sigma, sigmaY=sigma)
        else:
            img = Image.fromarray(m)
            img = img.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
            m = np.array(img, dtype=np.uint8)

    return m.astype(np.uint8)

# -----------------------------------------------------------------------------
# Prompt parsing (simple & predictable)
# -----------------------------------------------------------------------------

def parse_prompt_simple(prompt: str) -> dict:
    """
    Minimal rule-based parser.
    Keep it deterministic; avoid hidden heuristics that confuse debugging.

    Targets:
      - sleeve: sleeve edits (long sleeve -> short sleeve / sleeveless)
      - top: torso clothing edits
      - generic: fallback
    """
    p = (prompt or "").lower()

    # Explicit directive style
    if "target=sleeve" in p:
        target = "sleeve"
    elif "target=top" in p:
        target = "top"
    else:
        # Keyword inference
        sleeve_kw = ["sleeve", "sleeveless", "tank", "crop top", "short sleeve", "long sleeve"]
        top_kw = ["shirt", "t-shirt", "top", "blouse", "jacket", "hoodie", "sweater"]

        if any(k in p for k in sleeve_kw):
            target = "sleeve"
        elif any(k in p for k in top_kw):
            target = "top"
        else:
            target = "top"

    # Optional metadata extraction (best-effort; used only in UI status)
    color = None
    for c in ["black", "white", "red", "blue", "green", "gray", "brown", "beige"]:
        if c in p:
            color = c
            break

    garment = None
    for g in ["tank top", "t-shirt", "shirt", "blouse", "jacket", "hoodie", "sweater"]:
        if g in p:
            garment = g
            break

    return {"target": target, "color": color or "n/a", "garment": garment or "n/a"}

auto_enrich = gr.Checkbox(
    label="Auto-enrich prompt (natural language → diffusion tokens)",
    value=True,
    interactive=True
)

expanded_preview = gr.Textbox(
    label="Expanded Prompt Preview (read-only)",
    interactive=False,
    lines=6,
    max_lines=12,
    show_copy_button=True
)

# -----------------------------------------------------------------------------
# Model loaders
# -----------------------------------------------------------------------------

def load_pipe():
    """
    Loads SDXL inpaint pipeline.
    Keeps everything in one place so failures are obvious.
    """
    if MOCK_INPAINT:
        return None

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    if torch.cuda.is_available():
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[OPT] xformers enabled")
        except Exception as e:
            print("[OPT] xformers not available:", e)
    else:
        pipe = pipe.to("cpu")

    # Optional speedups
    try:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except Exception:
        pass

    # xFormers: optional; only works when installed with matching CUDA build
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    print("[PIPE] device:", pipe.device)
    print("[PIPE] dtype:", next(pipe.unet.parameters()).dtype)

    return pipe

PIPE = load_pipe()

# MediaPipe tasks helper (semantic masks)
mp_helper = None
MP_INIT_ERROR = None
try:
    mp_helper = MPTasksHelper(weights_dir=WEIGHTS_DIR)
except Exception as e:
    mp_helper = None
    MP_INIT_ERROR = repr(e)

print(f"[BOOT] mp_helper={'OK' if mp_helper else 'NONE'}")
if MP_INIT_ERROR:
    print(f"[BOOT] MP_INIT_ERROR={MP_INIT_ERROR}")

# SAM manager (manual masks)
sam_manager = SamMaskerManager(weights_dir=WEIGHTS_DIR, device=DEVICE if DEVICE == "cuda" else "cpu")

# -----------------------------------------------------------------------------
# Gradio callbacks
# -----------------------------------------------------------------------------

def debug_ping(user_prompt: str, auto_enrich_flag: bool):
    print("[PING] Auto button callback fired")
    print("[PING] prompt head:", (user_prompt or "")[:120])
    print("[PING] auto_enrich_flag:", auto_enrich_flag)
    print("[PING] working_np:", None if STATE["working_np"] is None else STATE["working_np"].shape)
    return [], "PING OK (callback fired)", user_prompt

def on_upload(img: Image.Image, working_long_side: int):
    if img is None:
        return None, None, "Upload an image first."

    orig = img.convert("RGB")
    working = resize_to_long_side(orig, int(working_long_side))

    STATE["orig_pil"] = orig
    STATE["working_pil"] = working
    STATE["working_np"] = to_rgb_np(working)
    STATE["mask_u8"] = None
    STATE["auto_mask_candidates"] = []

    return working, None, "Image loaded. Choose Auto Mask or click for Manual Mask."

def on_manual_click(evt: gr.SelectData, sam_model_type: str):
    """
    Manual mask via SAM:
    - single click -> mask
    - stores STATE['mask_u8']
    """
    if STATE["working_np"] is None:
        return None, None, "Upload an image first."

    x, y = evt.index[0], evt.index[1]

    masker = sam_manager.get(sam_model_type)
    mask_u8 = masker.predict_from_click(STATE["working_np"], x, y)

    STATE["mask_u8"] = mask_u8

    vis = overlay_mask(STATE["working_np"], mask_u8)
    return Image.fromarray(vis), Image.fromarray(mask_u8), f"Manual mask built (SAM {sam_model_type})."

import traceback
import time

def build_auto_candidates_v5(prompt: str, auto_enrich: bool, edit_mode: str):
    import time, traceback

    t0 = time.time()

    # 1) 입력 정리
    user_prompt = normalize_space(prompt)
    auto_enrich_flag = bool(auto_enrich)  # None -> False

    print("\n[AUTO] called")
    print("[AUTO] prompt head:", user_prompt[:160])
    print("[AUTO] auto_enrich:", auto_enrich_flag)
    print("[AUTO] edit_mode:", edit_mode)

    # 2) 이미지 존재 확인
    if STATE.get("working_pil") is None or STATE.get("working_np") is None:
        print("[AUTO] no image in STATE")
        return [], "Upload an image first.", ""

    print("[AUTO] working_pil size:", STATE["working_pil"].size)
    print("[AUTO] working_np shape:", STATE["working_np"].shape)

    # 3) 확장 프롬프트(미리보기용) 확정
    try:
        expanded = user_prompt
        if auto_enrich_flag:
            expanded, _info = enrich_positive(user_prompt)
        expanded = normalize_space(expanded)
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return [], f"[AUTO] enrich_prompt failed\n{tb}", ""

    print("[AUTO] expanded head:", expanded[:160])

    # 4) target 추출 (sleeve/top/pants/hair/background 등)
    try:
        info = parse_prompt_simple(expanded)  # 네 parse가 dict 반환하는 버전 기준
        target = info.get("target", "top")
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return [], f"[AUTO] parse_prompt_simple failed\n{tb}", expanded

    print("[AUTO] target:", target)

    # 5) mp_helper 확인
    if mp_helper is None:
        err = f"mp_helper unavailable: {MP_INIT_ERROR}"
        print("[AUTO]", err)
        return [], err, expanded

    # 6) 마스크 빌드
    try:
        if target == "sleeve":
            c = build_sleeve_mask_v5_tasks(STATE["working_pil"], mp_helper)
        elif target == "top":
            c = build_top_mask_v5_tasks(STATE["working_pil"], mp_helper)
        elif target == "pants":
            c = build_pants_mask_v5_tasks(STATE["working_pil"], mp_helper)
        elif target == "hair":
            c = build_hair_mask_v5_tasks(STATE["working_pil"], mp_helper)
        elif target == "background":
            c = build_background_mask_v5_tasks(STATE["working_pil"], mp_helper)
        else:
            # fallback
            c = build_top_mask_v5_tasks(STATE["working_pil"], mp_helper)

        # shape 정리: (H,W,1) -> (H,W)
        if c.ndim == 3 and c.shape[-1] == 1:
            c = c[..., 0]

        # dtype 정리: uint8 0/255 보장
        if c.dtype != np.uint8:
            c = c.astype(np.uint8)

        # 값이 0/255 아닐 수도 있으니 강제 (안전)
        c = np.where(c > 127, 255, 0).astype(np.uint8)

        print("[AUTO] mask dtype/shape:", c.dtype, c.shape, "unique:", np.unique(c)[:10])

        STATE["auto_mask_candidates"] = [c]
        dt = time.time() - t0

        # Gallery는 PIL 이미지 리스트로 반환
        return [Image.fromarray(c)], f"v5(mp-tasks) OK target={target} time={dt:.2f}s", expanded

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return [], tb, expanded

def select_candidate(evt: gr.SelectData):
    """
    Selects a candidate from the gallery and sets it as current mask.
    """
    idx = evt.index
    if idx is None:
        return None, "No selection."
    if not STATE["auto_mask_candidates"] or idx >= len(STATE["auto_mask_candidates"]):
        return None, "No candidates in state."

    mask_u8 = STATE["auto_mask_candidates"][idx]
    STATE["mask_u8"] = mask_u8

    vis = overlay_mask(STATE["working_np"], mask_u8)
    return Image.fromarray(vis), f"Candidate {idx+1} selected as active mask."

def clear_mask():
    STATE["mask_u8"] = None
    STATE["auto_mask_candidates"] = []
    if STATE["working_np"] is None:
        return None, None, None, "Cleared."
    return Image.fromarray(STATE["working_np"]), None, None, "Mask cleared."

def apply_inpaint(
    prompt: str,
    negative: str,
    steps: int,
    strength: float,
    guidance: float,
    expand_px: int,
    blur_px: int,
    seed: int,
    auto_enrich: bool,
    edit_mode: str,
):
    import time, traceback

    # 1) 입력 정리 / 방어
    user_prompt = normalize_space(prompt)
    user_negative = normalize_space(negative)
    auto_enrich_flag = bool(auto_enrich)  # None -> False

    if STATE.get("working_pil") is None or STATE.get("working_np") is None:
        return None, "Upload an image first.", "", "Error: working image missing."

    # 너 코드에서 마스크 키가 mask_u8인지 selected_mask인지 섞일 수 있어서 둘 다 처리
    mask_u8 = STATE.get("mask_u8", None)
    if mask_u8 is None:
        mask_u8 = STATE.get("selected_mask", None)

    if mask_u8 is None:
        return None, "Create or select a mask first.", "", "Error: mask missing."

    # 2) Prompt enrichment (실제 실행에 사용할 prompt 결정)
    try:
        expanded = user_prompt
        if auto_enrich_flag:
            expanded, _info = enrich_positive(user_prompt)
        expanded = normalize_space(expanded)
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return None, f"Prompt enrich failed:\n{tb}", "", "Error: enrich_prompt failed."

    # 3) Negative prompt (사용자 입력 + 모드별 기본 negative 합치기)
    try:
        mode_neg = build_default_negative(edit_mode)
        merged_negative = comma_join_unique([user_negative, build_default_negative(edit_mode)])
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        merged_negative = enrich_negative(merged_negative)  # fallback

    # 4) Mask postprocess (expand/blur 반영)
    try:
        # postprocess_mask가 있다면 그대로 사용
        # -> 품질 더 올리고 싶으면 refine_mask_soft로 교체 가능
        mask_pp = postprocess_mask(mask_u8, expand_px=int(expand_px), blur_px=int(blur_px))

        # safety: shape (H,W,1) -> (H,W)
        if isinstance(mask_pp, np.ndarray) and mask_pp.ndim == 3 and mask_pp.shape[-1] == 1:
            mask_pp = mask_pp[..., 0]

        # uint8 보장
        if isinstance(mask_pp, np.ndarray) and mask_pp.dtype != np.uint8:
            mask_pp = mask_pp.astype(np.uint8)

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return None, f"Mask postprocess failed:\n{tb}", expanded, "Error: postprocess_mask failed."

    # 5) MOCK path (diffusion 없이 overlay만 보여주기)
    if MOCK_INPAINT:
        vis = overlay_mask(STATE["working_np"], (mask_pp > 0).astype(np.uint8) * 255)
        run_msg = "MOCK_INPAINT=1 (no diffusion run)."
        global_msg = f"OK (mock). mode={edit_mode}, auto_enrich={auto_enrich_flag}"
        return Image.fromarray(vis), run_msg, expanded, global_msg

    # 6) Pipeline 체크
    if PIPE is None:
        return None, "Pipeline not loaded.", expanded, "Error: PIPE is None."

    # 7) Seed generator
    gen = None
    try:
        if int(seed) >= 0:
            gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
    except Exception:
        gen = None

    # ------------------------------------------------------------------
    # Smart parameter tuning by edit_mode (gentle clamp, not override)
    # ------------------------------------------------------------------
    # 원칙:
    #  - Remove Clothes: strength가 높으면 이질감/CG티가 폭발 → 상한 낮게, CFG는 살짝 상향
    #  - Wear/Change: 의상 형태 반영 위해 strength 상한을 조금 높게 허용
    #  - General: 입력값 존중
    #
    # NOTE: "override"가 아니라 "clamp" 위주로만 조정한다.
    # ------------------------------------------------------------------
    mode = (edit_mode or "").strip()

    # 기본 clamp 범위
    min_steps, max_steps = 10, 60
    steps = int(max(min_steps, min(max_steps, int(steps))))

    # guidance(=CFG) 범위
    guidance = float(guidance)
    guidance = max(1.0, min(12.0, guidance))

    strength = float(strength)
    strength = max(0.30, min(0.95, strength))

    # 모드별 미세 조정
    if mode == "Remove Clothes":
        # remove는 "연결(repaint)"이 핵심 → 너무 재창조되면 티가 남
        # - strength 상한을 낮춤
        # - CFG를 약간 올려 "no clothing" 같은 의도를 더 따르게 함
        strength = min(strength, 0.60)
        strength = max(strength, 0.40)  # 너무 낮으면 거의 변화가 없을 수 있음
        guidance = min(10.0, max(guidance, 7.2))

        # remove는 경계가 중요 → blur를 어느 정도 강제 확보
        blur_px = int(blur_px)
        blur_px = max(18, blur_px)
        blur_px = min(40, blur_px)

    elif mode == "Wear / Change Clothes":
        # 옷은 텍스처/형태 반영 위해 remove보다 strength 상한을 높게 허용
        strength = min(strength, 0.72)
        strength = max(strength, 0.48)
        guidance = min(10.0, max(guidance, 6.5))

        # 옷 변경은 가장자리 bleeding 방지를 위해 expand/blur가 너무 과하면 안 좋음
        expand_px = int(expand_px)
        expand_px = min(20, max(6, expand_px))

        blur_px = int(blur_px)
        blur_px = min(30, max(10, blur_px))

    else:
        # General Edit: 입력값 최대한 존중 (안전 범위 clamp만)
        expand_px = int(expand_px)
        expand_px = min(40, max(0, expand_px))

        blur_px = int(blur_px)
        blur_px = min(40, max(0, blur_px))

    # 조정된 값 로그 (디버깅/튜닝에 도움)
    print(f"[APPLY] mode={mode} steps={steps} strength={strength:.2f} cfg={guidance:.1f} expand={expand_px} blur={blur_px}")   

    # 8) Diffusers inpaint 실행
    t0 = time.time()
    try:
        image_pil = STATE["working_pil"].convert("RGB")

        # diffusers mask는 L 모드 권장 (0=keep, 255=paint)
        mask_pil = Image.fromarray(mask_pp).convert("L")

        # 실행
        out = PIPE(
            prompt=expanded,
            negative_prompt=merged_negative if merged_negative else None,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            strength=float(strength),
            generator=gen,
        )

        # diffusers 결과는 보통 out.images[0]
        result = out.images[0] if hasattr(out, "images") and out.images else out[0]

        dt = time.time() - t0
        run_msg = f"Done in {dt:.2f}s | mode={edit_mode} | steps={steps} strength={strength:.2f} cfg={guidance:.1f} | expand={expand_px} blur={blur_px}"
        global_msg = f"Apply OK | mode={edit_mode} auto_enrich={auto_enrich_flag}"

        return result, run_msg, expanded, global_msg

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return None, f"Inpaint failed:\n{tb}", expanded, "Error: diffusion run failed."

# -----------------------------------------------------------------------------
# UI (modernized)
# -----------------------------------------------------------------------------

# -----------------------------
# CSS (모던/깔끔)
# -----------------------------
CSS = """
:root{
  --radius: 14px;
  --border: rgba(0,0,0,0.10);
}
.gradio-container { max-width: 1280px !important; }

.card{
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  box-shadow: 0 6px 24px rgba(0,0,0,0.06);
  padding: 10px;
}

/* Gradio 내부 wrapper 구조 차이 대비 */
.card > .wrap, .card > .block, .card > div{
  border-radius: var(--radius) !important;
}

.section-title{
  font-size: 14px;
  font-weight: 650;
  opacity: 0.9;
  margin: 4px 0 8px 0;
}

label, .gr-label{
  font-size: 12.5px !important;
  opacity: 0.9;
}

textarea, input{
  font-size: 13px !important;
  line-height: 1.35 !important;
}
"""

# -----------------------------
# UI Builder
# -----------------------------
def build_ui():
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size="lg",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    with gr.Blocks(css=CSS, theme=theme, title="ImageAI Inpaint Lab") as demo:
        gr.Markdown("## ImageAI Inpaint Lab")

        # ===== Top: Status =====
        with gr.Row():
            global_status = gr.Textbox(
                label="Status / Logs",
                value="Ready.",
                interactive=False,
                lines=2,
                max_lines=6,
                elem_classes=["card"],
            )

        # ===== Main Layout =====
        with gr.Row():
            # ---------------- LEFT: Image + Mask Preview ----------------
            with gr.Column(scale=7):
                with gr.Group(elem_classes=["card"]):
                    gr.Markdown('<div class="section-title">Working</div>')
                    input_image = gr.Image(
                        label="Working Image (upload)",
                        type="pil",
                        height=520,
                    )

                with gr.Group(elem_classes=["card"]):
                    gr.Markdown('<div class="section-title">Mask (selected)</div>')
                    # overlay (원본 위에 마스크 표시)
                    mask_overlay = gr.Image(
                        label="Mask Overlay",
                        type="numpy",
                        height=260,
                    )
                    selected_mask_preview = gr.Image(
                        label="Selected Mask Preview",
                        type="numpy",
                        height=420,
                    )

            # ---------------- RIGHT: Auto Mask + Prompt + Controls ----------------
            with gr.Column(scale=5):
                with gr.Group(elem_classes=["card"]):
                    gr.Markdown('<div class="section-title">Auto Mask</div>')

                    # Auto candidates 크게
                    auto_gallery = gr.Gallery(
                        label="Auto Mask Candidates (v5 semantic)",
                        columns=4,
                        rows=1,
                        height=420,
                        allow_preview=True,
                        preview=True,
                    )

                    auto_status = gr.Textbox(
                        label="Auto Mask Status",
                        interactive=False,
                        lines=2,
                        max_lines=6
                    )

                    with gr.Row():
                        # NOTE: 기존엔 "Auto Mask (v5)" / "Clear Mask" 이런 버튼이었지
                        btn_auto = gr.Button("Auto Mask (v5)", variant="secondary")
                        btn_clear = gr.Button("Clear Mask", variant="secondary")

                with gr.Group(elem_classes=["card"]):
                    gr.Markdown('<div class="section-title">Prompt</div>')

                    #auto_enrich는 UI에 반드시 존재해야 함 (브라우저 null props 방지)
                    auto_enrich = gr.Checkbox(
                        label="Auto-enrich prompt (natural language → diffusion tokens)",
                        value=True,
                        interactive=True,
                    )

                    # (옵션) 편하게 모드 선택: remove/wear 품질 튜닝에 매우 도움됨
                    edit_mode = gr.Dropdown(
                        label="Edit Mode",
                        choices=["Wear / Change Clothes", "Remove Clothes", "General Edit"],
                        value="Wear / Change Clothes",
                        interactive=True,
                    )

                    prompt = gr.Textbox(
                        label="Prompt (Korean/English)",
                        placeholder="e.g. black tank top, sleeveless, photorealistic",
                        lines=3,
                        max_lines=10,
                        show_copy_button=True,
                    )

                    negative = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="e.g. extra limbs, blurry, plastic skin, artifacts ...",
                        lines=3,
                        max_lines=12,
                        show_copy_button=True,
                    )

                    #expanded_preview도 UI에 반드시 존재해야 함 (브라우저 null props 방지)
                    expanded_preview = gr.Textbox(
                        label="Expanded Prompt Preview (read-only)",
                        interactive=False,
                        lines=7,
                        max_lines=16,
                        show_copy_button=True,
                    )

                with gr.Group(elem_classes=["card"]):
                    gr.Markdown('<div class="section-title">Controls</div>')

                    # 기존에 있던 슬라이더들: 범위는 너 코드 기준 유지/조정
                    working_long_side = gr.Slider(
                        label="Working Long Side",
                        minimum=512,
                        maximum=1536,
                        step=64,
                        value=1024,
                    )

                    sam_model = gr.Dropdown(
                        label="SAM Model (speed vs accuracy)",
                        choices=["vit_b", "vit_l", "vit_h"],
                        value="vit_b",
                        interactive=True,
                    )

                    mask_expand = gr.Slider(
                        label="Mask Expand (px)",
                        minimum=0,
                        maximum=40,
                        step=1,
                        value=10,
                    )
                    mask_blur = gr.Slider(
                        label="Mask Blur (px)",
                        minimum=0,
                        maximum=40,
                        step=1,
                        value=18,
                    )

                    steps = gr.Slider(
                        label="Steps",
                        minimum=10,
                        maximum=60,
                        step=1,
                        value=28,
                    )
                    strength = gr.Slider(
                        label="Strength",
                        minimum=0.30,
                        maximum=0.95,
                        step=0.01,
                        value=0.55,
                    )
                    guidance = gr.Slider(
                        label="Guidance Scale (CFG)",
                        minimum=1.0,
                        maximum=12.0,
                        step=0.1,
                        value=7.0,
                    )
                    seed = gr.Number(
                        label="Seed (-1 = random)",
                        value=-1,
                        precision=0,
                    )

                    btn_apply = gr.Button("Apply", variant="primary")

                with gr.Group(elem_classes=["card"]):
                    gr.Markdown('<div class="section-title">Result</div>')
                    output = gr.Image(label="Result", type="pil", height=520)
                    run_status = gr.Textbox(label="Run Status", interactive=False, lines=2, max_lines=6)

    # ============================================================
    # Events
    # ============================================================
    # TODO(1): 네 app.py에 존재하는 함수 이름 그대로 유지해야 함
    #   - on_upload
    #   - on_manual_click
    #   - build_auto_candidates_v5  (또는 build_auto_candidates_v4/v5)
    #   - select_candidate
    #   - clear_mask
    #   - apply_inpaint

        # 업로드 처리
        input_image.upload(
            fn=on_upload,
            inputs=[input_image, working_long_side],
            outputs=[input_image, selected_mask_preview, auto_status, global_status],
        )

        # 수동 마스크(클릭)
        input_image.select(
            fn=on_manual_click,
            inputs=[sam_model],
            outputs=[mask_overlay, selected_mask_preview, auto_status, global_status],
        )

        # Auto mask 버튼
        # outputs에 expanded_preview 포함 (없으면 브라우저 에러 가능)
        btn_auto.click(
            fn=build_auto_candidates_v5,  # TODO(2): 네 실제 함수명으로 맞춰
            inputs=[prompt, auto_enrich, edit_mode],
            outputs=[auto_gallery, auto_status, expanded_preview],
        )

        # 갤러리에서 candidate 선택
        auto_gallery.select(
            fn=select_candidate,
            inputs=[],
            outputs=[mask_overlay, selected_mask_preview, auto_status, global_status],
        )

        # 마스크 클리어
        btn_clear.click(
            fn=clear_mask,
            inputs=None,
            outputs=[mask_overlay, selected_mask_preview, auto_gallery, auto_status, global_status],
        )

        # Apply (인페인트 실행)
        btn_apply.click(
            fn=apply_inpaint,
            inputs=[
                prompt, negative, steps, strength, guidance,
                mask_expand, mask_blur, seed, auto_enrich, edit_mode
            ],
            outputs=[output, run_status, expanded_preview, global_status],
        )

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
