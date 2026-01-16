# venv311\Scripts\activate
# app.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 에러 자세히 보기
import time
import psutil  # RAM 사용량 확인용
import re
import numpy as np
from PIL import Image, ImageFilter
import torch
torch.cuda.empty_cache()
from diffusers import (
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline
)
from diffusers.utils import load_image
from prompt_enricher import enrich_positive, enrich_negative

# opencv 있으면 더 좋고, 없으면 PIL로만 동작
try:
    import cv2
except Exception:
    cv2 = None

import cv2
from PIL import Image
import gradio as gr
import gradio as gr
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"]
)
import torch
import traceback

from sam_utils import SamMaskerManager
from mp_tasks_utils import MPTasksHelper, build_sleeve_mask_v5_tasks, build_top_mask_v5_tasks

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
MODELS_DIR = os.path.join(BASE_DIR, "models", "stable-diffusion-xl")

JUGGERNAUT_INPAINT = os.path.join(MODELS_DIR, "juggernautXL_ragnarokBy.safetensors")
DEFAULT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

CONTROLNET_DEPTH = os.path.join(BASE_DIR, "models", "ControlNet", "controlnet-depth-sdxl-1.0")
CONTROLNET_OPENPOSE = os.path.join(BASE_DIR, "models", "ControlNet", "controlnet-openpose-sdxl-1.0")

MOCK_INPAINT = os.getenv("MOCK_INPAINT", "0") == "1"

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = pick_device()

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

print(f"DEVICE={DEVICE}, MOCK_INPAINT={MOCK_INPAINT}")

# -----------------------------------------------------------------------------
# Global state
# -----------------------------------------------------------------------------

STATE = {
    "working_pil": None,
    "working_np": None,
    "orig_pil": None,
    "mask_u8": None,
    "selected_mask": None,
    "auto_mask_candidates": []
}

# Global pipelines
pipe = None
controlnet_pipes = {}
img2img_pipe = None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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
    wear_block = ["bare chest", "nipples", "shirtless"]
    remove_block = ["shirt", "t-shirt", "top", "clothing", "fabric", "textile", "sweater", "hoodie", "jacket", "vest"]

    if mode == "Wear / Change Clothes":
        return comma_join_unique(base + wear_block)
    if mode == "Remove Clothes":
        return comma_join_unique(base + remove_block)
    return comma_join_unique(base)

def enrich_prompt(prompt: str, mode: str) -> str:
    p = normalize_space(prompt)
    if not p:
        return ""

    common_quality = [
        "photorealistic", "high detail", "realistic lighting", "sharp focus",
        "natural skin texture", "consistent skin tone", "clean edges", "8k"
    ]

    wear_tokens = [
        "realistic fabric texture", "natural folds", "subtle wrinkles",
        "proper garment seam", "correct neckline", "clean shoulder line"
    ]

    remove_tokens = [
        "natural shoulders and arms", "realistic skin texture", "subsurface scattering",
        "skin pores", "fine body hair", "consistent shading", "anatomically plausible torso",
        "soft natural shadows", "visible skin details"
    ]

    if mode == "Remove Clothes":
        remove_goal = [
            "shirtless", "no shirt", "no clothing on upper body",
            "bare male torso", "natural male chest", "natural body hair"
        ]
        return comma_join_unique([p] + remove_goal + remove_tokens + common_quality)

    if mode == "Wear / Change Clothes":
        keep_identity = [
            "keep face unchanged", "keep hairstyle unchanged", "keep background unchanged",
            "preserve original pose"
        ]
        return comma_join_unique([p] + wear_tokens + common_quality + keep_identity)

    return comma_join_unique([p] + common_quality)

def resize_to_long_side(pil: Image.Image, long_side: int) -> Image.Image:
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
    vis = image_rgb.copy()
    red = np.zeros_like(vis)
    red[..., 0] = 255
    alpha = 0.45
    m = (mask_u8 > 0)
    vis[m] = (vis[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)
    return vis

def postprocess_mask(mask_u8: np.ndarray, expand_px: int, blur_px: int) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8) * 255

    if expand_px > 0:
        k = expand_px * 2 + 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8), iterations=1) if cv2 is not None else m

    if blur_px > 0:
        k = blur_px * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0) if cv2 is not None else m

    return np.clip(m, 0, 255).astype(np.uint8)

def refine_mask_soft(mask_u8: np.ndarray, expand_px: int, blur_px: int) -> np.ndarray:
    if mask_u8 is None:
        return None

    m = mask_u8
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 127).astype(np.uint8) * 255

    # Expand (dilation)
    if expand_px > 0:
        if cv2 is not None:
            k = max(1, int(expand_px))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
            m = cv2.dilate(m, kernel, iterations=1)
        else:
            img = Image.fromarray(m)
            size = int(expand_px) * 2 + 1
            img = img.filter(ImageFilter.MaxFilter(size=size))
            m = np.array(img, dtype=np.uint8)

    # Blur / Feather (soft edge)
    if blur_px > 0:
        if cv2 is not None:
            sigma = max(0.1, blur_px / 2.0)
            m = cv2.GaussianBlur(m, (0, 0), sigmaX=sigma, sigmaY=sigma)
        else:
            img = Image.fromarray(m)
            img = img.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
            m = np.array(img, dtype=np.uint8)

    return m.astype(np.uint8)

def preview_enriched_prompt(prompt: str, negative: str, auto_enrich: bool, edit_mode: str):
    """
    Preview 버튼 클릭 시 enrich된 positive/negative만 계산해서 보여줌
    실제 생성은 하지 않음
    """
    if not prompt.strip():
        return "Positive prompt is required!"

    positive_final_pos = prompt.strip()
    positive_final_neg = (negative or "").strip()

    if auto_enrich:
        positive_final_pos, _ = enrich_positive(prompt)
        positive_final_neg = enrich_negative(positive_final_neg)

    # enrich 후에도 기본 negative 추가 (필요 시)
    positive_final_neg = comma_join_unique([positive_final_neg, build_default_negative(edit_mode)])

    preview_text = (
        f"Positive (enriched):\n{positive_final_pos}\n\n"
        f"Negative (enriched):\n{positive_final_neg}"
    )
    return preview_text

def parse_prompt_simple(prompt: str) -> dict:
    p = (prompt or "").lower()

    sleeve_kw = ["sleeve", "sleeveless", "tank", "crop top", "short sleeve", "long sleeve"]
    top_kw = ["shirt", "t-shirt", "top", "blouse", "jacket", "hoodie", "sweater"]

    if any(k in p for k in sleeve_kw):
        target = "sleeve"
    elif any(k in p for k in top_kw):
        target = "top"
    else:
        target = "top"

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

# -----------------------------------------------------------------------------
# Model loaders
# -----------------------------------------------------------------------------

def load_pipe():
    global pipe
    if MOCK_INPAINT:
        print("[PIPE] MOCK_INPAINT mode - no real model loaded")
        return None

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    model_path = JUGGERNAUT_INPAINT if os.path.exists(JUGGERNAUT_INPAINT) else DEFAULT_MODEL

    print(f"[PIPE] Loading checkpoint from: {model_path}")
    print(f"[PIPE] Target device: {DEVICE}")
    print(f"[PIPE] Using dtype: {dtype}")

    try:
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(
            model_path,
            torch_dtype=dtype,
            variant="fp16" if "safetensors" in model_path.lower() and DEVICE == "cuda" else None,
            use_safetensors=True,
            safety_checker=None,
        )

        # GPU 최적화: offload 비활성화
        if DEVICE == "cuda":
            pipe.to("cuda")
            if hasattr(pipe, "disable_model_cpu_offload"):
                pipe.disable_model_cpu_offload()
            print("[GPU] Disabled cpu_offload - full GPU acceleration enabled")

            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("[OPT] xformers enabled - faster attention")
            except Exception:
                print("[OPT] xformers not available (fallback)")

        else:
            pipe.to("cpu")
            print("[CPU] Running on CPU - generation will be slow")

        # 컴포넌트 이동
        pipe.vae.to(DEVICE)
        pipe.text_encoder.to(DEVICE)
        pipe.text_encoder_2.to(DEVICE)
        pipe.unet.to(DEVICE)

        # Warm-up
        print("[PIPE] Running warm-up dummy inference...")
        try:
            dummy_image = Image.new("RGB", (512, 512), color="white")
            dummy_mask = Image.new("L", (512, 512), color=0)
            _ = pipe(
                prompt="a photo of a cat",
                image=dummy_image,
                mask_image=dummy_mask,
                num_inference_steps=1,
                strength=0.01,
                guidance_scale=1.0
            ).images[0]
            print("[PIPE] Warm-up success! GPU/VRAM ready")
        except Exception as warm_up_e:
            print(f"[PIPE] Warm-up failed (non-fatal): {str(warm_up_e)}")

        # 최종 상태
        print("[PIPE] Loaded successfully!")
        print(f"[PIPE] Pipeline class: {type(pipe).__name__}")
        print(f"[PIPE] Scheduler: {type(pipe.scheduler).__name__}")
        print(f"[PIPE] Device map: {pipe.device}")
        print(f"[PIPE] UNet dtype: {next(pipe.unet.parameters()).dtype}")

        # xformers (GPU에서만)
        if DEVICE == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("[OPT] xformers enabled - faster attention")
            except Exception:
                print("[OPT] xformers not available (fallback to default)")

        return pipe

    except Exception as e:
        print(f"[PIPE] Load failed: {str(e)}")
        print("[PIPE] Check model path, safetensors file, or diffusers version")
        return None

PIPE = load_pipe()

mp_helper = None
try:
    mp_helper = MPTasksHelper(weights_dir=WEIGHTS_DIR)
except Exception as e:
    print(f"[BOOT] MP init failed: {e}")

sam_manager = SamMaskerManager(weights_dir=WEIGHTS_DIR, device=DEVICE)

# -----------------------------------------------------------------------------
# Gradio callbacks
# -----------------------------------------------------------------------------

def on_upload(img: Image.Image, working_long_side: int):
    if img is None:
        return None, None, "Upload an image first.", "Error: No image uploaded."

    orig = img.convert("RGB")
    working = resize_to_long_side(orig, int(working_long_side))

    STATE["orig_pil"] = orig
    STATE["working_pil"] = working
    STATE["working_np"] = to_rgb_np(working)
    STATE["mask_u8"] = None
    STATE["selected_mask"] = None
    STATE["auto_mask_candidates"] = []

    status_msg = "Image loaded. Choose Auto Mask or click for Manual Mask."
    return working, None, status_msg, status_msg

def on_manual_click(evt: gr.SelectData, sam_model_type: str):
    if STATE["working_np"] is None:
        return None, None, "Upload an image first.", "Error: No working image."

    x, y = evt.index[0], evt.index[1]

    masker = sam_manager.get(sam_model_type)
    mask_u8 = masker.predict_from_click(STATE["working_np"], x, y)

    STATE["mask_u8"] = mask_u8
    STATE["selected_mask"] = mask_u8

    vis = overlay_mask(STATE["working_np"], mask_u8)
    mask_preview = Image.fromarray(mask_u8)

    status_msg = f"Manual mask built (SAM {sam_model_type})."
    return Image.fromarray(vis), mask_preview, status_msg, status_msg

def build_auto_candidates_v5(prompt: str, auto_enrich: bool, edit_mode: str):
    t0 = time.time()

    user_prompt = normalize_space(prompt)
    auto_enrich_flag = bool(auto_enrich)

    if STATE.get("working_pil") is None or STATE.get("working_np") is None:
        return [], "Upload an image first.", ""

    try:
        positive_final = user_prompt
        if auto_enrich_flag:
            positive_final, _info = enrich_positive(user_prompt)
        positive_final = normalize_space(positive_final)
    except Exception as e:
        return [], str(e), ""

    try:
        info = parse_prompt_simple(positive_final)
        target = info.get("target", "top")
    except Exception as e:
        return [], str(e), positive_final

    if mp_helper is None:
        return [], "mp_helper unavailable", positive_final

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
            c = build_top_mask_v5_tasks(STATE["working_pil"], mp_helper)

        if c.ndim == 3 and c.shape[-1] == 1:
            c = c[..., 0]

        if c.dtype != np.uint8:
            c = c.astype(np.uint8)

        c = np.where(c > 127, 255, 0).astype(np.uint8)

        STATE["auto_mask_candidates"] = [c]
        STATE["selected_mask"] = c  # 자동 선택

        dt = time.time() - t0
        return [Image.fromarray(c)], f"v5 OK target={target} time={dt:.2f}s", positive_final

    except Exception as e:
        return [], str(e), positive_final

def select_candidate(evt: gr.SelectData):
    idx = evt.index
    if idx is None or not STATE["auto_mask_candidates"]:
        return None, None, "No selection.", "No selection."

    mask_u8 = STATE["auto_mask_candidates"][idx]
    STATE["mask_u8"] = mask_u8
    STATE["selected_mask"] = mask_u8

    vis = overlay_mask(STATE["working_np"], mask_u8)
    mask_preview = Image.fromarray(mask_u8)

    status_msg = f"Candidate {idx+1} selected as active mask."
    return Image.fromarray(vis), mask_preview, status_msg, status_msg

def clear_mask():
    STATE["mask_u8"] = None
    STATE["selected_mask"] = None
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
    use_controlnet: bool = False,
    controlnet_type: str = "depth",
    do_refine: bool = False
):
    import time
    t0 = time.time()

    if MOCK_INPAINT:
        return None, "MOCK_INPAINT mode", "", "Mock mode active"

    # 변수 초기화 (모든 에러 방지)
    result = None
    refined = None
    gen = None  # generator도 초기화
    positive_final = (prompt or "").strip()
    negative_final = (negative or "").strip()
    expanded_preview = positive_final  # UI에 보여줄 기본값

    # Seed 기반 generator 생성
    if seed >= 0:
        gen = torch.Generator(DEVICE).manual_seed(seed)

    # 마스크 안전하게 가져오기
    mask_u8 = STATE.get("mask_u8")
    if mask_u8 is None:
        mask_u8 = STATE.get("selected_mask")

    if mask_u8 is None:
        return None, "Mask missing", "", "Error: No mask selected"

    # 마스크 postprocess
    mask_pp = postprocess_mask(mask_u8, int(expand_px), int(blur_px))
    mask_pil = Image.fromarray(mask_pp).convert("L")

    image_pil = STATE["working_pil"].convert("RGB")

    # 프롬프트 enrich (Apply 시점 실시간 적용)
    if auto_enrich:
        try:
            positive_final, _ = enrich_positive(prompt)
            expanded_preview = positive_final  # Preview에 enrich된 버전 보여줌
        except Exception as e:
            print(f"[WARN] Positive enrich failed: {e}")

        try:
            negative_final = enrich_negative(negative_final)
        except Exception as e:
            print(f"[WARN] Negative enrich failed: {e}")

    # 기본 negative 항상 추가
    negative_final = comma_join_unique([negative_final, build_default_negative(edit_mode)])

    # 모드별 clamp (Remove Clothes)
    if edit_mode == "Remove Clothes":
        strength = min(max(strength, 0.45), 0.92)
        guidance = max(guidance, 7.0)
        blur_px = max(blur_px, 12)

    print(f"[START] Generation start | mode={edit_mode} | controlnet={use_controlnet} | steps={steps}")

    # ControlNet 분기
    if use_controlnet:
        if controlnet_type == "depth":
            repo = CONTROLNET_DEPTH
        elif controlnet_type == "openpose":
            repo = CONTROLNET_OPENPOSE
        else:
            repo = "lllyasviel/control_v11p_sd15_inpaint"

        key = f"{controlnet_type}_{strength:.2f}"
        if key not in controlnet_pipes:
            print(f"[CONTROLNET] Loading {controlnet_type} from: {repo}")
            try:
                controlnet = ControlNetModel.from_pretrained(
                    repo,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    use_safetensors=True,
                    local_files_only=True
                )
                controlnet_pipes[key] = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
                    JUGGERNAUT_INPAINT,
                    controlnet=controlnet,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    use_safetensors=True,
                )
                controlnet_pipes[key].to(DEVICE)
                print(f"[CONTROLNET] Loaded successfully on {DEVICE}")
            except Exception as e:
                print(f"[CONTROLNET] Load failed: {type(e).__name__}: {str(e)}")
                use_controlnet = False

        if use_controlnet and key in controlnet_pipes:
            p = controlnet_pipes[key]
            print(f"[CONTROLNET] Using {controlnet_type} | type={type(p).__name__}")

            control_image = image_pil

            try:
                result = p(
                    prompt=positive_final,
                    negative_prompt=negative_final,
                    image=image_pil,
                    mask_image=mask_pil,
                    control_image=control_image,
                    controlnet_conditioning_scale=0.65,
                    num_inference_steps=int(steps),  # ← int로 강제
                    strength=strength,
                    guidance_scale=guidance,
                    generator=gen
                ).images[0]
                print("[CONTROLNET] Generation success!")
            except Exception as e:
                print(f"[CONTROLNET] Generation failed: {str(e)} → fallback to base")
                result = None

    # ControlNet 실패 시 기본 Inpaint
    if result is None:
        if pipe is None:
            load_pipe()

        print("[INPAINT] Using base Juggernaut XL Inpainting")
        try:
            result = pipe(
            prompt=positive_final,
            negative_prompt=negative_final,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=int(steps),  # ← int로 강제
            strength=strength,
            guidance_scale=guidance,
            generator=gen
        ).images[0]
            print("[INPAINT] Generation success!")
        except Exception as e:
            print(f"[INPAINT] Base generation failed: {str(e)}")
            return None, f"Generation failed: {str(e)}", positive_final, "Error"

    # Refine pass
        # Refine pass
    if do_refine and result is not None:
        print("[REFINE] Refine pass 시작")
        try:
            global img2img_pipe  # ← global 선언 (함수 내에서 수정 가능)
            if img2img_pipe is None:
                model_path = JUGGERNAUT_INPAINT if os.path.exists(JUGGERNAUT_INPAINT) else DEFAULT_MODEL
                img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    use_safetensors=True,
                )
                img2img_pipe.to(DEVICE)
                print("[REFINE] Img2Img pipeline loaded")

            refined = img2img_pipe(
                prompt=positive_final,
                image=result,
                strength=0.20,
                num_inference_steps=10,
                guidance_scale=guidance,
                generator=gen
            ).images[0]
            print("[REFINE] Refine completed")
        except Exception as e:
            print(f"[REFINE] Failed: {str(e)} → using first result")
            refined = result

    # 시간 계산 & 반환
    dt = time.time() - t0
    steps = int(steps)  # float → int 강제 변환 (20.1 → 20)
    strength = float(strength)  # strength는 float 그대로 OK
    final_image = refined if do_refine and refined is not None else result
    model_info = f"Juggernaut XL | ControlNet: {use_controlnet} ({controlnet_type}) | Refine: {do_refine}"
    run_msg = f"완료! {model_info} | Time: {int(dt // 60)}m {int(dt % 60)}s"

    # 1. VRAM 강제 청소 (OOM 방지)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # 더 확실하게 동기화
        print("[VRAM] Cleared cache after generation")

    # 2. 마지막 seed 기록 (재실행해도 날아가지 않게)
    with open("last_seed.txt", "w") as f:
        f.write(str(seed if seed >= 0 else "random"))

    # 3. (필요 시) 메모리 상태 로그
    print(f"[MEMORY] VRAM after clear: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    return final_image, run_msg, positive_final, model_info

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

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

.section-title{
  font-size: 14px;
  font-weight: 650;
  opacity: 0.9;
  margin: 4px 0 8px 0;
}
"""

def build_ui():
    with gr.Blocks(title="ImageAI Inpaint Lab") as demo:
        gr.Markdown("## ImageAI Inpaint Lab (Juggernaut XL + ControlNet)")

        with gr.Row():
            global_status = gr.Textbox(label="Status / Logs", value="Ready.", lines=2)

        with gr.Row():
            with gr.Column(scale=7):
                with gr.Group():
                    gr.Markdown("Working")
                    input_image = gr.Image(type="pil", height=520)

                with gr.Group():
                    gr.Markdown("Mask (selected)")
                    mask_overlay = gr.Image(type="numpy", height=260)
                    selected_mask_preview = gr.Image(type="numpy", height=420)

            with gr.Column(scale=5):
                with gr.Group():
                    gr.Markdown("Auto Mask")
                    auto_gallery = gr.Gallery(columns=4, height=420)
                    auto_status = gr.Textbox(lines=2)
                    with gr.Row():
                        btn_auto = gr.Button("Auto Mask (v5)")
                        btn_clear = gr.Button("Clear Mask")

                with gr.Group():
                    gr.Markdown("Prompt")
                    auto_enrich = gr.Checkbox(value=True, label="Auto-enrich prompt")
                    edit_mode = gr.Dropdown(
                        choices=["Wear / Change Clothes", "Remove Clothes", "General Edit"],
                        value="Remove Clothes"
                    )
                    prompt = gr.Textbox(lines=3, label="Positive Prompt")
                    negative = gr.Textbox(lines=3, label="Negative Prompt")
                    with gr.Row():
                        preview_btn = gr.Button("Preview Enriched Prompt", variant="secondary")
                        preview_output = gr.Textbox(
                            label="Preview (enriched prompt - Apply 전에 확인용)",
                            lines=8,
                            interactive=False,
                            placeholder="여기에 enrich된 프롬프트가 미리 표시됩니다"
                        )
                    positive_final_preview = gr.Textbox(lines=7, interactive=False, label="positive_final Prompt")

                with gr.Group():
                    gr.Markdown("Controls")
                    # Controls 그룹 안
                    working_long_side = gr.Slider(512, 1536, value=1024, step=64, label="Working Long Side (px)")
                    sam_model = gr.Dropdown(["vit_b", "vit_h"], value="vit_b", label="SAM Model (Manual Mask)")
                    mask_expand = gr.Slider(0, 40, value=10, label="Mask Expand (px) - blending 범위")
                    mask_blur = gr.Slider(0, 40, value=18, label="Mask Blur (px) - 부드러운 경계")
                    steps = gr.Slider(10, 60, value=28, label="Inference Steps")
                    strength = gr.Slider(0.3, 0.95, value=0.55, label="Strength (변경 강도)")
                    guidance = gr.Slider(1.0, 12.0, value=7.0, label="Guidance Scale (CFG) - 프롬프트 준수도")
                    seed = gr.Number(value=-1, label="Seed (-1 = random)")
                    seed_display = gr.Textbox(
                        label="Used Seed (생성 후 표시)",
                        interactive=False,
                        placeholder="생성 후 seed 값이 여기에 표시됩니다"
                    )
                    use_controlnet = gr.Checkbox(label="Use ControlNet (경계 자연스러움 향상, CPU 느림)", value=False)
                    controlnet_type = gr.Dropdown(["depth", "openpose", "inpaint"], value="depth", label="ControlNet Type")
                    do_refine = gr.Checkbox(label="Refine Pass (추가 10~30분, 퀄리티 ↑, CPU 느림)", value=False, interactive=True)
                    btn_apply = gr.Button("Apply", variant="primary")


                with gr.Group():
                    gr.Markdown("Result")
                    output = gr.Image(height=520)
                    run_status = gr.Textbox(lines=2)

        # Events
        input_image.upload(fn=on_upload, inputs=[input_image, working_long_side], outputs=[input_image, selected_mask_preview, auto_status, global_status])
        input_image.select(fn=on_manual_click, inputs=[sam_model], outputs=[mask_overlay, selected_mask_preview, auto_status, global_status])
        btn_auto.click(fn=build_auto_candidates_v5, inputs=[prompt, auto_enrich, edit_mode], outputs=[auto_gallery, auto_status, positive_final_preview])
        #auto_gallery.select(fn=select_candidate, outputs=[mask_overlay, selected_mask_preview, auto_status, global_status])
        btn_clear.click(fn=clear_mask, outputs=[mask_overlay, selected_mask_preview, auto_gallery, auto_status, global_status])

        btn_apply.click(
            fn=apply_inpaint,
            inputs=[
                prompt, negative, steps, strength, guidance,
                mask_expand, mask_blur, seed, auto_enrich, edit_mode,
                use_controlnet, controlnet_type,
                do_refine 
            ],
            outputs=[output, run_status, positive_final_preview, global_status, seed_display]
        )

        preview_btn.click(
            fn=preview_enriched_prompt,
            inputs=[prompt, negative, auto_enrich, edit_mode],
            outputs=[preview_output]
        )

    return demo

if __name__ == "__main__":
    demo = build_ui()
    # queue()에 timeout 관련 옵션 제거 (Gradio 6.0 호환)
    demo.queue(
        max_size=10,  # 큐 크기 제한 (대기열 관리용)
        # concurrency_count, timeout 등 제거
    )

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        debug=True,
        share=False,
        prevent_thread_lock=True,
        css=CSS,
        theme=theme
    )   