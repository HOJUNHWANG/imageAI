# venv311\Scripts\activate
# app.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 에러 자세히 보기

import time
import gc
import psutil  # RAM 사용량 확인용
import re
import numpy as np
from PIL import Image, ImageFilter
import torch
import traceback

# Speed boost on RTX 30/40 (minimal quality impact for inference)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

import gradio as gr

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"]
)

from sam_utils import SamMaskerManager
from mp_tasks_utils import MPTasksHelper
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
PIPE = None  # 유지 (기존 코드 호환)
controlnet_pipes = {}
img2img_pipe = None

# -----------------------------------------------------------------------------
# VRAM / RAM helpers
# -----------------------------------------------------------------------------

def _gb(x: int) -> float:
    return float(x) / (1024 ** 3)

def get_vram_text() -> str:
    if DEVICE != "cuda" or (not torch.cuda.is_available()):
        ram = psutil.virtual_memory()
        return (
            f"GPU: (CPU mode)\n"
            f"RAM Used: {_gb(ram.used):.2f} GB / {_gb(ram.total):.2f} GB\n"
        )

    torch.cuda.synchronize()
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    alloc = torch.cuda.memory_allocated(dev)
    resv = torch.cuda.memory_reserved(dev)
    max_alloc = torch.cuda.max_memory_allocated(dev)
    max_resv = torch.cuda.max_memory_reserved(dev)
    total = torch.cuda.get_device_properties(dev).total_memory

    return (
        f"GPU: {name}\n"
        f"Allocated: {_gb(alloc):.2f} GB\n"
        f"Reserved:  {_gb(resv):.2f} GB\n"
        f"MaxAlloc:  {_gb(max_alloc):.2f} GB\n"
        f"MaxResv:   {_gb(max_resv):.2f} GB\n"
        f"Total:     {_gb(total):.2f} GB\n"
    )

def soft_clear_vram():
    """
    🧹 Soft Clear: 캐시만 비움 (모델은 GPU에 유지 → 2번째 속도 유지)
    """
    msg = "[VRAM][SOFT] CPU mode (no CUDA)."
    if DEVICE == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # ipc_collect는 드물게 도움됨(특히 메모리 파편화/보류된 블록)
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            msg = "[VRAM][SOFT] empty_cache() + ipc_collect() done."
        except Exception as e:
            msg = f"[VRAM][SOFT] failed: {type(e).__name__}: {e}"
    return msg, get_vram_text()

def hard_clear_vram():
    """
    🔥 Hard Clear: 모델/ControlNet/Refine 파이프를 GPU에서 내리고 객체 제거
    - OOM 복구 확실
    - 다음 생성은 모델 재로딩 때문에 느려짐
    """
    global pipe, PIPE, controlnet_pipes, img2img_pipe

    if DEVICE != "cuda" or (not torch.cuda.is_available()):
        # CPU 모드면 그냥 객체만 정리
        pipe = None
        PIPE = None
        controlnet_pipes.clear()
        img2img_pipe = None
        gc.collect()
        return "[VRAM][HARD] CPU mode cleanup done.", get_vram_text()

    try:
        torch.cuda.synchronize()

        # 1) 메인 파이프라인 unload
        if pipe is not None:
            try:
                pipe.to("cpu")
            except Exception:
                pass
        pipe = None
        PIPE = None

        # 2) ControlNet 파이프들 unload
        if isinstance(controlnet_pipes, dict) and controlnet_pipes:
            for k, p in list(controlnet_pipes.items()):
                try:
                    p.to("cpu")
                except Exception:
                    pass
            controlnet_pipes.clear()

        # 3) Refine(img2img) unload
        if img2img_pipe is not None:
            try:
                img2img_pipe.to("cpu")
            except Exception:
                pass
        img2img_pipe = None

        # 4) 파이썬/토치 정리
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

        msg = "[VRAM][HARD] Unloaded pipelines + cleared CUDA cache. (Next run will reload models)"
        return msg, get_vram_text()

    except Exception as e:
        return f"[VRAM][HARD] failed: {type(e).__name__}: {e}", get_vram_text()

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

def preview_enriched_prompt(prompt: str, negative: str, auto_enrich: bool, edit_mode: str):
    if not (prompt or "").strip():
        return "Positive prompt is required!"

    positive_final_pos = prompt.strip()
    positive_final_neg = (negative or "").strip()

    if auto_enrich:
        positive_final_pos, _ = enrich_positive(prompt)
        positive_final_neg = enrich_negative(positive_final_neg)

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
    global pipe, PIPE
    if MOCK_INPAINT:
        print("[PIPE] MOCK_INPAINT mode - no real model loaded")
        pipe = None
        PIPE = None
        return None

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model_path = JUGGERNAUT_INPAINT if os.path.exists(JUGGERNAUT_INPAINT) else DEFAULT_MODEL

    print(f"[PIPE] Loading checkpoint from: {model_path}")
    print(f"[PIPE] Target device: {DEVICE}")
    print(f"[PIPE] Using dtype: {dtype}")

    try:
        p = StableDiffusionXLInpaintPipeline.from_single_file(
            model_path,
            torch_dtype=dtype,
            variant="fp16" if "safetensors" in model_path.lower() and DEVICE == "cuda" else None,
            use_safetensors=True,
            safety_checker=None,
        )

        if DEVICE == "cuda":
            p.to("cuda")
            if hasattr(p, "disable_model_cpu_offload"):
                p.disable_model_cpu_offload()
            print("[GPU] Disabled cpu_offload - full GPU acceleration enabled")

            try:
                p.enable_xformers_memory_efficient_attention()
                print("[OPT] xformers enabled - faster attention")
            except Exception:
                print("[OPT] xformers not available (fallback)")
        else:
            p.to("cpu")
            print("[CPU] Running on CPU - generation will be slow")

        # Warm-up (너무 공격적이면 여기만 꺼도 됨)
        print("[PIPE] Running warm-up dummy inference...")
        try:
            dummy_image = Image.new("RGB", (512, 512), color="white")
            dummy_mask = Image.new("L", (512, 512), color=0)
            _ = p(
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

        pipe = p
        PIPE = p
        print("[PIPE] Loaded successfully!")
        print(f"[PIPE] Pipeline class: {type(pipe).__name__}")
        print(f"[PIPE] Scheduler: {type(pipe.scheduler).__name__}")
        print(f"[PIPE] Device: {pipe.device}")
        print(f"[PIPE] UNet dtype: {next(pipe.unet.parameters()).dtype}")

        return pipe

    except Exception as e:
        print(f"[PIPE] Load failed: {str(e)}")
        print("[PIPE] Check model path, safetensors file, or diffusers version")
        pipe = None
        PIPE = None
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
    global pipe, PIPE, controlnet_pipes, img2img_pipe

    t0 = time.time()

    used_seed_str = str(seed if seed is not None and int(seed) >= 0 else "random")
    if MOCK_INPAINT:
        return None, "MOCK_INPAINT mode", "", "Mock mode active", used_seed_str

    # Seed 기반 generator 생성
    gen = None
    if seed is not None and int(seed) >= 0:
        gen = torch.Generator(DEVICE).manual_seed(int(seed))

    # 마스크 가져오기
    mask_u8 = STATE.get("mask_u8")
    if mask_u8 is None:
        mask_u8 = STATE.get("selected_mask")

    if mask_u8 is None:
        return None, "Mask missing", "", "Error: No mask selected", used_seed_str

    # 마스크 postprocess
    mask_pp = postprocess_mask(mask_u8, int(expand_px), int(blur_px))
    mask_pil = Image.fromarray(mask_pp).convert("L")
    image_pil = STATE["working_pil"].convert("RGB")

    # 프롬프트 enrich
    positive_final = (prompt or "").strip()
    negative_final = (negative or "").strip()

    if auto_enrich:
        try:
            positive_final, _ = enrich_positive(prompt)
        except Exception as e:
            print(f"[WARN] Positive enrich failed: {e}")

        try:
            negative_final = enrich_negative(negative_final)
        except Exception as e:
            print(f"[WARN] Negative enrich failed: {e}")

    negative_final = comma_join_unique([negative_final, build_default_negative(edit_mode)])

    # 모드별 clamp (Remove Clothes)
    if edit_mode == "Remove Clothes":
        strength = min(max(float(strength), 0.45), 0.92)
        guidance = max(float(guidance), 7.0)

    steps = int(steps)
    strength = float(strength)
    guidance = float(guidance)

    # --- 간단 병목 프로파일링(로그용) ---
    prof = {}
    def mark(k):
        prof[k] = time.time()

    mark("start")

    result = None
    refined = None

    print(f"[START] Generation start | mode={edit_mode} | controlnet={use_controlnet} | steps={steps} | seed={used_seed_str}")

    # ControlNet 분기
    if use_controlnet:
        if controlnet_type == "depth":
            repo = CONTROLNET_DEPTH
        elif controlnet_type == "openpose":
            repo = CONTROLNET_OPENPOSE
        else:
            repo = "lllyasviel/control_v11p_sd15_inpaint"

        # NOTE: key에 strength 넣으면 캐시가 과하게 늘어남 → type만으로 캐시
        key = f"{controlnet_type}"

        if key not in controlnet_pipes:
            print(f"[CONTROLNET] Loading {controlnet_type} from: {repo}")
            mark("cn_load_start")
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
            mark("cn_load_end")

        if use_controlnet and key in controlnet_pipes:
            p = controlnet_pipes[key]
            print(f"[CONTROLNET] Using {controlnet_type} | type={type(p).__name__}")
            mark("cn_run_start")
            try:
                result = p(
                    prompt=positive_final,
                    negative_prompt=negative_final,
                    image=image_pil,
                    mask_image=mask_pil,
                    control_image=image_pil,
                    controlnet_conditioning_scale=0.65,
                    num_inference_steps=steps,
                    strength=strength,
                    guidance_scale=guidance,
                    generator=gen
                ).images[0]
                print("[CONTROLNET] Generation success!")
            except Exception as e:
                print(f"[CONTROLNET] Generation failed: {str(e)} → fallback to base")
                result = None
            mark("cn_run_end")

    # 기본 Inpaint
    if result is None:
        if pipe is None:
            load_pipe()
        mark("base_run_start")
        try:
            result = pipe(
                prompt=positive_final,
                negative_prompt=negative_final,
                image=image_pil,
                mask_image=mask_pil,
                num_inference_steps=steps,
                strength=strength,
                guidance_scale=guidance,
                generator=gen
            ).images[0]
            print("[INPAINT] Generation success!")
        except Exception as e:
            print(f"[INPAINT] Base generation failed: {str(e)}")
            return None, f"Generation failed: {str(e)}", positive_final, "Error", used_seed_str
        mark("base_run_end")

    # Refine pass
    if do_refine and result is not None:
        print("[REFINE] Refine pass 시작")
        mark("refine_load_start")
        try:
            if img2img_pipe is None:
                model_path = JUGGERNAUT_INPAINT if os.path.exists(JUGGERNAUT_INPAINT) else DEFAULT_MODEL
                img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    use_safetensors=True,
                )
                img2img_pipe.to(DEVICE)
                print("[REFINE] Img2Img pipeline loaded")
            mark("refine_load_end")

            mark("refine_run_start")
            refined = img2img_pipe(
                prompt=positive_final,
                image=result,
                strength=0.20,
                num_inference_steps=10,
                guidance_scale=guidance,
                generator=gen
            ).images[0]
            mark("refine_run_end")
            print("[REFINE] Refine completed")
        except Exception as e:
            print(f"[REFINE] Failed: {str(e)} → using first result")
            refined = result

    final_image = refined if do_refine and refined is not None else result

    dt = time.time() - t0

    # 프로파일 로그(대략 어디서 막히는지)
    def _dur(a, b):
        if a in prof and b in prof:
            return prof[b] - prof[a]
        return None

    prof_msg_parts = []
    d_cn_load = _dur("cn_load_start", "cn_load_end")
    d_cn_run = _dur("cn_run_start", "cn_run_end")
    d_base = _dur("base_run_start", "base_run_end")
    d_ref_load = _dur("refine_load_start", "refine_load_end")
    d_ref_run = _dur("refine_run_start", "refine_run_end")

    if d_cn_load is not None: prof_msg_parts.append(f"cn_load:{d_cn_load:.2f}s")
    if d_cn_run is not None:  prof_msg_parts.append(f"cn_run:{d_cn_run:.2f}s")
    if d_base is not None:    prof_msg_parts.append(f"base_run:{d_base:.2f}s")
    if d_ref_load is not None:prof_msg_parts.append(f"ref_load:{d_ref_load:.2f}s")
    if d_ref_run is not None: prof_msg_parts.append(f"ref_run:{d_ref_run:.2f}s")
    prof_msg = " | ".join(prof_msg_parts) if prof_msg_parts else "profile:n/a"

    model_info = f"Juggernaut XL | ControlNet: {use_controlnet} ({controlnet_type}) | Refine: {do_refine}"
    run_msg = f"완료! {model_info} | Time: {int(dt // 60)}m {int(dt % 60)}s"
    global_status = f"{run_msg}\n{prof_msg}"

    # (기본) 생성 후 Soft Clear는 유지
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("[VRAM] Cleared cache after generation")

    # seed 기록
    try:
        with open("last_seed.txt", "w") as f:
            f.write(used_seed_str)
    except Exception:
        pass

    return final_image, run_msg, positive_final, global_status, used_seed_str

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

CSS = """
:root{
  --radius: 14px;
  --border: rgba(0,0,0,0.10);
}
.gradio-container { max-width: 1280px !important; }
"""

def build_ui():
    with gr.Blocks(title="ImageAI Inpaint Lab") as demo:
        gr.Markdown("## ImageAI Inpaint Lab (Juggernaut XL + ControlNet)")

        # 상단: Status + VRAM + 버튼들
        with gr.Row():
            with gr.Column(scale=7):
                global_status = gr.Textbox(label="Status / Logs", value="Ready.", lines=3)
            with gr.Column(scale=5):
                vram_box = gr.Textbox(label="GPU VRAM", value=get_vram_text(), lines=7, interactive=False)
                with gr.Row():
                    btn_vram_refresh = gr.Button("VRAM Refresh")
                    btn_soft_clear = gr.Button("🧹 Soft Clear")
                    btn_hard_clear = gr.Button("🔥 Hard Clear")

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
                    use_controlnet = gr.Checkbox(label="Use ControlNet", value=False)
                    controlnet_type = gr.Dropdown(["depth", "openpose", "inpaint"], value="depth", label="ControlNet Type")
                    do_refine = gr.Checkbox(label="Refine Pass", value=False, interactive=True)
                    btn_apply = gr.Button("Apply", variant="primary")

                with gr.Group():
                    gr.Markdown("Result")
                    output = gr.Image(height=520)
                    run_status = gr.Textbox(lines=2)

        # Events
        input_image.upload(fn=on_upload, inputs=[input_image, working_long_side], outputs=[input_image, selected_mask_preview, auto_status, global_status])
        input_image.select(fn=on_manual_click, inputs=[sam_model], outputs=[mask_overlay, selected_mask_preview, auto_status, global_status])

        btn_auto.click(fn=build_auto_candidates_v5, inputs=[prompt, auto_enrich, edit_mode], outputs=[auto_gallery, auto_status, positive_final_preview])
        btn_clear.click(fn=clear_mask, outputs=[mask_overlay, selected_mask_preview, auto_gallery, auto_status])

        preview_btn.click(
            fn=preview_enriched_prompt,
            inputs=[prompt, negative, auto_enrich, edit_mode],
            outputs=[preview_output]
        )

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

        # VRAM buttons
        btn_vram_refresh.click(fn=lambda: (get_vram_text(),), inputs=None, outputs=[vram_box])
        btn_soft_clear.click(fn=soft_clear_vram, inputs=None, outputs=[global_status, vram_box])
        btn_hard_clear.click(fn=hard_clear_vram, inputs=None, outputs=[global_status, vram_box])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.queue(max_size=10)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        debug=True,
        share=False,
        prevent_thread_lock=True,
        css=CSS,
        theme=theme
    )
