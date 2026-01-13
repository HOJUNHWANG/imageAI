# app.py
import os
import time
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import torch
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

expanded_preview = gr.Textbox(label="Expanded Prompt Preview (read-only)", interactive=False)


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
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )

    pipe = pipe.to(DEVICE)

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

def build_auto_candidates_v5(user_prompt: str, auto_enrich_flag: bool | None):
    auto_enrich_flag = bool(auto_enrich_flag)  # None -> False 로 강제
    t0 = time.time()
    try:
        print("\n[AUTO] called")
        print("[AUTO] prompt head:", (user_prompt or "")[:160])
        print("[AUTO] auto_enrich:", auto_enrich_flag)

        if STATE["working_pil"] is None or STATE["working_np"] is None:
            print("[AUTO] no image in STATE")
            return [], "Upload an image first.", ""

        print("[AUTO] working_pil size:", STATE["working_pil"].size)
        print("[AUTO] working_np shape:", STATE["working_np"].shape)

        # Prompt enrichment
        expanded = user_prompt
        if auto_enrich_flag:
            expanded, info = enrich_positive(user_prompt)
            target = info.target
        else:
            info = parse_prompt_simple(user_prompt)
            target = info["target"]

        print("[AUTO] target:", target)
        print("[AUTO] expanded head:", (expanded or "")[:160])

        if mp_helper is None:
            print("[AUTO] mp_helper NONE:", MP_INIT_ERROR)
            return [], f"mp_helper unavailable: {MP_INIT_ERROR}", expanded

        # Build mask
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

        print("[AUTO] mask dtype/shape:", c.dtype, c.shape, "unique:", np.unique(c)[:10])

        STATE["auto_mask_candidates"] = [c]
        dt = time.time() - t0
        return [Image.fromarray(c)], f"v5(mp-tasks) OK target={target} time={dt:.2f}s", expanded

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        # Status에 traceback을 넣어 UI에서 바로 보이게 함
        return [], tb, ""


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
    negative_prompt: str,
    steps: int,
    strength: float,
    guidance: float,
    expand_px: int,
    blur_px: int,
    seed: int,
    auto_enrich_flag: bool, ):

    auto_enrich_flag = bool(auto_enrich_flag)

    if STATE["working_pil"] is None or STATE["working_np"] is None:
        return None, "Upload an image first."

    if STATE["mask_u8"] is None:
        return None, "Create or select a mask first."

    # Post-process mask with user controls
    mask_pp = postprocess_mask(STATE["mask_u8"], expand_px=int(expand_px), blur_px=int(blur_px))

    if MOCK_INPAINT:
        # Fast local dev path (CPU-only laptops etc.)
        # Returns an overlay as a placeholder “result”.
        vis = overlay_mask(STATE["working_np"], (mask_pp > 0).astype(np.uint8) * 255)
        return Image.fromarray(vis), "MOCK_INPAINT=1 (no diffusion run)."

    if PIPE is None:
        return None, "Pipeline not loaded."

    gen = None
    if seed >= 0:
        gen = torch.Generator(device=DEVICE).manual_seed(int(seed))

    t0 = time.time()

    # ------------------------------------------------------------
    # Prompt enrichment:
    # - If enabled, convert natural language into diffusion-friendly tokens.
    # - Always normalize negative prompt so basic safety/quality tokens exist.
    # ------------------------------------------------------------
    pos_prompt = prompt
    neg_prompt = negative_prompt

    if auto_enrich_flag:
        # Returns (expanded_prompt, parse_info)
        pos_prompt, _info = enrich_positive(prompt)

    # Always ensure negative prompt has core “quality blockers”
    neg_prompt = enrich_negative(neg_prompt)

    result = PIPE(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        image=STATE["working_pil"],
        mask_image=Image.fromarray(mask_pp),
        num_inference_steps=int(steps),
        strength=float(strength),
        guidance_scale=float(guidance),
        generator=gen,
    ).images[0]

    dt = time.time() - t0
    return result, f"Done. time={dt:.2f}s, steps={steps}, strength={strength:.2f}, cfg={guidance:.1f}"

# -----------------------------------------------------------------------------
# UI (modernized)
# -----------------------------------------------------------------------------

CSS = """
:root {
  --radius: 14px;
}

#app-wrap { max-width: 1200px; margin: 0 auto; }
.header {
  padding: 14px 18px;
  border-radius: var(--radius);
  background: linear-gradient(90deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
}
.header h1 { margin: 0; font-size: 22px; }
.header p { margin: 6px 0 0; opacity: 0.8; font-size: 13px; }

.card {
  border-radius: var(--radius) !important;
}
"""

theme = gr.themes.Soft()

with gr.Blocks(title=APP_TITLE, theme=theme, css=CSS) as demo:
    with gr.Column(elem_id="app-wrap"):
        gr.HTML(f"""
        <div class="header">
          <h1>{APP_TITLE}</h1>
          <p>{APP_SUBTITLE}</p>
        </div>
        """)

        with gr.Accordion("How to / Parameter Guide", open=True):
            gr.Markdown(HOW_TO_MD)
            gr.Markdown(SLIDER_HINTS_MD)

        with gr.Row():
            # Left: inputs & masks
            with gr.Column(scale=6):
                with gr.Group(elem_classes=["card"]):
                    working_long_side = gr.Slider(
                        label="Working Long Side",
                        minimum=512,
                        maximum=1536,
                        step=64,
                        value=1024,
                    )

                    input_image = gr.Image(
                        label="Working Image (click for manual mask)",
                        type="pil",
                        interactive=True
                    )

                    mask_overlay = gr.Image(label="Overlay Preview", type="pil")
                    selected_mask_preview = gr.Image(label="Selected Mask Preview", type="pil")

                with gr.Group(elem_classes=["card"]):
                    gr.Markdown("### Auto Mask Candidates (v5 semantic)")
                    auto_gallery = gr.Gallery(
                        label="Candidates",
                        columns=3,
                        rows=1,
                        height=260,
                        allow_preview=True
                    )
                    auto_status = gr.Textbox(label="Status", interactive=False)

            # Right: controls & output
            with gr.Column(scale=6):
                with gr.Group(elem_classes=["card"]):
                    sam_model = gr.Dropdown(
                        choices=["vit_b", "vit_h"],
                        value="vit_b",
                        label="SAM Model (speed vs accuracy)",
                    )

                    mask_expand = gr.Slider(label="Mask Expand (px)", minimum=0, maximum=40, step=1, value=18)
                    mask_blur = gr.Slider(label="Mask Blur (px)", minimum=0, maximum=40, step=1, value=10)

                    prompt = gr.Textbox(
                        label="Prompt (Korean/English)",
                        value="black tank top, sleeveless, natural shoulders and arms, realistic skin texture, realistic fabric, photorealistic"
                    )
                    negative = gr.Textbox(
                        label="Negative Prompt",
                        value="bad anatomy, extra arms, extra hands, deformed, blurry, artifacts, low quality"
                    )

                    with gr.Row():
                        btn_auto = gr.Button("Auto Mask (v5)", variant="secondary")
                        btn_clear = gr.Button("Clear Mask", variant="secondary")

                with gr.Group(elem_classes=["card"]):
                    steps = gr.Slider(label="Steps", minimum=10, maximum=60, step=1, value=32)
                    strength = gr.Slider(label="Strength", minimum=0.30, maximum=0.95, step=0.01, value=0.86)
                    guidance = gr.Slider(label="Guidance Scale (CFG)", minimum=1.0, maximum=12.0, step=0.1, value=6.5)
                    seed = gr.Number(label="Seed (-1=random)", value=-1, precision=0)

                    btn_apply = gr.Button("Apply", variant="primary")

                    output = gr.Image(label="Result", type="pil")
                    run_status = gr.Textbox(label="Run Status", interactive=False)

        # Events
        input_image.upload(on_upload, inputs=[input_image, working_long_side], outputs=[input_image, selected_mask_preview, auto_status])

        input_image.select(on_manual_click, inputs=[sam_model], outputs=[mask_overlay, selected_mask_preview, auto_status])

        btn_auto.click(
            fn=build_auto_candidates_v5,
            inputs=[prompt, auto_enrich],
            outputs=[auto_gallery, auto_status, expanded_preview],
        )

        auto_gallery.select(select_candidate, inputs=None, outputs=[mask_overlay, auto_status])

        btn_clear.click(clear_mask, inputs=None, outputs=[mask_overlay, selected_mask_preview, auto_gallery, auto_status])

        btn_apply.click(
            apply_inpaint,
            inputs=[prompt, negative, steps, strength, guidance, mask_expand, mask_blur, seed, auto_enrich],
            outputs=[output, run_status]
        )



if __name__ == "__main__":
    demo.launch(show_error=True)