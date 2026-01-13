import os
import numpy as np
import gradio as gr
import torch
from PIL import Image

from diffusers import AutoPipelineForInpainting

from sam_utils import SamMaskerManager
from image_utils import (
    to_rgb_pil,
    resize_long_side,
    postprocess_mask,
    mask_to_pil,
    overlay_mask_on_image,
    sam_dict_to_uint8_mask,
    bbox_center,
    clamp01,
)

# -----------------------------------------------------------------------------
# Runtime flags
# -----------------------------------------------------------------------------
MOCK_INPAINT = os.environ.get("MOCK_INPAINT", "0") == "1"
MODEL_ID = os.environ.get(
    "INPAINT_MODEL_ID",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
)

# -----------------------------------------------------------------------------
# Device / perf knobs
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True

# -----------------------------------------------------------------------------
# Global state
# -----------------------------------------------------------------------------
STATE = {
    "working_pil": None,
    "working_np": None,
    "mask_np": None,
    "auto_mask_candidates": [],  # list of uint8 masks
}

# -----------------------------------------------------------------------------
# SAM manager (vit_b/vit_h)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
sam_manager = SamMaskerManager(weights_dir=WEIGHTS_DIR, device=DEVICE)

# -----------------------------------------------------------------------------
# Inpaint pipeline
# -----------------------------------------------------------------------------
pipe = None
if not MOCK_INPAINT:
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    variant = "fp16" if DEVICE == "cuda" else None

    pipe = AutoPipelineForInpainting.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant=variant,
        use_safetensors=True,
    ).to(DEVICE)

    if DEVICE == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[OK] xformers enabled")
        except Exception as e:
            print("[WARN] xformers not available:", e)

    pipe.set_progress_bar_config(disable=True)

print(f"DEVICE={DEVICE}, MOCK_INPAINT={MOCK_INPAINT}, MODEL_ID={MODEL_ID}")


# -----------------------------------------------------------------------------
# Prompt parsing (minimal, rule-based)
# -----------------------------------------------------------------------------
def infer_edit_target(prompt_ko: str) -> str:
    """
    Extremely small rule-based intent classifier.

    returns one of:
      - "sleeve"  : 긴팔/반팔/소매 중심
      - "top"     : 상의/셔츠/자켓
      - "pants"   : 바지/하의
      - "dress"   : 드레스/치마
      - "generic" : fallback
    """
    p = (prompt_ko or "").lower()

    sleeve_keywords = ["긴팔", "반팔", "소매", "sleeve", "short sleeve", "long sleeve"]
    top_keywords = ["상의", "셔츠", "자켓", "재킷", "후드", "hoodie", "shirt", "jacket", "top"]
    pants_keywords = ["바지", "하의", "pants", "trousers", "jeans"]
    dress_keywords = ["드레스", "치마", "skirt", "dress"]

    if any(k in p for k in sleeve_keywords):
        return "sleeve"
    if any(k in p for k in top_keywords):
        return "top"
    if any(k in p for k in pants_keywords):
        return "pants"
    if any(k in p for k in dress_keywords):
        return "dress"
    return "generic"


# -----------------------------------------------------------------------------
# Auto mask scoring
# -----------------------------------------------------------------------------
def score_mask(target: str, mask_dict: dict, img_w: int, img_h: int) -> float:
    """
    Score SAM auto masks by heuristics.

    The goal is not "semantic correctness"; it's selecting a plausible region quickly.
    - target="sleeve": prefer left/right upper arm regions, moderate area, upper-half.
    - target="top": prefer upper torso region, moderate-large area.
    - target="pants": prefer lower-half region.
    - generic: prefer medium area regions near center.

    returns: score (higher is better)
    """
    area = float(mask_dict.get("area", 0))
    bbox = mask_dict.get("bbox", [0, 0, 0, 0])
    cx, cy = bbox_center(bbox)

    # normalize center
    nx = cx / max(1.0, img_w)
    ny = cy / max(1.0, img_h)

    # area ratio
    a = area / max(1.0, float(img_w * img_h))

    # basic area preference: avoid tiny specks and huge background-like masks
    area_score = 1.0 - abs(a - 0.12) / 0.12  # peak at ~12%
    area_score = clamp01(area_score)

    # position scores
    upper_score = 1.0 - abs(ny - 0.35) / 0.35  # peak upper third
    upper_score = clamp01(upper_score)

    lower_score = 1.0 - abs(ny - 0.75) / 0.25
    lower_score = clamp01(lower_score)

    center_score = 1.0 - abs(nx - 0.50) / 0.50
    center_score = clamp01(center_score)

    # sleeve: favor left/right mid positions (arms) and upper region
    arm_left = 1.0 - abs(nx - 0.25) / 0.25
    arm_right = 1.0 - abs(nx - 0.75) / 0.25
    arm_score = clamp01(max(arm_left, arm_right))

    if target == "sleeve":
        return (0.45 * upper_score) + (0.35 * arm_score) + (0.20 * area_score)
    if target == "top":
        return (0.50 * upper_score) + (0.30 * center_score) + (0.20 * area_score)
    if target == "pants":
        return (0.55 * lower_score) + (0.25 * center_score) + (0.20 * area_score)
    if target == "dress":
        # dress often spans mid-to-lower; bias to mid-lower and larger area
        dress_area = 1.0 - abs(a - 0.22) / 0.22
        dress_area = clamp01(dress_area)
        mid_lower = 1.0 - abs(ny - 0.60) / 0.40
        mid_lower = clamp01(mid_lower)
        return (0.50 * mid_lower) + (0.30 * dress_area) + (0.20 * center_score)

    return (0.40 * center_score) + (0.40 * area_score) + (0.20 * upper_score)


def build_auto_candidates(sam_model_type: str, prompt_ko: str, top_k: int = 3):
    """
    Generate K candidate masks via SAM auto generator and heuristic scoring.
    Stores candidates in STATE["auto_mask_candidates"] as uint8 masks 0/255.
    """
    if STATE["working_np"] is None:
        return [], "Upload an image first."

    img_np = STATE["working_np"]
    h, w = img_np.shape[:2]

    target = infer_edit_target(prompt_ko)

    masker = sam_manager.get(sam_model_type)
    masks = masker.auto_masks(img_np)

    # sort by heuristic score
    scored = []
    for m in masks:
        s = score_mask(target, m, w, h)
        scored.append((s, m))
    scored.sort(key=lambda x: x[0], reverse=True)

    candidates = []
    for s, m in scored[:top_k]:
        candidates.append(sam_dict_to_uint8_mask(m))

    STATE["auto_mask_candidates"] = candidates
    return candidates, f"Auto masks generated. target={target}, candidates={len(candidates)}"


# -----------------------------------------------------------------------------
# Handlers
# -----------------------------------------------------------------------------
def on_upload(img_pil: Image.Image, work_long_side: int):
    if img_pil is None:
        return None, None, None, "Upload an image."

    img_pil = to_rgb_pil(img_pil)
    working = resize_long_side(img_pil, int(work_long_side))
    working_np = np.array(working)

    STATE["working_pil"] = working
    STATE["working_np"] = working_np
    STATE["mask_np"] = None
    STATE["auto_mask_candidates"] = []

    return working, None, None, f"Loaded. Working size={working.size}."

def on_click(evt: gr.SelectData, sam_model_type: str, expand_px: int, blur_px: int):
    if STATE["working_np"] is None:
        return None, "Upload an image first."

    x, y = evt.index
    img_np = STATE["working_np"]

    masker = sam_manager.get(sam_model_type)
    raw_mask = masker.mask_from_click(img_np, int(x), int(y))

    mask = postprocess_mask(raw_mask, int(expand_px), int(blur_px))
    STATE["mask_np"] = mask
    return mask_to_pil(mask), f"Manual mask set at ({x},{y})."

def on_auto_mask(sam_model_type: str, prompt_ko: str):
    candidates, msg = build_auto_candidates(sam_model_type, prompt_ko, top_k=3)
    if not candidates:
        return None, None, None, msg

    # 후보를 이미지로 반환(최대 3개)
    c1 = mask_to_pil(candidates[0]) if len(candidates) > 0 else None
    c2 = mask_to_pil(candidates[1]) if len(candidates) > 1 else None
    c3 = mask_to_pil(candidates[2]) if len(candidates) > 2 else None
    return c1, c2, c3, msg

def on_select_candidate(idx: int):
    """
    Candidate 선택 버튼 -> STATE["mask_np"]에 후보 마스크를 적용.
    """
    cands = STATE.get("auto_mask_candidates", [])
    if idx < 0 or idx >= len(cands):
        return None, "No such candidate."
    STATE["mask_np"] = cands[idx]
    return mask_to_pil(cands[idx]), f"Selected auto mask candidate #{idx+1}."

def clear_mask():
    STATE["mask_np"] = None
    STATE["auto_mask_candidates"] = []
    return None, None, None, "Mask cleared."

@torch.inference_mode()
def apply_edit(
    prompt: str,
    negative_prompt: str,
    steps: int,
    strength: float,
    guidance_scale: float,
    expand_px: int,
    blur_px: int,
):
    if STATE["working_pil"] is None:
        return None, "Upload an image first."
    if STATE["mask_np"] is None:
        return None, "Create/select a mask first."

    image = STATE["working_pil"]
    mask_np = postprocess_mask(STATE["mask_np"], int(expand_px), int(blur_px))
    STATE["mask_np"] = mask_np
    mask_pil = mask_to_pil(mask_np)

    if MOCK_INPAINT:
        preview = overlay_mask_on_image(image, mask_np)
        return preview, "MOCK mode: returned mask overlay preview."

    if prompt is None or prompt.strip() == "":
        return None, "Prompt is empty."

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_pil,
        num_inference_steps=int(steps),
        strength=float(strength),
        guidance_scale=float(guidance_scale),
    ).images[0]

    return result, "Done."


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
with gr.Blocks(title="Local Image Edit (Manual + Auto Mask)") as demo:
    gr.Markdown(
        "# Local Image Edit (SAM + SDXL Inpaint)\n"
        "- Manual mask: click-based\n"
        "- Auto mask: prompt-based heuristic selection over SAM auto proposals\n"
        "- vit_b: faster, vit_h: more precise (slower)\n"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload", type="pil")
            status = gr.Textbox(label="Status", interactive=False)

            work_long_side = gr.Slider(512, 1536, value=1024, step=64, label="Working Long Side")

            sam_model_type = gr.Dropdown(
                choices=["vit_b", "vit_h"],
                value="vit_b",
                label="SAM Model (speed vs accuracy)",
            )

            expand_px = gr.Slider(0, 40, value=12, step=1, label="Mask Expand(px)")
            blur_px = gr.Slider(0, 40, value=8, step=1, label="Mask Blur(px)")

            prompt = gr.Textbox(
                label="Prompt (Korean/English)",
                placeholder="예: 긴팔을 반팔로 바꿔줘, realistic fabric, natural arm",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="blurry, distorted, bad anatomy, extra limbs, deformed, low quality",
            )

            with gr.Row():
                btn_auto = gr.Button("Auto Mask (from prompt)")
                btn_apply = gr.Button("Apply", variant="primary")
                btn_clear = gr.Button("Clear Mask")

            steps = gr.Slider(10, 60, value=28, step=1, label="Steps")
            strength = gr.Slider(0.3, 0.95, value=0.78, step=0.01, label="Strength")
            guidance_scale = gr.Slider(1.0, 12.0, value=6.0, step=0.1, label="Guidance Scale")

        with gr.Column(scale=1):
            working_view = gr.Image(label="Working Image (click for manual mask)", interactive=True)
            mask_preview = gr.Image(label="Selected Mask Preview")
            output_img = gr.Image(label="Result")

            gr.Markdown("### Auto Mask Candidates")
            cand1 = gr.Image(label="Candidate 1")
            cand2 = gr.Image(label="Candidate 2")
            cand3 = gr.Image(label="Candidate 3")
            with gr.Row():
                pick1 = gr.Button("Use #1")
                pick2 = gr.Button("Use #2")
                pick3 = gr.Button("Use #3")

    # Upload
    input_img.change(
        fn=on_upload,
        inputs=[input_img, work_long_side],
        outputs=[working_view, mask_preview, output_img, status],
    )

    # Manual click mask
    working_view.select(
        fn=on_click,
        inputs=[sam_model_type, expand_px, blur_px],
        outputs=[mask_preview, status],
    )

    # Auto mask
    btn_auto.click(
        fn=on_auto_mask,
        inputs=[sam_model_type, prompt],
        outputs=[cand1, cand2, cand3, status],
    )

    # Pick candidate
    pick1.click(fn=lambda: on_select_candidate(0), inputs=[], outputs=[mask_preview, status])
    pick2.click(fn=lambda: on_select_candidate(1), inputs=[], outputs=[mask_preview, status])
    pick3.click(fn=lambda: on_select_candidate(2), inputs=[], outputs=[mask_preview, status])

    # Apply
    btn_apply.click(
        fn=apply_edit,
        inputs=[prompt, negative_prompt, steps, strength, guidance_scale, expand_px, blur_px],
        outputs=[output_img, status],
    )

    # Clear
    btn_clear.click(
        fn=clear_mask,
        inputs=[],
        outputs=[mask_preview, cand1, cand2, status],  # cand3는 아래에서 따로 초기화
    )
    btn_clear.click(
        fn=lambda: (None, "Mask cleared."),
        inputs=[],
        outputs=[cand3, status],
    )

if __name__ == "__main__":
    demo.launch(share=True)