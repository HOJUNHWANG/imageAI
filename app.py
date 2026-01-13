import os
import numpy as np
import gradio as gr
import torch
from PIL import Image
import re
from human_parsing import HumanParsing
from mp_tasks_utils import MPTasksHelper, build_sleeve_mask_v5_tasks, build_top_mask_v5_tasks


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
    roi_mask,
    overlap_ratio,
    union_masks,
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
    "auto_mask_candidates": [],
}

# -----------------------------------------------------------------------------
# SAM manager
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
sam_manager = SamMaskerManager(weights_dir=WEIGHTS_DIR, device=DEVICE)

# -----------------------------------------------------------------------------
# Inpaint pipeline (IMPORTANT: conditional import to avoid diffusers/xformers crash in MOCK mode)
# -----------------------------------------------------------------------------
pipe = None
if not MOCK_INPAINT:
    # diffusers는 여기서만 import
    from diffusers import AutoPipelineForInpainting

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    variant = "fp16" if DEVICE == "cuda" else None

    pipe = AutoPipelineForInpainting.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant=variant,
        use_safetensors=True,
    ).to(DEVICE)

    # xformers는 있으면 활성화, 없거나 깨져 있으면 무시
    if DEVICE == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    pipe.set_progress_bar_config(disable=True)

print(f"DEVICE={DEVICE}, MOCK_INPAINT={MOCK_INPAINT}, MODEL_ID={MODEL_ID}")

# -----------------------------------------------------------------------------
# Prompt parsing + scoring (v4)
# -----------------------------------------------------------------------------

mp_tasks = None
MP_TASKS_INIT_ERROR = None

try:
    mp_tasks = MPTasksHelper(weights_dir="weights")
    if not mp_tasks.enabled:
        MP_TASKS_INIT_ERROR = f"MPTasks disabled ({mp_tasks.diag}). Ensure model files exist under ./weights"
        mp_tasks = None
except Exception as e:
    mp_tasks = None
    MP_TASKS_INIT_ERROR = repr(e)

print(f"[BOOT] mp_tasks={'OK' if mp_tasks else 'NONE'}")
if MP_TASKS_INIT_ERROR:
    print(f"[BOOT] MP_TASKS_INIT_ERROR={MP_TASKS_INIT_ERROR}")

parsing = HumanParsing(device=DEVICE)


COLOR_KEYWORDS = {
    "black": ["검정", "검은", "블랙", "black"],
    "white": ["흰", "화이트", "white"],
    "gray":  ["회색", "그레이", "gray", "grey"],
    "red":   ["빨강", "레드", "red"],
    "blue":  ["파랑", "블루", "blue"],
    "green": ["초록", "그린", "green"],
    "navy":  ["네이비", "navy"],
    "beige": ["베이지", "beige"],
    "brown": ["갈색", "브라운", "brown"],
    "pink":  ["분홍", "핑크", "pink"],
    "purple":["보라", "퍼플", "purple"],
}

GARMENT_KEYWORDS = {
    "tank_top": ["민소매", "나시", "sleeveless", "tank", "tank top"],
    "tshirt":   ["티셔츠", "티", "t-shirt", "tee", "tshirt"],
    "shirt":    ["셔츠", "shirt", "dress shirt", "button up", "button-up"],
    "hoodie":   ["후드", "후드티", "hoodie"],
    "sweater":  ["니트", "스웨터", "sweater", "knit"],
    "jacket":   ["자켓", "재킷", "jacket"],
    "blazer":   ["블레이저", "정장", "blazer", "suit"],
}

def extract_first_match(prompt: str, mapping: dict) -> str | None:
    p = (prompt or "").lower()
    for key, words in mapping.items():
        if any(w.lower() in p for w in words):
            return key
    return None

def parse_prompt_simple(prompt: str) -> dict:
    """
    Minimal prompt parser.

    - Extract garment/color hints for logging/presets.
    - Keeps structure extensible for future (human parsing, grounded masks, etc).
    """
    return {
        "target": infer_edit_target(prompt),
        "color": extract_first_match(prompt, COLOR_KEYWORDS),
        "garment": extract_first_match(prompt, GARMENT_KEYWORDS),
        "raw": prompt or "",
    }

def infer_edit_target(prompt: str) -> str:
    """
    Rule-based intent classifier.

    - Keyword list is intentionally large to support casual sentence prompts.
    - Matching order matters: sleeve first (often overlaps with top), then top, pants, dress, generic.
    """
    p = (prompt or "").lower().strip()

    # Sleeve-focused edits (often implies top + arms)
    sleeve_keywords = [
        "긴팔", "반팔", "소매", "소매를", "팔부분", "팔 부분", "팔을", "팔만",
        "나시", "민소매",  # sleeveless implies sleeve edit
        "sleeve", "sleeves", "short sleeve", "long sleeve",
        "sleeveless", "tank", "tank top",
        "cap sleeve", "rolled sleeve", "roll up", "roll-up",
        "remove sleeves", "cut sleeves", "cut-off", "cutoff",
    ]

    # Top / upper-body clothing
    top_keywords = [
        "상의", "상체", "윗옷", "티", "티셔츠", "반팔티", "긴팔티", "셔츠", "블라우스",
        "후드", "후드티", "집업", "가디건", "니트", "스웨터", "맨투맨",
        "자켓", "재킷", "재킷을", "코트", "점퍼", "패딩", "바람막이",
        "조끼", "베스트", "vest",
        "정장", "수트", "블레이저", "blazer", "suit",
        "shirt", "t-shirt", "tee", "top", "upper", "hoodie", "sweater", "jacket", "coat", "cardigan"
    ]

    # Bottom / pants
    pants_keywords = [
        "바지", "하의", "팬츠", "청바지", "슬랙스", "레깅스", "반바지", "쇼츠",
        "pants", "trousers", "jeans", "slacks", "shorts", "leggings"
    ]

    # Dress / skirt
    dress_keywords = [
        "드레스", "원피스", "치마", "스커트",
        "dress", "skirt"
    ]

    # Order matters
    if any(k in p for k in sleeve_keywords):
        return "sleeve"
    if any(k in p for k in top_keywords):
        return "top"
    if any(k in p for k in pants_keywords):
        return "pants"
    if any(k in p for k in dress_keywords):
        return "dress"
    return "generic"

def score_mask_v4(target: str, mask_dict: dict, img_w: int, img_h: int) -> float:
    area = float(mask_dict.get("area", 0))
    bbox = mask_dict.get("bbox", [0, 0, 0, 0])
    cx, cy = bbox_center(bbox)

    nx = cx / max(1.0, img_w)
    ny = cy / max(1.0, img_h)

    a = area / max(1.0, float(img_w * img_h))

    if a < 0.003:
        return 0.0
    if a > 0.45:
        return 0.0

    area_peak = 0.12
    area_score = clamp01(1.0 - abs(a - area_peak) / area_peak)

    upper = clamp01(1.0 - abs(ny - 0.38) / 0.38)
    mid = clamp01(1.0 - abs(ny - 0.55) / 0.45)
    lower = clamp01(1.0 - abs(ny - 0.78) / 0.28)
    center = clamp01(1.0 - abs(nx - 0.50) / 0.50)

    seg_bool = mask_dict["segmentation"]

    # --- Top-specific ROIs (to avoid face/head candidates) -------------------
    # Torso ROI: upper-body region where shirts usually exist (below head, above hips).
    torso_roi = roi_mask(img_h, img_w, 0.18, 0.28, 0.82, 0.85)
    torso_overlap = overlap_ratio(seg_bool, torso_roi)

    # Head ROI: if a mask heavily covers the head, it's not a "top" candidate.
    head_roi_strict = roi_mask(img_h, img_w, 0.12, 0.00, 0.88, 0.26)
    head_leak_strict = float((seg_bool & head_roi_strict).sum()) / (float(seg_bool.sum()) + 1e-6)

    human_roi = roi_mask(img_h, img_w, 0.20, 0.18, 0.80, 0.95)
    human_overlap = overlap_ratio(seg_bool, human_roi)
    if human_overlap < 0.30:
        return 0.0

    head_roi = roi_mask(img_h, img_w, 0.15, 0.00, 0.85, 0.22)
    head_leak = float((seg_bool & head_roi).sum()) / (float(seg_bool.sum()) + 1e-6)
    head_penalty = clamp01(1.0 - head_leak * 2.5)

    if target == "sleeve":
        left_arm_roi = roi_mask(img_h, img_w, 0.00, 0.22, 0.38, 0.72)
        right_arm_roi = roi_mask(img_h, img_w, 0.62, 0.22, 1.00, 0.72)

        left_overlap = overlap_ratio(seg_bool, left_arm_roi)
        right_overlap = overlap_ratio(seg_bool, right_arm_roi)
        arm_score = clamp01(max(left_overlap, right_overlap) / 0.75)

        return (0.30 * upper) + (0.35 * arm_score) + (0.20 * area_score) + (0.15 * head_penalty)

    if target == "top":
    # Require torso overlap; otherwise candidates are often face/hair/ears.
        if torso_overlap < 0.35:
            return 0.0

    # Hard cut: if the mask is mostly head, reject for top edits.
        if head_leak_strict > 0.08:
            return 0.0

        return (
            0.45 * upper +
            0.20 * center +
            0.20 * area_score +
            0.15 * head_penalty
        )


    if target == "pants":
        return (0.45 * lower) + (0.20 * center) + (0.20 * area_score) + (0.15 * head_penalty)

    if target == "dress":
        return (0.35 * mid) + (0.25 * lower) + (0.25 * area_score) + (0.15 * head_penalty)

    return (0.35 * center) + (0.25 * upper) + (0.25 * area_score) + (0.15 * head_penalty)

def pick_left_right_sleeve_masks(masks, img_w, img_h):
    left_roi = roi_mask(img_h, img_w, 0.00, 0.22, 0.42, 0.74)
    right_roi = roi_mask(img_h, img_w, 0.58, 0.22, 1.00, 0.74)

    best_left = None
    best_left_score = -1.0
    best_right = None
    best_right_score = -1.0

    for m in masks:
        seg = m["segmentation"]
        left_overlap = overlap_ratio(seg, left_roi)
        right_overlap = overlap_ratio(seg, right_roi)

        if left_overlap > 0.35:
            s = left_overlap * 0.8 + (m.get("predicted_iou", 0.0) * 0.2)
            if s > best_left_score:
                best_left_score = s
                best_left = m

        if right_overlap > 0.35:
            s = right_overlap * 0.8 + (m.get("predicted_iou", 0.0) * 0.2)
            if s > best_right_score:
                best_right_score = s
                best_right = m

    return best_left, best_right

def build_auto_candidates_v4(sam_model_type: str, prompt: str, top_k: int = 3):
    if STATE["working_np"] is None:
        return [], "Upload an image first."

    info = parse_prompt_simple(prompt)
    target = info["target"]
    
    # --- v5: MediaPipe Tasks candidates (PoseLandmarker + ImageSegmenter) -----
    if mp_tasks is not None and STATE["working_pil"] is not None:
        if target == "sleeve":
            c = build_sleeve_mask_v5_tasks(STATE["working_pil"], mp_tasks)
            STATE["auto_mask_candidates"] = [c]
            return [c], f"v5(mp-tasks) sleeve candidate built. candidates=1"
        if target == "top":
            c = build_top_mask_v5_tasks(STATE["working_pil"], mp_tasks)
            STATE["auto_mask_candidates"] = [c]
            return [c], f"v5(mp-tasks) top candidate built. candidates=1"

    # --- optional: parsing backend (currently likely disabled) ----------------
    if parsing.enabled and STATE["working_pil"] is not None:
        masks = parsing.predict(STATE["working_pil"])
        if "person" in masks:
            person = masks["person"]
            h, w = person.shape[:2]
            torso = np.zeros_like(person, dtype=np.uint8)
            y0, y1 = int(0.28 * h), int(0.85 * h)
            x0, x1 = int(0.18 * w), int(0.82 * w)
            torso[y0:y1, x0:x1] = person[y0:y1, x0:x1]

            STATE["auto_mask_candidates"] = [torso]
            return [torso], f"v5(parsing) torso candidate built. candidates=1"

    # --- v4 fallback: SAM proposals ------------------------------------------
    img_np = STATE["working_np"]
    h, w = img_np.shape[:2]

    masker = sam_manager.get(sam_model_type)
    masks = masker.auto_masks(img_np)

    candidates = []

    if target == "sleeve":
        left_m, right_m = pick_left_right_sleeve_masks(masks, w, h)
        union = None
        if left_m is not None:
            union = union_masks(union, sam_dict_to_uint8_mask(left_m))
        if right_m is not None:
            union = union_masks(union, sam_dict_to_uint8_mask(right_m))
        if union is not None:
            candidates.append(union)

        scored = []
        for m in masks:
            s = score_mask_v4(target, m, w, h)
            if s > 0:
                scored.append((s, m))
        scored.sort(key=lambda x: x[0], reverse=True)

        for s, m in scored:
            if len(candidates) >= top_k:
                break
            candidates.append(sam_dict_to_uint8_mask(m))

        # De-dup
        uniq, seen = [], set()
        for c in candidates:
            key = int(c.sum())
            if key not in seen:
                uniq.append(c)
                seen.add(key)

        candidates = uniq[:top_k]
        STATE["auto_mask_candidates"] = candidates
        return candidates, f"Auto masks v4 (mp=OFF) target={target}, color={info['color']}, garment={info['garment']}, candidates={len(candidates)}"

    # non-sleeve targets
    scored = []
    for m in masks:
        s = score_mask_v4(target, m, w, h)
        if s > 0:
            scored.append((s, m))
    scored.sort(key=lambda x: x[0], reverse=True)

    for s, m in scored[:top_k]:
        candidates.append(sam_dict_to_uint8_mask(m))

    STATE["auto_mask_candidates"] = candidates
    return candidates, f"Auto masks v4 (mp=OFF) target={target}, candidates={len(candidates)}"

# -----------------------------------------------------------------------------
# UI handlers
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

def on_auto_mask(sam_model_type: str, prompt: str):
    candidates, msg = build_auto_candidates_v4(sam_model_type, prompt, top_k=3)
    if not candidates:
        return None, None, None, msg

    c1 = mask_to_pil(candidates[0]) if len(candidates) > 0 else None
    c2 = mask_to_pil(candidates[1]) if len(candidates) > 1 else None
    c3 = mask_to_pil(candidates[2]) if len(candidates) > 2 else None
    return c1, c2, c3, msg

def on_select_candidate(idx: int):
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
def apply_edit(prompt, negative_prompt, steps, strength, guidance_scale, expand_px, blur_px):
    if STATE["working_pil"] is None:
        return None, "Upload an image first."
    if STATE["mask_np"] is None:
        return None, "Create/select a mask first."

    image = STATE["working_pil"]
    mask_np = postprocess_mask(STATE["mask_np"], int(expand_px), int(blur_px))
    STATE["mask_np"] = mask_np

    if MOCK_INPAINT:
        preview = overlay_mask_on_image(image, mask_np)
        return preview, "MOCK mode: returned mask overlay preview."

    if prompt is None or prompt.strip() == "":
        return None, "Prompt is empty."

    mask_pil = mask_to_pil(mask_np)

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
with gr.Blocks(title="Local Image Edit (v4 Auto Mask)") as demo:
    gr.Markdown(
        "# Local Image Edit (v4 Auto Mask)\n"
        "- MOCK mode avoids importing diffusers/xformers entirely\n"
        "- Sleeve mode unions left+right candidates\n"
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
                placeholder="예: 긴팔을 반팔로 바꿔줘, short-sleeve, natural arm",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="blurry, distorted, bad anatomy, extra limbs, deformed, low quality",
            )

            with gr.Row():
                btn_auto = gr.Button("Auto Mask (v4)")
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

    input_img.change(fn=on_upload, inputs=[input_img, work_long_side],
                     outputs=[working_view, mask_preview, output_img, status])

    working_view.select(fn=on_click, inputs=[sam_model_type, expand_px, blur_px],
                        outputs=[mask_preview, status])

    btn_auto.click(fn=on_auto_mask, inputs=[sam_model_type, prompt],
                   outputs=[cand1, cand2, cand3, status])

    pick1.click(fn=lambda: on_select_candidate(0), inputs=[], outputs=[mask_preview, status])
    pick2.click(fn=lambda: on_select_candidate(1), inputs=[], outputs=[mask_preview, status])
    pick3.click(fn=lambda: on_select_candidate(2), inputs=[], outputs=[mask_preview, status])

    btn_apply.click(fn=apply_edit,
                    inputs=[prompt, negative_prompt, steps, strength, guidance_scale, expand_px, blur_px],
                    outputs=[output_img, status])

    btn_clear.click(fn=clear_mask, inputs=[], outputs=[mask_preview, cand1, cand2, status])
    btn_clear.click(fn=lambda: (None, "Mask cleared."), inputs=[], outputs=[cand3, status])

if __name__ == "__main__":
    demo.launch(share=True)