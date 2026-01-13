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
)

# -----------------------------------------------------------------------------
# Runtime switches (환경에 따라 동작을 바꾸기 위한 플래그들)
# -----------------------------------------------------------------------------
# GPU 없는 PC에서 테스트할 때:
#   MOCK_INPAINT=1 python app.py
# => Diffusion 모델 로딩/추론을 생략하고, 마스크 오버레이로 UI 동작만 확인 가능.
MOCK_INPAINT = os.environ.get("MOCK_INPAINT", "0") == "1"

# CPU에서 Diffusers를 억지로 돌리면 SDXL은 매우 느림.
# CPU 테스트를 "실제 생성"까지 하려면 작은 모델로 바꿔서 512 정도로 돌리는 게 현실적임.
# 필요하면 아래 MODEL_ID를 SD 1.5 inpaint 모델로 변경:
#   e.g. runwayml/stable-diffusion-inpainting (환경에 따라 접근 가능)
MODEL_ID = os.environ.get(
    "INPAINT_MODEL_ID",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
)

# -----------------------------------------------------------------------------
# Device / performance knobs
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TF32는 Ampere(30xx)에서 matmul 성능을 올려주는 옵션.
# 모델 정확도에 영향이 있을 수 있으나 보통 이미지 생성에서는 체감이 크지 않음.
torch.backends.cuda.matmul.allow_tf32 = True

# -----------------------------------------------------------------------------
# Global state (Gradio 이벤트 간 공유)
# -----------------------------------------------------------------------------
STATE = {
    "working_pil": None,    # SAM+Inpaint용 작업 이미지(리사이즈된 PIL)
    "working_np": None,     # SAM 입력용 numpy (RGB)
    "mask_np": None,        # uint8 (H, W), 0/255
}

# -----------------------------------------------------------------------------
# Load SAM manager (vit_b / vit_h)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

sam_manager = SamMaskerManager(weights_dir=WEIGHTS_DIR, device=DEVICE)

# -----------------------------------------------------------------------------
# Load inpaint pipeline (optional in MOCK mode)
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

    # xformers가 있으면 attention을 더 효율적으로 처리해서 빨라질 수 있음.
    if DEVICE == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[OK] xformers enabled")
        except Exception as e:
            print("[WARN] xformers not available:", e)

    pipe.set_progress_bar_config(disable=True)

print(f"DEVICE={DEVICE}, MOCK_INPAINT={MOCK_INPAINT}, MODEL_ID={MODEL_ID}")

# -----------------------------------------------------------------------------
# Handlers
# -----------------------------------------------------------------------------
def on_upload(img_pil: Image.Image, work_long_side: int):
    """
    업로드 시:
    - 입력 이미지를 RGB로 정규화
    - 작업 해상도(working)로 리사이즈해서 저장
    - 마스크 초기화
    """
    if img_pil is None:
        return None, None, "Upload an image."

    img_pil = to_rgb_pil(img_pil)

    # 작업 이미지 고정(좌표/마스크/모델 입력을 동일 스케일로 유지)
    working = resize_long_side(img_pil, int(work_long_side))
    working_np = np.array(working)  # RGB

    STATE["working_pil"] = working
    STATE["working_np"] = working_np
    STATE["mask_np"] = None

    return working, None, f"Loaded. Working size={working.size}. Click to create a mask."

def on_click(evt: gr.SelectData, sam_model_type: str, expand_px: int, blur_px: int):
    """
    클릭 시:
    - SAM으로 마스크 생성
    - 후처리(확장/블러)
    - 마스크 프리뷰 반환
    """
    if STATE["working_np"] is None:
        return None, "Upload an image first."

    x, y = evt.index
    img_np = STATE["working_np"]

    masker = sam_manager.get(sam_model_type)
    raw_mask = masker.mask_from_click(img_np, int(x), int(y))

    mask = postprocess_mask(raw_mask, int(expand_px), int(blur_px))
    STATE["mask_np"] = mask

    return mask_to_pil(mask), f"Mask created at ({x},{y})."

@torch.inference_mode()
def apply_edit(
    sam_model_type: str,
    prompt: str,
    negative_prompt: str,
    steps: int,
    strength: float,
    guidance_scale: float,
    expand_px: int,
    blur_px: int,
):
    """
    Apply 버튼:
    - 마스크 유효성 체크
    - (선택) 마스크 후처리 재적용
    - MOCK 모드면 오버레이만 반환
    - 실제 모드면 inpaint 실행
    """
    if STATE["working_pil"] is None:
        return None, "Upload an image first."
    if STATE["mask_np"] is None:
        return None, "Create a mask by clicking the image."

    image = STATE["working_pil"]
    mask_np = postprocess_mask(STATE["mask_np"], int(expand_px), int(blur_px))
    STATE["mask_np"] = mask_np
    mask_pil = mask_to_pil(mask_np)

    if MOCK_INPAINT:
        # 모델 없이 UI/합성 파이프 테스트용
        preview = overlay_mask_on_image(image, mask_np)
        return preview, "MOCK mode: returned mask overlay preview."

    # Safety: 빈 프롬프트면 실행 가치가 낮아서 early return
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

def clear_mask():
    """마스크만 초기화."""
    STATE["mask_np"] = None
    return None, "Mask cleared."

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
with gr.Blocks(title="Local Image Edit (Fast + vit_b/vit_h)") as demo:
    gr.Markdown(
        "# Local Image Edit (SAM + SDXL Inpaint)\n"
        "- SAM: vit_b / vit_h 선택 가능\n"
        "- set_image 캐시로 클릭 반복 시 속도 개선\n"
        "- working 해상도 고정으로 좌표 꼬임/늘어짐 감소\n"
        "- GPU 없는 환경은 MOCK_INPAINT=1로 UI 테스트 가능\n"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload", type="pil")
            status = gr.Textbox(label="Status", interactive=False)

            # 작업 해상도(긴 변). 기본 1024.
            work_long_side = gr.Slider(512, 1536, value=1024, step=64, label="Working Long Side")

            # SAM 모델 선택
            sam_model_type = gr.Dropdown(
                choices=["vit_b", "vit_h"],
                value="vit_b",
                label="SAM Model (speed vs accuracy)"
            )

            # 마스크 후처리
            expand_px = gr.Slider(0, 40, value=12, step=1, label="Mask Expand(px)")
            blur_px = gr.Slider(0, 40, value=8, step=1, label="Mask Blur(px)")

            # 프롬프트
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="e.g. short-sleeve shirt, natural arm, realistic fabric, photorealistic"
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="blurry, distorted, bad anatomy, extra limbs, deformed, low quality"
            )

            # 생성 파라미터 (속도/품질 트레이드오프)
            steps = gr.Slider(10, 60, value=28, step=1, label="Steps")
            strength = gr.Slider(0.3, 0.95, value=0.78, step=0.01, label="Strength")
            guidance_scale = gr.Slider(1.0, 12.0, value=6.0, step=0.1, label="Guidance Scale")

            with gr.Row():
                btn_apply = gr.Button("Apply", variant="primary")
                btn_clear = gr.Button("Clear Mask")

        with gr.Column(scale=1):
            working_view = gr.Image(label="Working Image (click here)", interactive=True)
            mask_preview = gr.Image(label="Mask Preview")
            output_img = gr.Image(label="Result")

    # 업로드 -> working 이미지 생성 후 working_view에 표시
    input_img.change(
        fn=on_upload,
        inputs=[input_img, work_long_side],
        outputs=[working_view, output_img, status],
    )

    # working_view에서 클릭 -> 마스크 생성
    working_view.select(
        fn=on_click,
        inputs=[sam_model_type, expand_px, blur_px],
        outputs=[mask_preview, status],
    )

    # apply
    btn_apply.click(
        fn=apply_edit,
        inputs=[sam_model_type, prompt, negative_prompt, steps, strength, guidance_scale, expand_px, blur_px],
        outputs=[output_img, status],
    )

    # clear mask
    btn_clear.click(fn=clear_mask, inputs=[], outputs=[mask_preview, status])

if __name__ == "__main__":
    demo.launch(share=True)