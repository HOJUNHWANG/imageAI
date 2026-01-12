import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import AutoPipelineForInpainting
from sam_utils import AutoMasker # 우리가 만든 SAM 헬퍼 클래스
import os

# 1. 모델 로드 (서버 시작 시 한 번만 실행)
base_path = os.path.dirname(__file__)
sam_checkpoint = os.path.join(base_path, "weights", "sam_vit_h_4b8939.pth")

print("AI 모델들을 로딩 중입니다... 잠시만 기다려주세요.")
masker = AutoMasker(sam_checkpoint)
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# 전역 변수로 마스크 저장 (클릭 시 생성된 마스크를 유지하기 위함)
current_mask = None

# 2. 핵심 로직 함수 (자바의 Service Method 역할)
def on_select(img, evt: gr.SelectData):
    global current_mask
    # img는 PIL 이미지 객체입니다. 이를 numpy 배열로 변환합니다.
    img_rgb = np.array(img)
    
    # 사용자가 클릭한 좌표(x, y)를 가져와서 SAM으로 마스크 생성
    mask_np = masker.generate_mask(img_rgb, evt.index[0], evt.index[1])
    current_mask = Image.fromarray(mask_np)
    
    # 마스크가 잘 잡혔는지 사용자에게 보여주기 위해 원본 위에 붉은색으로 덮어씌웁니다.
    overlay = img_rgb.copy()
    overlay[mask_np > 0] = [255, 0, 0] # 흰색 마스크 영역을 빨간색으로
    return Image.fromarray(overlay), "마스크 생성 완료! 이제 수정을 실행하세요."

def run_inpaint(img, prompt):
    if current_mask is None:
        return None, "먼저 이미지에서 수정할 부분을 클릭해 주세요!"
    
    # Stable Diffusion 실행
    result = pipe(
        prompt=prompt,
        image=img,
        mask_image=current_mask,
        num_inference_steps=30,
        strength=0.95
    ).images[0]
    
    return result, "이미지 수정이 완료되었습니다!"

# 3. Gradio UI 레이아웃 설정 (자바의 GUI 레이아웃 구성과 유사)
with gr.Blocks(title="AI Image Editor Portfolio") as demo:
    gr.Markdown("# 🎨 나만의 AI 이미지 편집 서비스")
    gr.Markdown("이미지를 업로드하고, 수정하고 싶은 옷 영역을 **클릭**한 뒤 프롬프트를 입력하세요.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Image", type="pil")
            mask_preview = gr.Image(label="Mask Preview (Click on the image)")
            status_text = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", placeholder="예: a blue silk dress")
            run_button = gr.Button("Apply Changes", variant="primary")
            output_img = gr.Image(label="Result")

    # 이벤트 연결 (자바의 Event Listener)
    # 이미지 클릭 시 on_select 함수 실행
    input_img.select(on_select, inputs=[input_img], outputs=[mask_preview, status_text])
    
    # 버튼 클릭 시 run_inpaint 함수 실행
    run_button.click(run_inpaint, inputs=[input_img, prompt_input], outputs=[output_img, status_text])

# 4. 서버 실행
if __name__ == "__main__":
    demo.launch(share=True) # share=True로 설정하면 외부에서 접속 가능한 링크가 생성됩니다.