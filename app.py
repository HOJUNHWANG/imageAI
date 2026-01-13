import gradio as gr
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForInpainting
from sam_utils import AutoMasker
import os

# 1. 환경 설정 및 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
base_path = os.path.dirname(__file__)
sam_checkpoint = os.path.join(base_path, "weights", "sam_vit_h_4b8939.pth")

print(f"현재 사용 중인 장치: {device}")
print("AI 모델들을 로딩 중입니다... (3080 Ti 최적화 모드)")

# SAM 모델 로드 (수정한 sam_utils 사용)
masker = AutoMasker(sam_checkpoint, device=device)

# SDXL Inpainting 모델 로드
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

if device == "cuda":
    # 1. 3080 Ti 메모리 최적화의 핵심
    # 모델 전체를 GPU에 올리지 않고, 실행되는 레이어만 순차적으로 올려서 VRAM 부족을 방지합니다.
    pipe.enable_sequential_cpu_offload() 
    
    # 2. 어텐션 연산을 쪼개서 처리하여 메모리 피크치를 낮춥니다.
    pipe.enable_attention_slicing()
    
    # 3. VAE(이미지 복원) 단계에서의 메모리 부족을 방지합니다.
    pipe.enable_vae_tiling()
    
    print("3080 Ti 최적화 모드(Sequential Offload) 활성화 완료.")

# 전역 변수로 마스크와 이미지 저장
current_mask = None
last_uploaded_img = None

# 2. 로직 함수
def on_select(img, evt: gr.SelectData):
    global current_mask, last_uploaded_img
    
    # PIL 이미지를 numpy로 변환
    img_rgb = np.array(img.convert("RGB"))
    
    # 마스크 생성 (이미지 임베딩은 바뀌었을 때만 sam_utils에서 처리됨)
    mask_np = masker.generate_mask(img_rgb, evt.index[0], evt.index[1])
    current_mask = Image.fromarray(mask_np)
    
    # 화면 표시용 오버레이 생성
    overlay = img_rgb.copy()
    overlay[mask_np > 0] = [255, 0, 0] # 마스크 영역을 빨간색으로 표시
    
    return Image.fromarray(overlay), "마스크 생성 완료! 프롬프트를 입력하고 Apply를 누르세요."

def run_inpaint(img, prompt):
    global current_mask
    if current_mask is None:
        return None, "오류: 먼저 이미지에서 수정할 부분을 클릭해 마스크를 만들어주세요!"
    
    print(f"이미지 생성 시작... 프롬프트: {prompt}")
    
    # 생성 실행 (3080 Ti 기준 약 10~15초 소요)
    result = pipe(
        prompt=prompt,
        image=img.convert("RGB"),
        mask_image=current_mask,
        num_inference_steps=30,  # 30단계면 충분히 고퀄리티가 나옵니다.
        strength=0.85,           # 원본과의 조화 정도 (0.7~0.9 권장)
        guidance_scale=7.5
    ).images[0]
    
    return result, "생성이 완료되었습니다!"

# 3. UI 구성
with gr.Blocks(title="3080 Ti AI Photo Editor") as demo:
    gr.Markdown("## ⚡ 3080 Ti 가속 AI 이미지 편집기")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="1. 원본 업로드", type="pil")
            mask_preview = gr.Image(label="2. 클릭하여 영역 지정 (빨간색)")
            status_text = gr.Textbox(label="상태 알림", interactive=False)
            
        with gr.Column():
            prompt_input = gr.Textbox(label="3. 변경할 내용 입력", placeholder="예: a fancy long sleeve black sweater")
            run_button = gr.Button("✨ 변환 실행 (Apply)", variant="primary")
            output_img = gr.Image(label="4. 결과물")

    # 이벤트 바인딩
    input_img.select(on_select, inputs=[input_img], outputs=[mask_preview, status_text])
    run_button.click(run_inpaint, inputs=[input_img, prompt_input], outputs=[output_img, status_text])

if __name__ == "__main__":
    # share=True로 하면 외부에서도 접속 가능한 주소가 나옵니다.
    demo.launch()