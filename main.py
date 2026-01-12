import os
import cv2
import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting
from sam_utils import AutoMasker # 이전에 만든 헬퍼 클래스

# 1. 경로 및 설정 (Configuration)
base_path = os.path.dirname(__file__)
sam_checkpoint = os.path.join(base_path, "weights", "sam_vit_h_4b8939.pth")
image_path = os.path.join(base_path, "inputs", "photo.png")

# 2. 모델 로드 (Initialization)
# 메모리 절약을 위해 GPU가 충분한지 확인하세요.
print("모델을 로드 중입니다...")
masker = AutoMasker(sam_checkpoint) # SAM 로드
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# 3. 이미지 읽기 및 SAM 실행
image_cv = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

# 사용자가 드레스 부분을 클릭했다고 가정 (x, y 좌표)
# 포트폴리오 시연 시 이 좌표값을 바꾸며 테스트할 수 있습니다.
click_x, click_y = 500, 800 
print(f"좌표 ({click_x}, {click_y})를 기준으로 마스크를 생성합니다...")

mask_np = masker.generate_mask(image_rgb, click_x, click_y)
mask_pil = Image.fromarray(mask_np) # Numpy 배열을 PIL 이미지 객체로 변환

# 4. 이미지 편집 실행 (Inpainting)
prompt = "a elegant emerald green velvet dress"
negative_prompt = "low quality, blurry, distorted"

print("AI가 옷을 갈아입히는 중입니다...")
final_result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=Image.fromarray(image_rgb),
    mask_image=mask_pil,
    num_inference_steps=30,
    strength=0.95
).images[0]

# 5. 결과 저장 및 확인
final_result.save(os.path.join(base_path, "final_dress_change.png"))
print("모든 작업이 완료되었습니다! 'final_dress_change.png'를 확인하세요.")