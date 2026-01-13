import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

class AutoMasker:
    def __init__(self, checkpoint_path, model_type="vit_h", device="cuda"):
        # GPU 사용 가능 여부 재확인
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"SAM 로딩 중... 장치: {actual_device}")
        
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=actual_device)
        self.predictor = SamPredictor(sam)
        self.current_image_hash = None # 이미지 변경 감지용

    def set_image(self, image_np):
        # 이미지가 바뀌었을 때만 임베딩을 새로 계산합니다 (속도 향상의 핵심)
        image_hash = hash(image_np.tostring())
        if self.current_image_hash != image_hash:
            self.predictor.set_image(image_np)
            self.current_image_hash = image_hash
            print("이미지 임베딩 완료 (새로운 이미지)")

    def generate_mask(self, image_np, x, y):
        # 1. 이미지 설정 (바뀌었을 때만 동작)
        self.set_image(image_np)

        # 2. 마스크 예측 (이 부분은 GPU에서 순식간에 끝납니다)
        input_point = np.array([[x, y]])
        input_label = np.array([1])

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        
        mask_image = (masks[0] * 255).astype(np.uint8)
        return mask_image