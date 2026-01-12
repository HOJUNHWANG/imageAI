import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# 자바의 Helper Class 같은 역할입니다.
class AutoMasker:
    def __init__(self, checkpoint_path, model_type="vit_h", device="cuda"):
        # 모델 로드 (생성자에서 한 번만 로드)
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def generate_mask(self, image_np, x, y):
        # 1. 원본 이미지 설정 (numpy 배열 형태)
        self.predictor.set_image(image_np)

        # 2. 클릭한 좌표를 바탕으로 마스크 예측
        # input_point: [x, y] 좌표 / input_label: 1은 해당 물체를 포함하라는 의미
        input_point = np.array([[x, y]])
        input_label = np.array([1])

        # masks: 결과 마스크 리스트 / scores: 예측 정확도 점수
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False, # 가장 정확한 마스크 하나만 가져옴
        )
        
        # 0과 1로 된 마스크를 0(검정)과 255(흰색)의 이미지 데이터로 변환
        mask_image = (masks[0] * 255).astype(np.uint8)
        return mask_image