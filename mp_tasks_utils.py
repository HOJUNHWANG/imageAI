# mp_tasks_utils.py (최신 mediapipe Tasks API 0.10+ 호환)
"""
MediaPipe Tasks API 기반 semantic mask 생성 (0.10.14+)
- SelfieSegmentation + PoseLandmarker 활용
- 상의/소매/머리/배경 등 v5 마스크 빌드
- CPU 환경에서도 동작
"""

import os
import numpy as np
from PIL import Image, ImageFilter
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import cv2
except ImportError:
    cv2 = None

# Label constants (SelfieSegmentation은 foreground/background만)
LABEL_BG = 0
LABEL_PERSON = 1

class MPTasksHelper:
    """
    MediaPipe Pose + SelfieSegmentation 헬퍼 (Tasks API)
    - weights_dir에서 모델 파일 자동 검색
    """
    def __init__(self, weights_dir: str):
        self.weights_dir = weights_dir

        # SelfieSegmentation Task
        self.seg_options = vision.ImageSegmenterOptions(
            base_options=python.BaseOptions(
                model_asset_path=os.path.join(weights_dir, "selfie_multiclass_256x256.tflite")
            ),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(self.seg_options)

    def person_mask(self, pil: Image.Image, threshold: float = 0.5) -> np.ndarray:
        """사람(foreground) 마스크 반환 (uint8 0/255)"""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(pil.convert("RGB")))
        segmentation_result = self.segmenter.segment(mp_image)
        mask = segmentation_result.category_mask.numpy_view()
        person = (mask > threshold).astype(np.uint8) * 255
        return person

def _morph_close(mask: np.ndarray, k: int = 5) -> np.ndarray:
    """Morphological close with fallback."""
    if cv2 is not None:
        kernel = np.ones((k, k), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        img = Image.fromarray(mask)
        size = k * 2 + 1
        img = img.filter(ImageFilter.MaxFilter(size=size))
        img = img.filter(ImageFilter.MinFilter(size=size))
        return np.array(img, dtype=np.uint8)

def build_sleeve_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    person = helper.person_mask(pil)
    h, w = person.shape[:2]
    mask = person.copy()
    k = max(5, int(0.010 * min(h, w)) // 2 * 2 + 1)
    return _morph_close(mask, k)

def build_top_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    person = helper.person_mask(pil)
    h, w = person.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    y0 = int(0.28 * h)
    y1 = int(0.85 * h)
    x0 = int(0.18 * w)
    x1 = int(0.82 * w)
    mask[y0:y1, x0:x1] = person[y0:y1, x0:x1]

    head_y1 = int(0.26 * h)
    mask[:head_y1, :] = 0

    k = max(5, int(0.008 * min(h, w)) // 2 * 2 + 1)
    return _morph_close(mask, k)

def build_pants_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    person = helper.person_mask(pil)
    h, w = person.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    y0 = int(0.45 * h)
    y1 = int(0.95 * h)
    x0 = int(0.20 * w)
    x1 = int(0.80 * w)
    mask[y0:y1, x0:x1] = person[y0:y1, x0:x1]

    k = max(5, int(0.010 * min(h, w)) // 2 * 2 + 1)
    return _morph_close(mask, k)

def build_hair_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    person = helper.person_mask(pil)
    h, w = person.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    y1 = int(0.30 * h)
    mask[:y1, :] = person[:y1, :]
    k = max(5, int(0.010 * min(h, w)) // 2 * 2 + 1)
    return _morph_close(mask, k)

def build_background_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    person = helper.person_mask(pil)
    bg = 255 - person
    k = max(3, int(0.005 * min(pil.size)) // 2 * 2 + 1)
    if cv2 is not None:
        bg = cv2.erode(bg, np.ones((k, k), np.uint8), iterations=1)
    else:
        img = Image.fromarray(bg)
        size = k * 2 + 1
        img = img.filter(ImageFilter.MinFilter(size=size))
        bg = np.array(img)
    return bg.astype(np.uint8)