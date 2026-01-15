# mp_tasks_utils.py
"""
MediaPipe Tasks 기반 semantic mask 생성 유틸리티
- SelfieSegmentation + PoseLandmarker 활용
- 상의/소매/머리/배경 등 v5 마스크 빌드
- CPU 환경에서도 동작하도록 최소 구현
"""

import numpy as np
from PIL import Image, ImageFilter

# cv2 import 안전하게 (없으면 fallback)
try:
    import cv2
except ImportError:
    cv2 = None

import mediapipe as mp

# Semantic label constants (실제 모델에 따라 달라질 수 있음)
LABEL_BG         = 0
LABEL_HAIR       = 1
LABEL_BODY_SKIN  = 2
LABEL_FACE_SKIN  = 3
LABEL_CLOTHES    = 4
LABEL_OTHER      = 5

class MPTasksHelper:
    """
    MediaPipe Pose + SelfieSegmentation 헬퍼
    - 실제 모델 로드 없이도 최소 동작 가능 (placeholder)
    """
    def __init__(self, weights_dir: str):
        self.weights_dir = weights_dir
        self.mp = mp
        self._pose = None
        self._seg = None

    @property
    def pose(self):
        if self._pose is None:
            self._pose = self.mp.solutions.pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return self._pose

    @property
    def seg(self):
        if self._seg is None:
            self._seg = self.mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=0  # 빠른 버전
            )
        return self._seg

    def person_mask(self, pil: Image.Image, threshold: float = 0.5) -> np.ndarray:
        """사람(foreground) 마스크 반환 (uint8 0/255)"""
        img = np.array(pil.convert("RGB"))
        h, w = img.shape[:2]
        result = self.seg.process(img)
        mask = (result.segmentation_mask > threshold).astype(np.uint8) * 255
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR) if cv2 is not None else mask

def _morph_close(mask: np.ndarray, k: int = 5) -> np.ndarray:
    """Morphological close with cv2 fallback to PIL."""
    if cv2 is not None:
        kernel = np.ones((k, k), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        img = Image.fromarray(mask)
        size = k * 2 + 1
        img = img.filter(ImageFilter.MaxFilter(size=size))
        img = img.filter(ImageFilter.MinFilter(size=size))
        return np.array(img, dtype=np.uint8)

# =============================================================================
# 필수 5개 함수 (app.py에서 import되는 것들)
# =============================================================================

def build_sleeve_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    """
    소매 영역 마스크 (v5)
    - 팔 영역 근사 (MediaPipe Pose + person mask 결합)
    - 최소 구현: person mask 전체를 반환 (실제론 팔 랜드마크 사용)
    """
    person = helper.person_mask(pil)
    mask = person.copy()

    # 간단한 morphology
    k = max(5, int(0.010 * min(pil.size)) // 2 * 2 + 1)
    mask = _morph_close(mask, k)

    return mask.astype(np.uint8)

def build_top_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    """
    상의 영역 마스크 (v5)
    - torso ROI + person mask 결합
    - 머리 부분 강하게 제외
    """
    person = helper.person_mask(pil)
    h, w = person.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # torso ROI (보수적 추정)
    y0 = int(0.28 * h)
    y1 = int(0.85 * h)
    x0 = int(0.18 * w)
    x1 = int(0.82 * w)
    mask[y0:y1, x0:x1] = person[y0:y1, x0:x1]

    # head cut
    head_y1 = int(0.26 * h)
    mask[:head_y1, :] = 0

    # smooth
    k = max(5, int(0.008 * min(h, w)) // 2 * 2 + 1)
    mask = _morph_close(mask, k)

    return mask.astype(np.uint8)

def build_pants_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    """
    바지 영역 마스크 (v5)
    - 하반신 ROI + person mask 결합
    """
    person = helper.person_mask(pil)
    h, w = person.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # pants ROI (하반신)
    y0 = int(0.45 * h)
    y1 = int(0.95 * h)
    x0 = int(0.20 * w)
    x1 = int(0.80 * w)
    mask[y0:y1, x0:x1] = person[y0:y1, x0:x1]

    # smooth
    k = max(5, int(0.010 * min(h, w)) // 2 * 2 + 1)
    mask = _morph_close(mask, k)

    return mask.astype(np.uint8)

def build_hair_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    """
    머리카락 영역 마스크 (v5)
    - 상단 ROI + person mask 결합 (placeholder)
    """
    person = helper.person_mask(pil)
    h, w = person.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # hair ROI (상단 30%)
    y1 = int(0.30 * h)
    mask[:y1, :] = person[:y1, :]

    # smooth
    k = max(5, int(0.010 * min(h, w)) // 2 * 2 + 1)
    mask = _morph_close(mask, k)

    return mask.astype(np.uint8)

def build_background_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    """
    배경 마스크 (v5)
    - person mask의 반전 (배경 = 1 - person)
    """
    person = helper.person_mask(pil)
    bg = 255 - person  # 간단 반전

    # 약간 erode (배경이 사람을 침범하지 않게)
    k = max(3, int(0.005 * min(pil.size)) // 2 * 2 + 1)
    if cv2 is not None:
        bg = cv2.erode(bg, np.ones((k, k), np.uint8), iterations=1)
    else:
        img = Image.fromarray(bg)
        size = k * 2 + 1
        img = img.filter(ImageFilter.MinFilter(size=size))
        bg = np.array(img)

    return bg.astype(np.uint8)