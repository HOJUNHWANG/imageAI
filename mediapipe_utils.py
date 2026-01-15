# mediapipe_utils.py
import numpy as np
import cv2
from PIL import Image

import mediapipe as mp

class MPHelper:
    def __init__(self):
        self.mp = mp
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)  # 0 for faster
        self.pose = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5)

    def person_mask(self, pil: Image.Image, threshold: float = 0.5) -> np.ndarray:
        """Generate person mask from image."""
        img = np.array(pil.convert("RGB"))
        h, w = img.shape[:2]
        result = self.seg.process(img)
        mask = (result.segmentation_mask > threshold).astype(np.uint8) * 255
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    def pose_landmarks(self, image_pil: Image.Image):
        """
        Returns:
          dict with pixel coords of key landmarks (if detected), else None
        """
        img = np.array(image_pil.convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        res = self.pose.process(img_bgr)
        if not res.pose_landmarks:
            return None

        h, w = img.shape[:2]
        lm = res.pose_landmarks.landmark

        # Landmark indices (MediaPipe Pose)
        # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
        idx = {
            "l_shoulder": 11,
            "r_shoulder": 12,
            "l_elbow": 13,
            "r_elbow": 14,
            "l_wrist": 15,
            "r_wrist": 16,
            "l_hip": 23,
            "r_hip": 24,
        }

        out = {}
        for k, i in idx.items():
            x = int(lm[i].x * w)
            y = int(lm[i].y * h)
            out[k] = (x, y)

        return out

def _clip_pt(x, y, w, h):
    return (int(max(0, min(w - 1, x))), int(max(0, min(h - 1, y))))

def _draw_thick_polyline(mask, pts, thickness):
    if len(pts) < 2:
        return
    for i in range(len(pts) - 1):
        cv2.line(mask, pts[i], pts[i + 1], 255, thickness=thickness, lineType=cv2.LINE_AA)

def build_sleeve_mask_v5(image_pil: Image.Image, mp_helper: MPHelper) -> np.ndarray:
    """
    "긴팔 -> 반팔 / 민소매" 케이스에서 유용한 마스크 생성.

    Strategy:
    - person segmentation으로 사람 영역만 제한
    - pose로 어깨/팔꿈치/손목을 얻고
    - 어깨~팔꿈치 구간을 중심으로 "소매/팔 노출 예상 밴드" 생성
    - 밴드를 사람 영역과 AND 해서 과도한 배경 확장을 방지
    """
    img = np.array(image_pil.convert("RGB"))
    h, w = img.shape[:2]

    person = mp_helper.person_mask(image_pil, threshold=0.2)

    lm = mp_helper.pose_landmarks(image_pil)
    if lm is None:
        # 포즈 실패 시: torso ROI 기반의 보수적 fallback
        mask = np.zeros((h, w), dtype=np.uint8)
        y0 = int(0.28 * h)
        y1 = int(0.85 * h)
        x0 = int(0.18 * w)
        x1 = int(0.82 * w)
        mask[y0:y1, x0:x1] = person[y0:y1, x0:x1]
        return mask

    # 팔 라인: 어깨 -> 팔꿈치 -> 손목
    ls = _clip_pt(*lm["l_shoulder"], w, h)
    le = _clip_pt(*lm["l_elbow"], w, h)
    lw = _clip_pt(*lm["l_wrist"], w, h)

    rs = _clip_pt(*lm["r_shoulder"], w, h)
    re = _clip_pt(*lm["r_elbow"], w, h)
    rw = _clip_pt(*lm["r_wrist"], w, h)

    # 두께(픽셀)는 이미지 크기 비례로 설정
    arm_th = max(12, int(0.04 * min(w, h)))

    arm_mask = np.zeros((h, w), dtype=np.uint8)
    _draw_thick_polyline(arm_mask, [ls, le, lw], thickness=arm_th)
    _draw_thick_polyline(arm_mask, [rs, re, rw], thickness=arm_th)

    # 소매/민소매 편집에서 핵심은 "어깨~팔 위쪽"이 잘 포함되는 것.
    # 팔 전체를 다 바꾸면 과편집이 될 수 있으므로, 손목 방향은 약하게 제한.
    upper_arm_mask = np.zeros((h, w), dtype=np.uint8)
    _draw_thick_polyline(upper_arm_mask, [ls, le], thickness=int(arm_th * 1.15))
    _draw_thick_polyline(upper_arm_mask, [rs, re], thickness=int(arm_th * 1.15))

    # torso ROI를 함께 사용해 "상의"쪽도 같이 포함
    torso = np.zeros((h, w), dtype=np.uint8)
    y0 = int(0.28 * h)
    y1 = int(0.85 * h)
    x0 = int(0.18 * w)
    x1 = int(0.82 * w)
    torso[y0:y1, x0:x1] = person[y0:y1, x0:x1]

    # 최종 마스크: (torso + upper arms) AND person
    combined = np.where((torso > 0) | (upper_arm_mask > 0), 255, 0).astype(np.uint8)
    combined = np.where(person > 0, combined, 0).astype(np.uint8)

    # 경계 정리: 작은 홀/거친 경계 보정
    k = max(5, int(0.01 * min(w, h)) // 2 * 2 + 1)  # odd kernel
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)

    return combined

def build_top_mask_v5(image_pil: Image.Image, mp_helper: MPHelper) -> np.ndarray:
    """
    상의 색/재질 변경용 마스크.
    - 사람 마스크를 torso ROI로 자른 "보수적 상의 후보"
    - 얼굴/머리 쪽은 강하게 제외
    """
    img = np.array(image_pil.convert("RGB"))
    h, w = img.shape[:2]

    person = mp_helper.person_mask(image_pil, threshold=0.2)

    mask = np.zeros((h, w), dtype=np.uint8)

    # torso ROI (below head)
    y0 = int(0.28 * h)
    y1 = int(0.85 * h)
    x0 = int(0.18 * w)
    x1 = int(0.82 * w)
    mask[y0:y1, x0:x1] = person[y0:y1, x0:x1]

    # head cut (very conservative)
    head_y1 = int(0.26 * h)
    mask[:head_y1, :] = 0

    # smooth
    k = max(5, int(0.01 * min(w, h)) // 2 * 2 + 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8), iterations=1)

    return mask