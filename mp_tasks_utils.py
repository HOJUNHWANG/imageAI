# mp_tasks_utils.py
import os
import glob
import numpy as np
import cv2
from PIL import Image

# -----------------------------------------------------------------------------
# MediaPipe Tasks helper (mediapipe==0.10.31)
#
# Notes:
# - Newer MediaPipe wheels may not ship `mediapipe.solutions`.
# - Tasks API is the supported path: PoseLandmarker + ImageSegmenter.
#
# This module:
# - Loads tasks models from ./weights (or env overrides)
# - Produces semantic masks: person/clothes/skin
# - Builds v5 masks for "top" and "sleeve" edits (diffusion-aware)
# -----------------------------------------------------------------------------

LABEL_BG = 0
LABEL_HAIR = 1
LABEL_BODY_SKIN = 2
LABEL_FACE_SKIN = 3
LABEL_CLOTHES = 4
LABEL_OTHER = 5

def _find_first(weights_dir: str, patterns: list[str]) -> str | None:
    for pat in patterns:
        hits = glob.glob(os.path.join(weights_dir, pat))
        if hits:
            return hits[0]
    return None

class MPTasksHelper:
    def __init__(self, weights_dir: str, pose_model_path: str | None = None, seg_model_path: str | None = None):
        self.weights_dir = weights_dir

        pose_env = os.getenv("MP_POSE_MODEL")
        seg_env = os.getenv("MP_SEG_MODEL")

        self.pose_model_path = pose_model_path or pose_env
        self.seg_model_path = seg_model_path or seg_env

        if self.pose_model_path is None:
            self.pose_model_path = _find_first(weights_dir, [
                "*pose_landmarker*lite*.task",
                "*pose_landmarker*full*.task",
                "*pose_landmarker*.task",
            ])

        if self.seg_model_path is None:
            self.seg_model_path = _find_first(weights_dir, [
                "*selfie_multiclass*256x256*.tflite",
                "*selfie_multiclass*.tflite",
                "*selfie_segmenter*square*.tflite",
                "*selfie_segmenter*.tflite",
                "*segmenter*.tflite",
                "*segmenter*.task",
            ])

        if not self.pose_model_path or not os.path.exists(self.pose_model_path):
            raise FileNotFoundError(
                f"Pose model not found. Set MP_POSE_MODEL or drop pose_landmarker_*.task into {weights_dir}"
            )

        if not self.seg_model_path or not os.path.exists(self.seg_model_path):
            raise FileNotFoundError(
                f"Seg model not found. Set MP_SEG_MODEL or drop selfie_multiclass_*.tflite into {weights_dir}"
            )

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions

        self.mp = mp
        self.BaseOptions = BaseOptions
        self.RunningMode = RunningMode
        self.PoseLandmarker = PoseLandmarker
        self.PoseLandmarkerOptions = PoseLandmarkerOptions
        self.ImageSegmenter = ImageSegmenter
        self.ImageSegmenterOptions = ImageSegmenterOptions

        self.pose = self.PoseLandmarker.create_from_options(
            self.PoseLandmarkerOptions(
                base_options=self.BaseOptions(model_asset_path=self.pose_model_path),
                running_mode=self.RunningMode.IMAGE,
            )
        )

        self.seg = self.ImageSegmenter.create_from_options(
            self.ImageSegmenterOptions(
                base_options=self.BaseOptions(model_asset_path=self.seg_model_path),
                running_mode=self.RunningMode.IMAGE,
                output_category_mask=True,
                output_confidence_masks=False,
            )
        )

    def _mp_image(self, pil: Image.Image):
        rgb = np.array(pil.convert("RGB"))
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), rgb

    def pose_landmarks_px(self, pil: Image.Image) -> dict | None:
        mp_img, rgb = self._mp_image(pil)
        h, w = rgb.shape[:2]

        res = self.pose.detect(mp_img)
        if not res.pose_landmarks:
            return None

        lm = res.pose_landmarks[0]
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

    def category_mask(self, pil: Image.Image) -> np.ndarray:
        mp_img, _ = self._mp_image(pil)
        res = self.seg.segment(mp_img)
        cat = res.category_mask.numpy_view()

        # Normalize to (H, W)
        cat = np.asarray(cat)
        if cat.ndim == 3 and cat.shape[-1] == 1:
            cat = cat[..., 0]

        return cat.astype(np.uint8)

    def masks_u8(self, pil: Image.Image) -> dict[str, np.ndarray]:
        cat = self.category_mask(pil)

        # Some MediaPipe wheels expose category_mask as (H, W, 1). Normalize to (H, W).
        if cat.ndim == 3 and cat.shape[-1] == 1:
            cat = cat[..., 0]

        person = (cat != LABEL_BG).astype(np.uint8) * 255
        clothes = (cat == LABEL_CLOTHES).astype(np.uint8) * 255
        skin = ((cat == LABEL_BODY_SKIN) | (cat == LABEL_FACE_SKIN)).astype(np.uint8) * 255

        return {"person": person, "clothes": clothes, "skin": skin, "cat": cat}

# ----------------------- v5 mask builders -----------------------------------

def _clip_pt(x, y, w, h):
    return (int(max(0, min(w - 1, x))), int(max(0, min(h - 1, y))))

def build_top_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    """
    Upper-clothes mask candidate.

    Design goals:
    - Focus edits on torso clothing (shirt/jacket/top).
    - Avoid repainting face/hair.
    - Avoid repainting pants/legs.

    This mask is intentionally conservative; users can expand via UI if needed.
    """
    rgb = np.array(pil.convert("RGB"))
    h, w = rgb.shape[:2]
    masks = helper.masks_u8(pil)

    clothes = masks["clothes"]
    person = masks["person"]

    roi = np.zeros((h, w), dtype=np.uint8)
    y0, y1 = int(0.18 * h), int(0.80 * h)
    x0, x1 = int(0.12 * w), int(0.88 * w)
    roi[y0:y1, x0:x1] = 255

    m = cv2.bitwise_and(clothes, roi)
    m = cv2.bitwise_and(m, person)

    k = max(7, int(0.012 * min(h, w)) // 2 * 2 + 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)
    return m

def build_sleeve_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    """
    Sleeve removal / sleeveless conversion mask candidate (diffusion-aware).

    Key problem:
    - Removing sleeves exposes skin that was not visible in the input image.
    - Inpainting must cover BOTH:
        (1) sleeve fabric region (clothes mask)
        (2) "to-be-exposed" upper arm region (skin + pose-driven band)

    Implementation strategy:
    - Start with clothes mask from multiclass segmentation.
    - Add a pose-driven arm band:
        * shoulder ellipse (adds shoulder volume)
        * shoulder→elbow thick line (adds upper arm corridor)
        * a small rectangular ROI around the upper arm to increase coverage stability
    - Add a small amount of skin around the arm band to encourage natural arm synthesis.
    """
    rgb = np.array(pil.convert("RGB"))
    h, w = rgb.shape[:2]

    sem = helper.masks_u8(pil)
    clothes = sem["clothes"]
    person = sem["person"]
    skin = sem["skin"]

    # Defensive: ensure all semantic masks are (H, W)
    if clothes.ndim == 3 and clothes.shape[-1] == 1:
        clothes = clothes[..., 0]
    if person.ndim == 3 and person.shape[-1] == 1:
        person = person[..., 0]
    if skin.ndim == 3 and skin.shape[-1] == 1:
        skin = skin[..., 0]

    lm = helper.pose_landmarks_px(pil)

    arm_band = np.zeros((h, w), dtype=np.uint8)

    if lm is not None:
        ls = _clip_pt(*lm["l_shoulder"], w, h)
        le = _clip_pt(*lm["l_elbow"], w, h)
        rs = _clip_pt(*lm["r_shoulder"], w, h)
        re = _clip_pt(*lm["r_elbow"], w, h)

        # Thickness proportional to image size; tuned for “arm exposure” edits.
        arm_th = max(22, int(0.065 * min(h, w)))

        # 1) Upper arm corridor (shoulder->elbow)
        cv2.line(arm_band, ls, le, 255, thickness=int(arm_th * 1.4), lineType=cv2.LINE_AA)
        cv2.line(arm_band, rs, re, 255, thickness=int(arm_th * 1.4), lineType=cv2.LINE_AA)

        # 2) Shoulder volume (ellipse) — critical for sleeveless realism
        for (sx, sy) in [ls, rs]:
            cv2.ellipse(
                arm_band,
                center=(sx, sy),
                axes=(int(arm_th * 1.25), int(arm_th * 0.95)),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=255,
                thickness=-1,
            )

        # 3) Upper-arm ROI (simple box around shoulder→elbow area)
        # This intentionally over-covers slightly; UI expand/blur can be reduced if needed.
        def _arm_roi(a, b, pad_x, pad_y):
            x0 = min(a[0], b[0]) - pad_x
            x1 = max(a[0], b[0]) + pad_x
            y0 = min(a[1], b[1]) - pad_y
            y1 = max(a[1], b[1]) + pad_y
            x0, y0 = _clip_pt(x0, y0, w, h)
            x1, y1 = _clip_pt(x1, y1, w, h)
            arm_band[y0:y1, x0:x1] = 255

        _arm_roi(ls, le, pad_x=int(arm_th * 0.9), pad_y=int(arm_th * 0.9))
        _arm_roi(rs, re, pad_x=int(arm_th * 0.9), pad_y=int(arm_th * 0.9))

    # Combine sleeve fabric + pose-driven exposure band
    m = np.maximum(clothes, arm_band).astype(np.uint8)

    # Add a small amount of skin near arms:
    # - This helps the model synthesize natural arm boundaries instead of “cut-out” seams.
    # - Restrict to arm_band vicinity to avoid repainting face/neck.
    skin_near_arms = cv2.bitwise_and(skin, cv2.dilate(arm_band, np.ones((25, 25), np.uint8), iterations=1))
    m = np.maximum(m, skin_near_arms).astype(np.uint8)

    # Keep strictly on the person region (avoid background repaint)
    m = cv2.bitwise_and(m, person)

    # Mild dilation to provide blending room. (UI also has expand; keep moderate here.)
    k1 = max(9, int(0.018 * min(h, w)) // 2 * 2 + 1)
    m = cv2.dilate(m, np.ones((k1, k1), np.uint8), iterations=1)

    # Clean up
    k2 = max(7, int(0.014 * min(h, w)) // 2 * 2 + 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k2, k2), np.uint8), iterations=1)

    return m

def build_pants_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    """
    Pants (lower-clothes) candidate.
    Note: multiclass selfie segmentation usually labels all garments as "clothes".
          So we carve lower-body ROI from person/clothes intersection.
    """
    rgb = np.array(pil.convert("RGB"))
    h, w = rgb.shape[:2]
    sem = helper.masks_u8(pil)

    person = sem["person"]
    clothes = sem["clothes"]

    # lower body ROI heuristic
    roi = np.zeros((h, w), dtype=np.uint8)
    y0, y1 = int(0.55 * h), int(0.98 * h)
    x0, x1 = int(0.10 * w), int(0.90 * w)
    roi[y0:y1, x0:x1] = 255

    m = cv2.bitwise_and(person, clothes)
    m = cv2.bitwise_and(m, roi)

    k = max(7, int(0.014 * min(h, w)) // 2 * 2 + 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)
    return m

def build_hair_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    """
    Hair candidate (uses category labels).
    """
    cat = helper.category_mask(pil)
    hair = (cat == LABEL_HAIR).astype(np.uint8) * 255

    h, w = hair.shape[:2]
    k = max(5, int(0.010 * min(h, w)) // 2 * 2 + 1)
    hair = cv2.morphologyEx(hair, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)
    return hair.astype(np.uint8)

def build_background_mask_v5_tasks(pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    """
    Background candidate.
    Background is label 0. Editing background often benefits from slight dilation inward
    (to avoid halo around the subject).
    """
    cat = helper.category_mask(pil)
    bg = (cat == LABEL_BG).astype(np.uint8) * 255

    h, w = bg.shape[:2]
    # slightly erode background so edges don't eat into the subject too aggressively
    k = max(5, int(0.008 * min(h, w)) // 2 * 2 + 1)
    bg = cv2.erode(bg, np.ones((k, k), np.uint8), iterations=1)
    return bg.astype(np.uint8)