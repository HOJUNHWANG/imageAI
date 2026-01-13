import os
import glob
import numpy as np
import cv2
from PIL import Image

# -----------------------------------------------------------------------------
# MediaPipe Tasks 기반 v5 마스킹
#
# Why:
# - mediapipe 0.10.31+ 환경에서는 mediapipe.solutions 가 없을 수 있음.
# - Tasks API(vision)로 PoseLandmarker / ImageSegmenter를 사용해야 안정적으로 동작.
#
# Model files expected under ./weights:
# - Pose: pose_landmarker_*.task (lite 추천)
# - Segmentation: selfie_multiclass_*.tflite 또는 selfie_segmenter_*.tflite (또는 .task)
#
# Behavior:
# - Segmentation으로 "person / clothes / skin"을 최대한 확보
# - Pose landmarks로 어깨/팔 위치를 얻어서 sleeve edit에서 "팔 노출" 영역을 마스크에 포함
# - 멀티클래스가 없으면(person only) 보수적으로 torso ROI + 상완 밴드로 대체
# -----------------------------------------------------------------------------

def _find_first(patterns):
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None

def _roi_mask(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), dtype=np.uint8)
    m[int(y0*h):int(y1*h), int(x0*w):int(x1*w)] = 255
    return m

def _clip_pt(x, y, w, h):
    return (int(max(0, min(w - 1, x))), int(max(0, min(h - 1, y))))

def _draw_thick_polyline(mask, pts, thickness):
    if len(pts) < 2:
        return
    for i in range(len(pts) - 1):
        cv2.line(mask, pts[i], pts[i + 1], 255, thickness=thickness, lineType=cv2.LINE_AA)

class MPTasksHelper:
    """
    Wrapper for MediaPipe Tasks:
    - PoseLandmarker: shoulder/elbow/wrist
    - ImageSegmenter: category_mask (preferred) or confidence_masks fallback

    Notes:
    - This class intentionally does NOT auto-download models.
    - If model paths are missing, helper disables itself gracefully.
    """

    def __init__(self, weights_dir: str = "weights"):
        self.weights_dir = weights_dir
        self.pose_path = None
        self.seg_path = None

        # Pose model (.task)
        self.pose_path = _find_first([
            os.path.join(weights_dir, "pose_landmarker_*.task"),
            os.path.join(weights_dir, "pose_*.task"),
        ])

        # Segmentation model (.tflite or .task)
        self.seg_path = _find_first([
            os.path.join(weights_dir, "selfie_multiclass*.tflite"),
            os.path.join(weights_dir, "selfie_multiclass*.task"),
            os.path.join(weights_dir, "selfie_segmenter*.tflite"),
            os.path.join(weights_dir, "selfie_segmenter*.task"),
            os.path.join(weights_dir, "segmenter*.tflite"),
            os.path.join(weights_dir, "segmenter*.task"),
        ])

        self.enabled = bool(self.pose_path and self.seg_path)

        self._pose = None
        self._seg = None

        # Diagnostics string (surface this in status/logs)
        self.diag = f"pose={'OK' if self.pose_path else 'MISSING'}, seg={'OK' if self.seg_path else 'MISSING'}"

    def _lazy_load(self):
        if not self.enabled:
            return

        if self._pose is not None and self._seg is not None:
            return

        # Local import to keep boot fast and avoid crashing when mediapipe isn't installed.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        # BaseOptions contains model_asset_path for local file usage.
        pose_base = mp_python.BaseOptions(model_asset_path=self.pose_path)
        seg_base = mp_python.BaseOptions(model_asset_path=self.seg_path)

        # PoseLandmarker
        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=pose_base,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose = mp_vision.PoseLandmarker.create_from_options(pose_opts)

        # ImageSegmenter
        # output_category_mask=True gives a per-pixel class id map if model supports it.
        seg_opts = mp_vision.ImageSegmenterOptions(
            base_options=seg_base,
            running_mode=mp_vision.RunningMode.IMAGE,
            output_category_mask=True,
            output_confidence_masks=False,
        )
        self._seg = mp_vision.ImageSegmenter.create_from_options(seg_opts)

    def _to_mp_image(self, image_pil: Image.Image):
        import mediapipe as mp
        rgb = np.array(image_pil.convert("RGB"))
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def pose_landmarks(self, image_pil: Image.Image):
        """
        Returns dict of landmark pixels (x,y) or None.
        Keys: l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip
        """
        if not self.enabled:
            return None

        self._lazy_load()
        mp_img = self._to_mp_image(image_pil)

        res = self._pose.detect(mp_img)
        if not res.pose_landmarks:
            return None

        # One pose expected
        lms = res.pose_landmarks[0]
        h, w = np.array(image_pil).shape[:2]

        # MediaPipe Pose landmark indices follow BlazePose convention.
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
            x = int(lms[i].x * w)
            y = int(lms[i].y * h)
            out[k] = _clip_pt(x, y, w, h)

        return out

    def category_mask(self, image_pil: Image.Image):
        """
        Returns uint8 label map (H,W) if available, else None.

        For most segmentation models, category_mask values are class indices.
        The mapping of index->semantic label depends on the model.
        """
        if not self.enabled:
            return None

        self._lazy_load()
        mp_img = self._to_mp_image(image_pil)

        res = self._seg.segment(mp_img)
        if res.category_mask is None:
            return None

        # category_mask is an mp.Image
        # numpy_view() returns view with dtype typically uint8 or int32
        label = res.category_mask.numpy_view()
        if label.ndim == 3:
            # Some builds return HxWx1
            label = label[:, :, 0]
        return label.astype(np.uint8)

# -----------------------------------------------------------------------------
# Mask builders
# -----------------------------------------------------------------------------

def build_top_mask_v5_tasks(image_pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    """
    Upper clothing edit mask.
    Goal: get a stable torso/upper-clothes region, avoid head/face.

    If multiclass segmentation is available, we *prefer* clothes-like classes.
    If not, fallback to (person AND torso ROI).
    """
    img = np.array(image_pil.convert("RGB"))
    h, w = img.shape[:2]

    label = helper.category_mask(image_pil)

    # Conservative head cut ROI
    head_cut = _roi_mask(h, w, 0.0, 0.0, 1.0, 0.26)

    if label is None:
        # Fallback: use torso ROI only
        torso = _roi_mask(h, w, 0.18, 0.28, 0.82, 0.85)
        torso[head_cut > 0] = 0
        return torso

    # Heuristic for class picking:
    # - We do not assume fixed indices across all models.
    # - For practical use, we treat "non-zero" as foreground and then crop to torso.
    #
    # If your selfie_multiclass model has known indices, you can hard-map them here.
    fg = (label != 0).astype(np.uint8) * 255

    torso = _roi_mask(h, w, 0.18, 0.28, 0.82, 0.85)
    m = np.where(torso > 0, fg, 0).astype(np.uint8)
    m[head_cut > 0] = 0

    # Smooth / cleanup
    k = max(7, int(0.015 * min(w, h)) // 2 * 2 + 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)

    return m

def build_sleeve_mask_v5_tasks(image_pil: Image.Image, helper: MPTasksHelper) -> np.ndarray:
    """
    Sleeve/tank/short-sleeve edit mask.
    Goal: include the garment boundary + newly exposed upper arms/shoulders region.

    Strategy:
    1) Start with a torso-ish region (segmentation foreground clipped to torso)
    2) Add upper-arm bands from pose landmarks (shoulder->elbow)
    3) Keep everything within a broad torso+arm ROI to prevent background repaint
    """
    img = np.array(image_pil.convert("RGB"))
    h, w = img.shape[:2]

    base = build_top_mask_v5_tasks(image_pil, helper)

    lm = helper.pose_landmarks(image_pil)

    # 기존 arm_band 위에 추가
    for (sx, sy) in [ls, rs]:
        # shoulder-centered ellipse
        cv2.ellipse(
            arm_band,
            center=(sx, sy),
            axes=(int(arm_th * 1.2), int(arm_th * 0.9)),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1,
        )

    # Upper arm band (pose-guided)
    arm_band = np.zeros((h, w), dtype=np.uint8)
    if lm is not None:
        ls, le = lm["l_shoulder"], lm["l_elbow"]
        rs, re = lm["r_shoulder"], lm["r_elbow"]

        arm_th = max(18, int(0.06 * min(w, h)))  # tuned for sleeve exposure
        _draw_thick_polyline(arm_band, [ls, le], thickness=int(arm_th * 1.4))
        _draw_thick_polyline(arm_band, [rs, re], thickness=int(arm_th * 1.4))

    # Broad ROI to avoid repainting too much (upper body band)
    upper_body_roi = _roi_mask(h, w, 0.08, 0.20, 0.92, 0.60)

    m = np.maximum(base, arm_band)
    m = np.where(upper_body_roi > 0, m, 0).astype(np.uint8)

    # Final dilation gives diffusion enough room to synthesize natural shoulder/arm transitions
    k = max(9, int(0.02 * min(w, h)) // 2 * 2 + 1)
    m = cv2.dilate(m, np.ones((k, k), np.uint8), iterations=1)

    return m
