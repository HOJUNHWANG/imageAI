# image_utils.py
import numpy as np
from PIL import Image, ImageFilter

# cv2 import 안전하게 (없으면 fallback)
try:
    import cv2
except ImportError:
    cv2 = None

def to_rgb_pil(pil: Image.Image) -> Image.Image:
    """Convert PIL image to RGB mode."""
    if pil is None:
        return None
    return pil.convert("RGB")

def resize_long_side(pil: Image.Image, long_side: int) -> Image.Image:
    """Resize image keeping aspect ratio, long side = long_side."""
    w, h = pil.size
    if max(w, h) == long_side:
        return pil

    scale = long_side / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # SDXL 호환: 64 배수로 맞춤
    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)

    return pil.resize((new_w, new_h), Image.LANCZOS)

def postprocess_mask(mask_np: np.ndarray, expand_px: int = 0, blur_px: int = 0) -> np.ndarray:
    """Expand (dilate) and blur mask. cv2 없어도 PIL fallback."""
    m = mask_np.copy().astype(np.uint8)

    if expand_px > 0:
        k = int(expand_px)
        if cv2 is not None:
            kernel = np.ones((k, k), np.uint8)
            m = cv2.dilate(m, kernel, iterations=1)
        else:
            img = Image.fromarray(m)
            size = k * 2 + 1
            img = img.filter(ImageFilter.MaxFilter(size=size))
            m = np.array(img)

    if blur_px > 0:
        k = int(blur_px)
        if cv2 is not None:
            m = cv2.GaussianBlur(m, (k, k), 0)
        else:
            img = Image.fromarray(m)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_px))
            m = np.array(img)

    return np.clip(m, 0, 255).astype(np.uint8)

def overlay_mask(image_rgb: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay red mask on RGB image for visualization."""
    vis = image_rgb.copy()
    red = np.zeros_like(vis)
    red[..., 0] = 255
    m = (mask_u8 > 0)
    vis[m] = (vis[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)
    return vis

def get_rect_roi_mask(h: int, w: int, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Rectangular ROI mask from normalized coordinates [0~1]."""
    xx0 = int(max(0, min(w - 1, round(x0 * w))))
    yy0 = int(max(0, min(h - 1, round(y0 * h))))
    xx1 = int(max(0, min(w, round(x1 * w))))
    yy1 = int(max(0, min(h, round(y1 * h))))

    m = np.zeros((h, w), dtype=bool)
    m[yy0:yy1, xx0:xx1] = True
    return m

def overlap_ratio(mask_bool: np.ndarray, roi_bool: np.ndarray) -> float:
    """Overlap ratio: |intersection| / |mask|."""
    denom = float(mask_bool.sum()) + 1e-6
    inter = float((mask_bool & roi_bool).sum())
    return inter / denom

def union_masks(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> np.ndarray:
    """Union two uint8 masks (0/255)."""
    if mask_a is None:
        return mask_b if mask_b is not None else np.zeros((1, 1), dtype=np.uint8)
    if mask_b is None:
        return mask_a
    return np.where((mask_a > 0) | (mask_b > 0), 255, 0).astype(np.uint8)