import numpy as np
import cv2
from PIL import Image


def to_rgb_pil(img_pil: Image.Image) -> Image.Image:
    if img_pil is None:
        return None
    return img_pil.convert("RGB")


def resize_long_side(pil_img: Image.Image, long_side: int) -> Image.Image:
    w, h = pil_img.size
    if max(w, h) == long_side:
        return pil_img

    scale = long_side / float(max(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)

    return pil_img.resize((new_w, new_h), resample=Image.LANCZOS)


def postprocess_mask(mask_np: np.ndarray, expand_px: int, blur_px: int) -> np.ndarray:
    m = mask_np.copy().astype(np.uint8)

    if expand_px > 0:
        k = int(expand_px)
        kernel = np.ones((k, k), np.uint8)
        m = cv2.dilate(m, kernel, iterations=1)

    if blur_px > 0:
        k = int(blur_px)
        if k % 2 == 0:
            k += 1
        m = cv2.GaussianBlur(m, (k, k), 0)

    return np.clip(m, 0, 255).astype(np.uint8)


def mask_to_pil(mask_np: np.ndarray) -> Image.Image:
    return Image.fromarray(mask_np).convert("L")


def overlay_mask_on_image(image_pil: Image.Image, mask_np: np.ndarray) -> Image.Image:
    img = np.array(image_pil).copy()
    if mask_np is None:
        return image_pil

    alpha = 0.45
    color = np.array([255, 0, 0], dtype=np.uint8)
    m = (mask_np > 0).astype(np.uint8)

    img[m == 1] = (img[m == 1] * (1 - alpha) + color * alpha).astype(np.uint8)
    return Image.fromarray(img)


def sam_dict_to_uint8_mask(sam_mask_dict: dict) -> np.ndarray:
    seg = sam_mask_dict["segmentation"]
    return (seg.astype(np.uint8) * 255)


def bbox_center(bbox):
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def roi_mask(h: int, w: int, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """
    Build a rectangular ROI mask in normalized coordinates.
    - (x0,y0,x1,y1) are normalized [0..1]
    returns bool mask (H,W)
    """
    xx0 = int(max(0, min(w - 1, round(x0 * w))))
    yy0 = int(max(0, min(h - 1, round(y0 * h))))
    xx1 = int(max(0, min(w, round(x1 * w))))
    yy1 = int(max(0, min(h, round(y1 * h))))

    m = np.zeros((h, w), dtype=bool)
    m[yy0:yy1, xx0:xx1] = True
    return m


def overlap_ratio(mask_bool: np.ndarray, roi_bool: np.ndarray) -> float:
    """
    overlap = |mask ∩ roi| / |mask|
    """
    denom = float(mask_bool.sum()) + 1e-6
    inter = float((mask_bool & roi_bool).sum())
    return inter / denom


def union_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """
    Union two uint8 masks (0/255).
    """
    if mask_a is None:
        return mask_b
    if mask_b is None:
        return mask_a
    return np.where((mask_a > 0) | (mask_b > 0), 255, 0).astype(np.uint8)