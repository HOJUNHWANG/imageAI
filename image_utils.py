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
    """
    SAM AutomaticMaskGenerator output -> uint8 mask 0/255.
    """
    seg = sam_mask_dict["segmentation"]  # bool (H,W)
    return (seg.astype(np.uint8) * 255)


def bbox_center(bbox):
    """
    bbox: [x, y, w, h]
    returns: (cx, cy)
    """
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))