import numpy as np
import cv2
from PIL import Image

def to_rgb_pil(img_pil: Image.Image) -> Image.Image:
    """PIL 이미지를 RGB로 정규화."""
    if img_pil is None:
        return None
    return img_pil.convert("RGB")

def resize_long_side(pil_img: Image.Image, long_side: int) -> Image.Image:
    """
    긴 변을 long_side로 맞추는 리사이즈.

    - 작업 해상도를 고정하면:
      1) SAM 임베딩 비용이 안정적으로 줄고
      2) SDXL inpaint도 일정한 해상도에서 퀄리티/속도가 안정적
      3) UI 표시/클릭 좌표와 모델 입력이 동일 스케일이라 좌표 꼬임이 줄어듦
    - SDXL 계열은 64의 배수 해상도에서 안정적인 편이라 64 배수로 스냅함.
    """
    w, h = pil_img.size
    if max(w, h) == long_side:
        return pil_img

    scale = long_side / float(max(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # 64 배수 스냅 (너무 작아지지 않도록 최소값도 보장)
    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)

    return pil_img.resize((new_w, new_h), resample=Image.LANCZOS)

def postprocess_mask(mask_np: np.ndarray, expand_px: int, blur_px: int) -> np.ndarray:
    """
    마스크 후처리: 팽창(dilate) + 블러(경계 부드럽게)

    - expand_px:
      마스크가 타이트하면 경계가 '똑' 끊기면서 합성이 티남.
      일정 픽셀만큼 팽창시키면 소매/피부 경계에서 자연스러워짐.
    - blur_px:
      마스크 경계를 부드럽게 해서 인페인팅 결과와 원본의 접합을 자연스럽게 만듦.

    반환: uint8 (H, W), 0~255
    """
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
    """numpy 마스크(uint8)를 PIL(L)로 변환."""
    return Image.fromarray(mask_np).convert("L")

def overlay_mask_on_image(image_pil: Image.Image, mask_np: np.ndarray) -> Image.Image:
    """
    GPU 없는 환경에서 모델을 돌리지 않고도 합성/UI 흐름을 테스트할 때 사용.
    - 마스크 영역을 반투명하게 칠한 미리보기 이미지를 반환.
    """
    img = np.array(image_pil).copy()
    if mask_np is None:
        return image_pil

    alpha = 0.45
    color = np.array([255, 0, 0], dtype=np.uint8)  # 고정 오버레이(테스트용)
    m = (mask_np > 0).astype(np.uint8)

    # 마스크 영역만 색을 섞음
    img[m == 1] = (img[m == 1] * (1 - alpha) + color * alpha).astype(np.uint8)
    return Image.fromarray(img)
