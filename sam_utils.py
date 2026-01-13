import os
import hashlib
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

class SamMasker:
    """
    SAM 마스커.

    속도 병목 포인트:
    - predictor.set_image()가 "이미지 임베딩"을 생성하는 과정이라 비용이 매우 큼.
    - 같은 이미지에서 여러 번 클릭할 때 set_image()를 매번 호출하면 클릭마다 느려짐.

    해결:
    - 마지막 set_image()에 사용된 이미지를 해시로 캐시하고, 동일하면 set_image() 생략.
    """

    def __init__(self, checkpoint_path: str, model_type: str, device: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing SAM checkpoint: {checkpoint_path}")

        self.model_type = model_type
        self.device = device

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)
        self._last_image_hash = None

    def _hash_image(self, image_np: np.ndarray) -> str:
        """
        이미지 동일성 체크용 해시.
        - 안정성을 위해 shape + 전체 bytes를 해시.
        - working 해상도를 1024로 고정하면 bytes 크기가 과도하게 커지지 않음.
        """
        h = hashlib.sha256()
        h.update(str(image_np.shape).encode("utf-8"))
        h.update(image_np.tobytes())
        return h.hexdigest()

    def set_image_if_needed(self, image_np: np.ndarray):
        """동일 이미지면 set_image() 호출을 생략."""
        img_hash = self._hash_image(image_np)
        if img_hash != self._last_image_hash:
            self.predictor.set_image(image_np)
            self._last_image_hash = img_hash

    def mask_from_click(self, image_np: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        클릭 좌표 기반 마스크 생성.

        반환: uint8 (H, W), 0 또는 255
        """
        self.set_image_if_needed(image_np)

        input_point = np.array([[x, y]], dtype=np.float32)
        input_label = np.array([1], dtype=np.int32)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        return (masks[0].astype(np.uint8) * 255)

class SamMaskerManager:
    """
    vit_b / vit_h를 모두 지원하는 매니저.

    - 두 모델을 동시에 GPU에 올리면 VRAM을 크게 잡아먹을 수 있음.
    - 그래서 '필요할 때만' 로드하고, 현재 선택된 모델 1개만 유지하는 방식.
    """

    def __init__(self, weights_dir: str, device: str):
        self.weights_dir = weights_dir
        self.device = device
        self._current = None
        self._current_type = None

        self._ckpt_map = {
            "vit_b": os.path.join(weights_dir, "sam_vit_b_01ec64.pth"),
            "vit_h": os.path.join(weights_dir, "sam_vit_h_4b8939.pth"),
        }

    def get(self, model_type: str) -> SamMasker:
        """
        model_type 변경 시에만 새로 로드.
        - 변경하면 기존 캐시/임베딩 상태도 새로 시작.
        """
        if model_type not in self._ckpt_map:
            raise ValueError(f"Unsupported SAM model_type: {model_type}")

        if self._current is not None and self._current_type == model_type:
            return self._current

        ckpt = self._ckpt_map[model_type]
        self._current = SamMasker(ckpt, model_type=model_type, device=self.device)
        self._current_type = model_type
        return self._current