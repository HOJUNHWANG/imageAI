import os
import hashlib
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


class SamMasker:
    """
    SAM wrapper.

    - Click mask: SamPredictor (point prompt)
    - Auto mask : SamAutomaticMaskGenerator (image-wide proposals)

    Caching:
    - set_image() 임베딩은 매우 비싸므로 동일 이미지에 대해서는 1회만 수행.
    - auto generator 결과도 이미지 단위로 캐시.
    """

    def __init__(self, checkpoint_path: str, model_type: str, device: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing SAM checkpoint: {checkpoint_path}")

        self.model_type = model_type
        self.device = device

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)

        # Auto mask generator parameters are tuned for speed-first demo.
        # points_per_side ↑ => better proposals but slower
        self.auto_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            crop_n_layers=0,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=256,
        )

        self._last_image_hash = None
        self._auto_masks_cache = None

    def _hash_image(self, image_np: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(str(image_np.shape).encode("utf-8"))
        h.update(image_np.tobytes())
        return h.hexdigest()

    def set_image_if_needed(self, image_np: np.ndarray):
        img_hash = self._hash_image(image_np)
        if img_hash != self._last_image_hash:
            self.predictor.set_image(image_np)
            self._last_image_hash = img_hash
            self._auto_masks_cache = None

    def mask_from_click(self, image_np: np.ndarray, x: int, y: int) -> np.ndarray:
        self.set_image_if_needed(image_np)

        input_point = np.array([[x, y]], dtype=np.float32)
        input_label = np.array([1], dtype=np.int32)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        return (masks[0].astype(np.uint8) * 255)

    def auto_masks(self, image_np: np.ndarray):
        img_hash = self._hash_image(image_np)
        if self._auto_masks_cache is not None and img_hash == self._last_image_hash:
            return self._auto_masks_cache

        self.set_image_if_needed(image_np)
        masks = self.auto_generator.generate(image_np)
        self._auto_masks_cache = masks
        return masks


class SamMaskerManager:
    """
    Lazy loader for vit_b / vit_h.
    Keeps one model resident to avoid VRAM spikes.
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
        if model_type not in self._ckpt_map:
            raise ValueError(f"Unsupported SAM model_type: {model_type}")

        if self._current is not None and self._current_type == model_type:
            return self._current

        ckpt = self._ckpt_map[model_type]
        self._current = SamMasker(ckpt, model_type=model_type, device=self.device)
        self._current_type = model_type
        return self._current