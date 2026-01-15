# sam_utils.py
import os
import numpy as np
from functools import lru_cache
from segment_anything import sam_model_registry, SamPredictor

class SamMasker:
    def __init__(self, model_type: str, checkpoint_path: str, device: str):
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def predict_from_click(self, image_rgb: np.ndarray, x: int, y: int) -> np.ndarray:
        """Generate mask from single positive click point."""
        self.predictor.set_image(image_rgb)
        point_coords = np.array([[x, y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        masks, _, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        return (masks[0] * 255).astype(np.uint8)

class SamMaskerManager:
    def __init__(self, weights_dir: str, device: str):
        self.weights_dir = weights_dir
        self.device = device

    @lru_cache(maxsize=4)
    def _load(self, model_type: str) -> SamMasker:
        ckpt_map = {
            "vit_b": os.path.join(self.weights_dir, "sam_vit_b_01ec64.pth"),
            "vit_h": os.path.join(self.weights_dir, "sam_vit_h_4b8939.pth"),
        }
        ckpt = ckpt_map.get(model_type)
        if not ckpt or not os.path.exists(ckpt):
            raise FileNotFoundError(f"SAM checkpoint not found for {model_type}: {ckpt}")

        return SamMasker(model_type, ckpt, self.device)

    def get(self, model_type: str) -> SamMasker:
        if model_type not in ["vit_b", "vit_h"]:
            raise ValueError(f"Unsupported SAM model_type: {model_type}")
        return self._load(model_type)