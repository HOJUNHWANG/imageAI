import os
import hashlib
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


class SamMasker:
    """
    SAM wrapper.

    - Click mask: SamPredictor 기반 (point prompt)
    - Auto mask: SamAutomaticMaskGenerator 기반 (image-wide proposals)

    Performance notes:
    - set_image()는 임베딩 생성 비용이 크므로 동일 이미지에 대해 캐시가 필요함.
    - AutomaticMaskGenerator도 내부적으로 임베딩이 필요하므로, "이미지 단위"로 결과 캐싱이 유효함.
    """

    def __init__(self, checkpoint_path: str, model_type: str, device: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing SAM checkpoint: {checkpoint_path}")

        self.model_type = model_type
        self.device = device

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)

        # Click-based predictor
        self.predictor = SamPredictor(sam)

        # Auto mask generator
        # - points_per_side를 줄이면 빠르지만 마스크 후보 품질/개수가 줄어듦.
        # - pred_iou_thresh / stability_score_thresh는 품질 필터.
        # - 포트폴리오 데모 목적이면 "속도 우선" 기본값을 추천.
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
        self._auto_masks_cache = None  # list[dict] from SAM generator

    def _hash_image(self, image_np: np.ndarray) -> str:
        """
        Image hash for cache invalidation.

        - working 해상도를 고정(예: 1024)하면 bytes 크기가 안정적이라 전체 bytes 해시가 부담이 덜함.
        - 캐시 히트율이 높을수록 set_image / auto generate 비용이 크게 절감됨.
        """
        h = hashlib.sha256()
        h.update(str(image_np.shape).encode("utf-8"))
        h.update(image_np.tobytes())
        return h.hexdigest()

    def set_image_if_needed(self, image_np: np.ndarray):
        """
        Predictor 임베딩 캐시.

        동일 이미지면 set_image()를 생략.
        """
        img_hash = self._hash_image(image_np)
        if img_hash != self._last_image_hash:
            self.predictor.set_image(image_np)
            self._last_image_hash = img_hash
            # 이미지가 바뀌면 auto 마스크 캐시도 무효화
            self._auto_masks_cache = None

    def mask_from_click(self, image_np: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Click -> mask.
        returns: uint8 mask (H, W) in {0,255}
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

    def auto_masks(self, image_np: np.ndarray):
        """
        Auto generate masks for the image.

        returns: list of SAM mask dicts:
          - 'segmentation' : np.ndarray bool (H, W)
          - 'area'         : int
          - 'bbox'         : [x, y, w, h]
          - plus other metadata
        """
        # 자동 마스크도 이미지 단위로 캐싱
        img_hash = self._hash_image(image_np)
        if self._auto_masks_cache is not None and img_hash == self._last_image_hash:
            return self._auto_masks_cache

        # Predictor 캐시와 해시를 통일해서 관리
        self.set_image_if_needed(image_np)

        masks = self.auto_generator.generate(image_np)
        self._auto_masks_cache = masks
        return masks


class SamMaskerManager:
    """
    vit_b / vit_h 모두 지원하는 lazy-loader.

    - 모델을 동시에 올리면 VRAM 부담이 커질 수 있어 1개만 유지.
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