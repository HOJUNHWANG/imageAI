import os
import numpy as np
import torch
from PIL import Image

# -----------------------------------------------------------------------------
# Human Parsing backend (optional)
#
# Goal:
# - Provide semantic masks like "upper-clothes" / "arms"
# - If weights are missing, fail gracefully and let the app fallback to SAM/manual.
#
# Notes:
# - There are multiple open-source human parsing models (SCHP, CIHP, LIP, etc.).
# - This module is designed as a plug-in point.
# -----------------------------------------------------------------------------

class HumanParsing:
    """
    Semantic human parsing wrapper.

    Contract:
    - predict(image_pil) -> dict[str, np.ndarray]
      returns masks as uint8 0/255 in image resolution.

    Current implementation:
    - Placeholder "adapter" that expects an ONNX model OR a torch model you provide.
    - In portfolio/dev, this is the correct place to swap parsing backends.
    """

    def __init__(self, device: str):
        self.device = device
        self.enabled = False

        # Path convention: weights/human_parsing.(onnx|pth)
        base_dir = os.path.dirname(__file__)
        self.weights_dir = os.path.join(base_dir, "weights")
        self.onnx_path = os.path.join(self.weights_dir, "human_parsing.onnx")
        self.pth_path = os.path.join(self.weights_dir, "human_parsing.pth")

        # Lightweight switch:
        # - If neither exists, disable parsing cleanly.
        if os.path.exists(self.onnx_path) or os.path.exists(self.pth_path):
            self.enabled = True

        # Lazy-loaded model handle(s)
        self._model = None
        self._backend = None  # "onnx" or "torch"

    def _lazy_load(self):
        if self._model is not None:
            return

        # IMPORTANT:
        # This project intentionally does not auto-download model weights to avoid
        # enterprise/EDR issues and to keep behavior explicit for users.
        if os.path.exists(self.onnx_path):
            # Optional: ONNX backend (recommended for CPU laptops)
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
            self._model = ort.InferenceSession(self.onnx_path, providers=providers)
            self._backend = "onnx"
            return

        if os.path.exists(self.pth_path):
            # Optional: Torch backend
            # You can wire your SCHP/CIHP/LIP model here.
            # For now, this is a stub to keep the app structure correct.
            raise RuntimeError(
                "human_parsing.pth exists but torch backend is not wired. "
                "Use ONNX for now or plug your torch model in _lazy_load()."
            )

        # No weights
        self.enabled = False

    def predict(self, image_pil: Image.Image) -> dict:
        """
        Returns:
          {
            "person": uint8 mask,
            "upper_clothes": uint8 mask (if available),
            "arms": uint8 mask (if available),
          }

        If parsing backend doesn't provide fine labels, return only "person".
        """
        if not self.enabled:
            return {}

        self._lazy_load()

        # Minimal preprocessing for ONNX:
        # - resize to fixed input
        # - normalize
        # - run inference
        #
        # The exact preprocessing depends on the ONNX model you choose.
        # Keep this method as the only place that "knows" preprocessing.
        if self._backend == "onnx":
            import cv2

            img = np.array(image_pil.convert("RGB"))
            h, w = img.shape[:2]

            # Common parsing models use 512x512 input
            inp = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            inp = (inp - 0.5) / 0.5  # typical [-1,1] normalization
            inp = np.transpose(inp, (2, 0, 1))[None, ...]  # NCHW

            # Model-specific: adjust input/output names accordingly.
            # You will set these once you pick a specific ONNX parsing model.
            input_name = self._model.get_inputs()[0].name
            out = self._model.run(None, {input_name: inp})[0]  # [1, C, H, W] or [1, H, W]
            out = out[0]

            # If output is logits [C,H,W], convert to label map by argmax
            if out.ndim == 3:
                label = np.argmax(out, axis=0).astype(np.uint8)
            else:
                label = out.astype(np.uint8)

            # Resize label back to original resolution
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

            # IMPORTANT:
            # Label mapping depends on the model/dataset.
            # Here we provide a conservative default:
            # - person: any non-background label
            # - upper_clothes/arms: left empty unless you set mappings
            person = (label != 0).astype(np.uint8) * 255

            return {
                "person": person,
                # These require dataset-specific label IDs:
                # "upper_clothes": ...,
                # "arms": ...,
            }

        return {}