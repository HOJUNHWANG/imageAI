# ui_text.py
# Centralized UI copy so the main app remains readable.
# Keeping all help/guide strings in one place makes iteration fast.

APP_TITLE = "ImageAI — Prompt-based Inpainting (Local)"
APP_SUBTITLE = "MediaPipe Tasks (semantic) + SAM (manual) + SDXL Inpainting"

HOW_TO_MD = r"""
### How to use
1) **Upload** an image.
2) Choose a mask method:
   - **Auto Mask (v5 semantic)**: best for clothing edits like *long sleeve → sleeveless*.
   - **Manual Mask (SAM click)**: click the area to edit; good for precise control.
3) Adjust **Mask Expand / Blur** to control blending.
4) Enter **Prompt** (+ optional **Negative Prompt**).
5) Click **Apply** to generate.

---

### Tips for garment edits (sleeve → short sleeve / sleeveless)
- Use prompts that describe both **garment** and **arm/shoulder realism**:
  - `black tank top, sleeveless, natural shoulders and arms, realistic fabric, photorealistic`
- Increase **Strength** when geometry changes (sleeves removed exposes new arm skin).
- Increase **Mask Expand** if boundaries look cut off.

---

### Parameters reference (what low/high means)

**Working Long Side**
- Low: faster, lower detail, fewer pixels to synthesize
- High: slower, more detail; VRAM usage increases

**Mask Expand (px)**
- Low: edits stay inside the mask; may leave hard seams
- High: edits spill outward; improves blending but may change nearby details

**Mask Blur (px)**
- Low: sharp boundaries; can look “cut out”
- High: smoother transitions; too high can cause “muddy” edges

**Steps**
- Low: faster; less stable detail
- High: slower; better detail (diminishing returns after ~35–40 on SDXL)

**Strength**
- Low: preserves original; small edits
- High: stronger edits; better for sleeve removal but can drift identity/details

**Guidance (CFG)**
- Low: more natural / less literal
- High: follows prompt more aggressively; too high can create artifacts
"""

SLIDER_HINTS_MD = r"""
### Recommended starting points (SDXL Inpainting)
- **Steps**: 28–34
- **Strength**:
  - color/material change: **0.55–0.70**
  - sleeve → sleeveless: **0.82–0.90**
- **Guidance (CFG)**: 5.5–7.0
- **Mask Expand**: 16–24
- **Mask Blur**: 8–14
"""
