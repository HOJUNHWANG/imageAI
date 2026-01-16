# prompt_enricher.py

QUALITY_TOKENS = [
    "photorealistic", "high detail", "realistic lighting", "sharp focus",
    "natural skin texture", "consistent skin tone", "clean edges", "8k"
]

ANATOMY_TOKENS = [
    "natural shoulders and arms", "realistic skin texture", "subsurface scattering",
    "skin pores", "fine body hair", "consistent shading", "anatomically plausible torso",
    "soft natural shadows", "visible skin details"
]

NEGATIVE_DEFAULT = "low quality, blurry, jpeg artifacts, bad anatomy, deformed, extra arms, extra hands, extra fingers, missing fingers, plastic skin, over-smoothed skin, uncanny, text, watermark, logo"

KOR_COLOR = {
    "검은": "black", "검정": "black", "블랙": "black",
    "하얀": "white", "흰": "white", "화이트": "white",
    "빨간": "red", "빨강": "red", "레드": "red",
    "파란": "blue", "파랑": "blue", "블루": "blue",
    "초록": "green", "녹색": "green", "그린": "green",
    "회색": "gray", "그레이": "gray",
    "갈색": "brown", "브라운": "brown",
    "베이지": "beige", "아이보리": "ivory",
    "노란": "yellow", "노랑": "yellow",
    "보라": "purple", "퍼플": "purple",
    "분홍": "pink", "핑크": "pink",
}

def enrich_positive(text: str | None) -> tuple[str, str]:
    """Natural language → diffusion-friendly prompt."""
    t = (text or "").strip().lower()

    # Parse target
    target = "top"
    if any(k in t for k in ["소매", "sleeve", "sleeveless", "tank"]):
        target = "sleeve"
    elif any(k in t for k in ["상의", "shirt", "t-shirt", "top", "blouse"]):
        target = "top"

    # Color
    color = None
    for kor, eng in KOR_COLOR.items():
        if kor in t:
            color = eng
            break

    # Garment
    garment = None
    for g in ["tank top", "t-shirt", "shirt", "blouse", "jacket"]:
        if g in t:
            garment = g
            break

    # Build tokens
    tokens = []
    if color:
        tokens.append(color)

    if target == "sleeve":
        tokens.append("sleeveless" if "remove" in t else "sleeve")
    elif target == "top":
        tokens.append("shirtless" if "remove" in t else "shirt")

    if target in ("sleeve", "top"):
        tokens.extend(ANATOMY_TOKENS)

    tokens.extend(QUALITY_TOKENS)

    expanded = ", ".join(set(tokens))  # 중복 제거

    # CLIP 77 토큰 제한 우회
    token_list = expanded.split(", ")
    if len(token_list) > 75:
        mid = len(token_list) // 2
        part1 = ", ".join(token_list[:mid])
        part2 = ", ".join(token_list[mid:])
        expanded = f"{part1} BREAK {part2}"  # BREAK로 multi-prompt 구분

    info = f"target={target}, color={color}, garment={garment}"
    return expanded, info

def enrich_negative(text: str | None) -> str:
    """Negative prompt enrichment."""
    t = (text or "").strip()
    if not t:
        return NEGATIVE_DEFAULT

    tokens = [tok.strip() for tok in t.split(",") if tok.strip()]
    default_tokens = [tok.strip() for tok in NEGATIVE_DEFAULT.split(",") if tok.strip()]
    combined = set(tokens + default_tokens)
    return ", ".join(combined)