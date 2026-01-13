# prompt_enricher.py
# -----------------------------------------------------------------------------
# Prompt enrichment module
#
# Goals:
# - Accept natural language (Korean/English) instructions.
# - Convert them into stable diffusion-friendly prompt tokens.
# - Keep it deterministic + debuggable (rule-based, no hidden "AI magic").
#
# Design:
# - Parse: detect intent (target), requested garment, color, action verbs.
# - Expand: add quality/detail tokens + anatomy/photorealism constraints.
# - Provide an explicit "Expanded Prompt Preview" for user trust.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
import re

@dataclass
class PromptParse:
    target: str            # sleeve/top/pants/hair/background
    action: str            # change/replace/remove
    color: str | None
    garment: str | None
    raw: str

# ---- dictionaries -----------------------------------------------------------

KOR_COLOR = {
    "검은": "black", "검정": "black", "블랙": "black",
    "하얀": "white", "흰": "white", "화이트": "white",
    "빨간": "red", "빨강": "red", "레드": "red",
    "파란": "blue", "파랑": "blue", "블루": "blue",
    "초록": "green", "그린": "green",
    "회색": "gray", "그레이": "gray",
    "베이지": "beige",
    "갈색": "brown", "브라운": "brown",
}

# Garment keywords (expand as needed)
GARMENT_SYNONYMS = [
    ("tank top", ["tank top", "sleeveless top", "민소매", "나시", "탱크탑"]),
    ("t-shirt", ["t-shirt", "tee", "반팔", "티셔츠", "티"]),
    ("shirt", ["shirt", "button-up", "dress shirt", "셔츠", "와이셔츠"]),
    ("hoodie", ["hoodie", "후드", "후드티"]),
    ("jacket", ["jacket", "자켓", "재킷"]),
    ("sweater", ["sweater", "니트", "스웨터"]),
    ("jeans", ["jeans", "denim", "청바지"]),
    ("trousers", ["trousers", "pants", "슬랙스", "바지"]),
]

# Target keywords (what region to edit)
TARGET_RULES = [
    ("sleeve",  ["sleeve", "sleeveless", "민소매", "나시", "반팔", "긴팔"]),
    ("top",     ["top", "shirt", "t-shirt", "blouse", "hoodie", "jacket", "상의", "셔츠", "티셔츠"]),
    ("pants",   ["pants", "trousers", "jeans", "슬랙스", "바지", "하의"]),
    ("hair",    ["hair", "hairstyle", "머리", "헤어", "염색", "컬러"]),
    ("background", ["background", "bg", "배경", "뒤", "백그라운드"]),
]

ACTION_RULES = [
    ("replace", ["바꿔", "변경", "교체", "replace", "change", "swap"]),
    ("remove",  ["없애", "지워", "remove", "delete"]),
    ("add",     ["추가", "add"]),
]

QUALITY_TOKENS = [
    "photorealistic",
    "high detail",
    "realistic lighting",
    "sharp focus",
    "realistic fabric texture",
]

ANATOMY_TOKENS = [
    "natural shoulders and arms",
    "realistic skin texture",
]

NEGATIVE_DEFAULT = "bad anatomy, extra arms, extra hands, deformed, blurry, artifacts, low quality"

# ---- parsing ----------------------------------------------------------------

def _lower(s: str) -> str:
    return (s or "").strip().lower()

def detect_color(text: str) -> str | None:
    t = _lower(text)
    for k, v in KOR_COLOR.items():
        if k in text:
            return v
    for c in ["black", "white", "red", "blue", "green", "gray", "brown", "beige"]:
        if re.search(rf"\b{c}\b", t):
            return c
    return None

def detect_garment(text: str) -> str | None:
    t = _lower(text)
    for canonical, keys in GARMENT_SYNONYMS:
        for k in keys:
            if k.lower() in t:
                return canonical
    return None

def detect_target(text: str) -> str:
    t = _lower(text)
    # explicit directive wins
    m = re.search(r"target\s*=\s*(\w+)", t)
    if m:
        return m.group(1)

    for target, keys in TARGET_RULES:
        if any(k.lower() in t for k in keys):
            return target
    # default
    return "top"

def detect_action(text: str) -> str:
    t = _lower(text)
    for a, keys in ACTION_RULES:
        if any(k.lower() in t for k in keys):
            return a
    return "replace"

def parse_prompt(text: str) -> PromptParse:
    return PromptParse(
        target=detect_target(text),
        action=detect_action(text),
        color=detect_color(text),
        garment=detect_garment(text),
        raw=text or "",
    )

# ---- enrichment -------------------------------------------------------------

def enrich_positive(text: str) -> tuple[str, PromptParse]:
    """
    Returns:
      (expanded_prompt, parse_info)

    Strategy:
    - If user already wrote diffusion-style tokens, keep them.
    - If user wrote natural language, append missing tokens.
    - Always keep it readable: join tokens with commas.
    """
    info = parse_prompt(text)
    base = (text or "").strip()

    tokens: list[str] = []

    # Garment phrasing depends on target
    if info.target in ("sleeve", "top"):
        if info.garment:
            tokens.append(info.garment)
        else:
            # fallback
            tokens.append("top")

        # Sleeve-specific realism tokens
        if info.target == "sleeve":
            tokens.append("sleeveless" if "sleeveless" in _lower(base) or "민소매" in base or info.garment == "tank top" else "short sleeve")

    if info.target == "pants":
        if info.garment in ("jeans", "trousers"):
            tokens.append(info.garment)
        else:
            tokens.append("trousers")

    if info.target == "hair":
        tokens.append("realistic hair")

    if info.target == "background":
        tokens.append("clean background")

    # Color token
    if info.color:
        tokens.insert(0, info.color)

    # Quality tokens
    tokens.extend(QUALITY_TOKENS)

    # Anatomy tokens only when clothing edits might expose skin
    if info.target in ("sleeve", "top"):
        tokens.extend(ANATOMY_TOKENS)

    # If user already provided token-like prompt, don't duplicate too aggressively.
    # Keep the original, append enrichment.
    expanded = base
    if expanded:
        expanded = expanded + ", " + ", ".join(tokens)
    else:
        expanded = ", ".join(tokens)

    # Normalize extra spaces/commas
    expanded = re.sub(r"\s*,\s*", ", ", expanded).strip(" ,")
    return expanded, info

def enrich_negative(text: str | None) -> str:
    t = (text or "").strip()
    if not t:
        return NEGATIVE_DEFAULT
    # Append defaults only if user didn't already include core anatomy blockers
    if "bad anatomy" not in t.lower():
        t = t + ", " + NEGATIVE_DEFAULT
    return re.sub(r"\s*,\s*", ", ", t).strip(" ,")