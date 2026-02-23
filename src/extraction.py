"""
Field extraction from OCR blocks using spatial heuristics.
Returns dict of field name -> { "value", "bbox" }.
"""
from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def _bbox_height(b: dict) -> int:
    box = b.get("bbox", [0, 0, 0, 0])
    return max(1, box[3] - box[1])


def _bbox_area(b: dict) -> int:
    box = b.get("bbox", [0, 0, 0, 0])
    return max(1, (box[2] - box[0]) * (box[3] - box[1]))


def _bbox_y_center(b: dict) -> float:
    box = b.get("bbox", [0, 0, 0, 0])
    return (box[1] + box[3]) / 2


def _bbox_x_center(b: dict) -> float:
    box = b.get("bbox", [0, 0, 0, 0])
    return (box[0] + box[2]) / 2


def _y_distance(a: dict, b: dict) -> float:
    return abs(_bbox_y_center(a) - _bbox_y_center(b))


def _x_overlap_ratio(a: dict, b: dict) -> float:
    """Fraction of horizontal overlap between two blocks (0-1)."""
    a_box, b_box = a.get("bbox", [0, 0, 0, 0]), b.get("bbox", [0, 0, 0, 0])
    overlap = max(0, min(a_box[2], b_box[2]) - max(a_box[0], b_box[0]))
    min_width = min(a_box[2] - a_box[0], b_box[2] - b_box[0])
    return overlap / max(1, min_width)


def _merge_bboxes(blocks: list[dict]) -> list[int] | None:
    """Compute a single bbox spanning all blocks."""
    boxes = [b.get("bbox") for b in blocks if b.get("bbox")]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


# ---------------------------------------------------------------------------
# Generic text filters
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(
    r"^(DISTILLED AND BOTTLED BY|BOTTLED BY|DISTILLED BY|PRODUCED BY|IMPORTED BY|"
    r"BREWED\s*[&]\s*BOTTLED BY|BREWED AND BOTTLED BY|BREWED BY|"
    r"GOVERNMENT WARNING|Brand Label|Back Label)[\s:]*$",
    re.I,
)

_BOTTLER_HEADER_RE = re.compile(
    r"(Distilled\s+and\s+Bottled\s+by|Bottled\s+by|Distilled\s+by|Produced\s+by"
    r"|Produced\s+and\s+Bottled\s+by|Imported\s+by"
    r"|Brewed\s+(?:and|&)\s+Bottled\s+by|Brewed\s+by"
    r"|Manufactured\s+by|Made\s+by"
    r"|Cellared\s+and\s+Bottled\s+by|Vinted\s+and\s+Bottled\s+by"
    r"|Blended\s+and\s+Bottled\s+by)",
    re.I,
)

# Brand suffixes kept in the brand name (domain-specific)
_BRAND_SUFFIXES = frozenset({
    "DISTILLERY", "DISTILLERS", "BREWING", "BREWERY", "WINERY", "VINEYARDS",
    "CELLARS", "IMPORTS", "SPIRITS", "ESTATES",
})

# Corporate suffixes stripped from brand (generic legal forms)
_CORP_SUFFIXES = frozenset({
    "COMPANY", "CO", "INC", "LLC", "LTD", "CORP", "CORPORATION",
})

_ALL_COMPANY_SUFFIXES = _BRAND_SUFFIXES | _CORP_SUFFIXES | {"RESERVE", "SELECTION"}

_WARNING_WORDS = frozenset({
    "GOVERNMENT", "WARNING", "ACCORDING", "SURGEON", "GENERAL", "WOMEN",
    "SHOULD", "DRINK", "ALCOHOLIC", "BEVERAGES", "DURING", "PREGNANCY",
    "BECAUSE", "RISK", "BIRTH", "DEFECTS", "CONSUMPTION", "IMPAIRS",
    "ABILITY", "DRIVE", "CAR", "OPERATE", "MACHINERY", "CAUSE", "HEALTH",
    "PROBLEMS",
})


_LABEL_MARKER_RE = re.compile(
    r"^[↑↓\s]*Brand\s+Labels?\s*[↑↓\s]*$|^[↑↓\s]*Back\s+Labels?\s*[↑↓\s]*$",
    re.I,
)


def _is_junk(text: str) -> bool:
    """Filter out likely OCR noise or non-brand text."""
    t = text.strip()
    if len(t) <= 1:
        return True
    alpha_ratio = sum(c.isalpha() for c in t) / max(1, len(t))
    if alpha_ratio < 0.4:
        return True
    if _HEADER_RE.match(t):
        return True
    if _LABEL_MARKER_RE.match(t):
        return True
    upper = t.upper()
    if upper in _WARNING_WORDS:
        return True
    if upper in {"AND", "THE", "OF", "OR", "BY", "NOT", "TO", "A", "IN", "ON", "AT", "FOR", "IT", "IS"}:
        return True
    return False


# ---------------------------------------------------------------------------
# Class-type keywords (Bug 7: expanded with beer styles)
# ---------------------------------------------------------------------------

_CLASS_KEYWORDS = (
    # --- Distilled Spirits (27 CFR Part 5 / TTB Spirits BAM Ch4) ---
    "Vodka", "Gin", "Distilled Gin", "Compounded Gin", "Redistilled Gin",
    "Rum", "Tequila", "Mezcal",
    "Whiskey", "Whisky", "Bourbon", "Bourbon Whisky", "Bourbon Whiskey",
    "Rye Whisky", "Rye Whiskey", "Wheat Whisky", "Wheat Whiskey",
    "Malt Whisky", "Malt Whiskey", "Corn Whisky", "Corn Whiskey",
    "Straight Bourbon Whisky", "Straight Bourbon Whiskey",
    "Straight Rye Whisky", "Straight Rye Whiskey",
    "Straight Wheat Whisky", "Straight Wheat Whiskey",
    "Straight Malt Whisky", "Straight Malt Whiskey",
    "Straight Corn Whisky", "Straight Corn Whiskey",
    "Straight Whisky", "Straight Whiskey",
    "Light Whisky", "Light Whiskey",
    "Blended Whisky", "Blended Whiskey",
    "Blended Bourbon Whisky", "Blended Bourbon Whiskey",
    "Blended Rye Whisky", "Blended Rye Whiskey",
    "Spirit Whisky", "Spirit Whiskey",
    "Scotch Whisky", "Irish Whiskey", "Canadian Whisky",
    "Kentucky Straight Bourbon Whiskey", "Tennessee Whiskey",
    "Single Malt", "Single Barrel", "Single Pot Still",
    "Brandy", "Cognac", "Armagnac", "Calvados", "Pisco", "Grappa",
    "Fruit Brandy", "Applejack",
    "Liqueur", "Cordial", "Sloe Gin", "Amaretto", "Triple Sec",
    "Sambuca", "Absinthe", "Bitters", "Aquavit",
    "Neutral Spirits", "Grain Spirits",
    # --- Wine (27 CFR Part 4 / TTB Wine BAM Ch5) ---
    "Table Wine", "Light Wine", "Dessert Wine",
    "Red Wine", "Rose Wine", "White Wine",
    "American Red Wine", "American White Wine",
    "Sparkling Wine", "Champagne", "Carbonated Wine",
    "Fortified Wine", "Sherry", "Port", "Madeira", "Marsala",
    "Vermouth", "Sake", "Mead", "Honey Wine",
    "Fruit Wine", "Citrus Wine", "Berry Wine",
    "Agricultural Wine", "Retsina", "Natural Wine", "Special Natural Wine",
    # --- Beer / Malt Beverages (27 CFR Part 7) ---
    "Beer", "Ale", "Lager", "Stout", "Porter",
    "Pale Ale", "India Pale Ale", "IPA",
    "Barleywine Ale", "Barleywine", "Barley Wine",
    "Pilsner", "Wheat Beer", "Hefeweizen",
    "Kolsch", "Saison", "Sour", "Gose",
    "Malt Liquor", "Malt Beverage", "Flavored Malt Beverage",
    "Hard Seltzer", "Hard Cider",
    # --- Generic qualifiers ---
    "Straight", "Blended", "Reserve", "Aged",
    "Scotch", "Irish", "Canadian", "Kentucky", "Tennessee",
    "Shochu", "Baijiu", "Cachaca",
)
_CLASS_KEYWORD_SET = frozenset(k.upper() for k in _CLASS_KEYWORDS)
_CLASS_RE = re.compile(
    r"(?:" + "|".join(re.escape(k) for k in sorted(_CLASS_KEYWORDS, key=len, reverse=True)) + r")",
    re.I,
)


# ---------------------------------------------------------------------------
# ABV regexes (Bug 1: two-pass strict then loose)
# ---------------------------------------------------------------------------

_ABV_STRICT_RE = re.compile(
    r"(?:ALC\.?\s*)"
    r"(\d+(?:\.\d+)?)\s*%\s*"
    r"(?:by\s+vol|Alc\.?/?Vol\.?|ALC/?VOL|alcohol\s+by\s+volume)",
    re.I,
)

_ABV_QUAL_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*"
    r"(?:Alc\.?/?Vol\.?|ALC/?VOL|by\s+vol|alcohol\s+by\s+volume)",
    re.I,
)

_ABV_LOOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%", re.I)
_PROOF_RE = re.compile(r"(\d+)\s*(?:Proof|PROOF)")

# ---------------------------------------------------------------------------
# Net contents regexes (Bug 2: imperial units)
# ---------------------------------------------------------------------------

_NET_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(mL\.?|ml\.?|ML\.?|L\.?|litre|liter"
    r"|FL\.?\s*OZ\.?|FLUID\s+OUNCES?|fl\.?\s*oz\.?"
    r"|QT\.?|QUART"
    r"|PT\.?|PINT"
    r"|GAL\.?|GALLON)\b",
    re.I,
)

_COMPOUND_NET_RE = re.compile(
    r"(\d+)\s*(PINT|PT\.?)\s+(\d+)\s*(FL\.?\s*OZ\.?|FLUID\s+OUNCES?)",
    re.I,
)

_LOCATION_RE = re.compile(r"[A-Z][a-z]+,\s*[A-Z]{2}\b")
_COUNTRY_RE = re.compile(r"Product\s+of\s+(.+)", re.I)


def _is_stop_content(text: str) -> bool:
    """Return True if text is clearly not a class/type continuation."""
    t = text.strip()
    upper = t.upper()
    if _ABV_QUAL_RE.search(t) or _ABV_STRICT_RE.search(t):
        return True
    if _BOTTLER_HEADER_RE.search(t):
        return True
    if "IMPORTED BY" in upper:
        return True
    if _NET_RE.search(t):
        return True
    if _COUNTRY_RE.search(t):
        return True
    if upper.startswith("CONTAINS"):
        return True
    if _LOCATION_RE.search(t) and not _CLASS_RE.search(t):
        return True
    if t in ("Brand Label", "Back Label"):
        return True
    if _LABEL_MARKER_RE.match(t):
        return True
    return False


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_fields(ocr_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Extract: brand_name, class_type, alcohol_pct, proof, net_contents,
    government_warning, bottler, country_of_origin.
    """
    if not ocr_blocks:
        return _empty_extracted()

    sorted_blocks = sorted(
        ocr_blocks,
        key=lambda b: (b.get("bbox", [0, 0, 0, 0])[1], b.get("bbox", [0])[0]),
    )

    out: dict[str, Any] = {}

    out["brand_name"] = _extract_brand(sorted_blocks)
    out["class_type"] = _extract_class_type(sorted_blocks)
    out.update(_extract_abv_proof(sorted_blocks))
    out["net_contents"] = _extract_net_contents(sorted_blocks)
    out["government_warning"] = _extract_warning(sorted_blocks)
    out["bottler"] = _extract_bottler(sorted_blocks)
    out["country_of_origin"] = _extract_country(sorted_blocks)
    out["_all_blocks"] = ocr_blocks
    return out


# ---------------------------------------------------------------------------
# Brand name (Bug 6: keep domain suffixes, strip corp suffixes)
# ---------------------------------------------------------------------------

def _extract_brand(blocks: list[dict]) -> dict[str, Any]:
    for i, b in enumerate(blocks):
        text_upper = (b.get("text") or "").strip().upper()
        words = text_upper.split()
        for word in words:
            if word in _ALL_COMPANY_SUFFIXES:
                if word in _BRAND_SUFFIXES:
                    full = (b.get("text") or "").strip()
                    if len(full) >= 3:
                        return {"value": full, "bbox": b.get("bbox")}
                prefix = text_upper.split(word)[0].strip()
                if prefix and len(prefix) >= 2:
                    return {"value": prefix, "bbox": b.get("bbox")}
                if i > 0:
                    prev = (blocks[i - 1].get("text") or "").strip()
                    if prev and not _is_junk(prev) and len(prev) >= 2:
                        return {"value": prev, "bbox": blocks[i - 1].get("bbox")}

    if not blocks:
        return {"value": "", "bbox": None}
    max_y = max(b.get("bbox", [0, 0, 0, 0])[3] for b in blocks)
    top_half = [b for b in blocks if _bbox_y_center(b) < max_y * 0.55]
    if not top_half:
        top_half = blocks[:max(1, len(blocks) // 2)]

    best_score = 0
    best_block = None
    for b in top_half:
        t = (b.get("text") or "").strip()
        if _is_junk(t):
            continue
        if _CLASS_RE.search(t):
            continue
        if re.match(r"^\d", t):
            continue
        score = _bbox_height(b) * len(t)
        if score > best_score:
            best_score = score
            best_block = b

    if best_block:
        return {"value": (best_block["text"] or "").strip(), "bbox": best_block.get("bbox")}
    return {"value": "", "bbox": None}


# ---------------------------------------------------------------------------
# Class / type (Bug 3: stop conditions)
# ---------------------------------------------------------------------------

def _extract_class_type(blocks: list[dict]) -> dict[str, Any]:
    anchor_idx = None
    for i, b in enumerate(blocks):
        if _CLASS_RE.search(b.get("text", "")):
            if not _is_stop_content(b.get("text", "")):
                anchor_idx = i
                break
            if _CLASS_RE.search(b.get("text", "")) and not _ABV_QUAL_RE.search(b.get("text", "")):
                anchor_idx = i
                break

    if anchor_idx is None:
        combined = " ".join(b.get("text", "") for b in blocks)
        m = _CLASS_RE.search(combined)
        if m:
            return {"value": m.group(0).strip(), "bbox": None}
        return {"value": "", "bbox": None}

    anchor = blocks[anchor_idx]
    y_thresh = _bbox_height(anchor) * 3
    collected = [anchor]

    _CLASS_ADJ = {"SINGLE", "BARREL", "STRAIGHT", "BLENDED", "DOUBLE", "TRIPLE",
                  "SMALL", "BATCH", "RESERVE", "AGED", "OLD", "AMERICAN",
                  "WHISKEY", "WHISKY", "WINE", "ALE"}

    for j in range(anchor_idx - 1, max(anchor_idx - 4, -1), -1):
        b = blocks[j]
        if _y_distance(b, anchor) > y_thresh:
            break
        t = (b.get("text") or "").strip()
        if _is_stop_content(t):
            break
        upper = t.upper()
        if _CLASS_RE.search(t) or upper in _CLASS_ADJ:
            collected.insert(0, b)
        else:
            break

    for j in range(anchor_idx + 1, min(anchor_idx + 4, len(blocks))):
        b = blocks[j]
        if _y_distance(b, anchor) > y_thresh:
            break
        t = (b.get("text") or "").strip()
        if _is_stop_content(t):
            break
        upper = t.upper()
        if _CLASS_RE.search(t) or upper in _CLASS_ADJ:
            collected.append(b)
        else:
            break

    full_text = " ".join((b.get("text") or "").strip() for b in collected)
    return {"value": full_text, "bbox": _merge_bboxes(collected)}


# ---------------------------------------------------------------------------
# ABV / Proof (Bug 1: strict-first extraction)
# ---------------------------------------------------------------------------

def _extract_abv_proof(blocks: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # Strict per-block: "ALC. X% BY VOL" or "X% ALC/VOL"
    for b in blocks:
        t = b.get("text", "")
        m = _ABV_STRICT_RE.search(t)
        if not m:
            m = _ABV_QUAL_RE.search(t)
        if m and "alcohol_pct" not in out:
            out["alcohol_pct"] = {"value": m.group(1), "bbox": b.get("bbox")}
        m2 = _PROOF_RE.search(t)
        if m2 and "proof" not in out:
            out["proof"] = {"value": m2.group(1), "bbox": b.get("bbox")}

    # Strict combined text fallback
    if "alcohol_pct" not in out:
        combined = " ".join(b.get("text", "") for b in blocks)
        m = _ABV_STRICT_RE.search(combined)
        if not m:
            m = _ABV_QUAL_RE.search(combined)
        if m:
            pct = m.group(1)
            bbox = None
            for b in blocks:
                if pct in b.get("text", "") or "%" in b.get("text", ""):
                    bbox = b.get("bbox")
                    break
            out["alcohol_pct"] = {"value": pct, "bbox": bbox}

    # Loose fallback only if strict found nothing
    if "alcohol_pct" not in out:
        for b in blocks:
            t = b.get("text", "")
            m = _ABV_LOOSE_RE.search(t)
            if m:
                out["alcohol_pct"] = {"value": m.group(1), "bbox": b.get("bbox")}
                break

    if "proof" not in out:
        combined = " ".join(b.get("text", "") for b in blocks)
        m = _PROOF_RE.search(combined)
        if m:
            out["proof"] = {"value": m.group(1), "bbox": None}

    out.setdefault("alcohol_pct", {"value": "", "bbox": None})
    out.setdefault("proof", {"value": "", "bbox": None})
    return out


# ---------------------------------------------------------------------------
# Net contents (Bug 2: imperial units)
# ---------------------------------------------------------------------------

def _extract_net_contents(blocks: list[dict]) -> dict[str, Any]:
    # Check compound expressions first (e.g. "1 PINT 8 FL OZ")
    combined = " ".join(b.get("text", "") for b in blocks)
    mc = _COMPOUND_NET_RE.search(combined)
    if mc:
        bbox = None
        for b in blocks:
            if "PINT" in b.get("text", "").upper() or "FL" in b.get("text", "").upper():
                bbox = b.get("bbox")
                break
        pints = int(mc.group(1))
        oz = int(mc.group(3))
        total_oz = pints * 16 + oz
        return {"value": f"{total_oz} fl oz", "bbox": bbox}

    for b in blocks:
        m = _NET_RE.search(b.get("text", ""))
        if m:
            return _format_net(m, b.get("bbox"))

    m = _NET_RE.search(combined)
    if m:
        bbox = None
        num = m.group(1)
        for b in blocks:
            t = b.get("text", "")
            if num in t or "ml" in t.lower() or "oz" in t.lower() or "qt" in t.lower():
                bbox = b.get("bbox")
                break
        return _format_net(m, bbox)

    return {"value": "", "bbox": None}


def _format_net(m: re.Match, bbox: Any) -> dict[str, Any]:
    num, unit_raw = m.group(1), m.group(2).rstrip(".").strip().lower()
    unit_raw = re.sub(r"\s+", " ", unit_raw)
    if unit_raw in ("l", "litre", "liter"):
        val = f"{num} L"
    elif unit_raw in ("ml",):
        val = f"{num} mL"
    elif "fl" in unit_raw or "fluid" in unit_raw:
        val = f"{num} fl oz"
    elif unit_raw in ("qt", "quart"):
        val = f"{num} qt"
    elif unit_raw in ("pt", "pint"):
        val = f"{num} pt"
    elif unit_raw in ("gal", "gallon"):
        val = f"{num} gal"
    else:
        val = f"{num} mL"
    return {"value": val, "bbox": bbox}


# ---------------------------------------------------------------------------
# Government warning (Bug 4: tighter stop conditions)
# ---------------------------------------------------------------------------

def _extract_warning(blocks: list[dict]) -> dict[str, Any]:
    anchor = None
    anchor_idx = None
    for i, b in enumerate(blocks):
        t = (b.get("text") or "").upper()
        if "GOVERNMENT" in t and "WARNING" in t:
            anchor = b
            anchor_idx = i
            break
        if "GOVERNMENT" in t:
            next_t = blocks[i + 1].get("text", "").upper() if i + 1 < len(blocks) else ""
            if "WARNING" in next_t:
                anchor = b
                anchor_idx = i
                break

    if anchor is None:
        return {"value": "", "bbox": None}

    anchor_box = anchor.get("bbox", [0, 0, 0, 0])
    anchor_x_min = anchor_box[0]
    anchor_x_max = anchor_box[2]
    col_width = anchor_x_max - anchor_x_min
    x_margin = max(col_width * 0.3, 30)

    collected_blocks = []
    last_y = anchor_box[3]
    line_height = _bbox_height(anchor)
    max_gap = line_height * 3

    for b in blocks[anchor_idx:]:
        box = b.get("bbox", [0, 0, 0, 0])
        if box[0] > anchor_x_max + x_margin:
            continue
        if box[2] < anchor_x_min - x_margin:
            continue
        if box[1] > last_y + max_gap and collected_blocks:
            break

        t = (b.get("text") or "").strip()
        upper = t.upper()

        if t in ("Brand Label", "Back Label"):
            break
        if upper.startswith("CONTAINS"):
            break
        if _NET_RE.search(t) and "GOVERNMENT" not in upper:
            break
        if _ABV_QUAL_RE.search(t) and "GOVERNMENT" not in upper:
            break
        if _CLASS_RE.search(t) and not any(w in upper for w in ("ALCOHOLIC", "BEVERAGES", "HEALTH", "PROBLEMS")):
            break

        if len(t) < 2 and not t.isdigit():
            continue

        collected_blocks.append(b)
        last_y = max(last_y, box[3])

    parts = [(b.get("text") or "").strip() for b in collected_blocks]
    return {
        "value": " ".join(parts) if parts else "",
        "bbox": _merge_bboxes(collected_blocks) if collected_blocks else None,
    }


# ---------------------------------------------------------------------------
# Bottler (Bug 5: stop conditions, Bug 8: more patterns)
# ---------------------------------------------------------------------------

_BOTTLER_FALLBACK_RE = re.compile(
    r"([\w\s&']+(?:Brewery|Distillery|Winery|Cellars|Imports|Vineyards|Brewing\s+Company|Company))"
    r"[\s,]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*,\s*([A-Z]{2})\b",
    re.I,
)


def _extract_bottler(blocks: list[dict]) -> dict[str, Any]:
    for i, b in enumerate(blocks):
        t = (b.get("text") or "").strip()
        m = _BOTTLER_HEADER_RE.search(t)
        if m:
            collected = [b]
            line_h = _bbox_height(b)
            for j in range(i + 1, min(i + 4, len(blocks))):
                nb = blocks[j]
                if _y_distance(nb, b) > line_h * 4:
                    break
                nt = (nb.get("text") or "").strip()
                upper_nt = nt.upper()
                if _is_junk(nt) and len(nt) < 3:
                    continue
                if "GOVERNMENT" in upper_nt or "WARNING" in upper_nt:
                    break
                if re.match(r"^\d+\s*%", nt):
                    break
                if _NET_RE.match(nt):
                    break
                if _COUNTRY_RE.search(nt):
                    break
                if upper_nt.startswith("CONTAINS"):
                    break
                collected.append(nb)
            parts = [(cb.get("text") or "").strip() for cb in collected]
            return {"value": " ".join(parts), "bbox": _merge_bboxes(collected)}

    # Fallback: look for "CompanyName, City, ST" pattern without a header
    combined = " ".join((b.get("text") or "").strip() for b in blocks)
    m = _BOTTLER_FALLBACK_RE.search(combined)
    if m:
        name = m.group(1).strip()
        city = m.group(2).strip()
        state = m.group(3).strip()
        bbox = None
        for b in blocks:
            if name.split()[0] in (b.get("text") or ""):
                bbox = b.get("bbox")
                break
        return {"value": f"{name}, {city}, {state}", "bbox": bbox}

    return {"value": "", "bbox": None}


# ---------------------------------------------------------------------------
# Country of origin
# ---------------------------------------------------------------------------

def _extract_country(blocks: list[dict]) -> dict[str, Any]:
    for b in blocks:
        m = _COUNTRY_RE.search(b.get("text", ""))
        if m:
            return {"value": m.group(0).strip(), "bbox": b.get("bbox")}
    return {"value": "", "bbox": None}


# ---------------------------------------------------------------------------
# Empty fallback
# ---------------------------------------------------------------------------

def _empty_extracted() -> dict[str, Any]:
    return {
        "brand_name": {"value": "", "bbox": None},
        "class_type": {"value": "", "bbox": None},
        "alcohol_pct": {"value": "", "bbox": None},
        "proof": {"value": "", "bbox": None},
        "net_contents": {"value": "", "bbox": None},
        "government_warning": {"value": "", "bbox": None},
        "bottler": {"value": "", "bbox": None},
        "country_of_origin": {"value": "", "bbox": None},
        "_all_blocks": [],
    }
