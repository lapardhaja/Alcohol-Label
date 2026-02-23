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


# ---------------------------------------------------------------------------
# Generic text filters
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(
    r"^(DISTILLED AND BOTTLED BY|BOTTLED BY|DISTILLED BY|PRODUCED BY|IMPORTED BY|"
    r"GOVERNMENT WARNING|Brand Label|Back Label)[\s:]*$",
    re.I,
)

_BOTTLER_HEADER_RE = re.compile(
    r"(Distilled\s+and\s+Bottled\s+by|Bottled\s+by|Distilled\s+by|Produced\s+by|Imported\s+by)",
    re.I,
)

_COMPANY_SUFFIXES = frozenset({
    "DISTILLERY", "DISTILLERS", "BREWING", "BREWERY", "VINEYARDS", "WINERY",
    "CELLARS", "IMPORTS", "SPIRITS", "COMPANY", "CO", "INC", "LLC", "LTD",
    "CORP", "CORPORATION", "ESTATES", "RESERVE", "SELECTION",
})

_WARNING_WORDS = frozenset({
    "GOVERNMENT", "WARNING", "ACCORDING", "SURGEON", "GENERAL", "WOMEN",
    "SHOULD", "DRINK", "ALCOHOLIC", "BEVERAGES", "DURING", "PREGNANCY",
    "BECAUSE", "RISK", "BIRTH", "DEFECTS", "CONSUMPTION", "IMPAIRS",
    "ABILITY", "DRIVE", "CAR", "OPERATE", "MACHINERY", "CAUSE", "HEALTH",
    "PROBLEMS",
})


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
    upper = t.upper()
    if upper in _WARNING_WORDS:
        return True
    if upper in {"AND", "THE", "OF", "OR", "BY", "NOT", "TO", "A", "IN", "ON", "AT", "FOR", "IT", "IS"}:
        return True
    return False


# ---------------------------------------------------------------------------
# Spirit / class-type keywords
# ---------------------------------------------------------------------------

_CLASS_KEYWORDS = (
    "Vodka", "Gin", "Rum", "Whiskey", "Whisky", "Bourbon", "Rye",
    "Tequila", "Mezcal", "Brandy", "Cognac", "Armagnac", "Liqueur",
    "Neutral Spirits", "Scotch", "Irish", "Canadian", "Blended",
    "Single Malt", "Single Barrel", "Single Pot Still",
    "Tennessee", "Kentucky", "Straight", "Reserve", "Aged",
    "Grappa", "Absinthe", "Port", "Sherry", "Vermouth", "Sake",
    "Shochu", "Baijiu", "Cachaca", "Pisco",
)
_CLASS_KEYWORD_SET = frozenset(k.upper() for k in _CLASS_KEYWORDS)
_CLASS_RE = re.compile(
    r"(?:" + "|".join(re.escape(k) for k in _CLASS_KEYWORDS) + r")",
    re.I,
)


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
# Brand name
# ---------------------------------------------------------------------------

def _extract_brand(blocks: list[dict]) -> dict[str, Any]:
    """
    Strategy:
    1. Look for a block adjacent to a company-suffix block (e.g. "ABC" before "DISTILLERY").
    2. Fall back to the most prominent text in the top half (largest bbox height * text length).
    """
    # Strategy 1: company-suffix adjacency
    for i, b in enumerate(blocks):
        text_upper = (b.get("text") or "").strip().upper()
        words = text_upper.split()
        for word in words:
            if word in _COMPANY_SUFFIXES:
                # The brand might be part of this block (e.g. "ABC DISTILLERY")
                prefix = text_upper.split(word)[0].strip()
                if prefix and len(prefix) >= 2:
                    return {"value": prefix, "bbox": b.get("bbox")}
                # Or the previous block
                if i > 0:
                    prev = (blocks[i - 1].get("text") or "").strip()
                    if prev and not _is_junk(prev) and len(prev) >= 2:
                        return {"value": prev, "bbox": blocks[i - 1].get("bbox")}

    # Strategy 2: prominence in top half — largest bbox_height * len(text)
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
# Class / type
# ---------------------------------------------------------------------------

def _extract_class_type(blocks: list[dict]) -> dict[str, Any]:
    """
    Find spirit type keywords and combine adjacent blocks to build the full designation.
    E.g. "SINGLE BARREL" + "STRAIGHT RYE WHISKY" -> "SINGLE BARREL STRAIGHT RYE WHISKY".
    """
    anchor_idx = None
    for i, b in enumerate(blocks):
        if _CLASS_RE.search(b.get("text", "")):
            anchor_idx = i
            break
    if anchor_idx is None:
        # Try combined text of nearby blocks
        combined = " ".join(b.get("text", "") for b in blocks)
        m = _CLASS_RE.search(combined)
        if m:
            return {"value": m.group(0).strip(), "bbox": None}
        return {"value": "", "bbox": None}

    anchor = blocks[anchor_idx]
    y_thresh = _bbox_height(anchor) * 3
    collected = [anchor]

    # Look backward for adjacent class-related blocks
    for j in range(anchor_idx - 1, max(anchor_idx - 4, -1), -1):
        b = blocks[j]
        if _y_distance(b, anchor) > y_thresh:
            break
        t = (b.get("text") or "").strip().upper()
        if _CLASS_RE.search(t) or t in ("SINGLE", "BARREL", "STRAIGHT", "BLENDED", "DOUBLE", "TRIPLE", "SMALL", "BATCH", "RESERVE", "AGED", "OLD"):
            collected.insert(0, b)
        else:
            break

    # Look forward for continuation
    for j in range(anchor_idx + 1, min(anchor_idx + 4, len(blocks))):
        b = blocks[j]
        if _y_distance(b, anchor) > y_thresh:
            break
        t = (b.get("text") or "").strip().upper()
        if _CLASS_RE.search(t) or t in ("SINGLE", "BARREL", "STRAIGHT", "BLENDED", "DOUBLE", "TRIPLE", "SMALL", "BATCH", "RESERVE", "AGED", "OLD", "WHISKEY", "WHISKY"):
            collected.append(b)
        else:
            break

    full_text = " ".join((b.get("text") or "").strip() for b in collected)
    first_bbox = collected[0].get("bbox")
    return {"value": full_text, "bbox": first_bbox}


# ---------------------------------------------------------------------------
# ABV / Proof
# ---------------------------------------------------------------------------

_ABV_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:Alc\.?/?Vol\.?|ALC/?VOL|Alcohol\s+by\s+volume)?", re.I)
_PROOF_RE = re.compile(r"(\d+)\s*(?:Proof|PROOF)")


def _extract_abv_proof(blocks: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # Per-block search
    for b in blocks:
        t = b.get("text", "")
        m = _ABV_RE.search(t)
        if m and "alcohol_pct" not in out:
            out["alcohol_pct"] = {"value": m.group(1), "bbox": b.get("bbox")}
        m = _PROOF_RE.search(t)
        if m and "proof" not in out:
            out["proof"] = {"value": m.group(1), "bbox": b.get("bbox")}

    # Combined text fallback (OCR splits "45" and "%" into separate blocks)
    if "alcohol_pct" not in out:
        combined = " ".join(b.get("text", "") for b in blocks)
        m = _ABV_RE.search(combined)
        if m:
            pct = m.group(1)
            bbox = None
            for b in blocks:
                if pct in b.get("text", "") or "%" in b.get("text", ""):
                    bbox = b.get("bbox")
                    break
            out["alcohol_pct"] = {"value": pct, "bbox": bbox}

    if "proof" not in out:
        combined = " ".join(b.get("text", "") for b in blocks)
        m = _PROOF_RE.search(combined)
        if m:
            out["proof"] = {"value": m.group(1), "bbox": None}

    out.setdefault("alcohol_pct", {"value": "", "bbox": None})
    out.setdefault("proof", {"value": "", "bbox": None})
    return out


# ---------------------------------------------------------------------------
# Net contents
# ---------------------------------------------------------------------------

_NET_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mL\.?|ml\.?|ML\.?|L\.?|litre|liter)\b", re.I)


def _extract_net_contents(blocks: list[dict]) -> dict[str, Any]:
    # Per-block
    for b in blocks:
        m = _NET_RE.search(b.get("text", ""))
        if m:
            return _format_net(m, b.get("bbox"))

    # Combined text fallback (OCR splits "750" and "ML")
    combined = " ".join(b.get("text", "") for b in blocks)
    m = _NET_RE.search(combined)
    if m:
        bbox = None
        num = m.group(1)
        for b in blocks:
            t = b.get("text", "")
            if num in t or "ml" in t.lower() or t.strip() == "L":
                bbox = b.get("bbox")
                break
        return _format_net(m, bbox)

    return {"value": "", "bbox": None}


def _format_net(m: re.Match, bbox: Any) -> dict[str, Any]:
    num, unit = m.group(1), m.group(2).rstrip(".").lower()
    val = f"{num} L" if unit in ("l", "litre", "liter") else f"{num} mL"
    return {"value": val, "bbox": bbox}


# ---------------------------------------------------------------------------
# Government warning — spatial grouping
# ---------------------------------------------------------------------------

def _extract_warning(blocks: list[dict]) -> dict[str, Any]:
    """
    Find "GOVERNMENT WARNING" anchor, then collect all blocks that are:
    - Below or at the same y as the anchor
    - Within a reasonable x-range (same column)
    - Until a large y-gap or non-warning content appears
    """
    anchor = None
    anchor_idx = None
    for i, b in enumerate(blocks):
        t = (b.get("text") or "").upper()
        if "GOVERNMENT" in t:
            next_t = blocks[i + 1].get("text", "").upper() if i + 1 < len(blocks) else ""
            if "WARNING" in t or "WARNING" in next_t:
                anchor = b
                anchor_idx = i
                break

    if anchor is None:
        return {"value": "", "bbox": None}

    anchor_box = anchor.get("bbox", [0, 0, 0, 0])
    anchor_x_min = anchor_box[0]
    anchor_x_max = anchor_box[2]
    col_width = anchor_x_max - anchor_x_min
    x_margin = max(col_width * 0.5, 50)

    # Collect blocks spatially: same column, below anchor, within reasonable y-gap
    parts = []
    last_y = anchor_box[3]
    line_height = _bbox_height(anchor)
    max_gap = line_height * 3

    for b in blocks[anchor_idx:]:
        box = b.get("bbox", [0, 0, 0, 0])
        # Must be roughly in the same column
        if box[0] > anchor_x_max + x_margin:
            continue
        if box[2] < anchor_x_min - x_margin:
            continue
        # Must not be too far below the last collected block
        if box[1] > last_y + max_gap and parts:
            break
        t = (b.get("text") or "").strip()
        # Stop on obvious non-warning markers
        if t in ("Brand Label", "Back Label"):
            break
        parts.append(t)
        last_y = max(last_y, box[3])

    return {
        "value": " ".join(parts) if parts else "",
        "bbox": anchor_box,
    }


# ---------------------------------------------------------------------------
# Bottler — multi-line
# ---------------------------------------------------------------------------

def _extract_bottler(blocks: list[dict]) -> dict[str, Any]:
    """
    Find "Bottled by" / "Distilled and Bottled by" header, then collect
    subsequent blocks (name, city, state) within vertical proximity.
    """
    for i, b in enumerate(blocks):
        t = (b.get("text") or "").strip()
        m = _BOTTLER_HEADER_RE.search(t)
        if m:
            # Inline content after the header phrase (e.g. "Bottled by Old Tom Distillery, KY")
            after = t[m.end():].strip().strip(":").strip()
            parts = [t] if after else [t]
            bbox = b.get("bbox")
            line_h = _bbox_height(b)
            # Collect next 1-3 blocks below this header
            for j in range(i + 1, min(i + 4, len(blocks))):
                nb = blocks[j]
                if _y_distance(nb, b) > line_h * 4:
                    break
                nt = (nb.get("text") or "").strip()
                if _is_junk(nt) and len(nt) < 3:
                    continue
                # Stop if we hit another section
                if "GOVERNMENT" in nt.upper() or "WARNING" in nt.upper():
                    break
                if re.match(r"^\d+\s*%", nt) or _NET_RE.match(nt):
                    break
                parts.append(nt)
            return {"value": " ".join(parts), "bbox": bbox}

    return {"value": "", "bbox": None}


# ---------------------------------------------------------------------------
# Country of origin
# ---------------------------------------------------------------------------

_COUNTRY_RE = re.compile(r"Product\s+of\s+(.+)", re.I)


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
