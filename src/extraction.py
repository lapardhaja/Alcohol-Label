"""
Region heuristics and field extraction from OCR blocks.
Returns dict of field name -> { "value", "bbox" } (or list of candidates).
"""
from __future__ import annotations

import re
from typing import Any


def extract_fields(ocr_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Extract candidates for: brand_name, class_type, alcohol_pct, proof, net_contents,
    government_warning, bottler, country_of_origin.
    """
    if not ocr_blocks:
        return _empty_extracted()

    full_text = " ".join(b.get("text", "") for b in ocr_blocks)
    # Sort by vertical position (y) then horizontal (x) for "front" order
    sorted_blocks = sorted(ocr_blocks, key=lambda b: (b.get("bbox", [0, 0, 0, 0])[1], b.get("bbox", [0])[0]))

    out: dict[str, Any] = {}

    # Brand: often in top third, prominent (first substantial line)
    top_blocks = sorted_blocks[: max(1, len(sorted_blocks) // 3)]
    brand_candidates = [b["text"] for b in top_blocks if len(b.get("text", "")) >= 2 and b["text"] != "GOVERNMENT WARNING"]
    if brand_candidates:
        out["brand_name"] = {"value": brand_candidates[0], "bbox": top_blocks[0].get("bbox")}
    else:
        out["brand_name"] = {"value": "", "bbox": None}

    # Class/type: known phrases
    class_pattern = re.compile(
        r"(Vodka|Gin|Rum|Whiskey|Whisky|Bourbon|Kentucky Straight Bourbon Whiskey|"
        r"Tennessee Whiskey|Tequila|Brandy|Liqueur|Neutral Spirits)[^.]*",
        re.I,
    )
    for b in sorted_blocks:
        m = class_pattern.search(b.get("text", ""))
        if m:
            out["class_type"] = {"value": m.group(0).strip(), "bbox": b.get("bbox")}
            break
    if "class_type" not in out:
        out["class_type"] = {"value": "", "bbox": None}

    # ABV and proof
    abv_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:Alc\.?/Vol\.?|ALC/VOL|Alcohol by volume)?", re.I)
    proof_pattern = re.compile(r"(\d+)\s*(?:Proof|PROOF)")
    for b in sorted_blocks:
        t = b.get("text", "")
        ma = abv_pattern.search(t)
        if ma:
            out["alcohol_pct"] = {"value": ma.group(1), "bbox": b.get("bbox")}
            break
    if "alcohol_pct" not in out:
        out["alcohol_pct"] = {"value": "", "bbox": None}
    for b in sorted_blocks:
        t = b.get("text", "")
        mp = proof_pattern.search(t)
        if mp:
            out["proof"] = {"value": mp.group(1), "bbox": b.get("bbox")}
            break
    if "proof" not in out:
        out["proof"] = {"value": "", "bbox": None}

    # Net contents
    net_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(mL|ml|L)", re.I)
    for b in sorted_blocks:
        m = net_pattern.search(b.get("text", ""))
        if m:
            out["net_contents"] = {"value": f"{m.group(1)} {m.group(2)}", "bbox": b.get("bbox")}
            break
    if "net_contents" not in out:
        out["net_contents"] = {"value": "", "bbox": None}

    # Government warning: block containing GOVERNMENT WARNING
    warning_parts = []
    warning_bbox = None
    for b in sorted_blocks:
        t = b.get("text", "")
        if "GOVERNMENT WARNING" in t.upper():
            warning_parts.append(t)
            if warning_bbox is None:
                warning_bbox = b.get("bbox")
        elif warning_parts:
            warning_parts.append(t)
    out["government_warning"] = {
        "value": " ".join(warning_parts) if warning_parts else "",
        "bbox": warning_bbox,
    }

    # Bottler: line with Bottled by / Distilled by / Produced by
    bottler_pattern = re.compile(r"(?:Bottled by|Distilled by|Produced by)\s*(.+)", re.I)
    for b in sorted_blocks:
        m = bottler_pattern.search(b.get("text", ""))
        if m:
            out["bottler"] = {"value": m.group(0).strip(), "bbox": b.get("bbox")}
            break
    if "bottler" not in out:
        out["bottler"] = {"value": "", "bbox": None}

    # Country of origin
    country_pattern = re.compile(r"Product of\s+(.+)", re.I)
    for b in sorted_blocks:
        m = country_pattern.search(b.get("text", ""))
        if m:
            out["country_of_origin"] = {"value": m.group(0).strip(), "bbox": b.get("bbox")}
            break
    if "country_of_origin" not in out:
        out["country_of_origin"] = {"value": "", "bbox": None}

    out["_all_blocks"] = ocr_blocks
    return out


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
