"""
Single pipeline: image + application data -> OCR -> extraction -> rules -> scoring.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .ocr import run_ocr, OcrUnavailableError
from .extraction import extract_fields
from .rules.engine import run_rules
from .scoring import compute_overall_status


def run_pipeline(image_input: Any, app_data: dict[str, Any]) -> dict[str, Any]:
    """
    image_input: file path (str/Path), bytes, or PIL Image
    app_data: dict with brand_name, class_type, alcohol_pct, proof, net_contents_ml,
              bottler_name, bottler_city, bottler_state, imported, country_of_origin,
              sulfites_required, fd_c_yellow_5_required, carmine_required,
              wood_treatment_required, age_statement_required, neutral_spirits_required
    Returns: {
        "ocr_blocks": [...],
        "extracted": {...},
        "rule_results": [...],
        "overall_status": "Ready to approve" | "Needs review" | "Critical issues",
        "counts": {"pass": N, "needs_review": N, "fail": N},
        "image": PIL Image (for display/overlay)
    }
    """
    from PIL import Image
    import io

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
    else:
        img = image_input.convert("RGB") if hasattr(image_input, "convert") else image_input

    try:
        ocr_blocks = run_ocr(img)
    except OcrUnavailableError as e:
        return {
            "ocr_blocks": [],
            "extracted": {},
            "rule_results": [],
            "overall_status": "Critical issues",
            "counts": {"pass": 0, "needs_review": 0, "fail": 0},
            "image": img,
            "error": str(e),
        }

    ocr_fallback = any(b.get("_ocr_fallback") for b in ocr_blocks)
    extracted = extract_fields(ocr_blocks)
    rule_results = run_rules(extracted, app_data)
    overall, counts = compute_overall_status(rule_results)

    result: dict[str, Any] = {
        "ocr_blocks": ocr_blocks,
        "extracted": extracted,
        "rule_results": rule_results,
        "overall_status": overall,
        "counts": counts,
        "image": img,
    }
    if ocr_fallback:
        result["ocr_fallback_warning"] = (
            "Primary OCR engine (EasyOCR) was unavailable. Using Tesseract fallback "
            "-- accuracy may be reduced for curved text, stylized fonts, and multi-panel labels."
        )
    return result
