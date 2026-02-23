"""Tests for full pipeline."""
import pytest
from PIL import Image

from src.pipeline import run_pipeline


def test_pipeline_accepts_pil_image(white_image, app_data):
    result = run_pipeline(white_image, app_data)
    assert "overall_status" in result
    assert result["overall_status"] in ("Ready to approve", "Needs review", "Critical issues")
    assert "counts" in result
    assert "rule_results" in result
    assert "ocr_blocks" in result
    assert "extracted" in result
    assert "image" in result


def test_pipeline_accepts_bytes(app_data):
    from io import BytesIO
    img = Image.new("RGB", (200, 100), color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    result = run_pipeline(buf.getvalue(), app_data)
    assert "overall_status" in result
    assert result["image"] is not None


def test_pipeline_counts_sum_to_rules(white_image, app_data):
    result = run_pipeline(white_image, app_data)
    n = len(result["rule_results"])
    c = result["counts"]
    assert c["pass"] + c["needs_review"] + c["fail"] == n


def test_pipeline_overall_consistent_with_counts(white_image, app_data):
    result = run_pipeline(white_image, app_data)
    if result.get("error"):
        assert result["overall_status"] == "Critical issues"
        return
    counts = result["counts"]
    overall = result["overall_status"]
    if counts["fail"] > 0:
        assert overall == "Critical issues"
    elif counts["needs_review"] > 0:
        assert overall == "Needs review"
    else:
        assert overall == "Ready to approve"
