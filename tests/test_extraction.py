"""Tests for field extraction from OCR blocks."""
import pytest

from src.extraction import extract_fields, _empty_extracted


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------

def test_empty_blocks_returns_empty_extracted(empty_ocr_blocks):
    out = extract_fields(empty_ocr_blocks)
    assert out["brand_name"]["value"] == ""
    assert out["class_type"]["value"] == ""
    assert out["government_warning"]["value"] == ""


def test_extraction_has_required_keys(ocr_blocks):
    out = extract_fields(ocr_blocks)
    for key in ("brand_name", "class_type", "alcohol_pct", "proof", "net_contents",
                "government_warning", "bottler", "country_of_origin"):
        assert key in out
        assert "value" in out[key]
    assert "_all_blocks" in out


def test_empty_extracted_structure():
    out = _empty_extracted()
    assert out["brand_name"]["bbox"] is None
    assert out["_all_blocks"] == []


# ---------------------------------------------------------------------------
# OLD TOM DISTILLERY (single-line blocks)
# ---------------------------------------------------------------------------

def test_extracts_brand_from_company_suffix(ocr_blocks):
    out = extract_fields(ocr_blocks)
    assert "OLD TOM" in out["brand_name"]["value"]


def test_extracts_class_type(ocr_blocks):
    out = extract_fields(ocr_blocks)
    assert "Bourbon" in out["class_type"]["value"] or "Whiskey" in out["class_type"]["value"]


def test_extracts_abv_and_proof(ocr_blocks):
    out = extract_fields(ocr_blocks)
    assert out["alcohol_pct"]["value"] == "45"
    assert out["proof"]["value"] == "90"


def test_extracts_net_contents(ocr_blocks):
    out = extract_fields(ocr_blocks)
    assert "750" in out["net_contents"]["value"]
    assert "ml" in out["net_contents"]["value"].lower()


def test_extracts_government_warning(ocr_blocks):
    out = extract_fields(ocr_blocks)
    assert "GOVERNMENT WARNING" in out["government_warning"]["value"].upper()


def test_extracts_bottler(ocr_blocks):
    out = extract_fields(ocr_blocks)
    assert "Bottled" in out["bottler"]["value"] or "Old Tom" in out["bottler"]["value"]


# ---------------------------------------------------------------------------
# ABC DISTILLERY (multi-block, two-panel)
# ---------------------------------------------------------------------------

def test_abc_brand_detection(ocr_blocks_abc):
    out = extract_fields(ocr_blocks_abc)
    assert out["brand_name"]["value"] == "ABC"
    assert out["brand_name"]["bbox"] is not None


def test_abc_class_type_multi_block(ocr_blocks_abc):
    out = extract_fields(ocr_blocks_abc)
    val = out["class_type"]["value"].upper()
    assert "STRAIGHT" in val
    assert "RYE" in val
    assert "WHISKY" in val


def test_abc_abv(ocr_blocks_abc):
    out = extract_fields(ocr_blocks_abc)
    assert out["alcohol_pct"]["value"] == "45"


def test_abc_net_contents(ocr_blocks_abc):
    out = extract_fields(ocr_blocks_abc)
    assert "750" in out["net_contents"]["value"]


def test_abc_warning_spatial(ocr_blocks_abc):
    out = extract_fields(ocr_blocks_abc)
    w = out["government_warning"]["value"]
    assert "GOVERNMENT WARNING" in w.upper()
    assert "Surgeon General" in w


def test_abc_bottler_multi_line(ocr_blocks_abc):
    out = extract_fields(ocr_blocks_abc)
    b = out["bottler"]["value"]
    assert "DISTILLED AND BOTTLED BY" in b.upper() or "ABC" in b.upper()


# ---------------------------------------------------------------------------
# Split ABV blocks
# ---------------------------------------------------------------------------

def test_split_abv_combined_text(ocr_blocks_split_abv):
    out = extract_fields(ocr_blocks_split_abv)
    assert out["alcohol_pct"]["value"] == "45"


def test_split_net_contents_combined(ocr_blocks_split_abv):
    out = extract_fields(ocr_blocks_split_abv)
    assert "750" in out["net_contents"]["value"]


# ---------------------------------------------------------------------------
# Brand prominence heuristic (no company suffix)
# ---------------------------------------------------------------------------

def test_brand_prominence_no_suffix():
    blocks = [
        {"text": "SMALL TEXT", "bbox": [10, 10, 100, 25], "confidence": 88},
        {"text": "BIG BRAND", "bbox": [10, 40, 300, 120], "confidence": 92},
        {"text": "Vodka", "bbox": [10, 130, 100, 155], "confidence": 90},
    ]
    out = extract_fields(blocks)
    assert out["brand_name"]["value"] == "BIG BRAND"
