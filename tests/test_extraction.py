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
    assert "ABC" in out["brand_name"]["value"]
    assert "DISTILLERY" in out["brand_name"]["value"]
    assert out["brand_name"]["bbox"] is not None


def test_abc_class_type_multi_block(ocr_blocks_abc):
    out = extract_fields(ocr_blocks_abc)
    val = out["class_type"]["value"].upper()
    assert "STRAIGHT" in val
    assert "RYE" in val
    assert "WHISKY" in val


def test_abc_class_type_merged_bbox(ocr_blocks_abc):
    """Bbox should span all collected class/type blocks, not just the first."""
    out = extract_fields(ocr_blocks_abc)
    bbox = out["class_type"]["bbox"]
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert x2 - x1 > 50
    assert y2 > y1


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


def test_abc_warning_merged_bbox(ocr_blocks_abc):
    """Warning bbox should span from header to last collected line."""
    out = extract_fields(ocr_blocks_abc)
    bbox = out["government_warning"]["bbox"]
    assert bbox is not None
    assert bbox[3] > bbox[1] + 30


def test_abc_bottler_multi_line(ocr_blocks_abc):
    out = extract_fields(ocr_blocks_abc)
    b = out["bottler"]["value"]
    assert "DISTILLED AND BOTTLED BY" in b.upper() or "ABC" in b.upper()


def test_abc_bottler_merged_bbox(ocr_blocks_abc):
    """Bottler bbox should span header + subsequent lines."""
    out = extract_fields(ocr_blocks_abc)
    bbox = out["bottler"]["bbox"]
    assert bbox is not None
    assert bbox[3] > bbox[1]


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


# ---------------------------------------------------------------------------
# Bug 1: ABV strict vs loose extraction
# ---------------------------------------------------------------------------

def test_abv_strict_prefers_qualified():
    """15% from 'FL OZ15%' should NOT be extracted when '5% ALC/VOL' exists."""
    blocks = [
        {"text": "BRAND", "bbox": [10, 10, 100, 30], "confidence": 90},
        {"text": "1 PINT 8 FL OZ15%", "bbox": [10, 40, 200, 60], "confidence": 88},
        {"text": "5% ALC/VOL", "bbox": [10, 70, 120, 90], "confidence": 92},
    ]
    out = extract_fields(blocks)
    assert out["alcohol_pct"]["value"] == "5"


def test_abv_strict_alc_dot_pct_by_vol():
    blocks = [
        {"text": "ALC. 12% BY VOL.", "bbox": [10, 10, 180, 30], "confidence": 90},
    ]
    out = extract_fields(blocks)
    assert out["alcohol_pct"]["value"] == "12"


def test_abv_loose_fallback_when_no_qualifier():
    blocks = [
        {"text": "BRAND", "bbox": [10, 10, 100, 30], "confidence": 90},
        {"text": "40%", "bbox": [10, 40, 60, 60], "confidence": 88},
    ]
    out = extract_fields(blocks)
    assert out["alcohol_pct"]["value"] == "40"


def test_abv_ignores_non_abv_percentage():
    """50% MERLOT should not be ABV when 13% ALC./VOL. exists."""
    blocks = [
        {"text": "50% MERLOT", "bbox": [10, 10, 120, 30], "confidence": 90},
        {"text": "13% ALC./VOL.", "bbox": [10, 40, 150, 60], "confidence": 92},
    ]
    out = extract_fields(blocks)
    assert out["alcohol_pct"]["value"] == "13"


# ---------------------------------------------------------------------------
# Bug 2: Imperial net contents
# ---------------------------------------------------------------------------

def test_net_contents_fl_oz(ocr_blocks_imperial_net):
    out = extract_fields(ocr_blocks_imperial_net)
    val = out["net_contents"]["value"]
    assert "fl oz" in val.lower() or "24" in val


def test_net_contents_quart():
    blocks = [
        {"text": "BRAND", "bbox": [10, 10, 100, 30], "confidence": 90},
        {"text": "1 QT.", "bbox": [10, 40, 70, 60], "confidence": 92},
    ]
    out = extract_fields(blocks)
    assert "qt" in out["net_contents"]["value"].lower()


def test_net_contents_fluid_ounces():
    blocks = [
        {"text": "BRAND", "bbox": [10, 10, 100, 30], "confidence": 90},
        {"text": "12 FLUID OUNCES", "bbox": [10, 40, 160, 60], "confidence": 91},
    ]
    out = extract_fields(blocks)
    assert "fl oz" in out["net_contents"]["value"].lower()


def test_net_contents_compound_pint_oz(ocr_blocks_imperial_net):
    out = extract_fields(ocr_blocks_imperial_net)
    val = out["net_contents"]["value"]
    assert "24" in val or "fl oz" in val.lower()


# ---------------------------------------------------------------------------
# Bug 3: Class/type overcollection stop conditions
# ---------------------------------------------------------------------------

def test_class_stops_at_abv(ocr_blocks_overcollect):
    out = extract_fields(ocr_blocks_overcollect)
    class_val = out["class_type"]["value"].upper()
    assert "ALC" not in class_val
    assert "IMPORTED" not in class_val


def test_class_stops_at_bottler():
    blocks = [
        {"text": "PALE ALE", "bbox": [10, 10, 100, 30], "confidence": 90},
        {"text": "Bottled by XYZ", "bbox": [10, 35, 150, 55], "confidence": 88},
    ]
    out = extract_fields(blocks)
    assert "Bottled" not in out["class_type"]["value"]


# ---------------------------------------------------------------------------
# Bug 5: Bottler overcollection stop conditions
# ---------------------------------------------------------------------------

def test_bottler_stops_at_product_of(ocr_blocks_overcollect):
    out = extract_fields(ocr_blocks_overcollect)
    bottler_val = out["bottler"]["value"].upper()
    assert "PRODUCT OF" not in bottler_val
    assert "CONTAINS" not in bottler_val
    assert "750" not in bottler_val


# ---------------------------------------------------------------------------
# Bug 6: Brand keeps domain suffixes
# ---------------------------------------------------------------------------

def test_brand_keeps_winery():
    blocks = [
        {"text": "ABC WINERY", "bbox": [10, 10, 200, 50], "confidence": 92},
        {"text": "Red Wine", "bbox": [10, 55, 100, 75], "confidence": 88},
    ]
    out = extract_fields(blocks)
    assert "WINERY" in out["brand_name"]["value"].upper()


def test_brand_keeps_distillery():
    blocks = [
        {"text": "ABC DISTILLERY", "bbox": [10, 10, 200, 50], "confidence": 92},
        {"text": "Bourbon", "bbox": [10, 55, 100, 75], "confidence": 88},
    ]
    out = extract_fields(blocks)
    assert "DISTILLERY" in out["brand_name"]["value"].upper()


def test_brand_strips_inc():
    blocks = [
        {"text": "ACME INC", "bbox": [10, 10, 200, 50], "confidence": 92},
        {"text": "Vodka", "bbox": [10, 55, 100, 75], "confidence": 88},
    ]
    out = extract_fields(blocks)
    assert "ACME" in out["brand_name"]["value"]
    assert "INC" not in out["brand_name"]["value"]


# ---------------------------------------------------------------------------
# Bug 7: Barleywine Ale keyword
# ---------------------------------------------------------------------------

def test_class_finds_barleywine(ocr_blocks_barleywine):
    out = extract_fields(ocr_blocks_barleywine)
    assert "barleywine" in out["class_type"]["value"].lower()


# ---------------------------------------------------------------------------
# Bug 8: BREWED & BOTTLED BY bottler pattern
# ---------------------------------------------------------------------------

def test_bottler_brewed_and_bottled(ocr_blocks_barleywine):
    out = extract_fields(ocr_blocks_barleywine)
    assert "BREWED" in out["bottler"]["value"].upper() or "Tiger" in out["bottler"]["value"]
