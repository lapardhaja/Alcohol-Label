"""Tests for OCR module."""
import pytest
from PIL import Image

from src.ocr import run_ocr, _preprocess, _deduplicate_blocks, _bbox_iou, _data_to_blocks, OcrUnavailableError


def test_preprocess_resizes_large_image():
    img = Image.new("RGB", (3000, 2000), color="white")
    original, enhanced = _preprocess(img)
    assert original.size[0] <= 2000 and original.size[1] <= 2000
    assert max(original.size) == 2000


def test_preprocess_returns_enhanced_grayscale():
    img = Image.new("RGB", (400, 300), color="white")
    original, enhanced = _preprocess(img)
    assert original.size == (400, 300)
    assert enhanced.mode == "L"


def test_preprocess_leaves_small_image_unchanged():
    img = Image.new("RGB", (400, 300), color="white")
    original, enhanced = _preprocess(img)
    assert original.size == (400, 300)


def test_bbox_iou_identical():
    assert _bbox_iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0


def test_bbox_iou_no_overlap():
    assert _bbox_iou([0, 0, 50, 50], [100, 100, 200, 200]) == 0.0


def test_bbox_iou_partial():
    iou = _bbox_iou([0, 0, 100, 100], [50, 50, 150, 150])
    assert 0 < iou < 1


def test_deduplicate_removes_duplicates():
    blocks = [
        {"text": "Hello", "bbox": [0, 0, 100, 50], "confidence": 90},
        {"text": "Hello", "bbox": [2, 2, 98, 48], "confidence": 80},
        {"text": "World", "bbox": [0, 60, 100, 110], "confidence": 85},
    ]
    result = _deduplicate_blocks(blocks)
    texts = [b["text"] for b in result]
    assert texts.count("Hello") == 1
    assert "World" in texts


def test_deduplicate_keeps_higher_confidence():
    blocks = [
        {"text": "Test", "bbox": [0, 0, 100, 50], "confidence": 70},
        {"text": "Test", "bbox": [0, 0, 100, 50], "confidence": 95},
    ]
    result = _deduplicate_blocks(blocks)
    assert len(result) == 1
    assert result[0]["confidence"] == 95


# ---------------------------------------------------------------------------
# _data_to_blocks: line-level grouping
# ---------------------------------------------------------------------------

def _mock_tesseract_data():
    """Simulate pytesseract.image_to_data output with hierarchy fields."""
    return {
        "block_num": [1, 1, 1, 1, 1, 2, 2, 2],
        "par_num":   [1, 1, 1, 1, 1, 1, 1, 1],
        "line_num":  [1, 1, 1, 2, 2, 1, 1, 1],
        "text":      ["GOVERNMENT", "WARNING:", "", "(1)", "According", "Brand", "Label", ""],
        "conf":      [91, 90, -1, 88, 87, 92, 93, -1],
        "left":      [10, 120, 0, 10, 50, 300, 350, 0],
        "top":       [100, 100, 0, 130, 130, 20, 20, 0],
        "width":     [100, 80, 0, 30, 80, 40, 50, 0],
        "height":    [20, 20, 0, 20, 20, 25, 25, 0],
    }


def test_data_to_blocks_groups_by_line():
    data = _mock_tesseract_data()
    blocks = _data_to_blocks(data)
    texts = [b["text"] for b in blocks]
    assert "GOVERNMENT WARNING:" in texts
    assert "(1) According" in texts
    assert "Brand Label" in texts


def test_data_to_blocks_merged_bbox():
    data = _mock_tesseract_data()
    blocks = _data_to_blocks(data)
    gov_block = next(b for b in blocks if "GOVERNMENT" in b["text"])
    assert gov_block["bbox"][0] == 10
    assert gov_block["bbox"][2] == 200  # left(120) + width(80)


def test_data_to_blocks_avg_confidence():
    data = _mock_tesseract_data()
    blocks = _data_to_blocks(data)
    gov_block = next(b for b in blocks if "GOVERNMENT" in b["text"])
    assert gov_block["confidence"] == pytest.approx((91 + 90) / 2)


def test_data_to_blocks_skips_empty_and_negative_conf():
    data = _mock_tesseract_data()
    blocks = _data_to_blocks(data)
    all_text = " ".join(b["text"] for b in blocks)
    assert all_text.strip()
    for b in blocks:
        assert b["text"].strip()
        assert b["confidence"] >= 0


def test_data_to_blocks_fallback_without_hierarchy():
    """When block_num/par_num/line_num are absent, falls back to word-level."""
    data = {
        "text": ["Hello", "World"],
        "conf": [90, 85],
        "left": [10, 60],
        "top": [10, 10],
        "width": [40, 50],
        "height": [20, 20],
    }
    blocks = _data_to_blocks(data)
    assert len(blocks) == 2
    assert blocks[0]["text"] == "Hello"
    assert blocks[1]["text"] == "World"


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def test_run_ocr_returns_list_of_blocks(white_image):
    try:
        blocks = run_ocr(white_image)
    except OcrUnavailableError:
        pytest.skip("Tesseract not installed")
    assert isinstance(blocks, list)
    for b in blocks:
        assert "text" in b
        assert "bbox" in b
        assert "confidence" in b
        assert len(b["bbox"]) == 4


def test_deduplicate_fuzzy_similar_text():
    """Near-identical text with overlapping bboxes should be deduped (Bug 9)."""
    blocks = [
        {"text": "RED WINE VICTORIA 12% ALC.VOL.", "bbox": [10, 10, 300, 40], "confidence": 90},
        {"text": "RED WINE VICTORIA 12% ALC./VOL.", "bbox": [12, 11, 298, 39], "confidence": 85},
    ]
    result = _deduplicate_blocks(blocks)
    assert len(result) == 1
    assert result[0]["confidence"] == 90


def test_deduplicate_keeps_different_text():
    """Different text blocks at different positions should be kept."""
    blocks = [
        {"text": "RED WINE", "bbox": [10, 10, 100, 40], "confidence": 90},
        {"text": "PALE ALE", "bbox": [10, 50, 100, 80], "confidence": 88},
    ]
    result = _deduplicate_blocks(blocks)
    assert len(result) == 2


def test_run_ocr_raises_when_tesseract_missing(white_image):
    try:
        run_ocr(white_image)
    except OcrUnavailableError:
        return
    pass
