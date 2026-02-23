"""Tests for OCR module."""
import pytest
from PIL import Image
from unittest.mock import patch

from src.ocr import (
    run_ocr,
    _preprocess_for_tesseract,
    _deduplicate_blocks,
    _bbox_iou,
    _data_to_blocks,
    _resize,
    _split_line_by_gaps,
    OcrUnavailableError,
)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_resize_caps_large_image():
    img = Image.new("RGB", (3000, 2000), color="white")
    resized = _resize(img)
    assert max(resized.size) == 2000

def test_resize_upscales_small_image():
    img = Image.new("RGB", (400, 300), color="white")
    resized = _resize(img)
    assert max(resized.size) >= 1000

def test_resize_leaves_normal_image():
    img = Image.new("RGB", (1500, 1200), color="white")
    resized = _resize(img)
    assert resized.size == (1500, 1200)

def test_preprocess_for_tesseract_returns_three_images():
    img = Image.new("RGB", (1500, 1000), color="white")
    original, sharpened, binary = _preprocess_for_tesseract(img)
    assert original.mode == "RGB"
    assert sharpened.mode == "L"
    assert binary.mode == "L"

def test_preprocess_for_tesseract_no_gaussian_blur():
    """Verify GaussianBlur is NOT applied (sharpening kernel instead)."""
    import numpy as np
    img = Image.new("RGB", (1500, 1000), color="white")
    _, sharpened, _ = _preprocess_for_tesseract(img)
    arr = np.array(sharpened)
    assert arr.shape == (1000, 1500)


# ---------------------------------------------------------------------------
# Tesseract block splitting
# ---------------------------------------------------------------------------

def test_split_line_by_gaps_no_split():
    """Closely spaced words should not be split."""
    data = {
        "text": ["STRAIGHT", "RYE", "WHISKY"],
        "conf": [90, 91, 92],
        "left": [10, 90, 130],
        "top": [100, 100, 100],
        "width": [70, 30, 50],
        "height": [20, 20, 20],
    }
    blocks = _split_line_by_gaps(data, [0, 1, 2])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "STRAIGHT RYE WHISKY"

def test_split_line_by_gaps_splits_wide_gap():
    """Words with a large horizontal gap should be split into separate blocks."""
    data = {
        "text": ["FREDERICK,MD", "STRAIGHT", "RYE", "WHISKY"],
        "conf": [90, 91, 92, 93],
        "left": [10, 300, 380, 420],
        "top": [100, 100, 100, 100],
        "width": [100, 70, 30, 50],
        "height": [20, 20, 20, 20],
    }
    blocks = _split_line_by_gaps(data, [0, 1, 2, 3])
    assert len(blocks) == 2
    assert blocks[0]["text"] == "FREDERICK,MD"
    assert blocks[1]["text"] == "STRAIGHT RYE WHISKY"

def test_split_line_single_word():
    data = {
        "text": ["HELLO"],
        "conf": [95],
        "left": [10],
        "top": [10],
        "width": [80],
        "height": [20],
    }
    blocks = _split_line_by_gaps(data, [0])
    assert len(blocks) == 1
    assert blocks[0]["text"] == "HELLO"

def test_data_to_blocks_with_gap_splitting():
    """Integration: _data_to_blocks should split wide lines via gap detection."""
    data = {
        "block_num": [1, 1, 1, 1],
        "par_num": [1, 1, 1, 1],
        "line_num": [1, 1, 1, 1],
        "text": ["LEFT", "PANEL", "RIGHT", "PANEL"],
        "conf": [90, 91, 92, 93],
        "left": [10, 60, 400, 460],
        "top": [100, 100, 100, 100],
        "width": [40, 50, 50, 50],
        "height": [20, 20, 20, 20],
    }
    blocks = _data_to_blocks(data)
    assert len(blocks) == 2
    texts = [b["text"] for b in blocks]
    assert "LEFT PANEL" in texts
    assert "RIGHT PANEL" in texts


# ---------------------------------------------------------------------------
# bbox / dedup (unchanged)
# ---------------------------------------------------------------------------

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

def test_deduplicate_fuzzy_similar_text():
    blocks = [
        {"text": "RED WINE VICTORIA 12% ALC.VOL.", "bbox": [10, 10, 300, 40], "confidence": 90},
        {"text": "RED WINE VICTORIA 12% ALC./VOL.", "bbox": [12, 11, 298, 39], "confidence": 85},
    ]
    result = _deduplicate_blocks(blocks)
    assert len(result) == 1
    assert result[0]["confidence"] == 90

def test_deduplicate_keeps_different_text():
    blocks = [
        {"text": "RED WINE", "bbox": [10, 10, 100, 40], "confidence": 90},
        {"text": "PALE ALE", "bbox": [10, 50, 100, 80], "confidence": 88},
    ]
    result = _deduplicate_blocks(blocks)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _data_to_blocks: line-level grouping
# ---------------------------------------------------------------------------

def _mock_tesseract_data():
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
    assert gov_block["bbox"][2] == 200

def test_data_to_blocks_avg_confidence():
    data = _mock_tesseract_data()
    blocks = _data_to_blocks(data)
    gov_block = next(b for b in blocks if "GOVERNMENT" in b["text"])
    assert gov_block["confidence"] == pytest.approx((91 + 90) / 2)

def test_data_to_blocks_skips_empty_and_negative_conf():
    data = _mock_tesseract_data()
    blocks = _data_to_blocks(data)
    for b in blocks:
        assert b["text"].strip()
        assert b["confidence"] >= 0

def test_data_to_blocks_fallback_without_hierarchy():
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
# run_ocr integration (Tesseract only)
# ---------------------------------------------------------------------------

def test_run_ocr_uses_tesseract(white_image):
    """run_ocr should call Tesseract and return blocks."""
    mock_blocks = [
        {"text": "MOCK TEXT", "bbox": [10, 10, 100, 40], "confidence": 95.0},
    ]
    with patch("src.ocr._run_tesseract_ocr", return_value=mock_blocks):
        blocks = run_ocr(white_image)
    assert blocks == mock_blocks
    assert len(blocks) == 1
    assert blocks[0]["text"] == "MOCK TEXT"

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
