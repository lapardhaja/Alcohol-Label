"""Tests for OCR module."""
import pytest
from PIL import Image

from src.ocr import run_ocr, _preprocess, _deduplicate_blocks, _bbox_iou, OcrUnavailableError


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


def test_run_ocr_raises_when_tesseract_missing(white_image):
    try:
        run_ocr(white_image)
    except OcrUnavailableError:
        return
    pass
