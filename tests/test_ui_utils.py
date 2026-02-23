"""Tests for UI helpers (bbox overlay)."""
from PIL import Image

from src.ui_utils import draw_bbox_on_image


def test_draw_bbox_returns_copy():
    img = Image.new("RGB", (100, 100), color="white")
    out = draw_bbox_on_image(img, [10, 10, 50, 50])
    assert out is not img
    assert out.size == img.size


def test_draw_bbox_none_returns_unchanged_copy():
    img = Image.new("RGB", (100, 100), color="white")
    out = draw_bbox_on_image(img, None)
    assert out is not img
    assert out.size == img.size


def test_draw_bbox_short_list_returns_unchanged():
    img = Image.new("RGB", (100, 100), color="white")
    out = draw_bbox_on_image(img, [1, 2])
    assert out is not img
