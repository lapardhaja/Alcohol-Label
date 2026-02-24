"""Tests for OCR confusable detection."""
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.rules.engine import _is_ocr_confusable, run_rules


class TestOCRConfusable:
    def test_identical_strings(self):
        assert _is_ocr_confusable("hello", "hello") is True

    def test_l_vs_1(self):
        assert _is_ocr_confusable("label", "1abel") is True

    def test_O_vs_0(self):
        assert _is_ocr_confusable("Tom", "T0m") is True

    def test_comma_vs_period(self):
        assert _is_ocr_confusable("45,0", "45.0") is True

    def test_S_vs_5(self):
        assert _is_ocr_confusable("St", "5t") is True

    def test_completely_different(self):
        assert _is_ocr_confusable("hello", "world") is False

    def test_empty_strings(self):
        assert _is_ocr_confusable("", "hello") is False
        assert _is_ocr_confusable("hello", "") is False

    def test_single_char_diff(self):
        assert _is_ocr_confusable("ABC", "A8C") is True

    def test_two_confusable_diffs(self):
        assert _is_ocr_confusable("IO", "10") is True

    def test_too_many_diffs(self):
        assert _is_ocr_confusable("ABCD", "1234") is False


class TestOCRConfusableInRules:
    def test_brand_ocr_misread_not_hard_fail(self):
        """Brand with OCR confusable diff should be needs_review, not fail."""
        extracted = {
            "brand_name": {"value": "T0m's", "bbox": [0, 0, 100, 20]},
            "class_type": {"value": "Vodka", "bbox": [0, 25, 80, 45]},
            "alcohol_pct": {"value": "40", "bbox": [0, 50, 60, 65]},
            "proof": {"value": "", "bbox": None},
            "net_contents": {"value": "750 mL", "bbox": [0, 70, 80, 85]},
            "government_warning": {"value": "GOVERNMENT WARNING: (1) test", "bbox": [0, 90, 200, 120]},
            "bottler": {"value": "Bottled by TestCo", "bbox": [0, 130, 150, 150]},
            "country_of_origin": {"value": "", "bbox": None},
            "_all_blocks": [{"text": "T0m's Vodka 40%"}],
        }
        app_data = {"beverage_type": "spirits", "brand_name": "Tom's", "class_type": "Vodka"}
        results = run_rules(extracted, app_data)
        brand = [r for r in results if "brand" in r["rule_id"].lower()]
        assert len(brand) >= 1
        statuses = {r["status"] for r in brand}
        assert "fail" not in statuses

    def test_abv_ocr_confusable_message(self):
        """ABV with OCR-like diff should mention 'OCR misread' in message."""
        extracted = {
            "brand_name": {"value": "TestBrand", "bbox": [0, 0, 100, 20]},
            "class_type": {"value": "Vodka", "bbox": [0, 25, 80, 45]},
            "alcohol_pct": {"value": "4O", "bbox": [0, 50, 60, 65]},
            "proof": {"value": "", "bbox": None},
            "net_contents": {"value": "750 mL", "bbox": [0, 70, 80, 85]},
            "government_warning": {"value": "GOVERNMENT WARNING: (1) test", "bbox": [0, 90, 200, 120]},
            "bottler": {"value": "Bottled by TestCo", "bbox": [0, 130, 150, 150]},
            "country_of_origin": {"value": "", "bbox": None},
            "_all_blocks": [{"text": "TestBrand Vodka 4O%"}],
        }
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand", "alcohol_pct": "40"}
        results = run_rules(extracted, app_data)
        abv = [r for r in results if "alcohol content" in r["rule_id"].lower()]
        assert len(abv) >= 1
