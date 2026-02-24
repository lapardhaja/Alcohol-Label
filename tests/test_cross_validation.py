"""Tests for cross-validation: numeric ABV, proof-to-ABV, country of origin, bottler city/state."""
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.rules.engine import run_rules, _parse_abv_float


def _minimal_extracted(**overrides):
    base = {
        "brand_name": {"value": "TestBrand", "bbox": [0, 0, 100, 20]},
        "class_type": {"value": "Vodka", "bbox": [0, 25, 80, 45]},
        "alcohol_pct": {"value": "40", "bbox": [0, 50, 60, 65]},
        "proof": {"value": "", "bbox": None},
        "net_contents": {"value": "750 mL", "bbox": [0, 70, 80, 85]},
        "government_warning": {"value": "GOVERNMENT WARNING: (1) test", "bbox": [0, 90, 200, 120]},
        "bottler": {"value": "Bottled by TestCo, Springfield, IL", "bbox": [0, 130, 150, 150]},
        "country_of_origin": {"value": "", "bbox": None},
        "_all_blocks": [{"text": "TestBrand Vodka 40% 750 mL GOVERNMENT WARNING: (1) test Bottled by TestCo Springfield IL"}],
    }
    base.update(overrides)
    return base


class TestNumericABV:
    def test_exact_string_match(self):
        extracted = _minimal_extracted()
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand", "alcohol_pct": "40"}
        results = run_rules(extracted, app_data)
        abv = [r for r in results if "alcohol content" in r["rule_id"].lower()]
        assert any(r["status"] == "pass" for r in abv)

    def test_numeric_tolerance_passes(self):
        """'40.0' vs '40' should pass (within 0.15 tolerance)."""
        extracted = _minimal_extracted()
        extracted["alcohol_pct"] = {"value": "40.0", "bbox": [0, 50, 60, 65]}
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand", "alcohol_pct": "40"}
        results = run_rules(extracted, app_data)
        abv = [r for r in results if "alcohol content" in r["rule_id"].lower()]
        assert any(r["status"] == "pass" for r in abv)

    def test_different_abv_fails(self):
        """'45' vs '40' should be needs_review."""
        extracted = _minimal_extracted()
        extracted["alcohol_pct"] = {"value": "45", "bbox": [0, 50, 60, 65]}
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand", "alcohol_pct": "40"}
        results = run_rules(extracted, app_data)
        abv = [r for r in results if "alcohol content" in r["rule_id"].lower()]
        assert any(r["status"] == "needs_review" for r in abv)

    def test_parse_abv_float_basic(self):
        assert _parse_abv_float("45") == 45.0
        assert _parse_abv_float("5.5") == 5.5
        assert _parse_abv_float("12.0%") == 12.0
        assert _parse_abv_float("") is None
        assert _parse_abv_float("abc") is None


class TestProofABVConsistency:
    def test_consistent_proof(self):
        """90 proof and 45% ABV should pass."""
        extracted = _minimal_extracted()
        extracted["alcohol_pct"] = {"value": "45", "bbox": [0, 50, 60, 65]}
        extracted["proof"] = {"value": "90", "bbox": [0, 55, 60, 68]}
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand", "alcohol_pct": "45", "proof": "90"}
        results = run_rules(extracted, app_data)
        consistency = [r for r in results if "consistency" in r["rule_id"].lower()]
        assert any(r["status"] == "pass" for r in consistency)

    def test_inconsistent_proof(self):
        """80 proof and 45% ABV should flag needs_review."""
        extracted = _minimal_extracted()
        extracted["alcohol_pct"] = {"value": "45", "bbox": [0, 50, 60, 65]}
        extracted["proof"] = {"value": "80", "bbox": [0, 55, 60, 68]}
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand", "alcohol_pct": "45", "proof": "80"}
        results = run_rules(extracted, app_data)
        consistency = [r for r in results if "consistency" in r["rule_id"].lower()]
        assert any(r["status"] == "needs_review" for r in consistency)

    def test_no_proof_no_check(self):
        """No proof on label — consistency check should not appear."""
        extracted = _minimal_extracted()
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand", "alcohol_pct": "40"}
        results = run_rules(extracted, app_data)
        consistency = [r for r in results if "consistency" in r["rule_id"].lower()]
        assert len(consistency) == 0


class TestCountryOfOriginComparison:
    def test_matching_country(self):
        extracted = _minimal_extracted()
        extracted["country_of_origin"] = {"value": "Product of Australia", "bbox": [0, 160, 200, 180]}
        app_data = {"beverage_type": "wine", "brand_name": "TestBrand", "imported": True,
                    "country_of_origin": "Australia"}
        results = run_rules(extracted, app_data)
        co_rules = [r for r in results if "country" in r["rule_id"].lower()]
        assert any(r["status"] == "pass" for r in co_rules)

    def test_mismatching_country(self):
        extracted = _minimal_extracted()
        extracted["country_of_origin"] = {"value": "Product of France", "bbox": [0, 160, 200, 180]}
        app_data = {"beverage_type": "wine", "brand_name": "TestBrand", "imported": True,
                    "country_of_origin": "Australia"}
        results = run_rules(extracted, app_data)
        co_rules = [r for r in results if "country" in r["rule_id"].lower()]
        assert any(r["status"] == "needs_review" for r in co_rules)

    def test_missing_country_on_import(self):
        extracted = _minimal_extracted()
        app_data = {"beverage_type": "wine", "brand_name": "TestBrand", "imported": True,
                    "country_of_origin": "Australia"}
        results = run_rules(extracted, app_data)
        co_rules = [r for r in results if "country" in r["rule_id"].lower()]
        assert any(r["status"] == "fail" for r in co_rules)


class TestBottlerCityState:
    def test_city_state_found(self):
        extracted = _minimal_extracted()
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand",
                    "bottler_name": "TestCo", "bottler_city": "Springfield", "bottler_state": "IL"}
        results = run_rules(extracted, app_data)
        addr_rules = [r for r in results if "bottler address" in r["rule_id"].lower()]
        assert any(r["status"] == "pass" for r in addr_rules)

    def test_city_missing(self):
        extracted = _minimal_extracted()
        extracted["bottler"] = {"value": "Bottled by TestCo, IL", "bbox": [0, 130, 150, 150]}
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand",
                    "bottler_name": "TestCo", "bottler_city": "Springfield", "bottler_state": "IL"}
        results = run_rules(extracted, app_data)
        addr_rules = [r for r in results if "bottler address" in r["rule_id"].lower()]
        assert any(r["status"] == "needs_review" for r in addr_rules)

    def test_no_city_state_in_app_no_check(self):
        """If app has no city/state, don't emit bottler address rule."""
        extracted = _minimal_extracted()
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand",
                    "bottler_name": "TestCo", "bottler_city": "", "bottler_state": ""}
        results = run_rules(extracted, app_data)
        addr_rules = [r for r in results if "bottler address" in r["rule_id"].lower()]
        assert len(addr_rules) == 0


class TestBoldWarningNote:
    def test_bold_warning_always_present(self):
        """The bold warning note should always appear when warning is found."""
        extracted = _minimal_extracted()
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand"}
        results = run_rules(extracted, app_data)
        bold = [r for r in results if "bold" in r["rule_id"].lower()]
        assert len(bold) == 1
        assert bold[0]["status"] == "needs_review"
