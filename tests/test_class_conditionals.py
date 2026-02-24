"""Tests for spirit-class-driven conditional auto-selection."""
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.rules.engine import run_rules, _infer_conditionals_from_class, _load_config


@pytest.fixture
def config():
    return _load_config()


def _minimal_extracted(class_type="Vodka"):
    return {
        "brand_name": {"value": "TestBrand", "bbox": [0, 0, 100, 20]},
        "class_type": {"value": class_type, "bbox": [0, 25, 80, 45]},
        "alcohol_pct": {"value": "40", "bbox": [0, 50, 60, 65]},
        "proof": {"value": "", "bbox": None},
        "net_contents": {"value": "750 mL", "bbox": [0, 70, 80, 85]},
        "government_warning": {"value": "GOVERNMENT WARNING: (1) test", "bbox": [0, 90, 200, 120]},
        "bottler": {"value": "Bottled by TestCo", "bbox": [0, 130, 150, 150]},
        "country_of_origin": {"value": "", "bbox": None},
        "_all_blocks": [{"text": "TestBrand Vodka 40% 750 mL GOVERNMENT WARNING: (1) test Bottled by TestCo"}],
    }


class TestInferConditionals:
    def test_vodka_requires_neutral_spirits(self, config):
        inferred = _infer_conditionals_from_class("Vodka", config)
        assert "neutral_spirits" in inferred

    def test_straight_bourbon_requires_state_not_age(self, config):
        """27 CFR 5.40(a): age is now age-based, not class-based."""
        inferred = _infer_conditionals_from_class("Straight Bourbon Whiskey", config)
        assert "age_statement" not in inferred
        assert "state_of_distillation" in inferred

    def test_tennessee_whiskey_requires_state_not_age(self, config):
        """27 CFR 5.40(a): age is now age-based, not class-based."""
        inferred = _infer_conditionals_from_class("Tennessee Whiskey", config)
        assert "age_statement" not in inferred
        assert "state_of_distillation" in inferred

    def test_bourbon_requires_state(self, config):
        inferred = _infer_conditionals_from_class("Bourbon Whiskey", config)
        assert "state_of_distillation" in inferred

    def test_neutral_spirits_requires_neutral(self, config):
        inferred = _infer_conditionals_from_class("Neutral Spirits", config)
        assert "neutral_spirits" in inferred

    def test_grain_spirits_requires_neutral(self, config):
        inferred = _infer_conditionals_from_class("Grain Spirits", config)
        assert "neutral_spirits" in inferred

    def test_scotch_requires_country(self, config):
        inferred = _infer_conditionals_from_class("Scotch Whisky", config)
        assert "country_of_origin" in inferred

    def test_beer_has_no_inferred(self, config):
        inferred = _infer_conditionals_from_class("Pale Ale", config)
        assert len(inferred) == 0

    def test_empty_class(self, config):
        inferred = _infer_conditionals_from_class("", config)
        assert len(inferred) == 0


class TestClassDrivenRules:
    def test_vodka_neutral_spirits_auto_required(self):
        """Vodka should auto-require neutral spirits even without explicit flag."""
        extracted = _minimal_extracted("Vodka")
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand", "class_type": "Vodka"}
        results = run_rules(extracted, app_data)
        neutral_rules = [r for r in results if "neutral" in r["rule_id"].lower()]
        assert len(neutral_rules) >= 1
        assert any(r["status"] == "fail" for r in neutral_rules)

    def test_whisky_age_unknown_requires_age_statement(self):
        """27 CFR 5.40(a): Whisky with unknown age (< 4) requires age statement."""
        extracted = _minimal_extracted("Straight Bourbon Whiskey")
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand",
                    "class_type": "Straight Bourbon Whiskey"}
        results = run_rules(extracted, app_data)
        age_rules = [r for r in results if "age" in r["rule_id"].lower()]
        assert len(age_rules) >= 1
        assert any(r["status"] == "fail" for r in age_rules)

    def test_whisky_age_4_years_optional_passes(self):
        """27 CFR 5.40(a): Whisky aged 4+ years - age statement optional; if present, passes."""
        extracted = _minimal_extracted("Straight Bourbon Whiskey")
        extracted["_all_blocks"] = [
            {"text": "Aged 4 years TestBrand Straight Bourbon Whiskey GOVERNMENT WARNING: (1) test Bottled by TestCo 750 mL 40%"}
        ]
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand",
                    "class_type": "Straight Bourbon Whiskey"}
        results = run_rules(extracted, app_data)
        age_rules = [r for r in results if "age" in r["rule_id"].lower()]
        # With "4 years" on label, age is parsed; >= 4 so rule not required - no age rule, or pass if run
        assert not any(r["status"] == "fail" for r in age_rules)

    def test_whisky_age_3_years_requires_age_statement(self):
        """27 CFR 5.40(a): Whisky aged < 4 years requires age statement."""
        extracted = _minimal_extracted("Bourbon Whiskey")
        extracted["_all_blocks"] = [
            {"text": "TestBrand Bourbon Whiskey GOVERNMENT WARNING: (1) test Bottled by TestCo 750 mL 40%"}
        ]
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand",
                    "class_type": "Bourbon Whiskey", "age_years": 3}
        results = run_rules(extracted, app_data)
        age_rules = [r for r in results if "age" in r["rule_id"].lower()]
        assert len(age_rules) >= 1
        assert any(r["status"] == "fail" for r in age_rules)

    def test_whisky_age_4_app_data_optional(self):
        """27 CFR 5.40(a): Whisky with age_years=4 in app_data - age statement optional."""
        extracted = _minimal_extracted("Bourbon Whiskey")
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand",
                    "class_type": "Bourbon Whiskey", "age_years": 4}
        results = run_rules(extracted, app_data)
        age_rules = [r for r in results if "age" in r["rule_id"].lower()]
        # age >= 4: not required, so no age rule should run (or pass)
        assert not any(r["status"] == "fail" for r in age_rules)

    def test_ale_no_spirits_conditionals(self):
        """Beer (Ale) should not trigger spirits-only conditionals."""
        extracted = _minimal_extracted("Ale")
        app_data = {"beverage_type": "beer", "brand_name": "TestBrand", "class_type": "Ale"}
        results = run_rules(extracted, app_data)
        neutral_rules = [r for r in results if "neutral" in r["rule_id"].lower()]
        age_rules = [r for r in results if "age" in r["rule_id"].lower()]
        wood_rules = [r for r in results if "wood" in r["rule_id"].lower()]
        assert len(neutral_rules) == 0
        assert len(age_rules) == 0
        assert len(wood_rules) == 0
