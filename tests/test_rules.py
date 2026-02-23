"""Tests for rule engine."""
import pytest

from src.rules.engine import (
    run_rules, _norm, _similarity, _net_contents_to_ml,
    _smart_match, _tokens_found_in_text,
)


def test_norm():
    assert _norm("  foo   bar  ") == "foo bar"
    assert _norm("") == ""
    assert _norm(None) == ""


def test_similarity_exact():
    assert _similarity("Stones Throw", "Stones Throw") >= 0.99


def test_similarity_case_insensitive():
    assert _similarity("STONES THROW", "Stones Throw") >= 0.9


def test_similarity_empty():
    assert _similarity("", "x") == 0.0
    assert _similarity("x", "") == 0.0


def test_run_rules_returns_list(ocr_blocks, app_data):
    from src.extraction import extract_fields
    extracted = extract_fields(ocr_blocks)
    results = run_rules(extracted, app_data)
    assert isinstance(results, list)
    for r in results:
        assert "rule_id" in r
        assert "category" in r
        assert r["status"] in ("pass", "needs_review", "fail")
        assert "message" in r


def test_run_rules_identity_pass_when_matching(ocr_blocks, app_data):
    from src.extraction import extract_fields
    extracted = extract_fields(ocr_blocks)
    results = run_rules(extracted, app_data)
    identity = [r for r in results if r["category"] == "Identity"]
    assert len(identity) >= 2
    # Brand and class should pass or need review (fuzzy match)
    statuses = [r["status"] for r in identity]
    assert "fail" not in statuses or any(s == "pass" for s in statuses)


def test_run_rules_warning_category(ocr_blocks, app_data):
    from src.extraction import extract_fields
    extracted = extract_fields(ocr_blocks)
    results = run_rules(extracted, app_data)
    warning = [r for r in results if r["category"] == "Warning"]
    assert len(warning) >= 1


def test_run_rules_origin_imported_requires_country(app_data_imported):
    from src.extraction import extract_fields
    blocks = [
        {"text": "Brand", "bbox": [0, 0, 50, 20], "confidence": 90},
        {"text": "Vodka", "bbox": [0, 25, 40, 45], "confidence": 90},
        {"text": "Bottled by X", "bbox": [0, 50, 100, 70], "confidence": 90},
    ]
    extracted = extract_fields(blocks)
    results = run_rules(extracted, app_data_imported)
    origin = [r for r in results if r["category"] == "Origin"]
    assert len(origin) >= 1


# ---------------------------------------------------------------------------
# Beverage-type-specific rule tests
# ---------------------------------------------------------------------------

def _minimal_extracted():
    """Extracted data with all fields present."""
    return {
        "brand_name": {"value": "TestBrand", "bbox": [0, 0, 100, 20]},
        "class_type": {"value": "Vodka", "bbox": [0, 25, 80, 45]},
        "alcohol_pct": {"value": "40", "bbox": [0, 50, 60, 65]},
        "proof": {"value": "", "bbox": None},
        "net_contents": {"value": "750 mL", "bbox": [0, 70, 80, 85]},
        "government_warning": {"value": "GOVERNMENT WARNING: (1) test", "bbox": [0, 90, 200, 120]},
        "bottler": {"value": "Bottled by TestCo", "bbox": [0, 130, 150, 150]},
        "country_of_origin": {"value": "", "bbox": None},
        "_all_blocks": [{"text": "TestBrand Vodka 40% 750 mL GOVERNMENT WARNING: (1) test Bottled by TestCo"}],
    }


def test_beer_abv_not_mandatory():
    """For beer, ABV is optional unless flavored."""
    extracted = _minimal_extracted()
    extracted["alcohol_pct"] = {"value": "", "bbox": None}
    app_data = {"beverage_type": "beer", "brand_name": "TestBrand", "class_type": "Beer"}
    results = run_rules(extracted, app_data)
    abv_rules = [r for r in results if "alcohol content" in r["rule_id"].lower()]
    assert any(r["status"] == "pass" for r in abv_rules)


def test_spirits_abv_mandatory():
    """For spirits, missing ABV is a fail."""
    extracted = _minimal_extracted()
    extracted["alcohol_pct"] = {"value": "", "bbox": None}
    app_data = {"beverage_type": "spirits", "brand_name": "TestBrand", "class_type": "Vodka"}
    results = run_rules(extracted, app_data)
    abv_rules = [r for r in results if "alcohol content" in r["rule_id"].lower()]
    assert any(r["status"] == "fail" for r in abv_rules)


def test_beer_proof_not_applicable():
    """Beer labels should not require proof."""
    extracted = _minimal_extracted()
    app_data = {"beverage_type": "beer", "brand_name": "TestBrand", "proof": "80"}
    results = run_rules(extracted, app_data)
    proof_rules = [r for r in results if "proof" in r["rule_id"].lower()]
    assert all(r["status"] == "pass" for r in proof_rules)


def test_wine_sulfites_default_required():
    """Wine: sulfites declaration is required by default."""
    extracted = _minimal_extracted()
    app_data = {"beverage_type": "wine", "brand_name": "TestBrand"}
    results = run_rules(extracted, app_data)
    sulfite_rules = [r for r in results if "sulfite" in r["rule_id"].lower()]
    assert any(r["status"] == "fail" for r in sulfite_rules)


def test_wine_sulfites_pass_when_present():
    """Wine: sulfites pass when statement is on label."""
    extracted = _minimal_extracted()
    extracted["_all_blocks"] = [{"text": "Contains Sulfites"}]
    app_data = {"beverage_type": "wine", "brand_name": "TestBrand"}
    results = run_rules(extracted, app_data)
    sulfite_rules = [r for r in results if "sulfite" in r["rule_id"].lower()]
    assert any(r["status"] == "pass" for r in sulfite_rules)


# ---------------------------------------------------------------------------
# Imperial net contents conversion
# ---------------------------------------------------------------------------

def test_net_contents_to_ml_metric():
    assert _net_contents_to_ml("750 mL") == 750
    assert _net_contents_to_ml("1 L") == 1000


def test_net_contents_to_ml_fl_oz():
    ml = _net_contents_to_ml("12 fl oz")
    assert ml is not None
    assert 350 < ml < 360


def test_net_contents_to_ml_quart():
    ml = _net_contents_to_ml("1 qt")
    assert ml is not None
    assert 940 < ml < 950


def test_net_contents_to_ml_pint():
    ml = _net_contents_to_ml("1 pt")
    assert ml is not None
    assert 470 < ml < 480


def test_net_contents_to_ml_empty():
    assert _net_contents_to_ml("") is None
    assert _net_contents_to_ml("unknown") is None


# ---------------------------------------------------------------------------
# Rule results include extracted_value and app_value (Bug 12)
# ---------------------------------------------------------------------------

def test_rule_results_have_values(ocr_blocks, app_data):
    from src.extraction import extract_fields
    extracted = extract_fields(ocr_blocks)
    results = run_rules(extracted, app_data)
    for r in results:
        assert "extracted_value" in r, f"Missing extracted_value in rule: {r['rule_id']}"
        assert "app_value" in r, f"Missing app_value in rule: {r['rule_id']}"


def test_rule_brand_mismatch_shows_values():
    extracted = _minimal_extracted()
    extracted["brand_name"]["value"] = "WrongBrand"
    app_data = {"beverage_type": "spirits", "brand_name": "CorrectBrand", "class_type": "Vodka"}
    results = run_rules(extracted, app_data)
    brand = [r for r in results if "brand" in r["rule_id"].lower()][0]
    assert brand["extracted_value"] == "WrongBrand"
    assert brand["app_value"] == "CorrectBrand"
    assert brand["status"] == "fail"


def test_rule_net_imperial_comparison():
    """Imperial net contents should compare correctly with metric application data."""
    extracted = _minimal_extracted()
    extracted["net_contents"]["value"] = "24 fl oz"
    app_data = {"beverage_type": "beer", "brand_name": "TestBrand", "net_contents_ml": "710 mL"}
    results = run_rules(extracted, app_data)
    net_rules = [r for r in results if "net contents" in r["rule_id"].lower()]
    assert len(net_rules) >= 1
    assert any(r.get("extracted_value") == "24 fl oz" for r in net_rules)


# ---------------------------------------------------------------------------
# Smart matching tests
# ---------------------------------------------------------------------------

class TestSmartMatch:
    """Tests for multi-strategy _smart_match()."""

    def test_exact_match(self):
        score, reason = _smart_match("ABC Distillery", "ABC Distillery")
        assert score == 1.0
        assert reason == "exact"

    def test_case_insensitive_exact(self):
        score, reason = _smart_match("ABC DISTILLERY", "abc distillery")
        assert score == 1.0
        assert reason == "exact"

    def test_token_containment_rye_in_straight_rye(self):
        """App says 'Rye' and label has 'Straight Rye Whisky' -> should pass."""
        score, reason = _smart_match("Rye", "Straight Rye Whisky")
        assert score >= 0.90
        assert reason == "token_containment"

    def test_token_containment_full_brand(self):
        score, reason = _smart_match("ABC Distillery", "ABC DISTILLERY")
        assert score >= 0.95

    def test_token_containment_ale(self):
        """App says 'Ale' and label has 'Pale Ale' -> should pass via containment."""
        score, reason = _smart_match("Ale", "Pale Ale")
        assert score >= 0.90

    def test_reverse_containment(self):
        """Label has fewer words than app."""
        score, reason = _smart_match("Single Barrel Straight Rye Whisky", "Straight Rye Whisky")
        assert score >= 0.88

    def test_substring_match(self):
        score, reason = _smart_match("Pale", "Pale Ale")
        assert score >= 0.90

    def test_fuzzy_token_ocr_typo(self):
        """OCR produces 'DISTILERY' instead of 'DISTILLERY'."""
        score, reason = _smart_match("ABC Distillery", "ABC DISTILERY")
        assert score >= 0.85

    def test_genuine_mismatch(self):
        """Merlot vs American Red Wine should score low."""
        score, reason = _smart_match("Merlot", "American Red Wine")
        assert score < 0.70

    def test_empty_values(self):
        score, _ = _smart_match("", "something")
        assert score == 0.0
        score, _ = _smart_match("something", "")
        assert score == 0.0

    def test_stones_throw_case_variation(self):
        """Dave's example from the project brief."""
        score, reason = _smart_match("STONE'S THROW", "Stone's Throw")
        assert score >= 0.90


class TestTokensFoundInText:
    def test_merlot_in_body_text(self):
        text = "This red wine is a blend of 50% MERLOT and 50% CABERNET SAUVIGNON"
        assert _tokens_found_in_text("Merlot", text) is True

    def test_missing_token(self):
        text = "This red wine is a blend of CABERNET SAUVIGNON"
        assert _tokens_found_in_text("Merlot", text) is False

    def test_multi_token(self):
        text = "ABC DISTILLERY Frederick MD 750 mL"
        assert _tokens_found_in_text("ABC Distillery", text) is True


class TestSmartMatchInRules:
    """Integration: _smart_match used by run_rules for identity checks."""

    def test_class_type_token_containment_passes(self):
        """App: 'Rye', Label extracted: 'Straight Rye Whisky' -> should pass."""
        extracted = _minimal_extracted()
        extracted["class_type"]["value"] = "Straight Rye Whisky"
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand", "class_type": "Rye"}
        results = run_rules(extracted, app_data)
        class_rules = [r for r in results if "class" in r["rule_id"].lower()]
        assert any(r["status"] == "pass" for r in class_rules)

    def test_class_merlot_vs_red_wine_fallback(self):
        """App: 'Merlot', Label: 'American Red Wine', but MERLOT in body text -> needs_review."""
        extracted = _minimal_extracted()
        extracted["class_type"]["value"] = "American Red Wine"
        extracted["_all_blocks"] = [
            {"text": "ABC WINERY AMERICAN RED WINE"},
            {"text": "This red wine is a blend of 50% MERLOT and 50% CABERNET SAUVIGNON"},
            {"text": "GOVERNMENT WARNING: (1) test"},
        ]
        app_data = {"beverage_type": "wine", "brand_name": "ABC Winery", "class_type": "Merlot"}
        results = run_rules(extracted, app_data)
        class_rules = [r for r in results if "class" in r["rule_id"].lower()]
        assert any(r["status"] == "needs_review" for r in class_rules)

    def test_brand_case_insensitive_pass(self):
        extracted = _minimal_extracted()
        extracted["brand_name"]["value"] = "ABC DISTILLERY"
        app_data = {"beverage_type": "spirits", "brand_name": "ABC Distillery", "class_type": "Vodka"}
        results = run_rules(extracted, app_data)
        brand_rules = [r for r in results if "brand" in r["rule_id"].lower()]
        assert any(r["status"] == "pass" for r in brand_rules)

    def test_bottler_smart_match_token_containment(self):
        """Bottler label has header + name, app has just name -> should pass."""
        extracted = _minimal_extracted()
        extracted["bottler"]["value"] = "Bottled by ABC Distillery Frederick MD"
        app_data = {"beverage_type": "spirits", "brand_name": "TestBrand", "bottler_name": "ABC Distillery"}
        results = run_rules(extracted, app_data)
        origin_rules = [r for r in results if r["category"] == "Origin"]
        bottler_rules = [r for r in origin_rules if "bottler" in r["rule_id"].lower()]
        assert any(r["status"] == "pass" for r in bottler_rules)
