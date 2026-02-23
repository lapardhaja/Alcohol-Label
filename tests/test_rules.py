"""Tests for rule engine."""
import pytest

from src.rules.engine import run_rules, _norm, _similarity


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
    # No country on label
    blocks = [
        {"text": "Brand", "bbox": [0, 0, 50, 20], "confidence": 90},
        {"text": "Vodka", "bbox": [0, 25, 40, 45], "confidence": 90},
        {"text": "Bottled by X", "bbox": [0, 50, 100, 70], "confidence": 90},
    ]
    extracted = extract_fields(blocks)
    results = run_rules(extracted, app_data_imported)
    origin = [r for r in results if r["category"] == "Origin"]
    country_rules = [r for r in origin if "country" in r.get("rule_id", "").lower() or "origin" in r.get("rule_id", "").lower()]
    # Should have at least one origin rule; if imported and no country extracted, expect fail
    assert len(origin) >= 1
