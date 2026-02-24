"""Run pipeline on test_2 and check suspicious token logic for government warning."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

# Simulated OCR output from test_2 (from user's earlier report)
# "GOVERNMENT WARNING: QB WARNING: (1) According to the Surgeon General..."
# Extra: QB (OCR for GOVERNMENT), CIG (OCR for cause)
# Version A: has both statements but extra words -> hits suspicious branch
gov_warning_extracted = (
    "GOVERNMENT WARNING: QB WARNING: (1) According to the Surgeon General, "
    "women should not drink alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or operate machinery, "
    "and may cause cig health problems."
)

required = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink "
    "alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or operate "
    "machinery, and may cause health problems."
)


def _warning_words(s):
    import re
    return re.findall(r"\b\w+\b", (s or "").upper())


def main():
    from src.rules.engine import _get_suspicious_warning_tokens, run_rules

    req_words = _warning_words(required)
    ext_words = _warning_words(gov_warning_extracted)
    from collections import Counter
    req_counts = Counter(req_words)
    extra_unique = [w for w in set(ext_words) if w not in req_counts]

    print("=== Extracted gov warning (test_2 style) ===")
    print(gov_warning_extracted[:120] + "...")
    print()
    print("=== Extra tokens (in extracted, not in reference) ===")
    print(extra_unique)
    print()

    suspicious = _get_suspicious_warning_tokens(extra_unique, set(req_words))
    print("=== Suspicious tokens (not in dictionary) ===")
    print(suspicious)
    print()

    # Full pipeline check
    extracted = {
        "government_warning": {"value": gov_warning_extracted, "bbox": [0, 0, 200, 300]},
        "brand_name": {"value": "Malt & Hop Brewery", "bbox": None},
        "class_type": {"value": "PALE ALE", "bbox": None},
        "alcohol_pct": {"value": "5", "bbox": None},
        "proof": {"value": "", "bbox": None},
        "net_contents": {"value": "24 fl oz", "bbox": None},
        "bottler": {"value": "Malt & Hop Brewery", "bbox": None},
        "country_of_origin": {"value": "", "bbox": None},
        "_all_blocks": [],
    }
    app_data = {
        "beverage_type": "beer",
        "brand_name": "Malt & Hop Brewery",
        "class_type": "Pale Ale",
        "alcohol_pct": "5",
        "net_contents_ml": "24 fl oz",
        "bottler_name": "Malt & Hop Brewery",
    }
    results = run_rules(extracted, app_data)
    warn_rules = [r for r in results if "warning" in r["rule_id"].lower()]
    print("=== Government warning rule results ===")
    for r in warn_rules:
        print(f"  {r['rule_id']}: {r['status']}")
        print(f"    message: {r['message'][:80]}...")
        if "extracted_value" in r and r["extracted_value"]:
            print(f"    extracted_value (filtered): {r['extracted_value'][:100]}...")


if __name__ == "__main__":
    main()
