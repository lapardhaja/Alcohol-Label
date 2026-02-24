"""
Run pipeline on each test image and compare extracted output to expected (from batch_test.csv).
Usage: from project root, run: python -m scripts.compare_test_images
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Expected application data per test (from batch_test.csv / plan)
EXPECTED = {
    "test_1": {
        "brand_name": "ABC Distillery",
        "class_type": "Single Barrel Straight Rye Whisky",
        "alcohol_pct": "45",
        "proof": "90",
        "net_contents_ml": "750 mL",
        "bottler_name": "ABC Distillery",
        "bottler_city": "Frederick",
        "bottler_state": "MD",
        "imported": False,
        "country_of_origin": "",
        "beverage_type": "spirits",
    },
    "test_2": {
        "brand_name": "Malt & Hop Brewery",
        "class_type": "Pale Ale",
        "alcohol_pct": "5",
        "proof": "",
        "net_contents_ml": "24 fl oz",
        "bottler_name": "Malt & Hop Brewery",
        "bottler_city": "Hyattsville",
        "bottler_state": "MD",
        "imported": False,
        "country_of_origin": "",
        "beverage_type": "beer",
    },
    "test_3": {
        "brand_name": "Milo's Ale",
        "class_type": "Ale",
        "alcohol_pct": "5",
        "proof": "",
        "net_contents_ml": "1 qt",
        "bottler_name": "Example Brewing Company",
        "bottler_city": "",
        "bottler_state": "",
        "imported": False,
        "country_of_origin": "",
        "beverage_type": "beer",
    },
    "test_4": {
        "brand_name": "Malt & Hop Brewery",
        "class_type": "Barleywine Ale",
        "alcohol_pct": "9",
        "proof": "",
        "net_contents_ml": "12 fl oz",
        "bottler_name": "Malt & Hop Brewery",
        "bottler_city": "",
        "bottler_state": "",
        "imported": False,
        "country_of_origin": "",
        "beverage_type": "beer",
    },
    "test_5": {
        "brand_name": "Downunder Winery",
        "class_type": "Red Wine",
        "alcohol_pct": "12",
        "proof": "",
        "net_contents_ml": "750 mL",
        "bottler_name": "OZ Imports",
        "bottler_city": "",
        "bottler_state": "",
        "imported": True,
        "country_of_origin": "Australia",
        "beverage_type": "wine",
    },
    "test_6": {
        "brand_name": "ABC Winery",
        "class_type": "Merlot",
        "alcohol_pct": "13",
        "proof": "",
        "net_contents_ml": "750 mL",
        "bottler_name": "ABC Winery",
        "bottler_city": "",
        "bottler_state": "",
        "imported": False,
        "country_of_origin": "",
        "beverage_type": "wine",
    },
    "test_7": {
        "brand_name": "Woodford Reserve",
        "class_type": "Bourbon Whiskey",
        "alcohol_pct": "45.2",
        "proof": "90.4",
        "net_contents_ml": "375 mL",
        "bottler_name": "Woodford Reserve",
        "bottler_city": "",
        "bottler_state": "KY",
        "imported": False,
        "country_of_origin": "",
        "beverage_type": "spirits",
    },
}


def _app_data_from_expected(test_id: str) -> dict:
    e = EXPECTED[test_id]
    return {
        "beverage_type": e["beverage_type"],
        "brand_name": e["brand_name"],
        "class_type": e["class_type"],
        "alcohol_pct": e["alcohol_pct"],
        "proof": e["proof"] or "",
        "net_contents_ml": e["net_contents_ml"],
        "bottler_name": e["bottler_name"],
        "bottler_city": e["bottler_city"],
        "bottler_state": e["bottler_state"],
        "imported": e["imported"],
        "country_of_origin": e["country_of_origin"],
        "sulfites_required": test_id in ("test_5", "test_6"),
        "fd_c_yellow_5_required": False,
        "carmine_required": False,
        "wood_treatment_required": False,
        "age_statement_required": False,
        "neutral_spirits_required": False,
        "aspartame_required": False,
        "appellation_required": False,
        "varietal_required": False,
    }


def _get_extracted_value(extracted: dict, key: str) -> str:
    v = extracted.get(key)
    if isinstance(v, dict):
        return (v.get("value") or "").strip()
    return (v or "").strip()


def main():
    from src.pipeline import run_pipeline

    test_dir = _root / "sample_data"
    if not test_dir.exists():
        test_dir = _root / "assets"
    if not test_dir.exists():
        test_dir = _root

    # Possible filenames per test
    candidates = [
        "test_1.jpg",
        "test_1.png",
        "test_1.jpeg",
        "test_2.jpg",
        "test_2.png",
        "test_2.jpeg",
        "test_3.jpg",
        "test_3.png",
        "test_3.jpeg",
        "test_4.jpg",
        "test_4.png",
        "test_4.jpeg",
        "test_5.jpg",
        "test_5.png",
        "test_5.jpeg",
        "test_6.jpg",
        "test_6.png",
        "test_6.jpeg",
        "test_7.jpg",
        "test_7.png",
        "test_7.jpeg",
    ]

    found = {}
    for c in candidates:
        p = test_dir / c
        if p.exists():
            tid = c.split(".")[0]
            if tid not in found:
                found[tid] = p

    if not found:
        print("No test_1..test_6 images found. Place images in:", test_dir)
        return

    print("=" * 80)
    print("EXTRACTED vs EXPECTED — per test image")
    print("=" * 80)

    fields_compare = [
        ("brand_name", "Brand"),
        ("class_type", "Class/Type"),
        ("alcohol_pct", "ABV %"),
        ("net_contents", "Net contents"),
        ("bottler", "Bottler"),
        ("government_warning", "Gov. warning (len)"),
    ]

    for test_id in sorted(found.keys(), key=lambda x: int(x.split("_")[1])):
        path = found[test_id]
        app_data = _app_data_from_expected(test_id)
        print(f"\n--- {test_id} ({path.name}) ---")
        try:
            result = run_pipeline(str(path), app_data)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        if result.get("error"):
            print(f"  OCR error: {result['error']}")
            continue

        extracted = result.get("extracted", {})
        expected = EXPECTED[test_id]

        for field_key, label in fields_compare:
            if field_key == "government_warning":
                ext_val = _get_extracted_value(extracted, field_key)
                ext_display = f"{len(ext_val)} chars" if ext_val else "(empty)"
                exp_display = "present (full text)"
                match = "OK" if len(ext_val) > 50 else "CHECK"
            else:
                ext_val = _get_extracted_value(extracted, field_key)
                if field_key == "net_contents":
                    app_val = expected.get("net_contents_ml", "")
                elif field_key == "bottler":
                    app_val = expected.get("bottler_name", "")
                elif field_key == "alcohol_pct":
                    app_val = expected.get("alcohol_pct", "")
                elif field_key == "class_type":
                    app_val = expected.get("class_type", "")
                elif field_key == "brand_name":
                    app_val = expected.get("brand_name", "")
                else:
                    app_val = ""
                ext_display = ext_val or "(empty)"
                exp_display = app_val or "(none)"
                match = (
                    "OK"
                    if ext_val
                    and (
                        not app_val
                        or app_val.lower() in ext_val.lower()
                        or ext_val.lower() in app_val.lower()
                    )
                    else "DIFF"
                )
                # Stricter: for brand/class/ABV/net we want meaningful match
                if (
                    field_key
                    in ("brand_name", "class_type", "alcohol_pct", "net_contents")
                    and app_val
                ):
                    if field_key == "alcohol_pct":
                        match = "OK" if ext_val == app_val else "DIFF"
                    elif field_key == "brand_name":
                        match = (
                            "OK"
                            if ext_val
                            and app_val
                            and ext_val.upper().replace(" ", "")
                            == app_val.upper().replace(" ", "")
                            or app_val.upper() in ext_val.upper()
                            else "DIFF"
                        )
                    elif field_key == "net_contents":
                        match = "OK" if ext_val and app_val else "DIFF"

            print(
                f"  {label}: extracted={ext_display!r}  expected={exp_display!r}  [{match}]"
            )

        # Rule summary
        counts = result.get("counts", {})
        print(
            f"  Rules: pass={counts.get('pass', 0)}  review={counts.get('needs_review', 0)}  fail={counts.get('fail', 0)}  -> {result.get('overall_status', '?')}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
