"""
Debug: Trace OCR outputs and how warning text is combined.
Run: python scripts/debug_warning_ocr.py [image_path]
Shows: raw OCR by pass, dedup result, extraction candidates, overlap trim, final value.
"""
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

_candidates = [
    _root / "Test Images" / "test_2.png",
    _root / "test_2.png",
    _root.parent / "Test Images" / "test_2.png",
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        for p in _candidates:
            if p.resolve().exists():
                path = p.resolve()
                break
    if not path or not Path(path).exists():
        print("Usage: python scripts/debug_warning_ocr.py [image_path]")
        print("No image found. Tried:", [str(p) for p in _candidates])
        return

    from PIL import Image
    from src.ocr import run_ocr_debug
    from src.extraction import debug_warning_extraction

    img = Image.open(path).convert("RGB")
    print(f"=== Image: {path} ===\n")

    # 1. OCR by pass
    deduped, by_pass = run_ocr_debug(img)
    print("--- OCR BY PASS (before dedup) ---")
    for pass_name, blks in by_pass:
        print(f"\n  [{pass_name}] {len(blks)} blocks:")
        for b in blks:
            t = (b.get("text") or "").strip()
            if not t:
                continue
            box = b.get("bbox", [0, 0, 0, 0])
            # Only show blocks that might be warning-related
            if any(
                kw in t.upper()
                for kw in (
                    "GOVERNMENT",
                    "WARNING",
                    "ALCOHOLIC",
                    "CONSUMPTION",
                    "IMPAIRS",
                    "DRIVE",
                    "OPERATE",
                    "MACHINERY",
                    "HEALTH",
                    "PROBLEMS",
                )
            ):
                print(f"    [{box[0]},{box[1]}] {t[:70]}")

    print("\n--- DEDUPLICATED BLOCKS (final OCR) ---")
    for b in deduped:
        t = (b.get("text") or "").strip()
        if not t:
            continue
        box = b.get("bbox", [0, 0, 0, 0])
        if any(
            kw in t.upper()
            for kw in (
                "GOVERNMENT",
                "WARNING",
                "ALCOHOLIC",
                "CONSUMPTION",
                "IMPAIRS",
                "DRIVE",
                "OPERATE",
                "MACHINERY",
                "HEALTH",
                "PROBLEMS",
            )
        ):
            print(f"  [{box[0]},{box[1]}] {t[:70]}")

    # 2. Extraction trace
    ref = (
        "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink "
        "alcoholic beverages during pregnancy because of the risk of birth defects. "
        "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
        "operate machinery, and may cause health problems."
    )
    result = debug_warning_extraction(deduped, ref)

    print("\n--- EXTRACTION TRACE (blocks with warning-related content) ---")
    for row in result.get("trace", []):
        t = row.get("text", "")
        if not t:
            continue
        # Show if has warning keywords or was candidate
        has_kw = any(
            kw in t.upper()
            for kw in (
                "GOVERNMENT",
                "WARNING",
                "ALCOHOLIC",
                "CONSUMPTION",
                "IMPAIRS",
                "DRIVE",
                "OPERATE",
                "MACHINERY",
                "HEALTH",
                "PROBLEMS",
                "BIRTH",
                "DEFECTS",
            )
        )
        if has_kw or "candidate" in row.get("reason", ""):
            print(f"  [{row.get('idx')}] {t[:60]} | {row.get('reason')}")

    print("\nCandidates (included):")
    for c in result.get("candidates", []):
        print(f"  - {c}")

    print("\nOrdered (after sort by reference position):")
    for o in result.get("ordered", []):
        print(f"  - {o}")

    print("\nCombine step (overlap trim, added, skipped):")
    for ct in result.get("combine_trace", []):
        print(f"  block: {ct.get('block', '')[:55]}")
        if ct.get("overlap_trimmed"):
            print(f"    overlap_trimmed: '{ct['overlap_trimmed']}'")
        if ct.get("action") == "skipped":
            print(f"    -> SKIPPED")
        else:
            print(f"    -> added: {ct.get('added', '')[:55]}")

    print("\nFinal parts:")
    for p in result.get("parts", []):
        print(f"  - {p}")

    print("\n--- FINAL EXTRACTED VALUE ---")
    print(result.get("value", ""))


if __name__ == "__main__":
    main()
