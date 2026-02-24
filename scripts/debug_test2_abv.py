"""
Debug script: Trace where ABV extraction gets its value from test_2.
Run: python scripts/debug_test2_abv.py
"""
from pathlib import Path
import sys
import re

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from PIL import Image
import numpy as np

_candidates = [
    _root / "Test Images" / "test_2.png",
    _root.parent / "Test Images" / "test_2.png",
]


def find_test2() -> Path | None:
    for p in _candidates:
        if p.resolve().exists():
            return p.resolve()
    return None


# Copy regexes from extraction
_ABV_STRICT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*(?:Alc\.?/?Vol\.?|ALC/?VOL|Alcohol by volume)?",
    re.I,
)
_ABV_QUAL_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*(?:Alc\.?/?Vol\.?|ALC/?VOL|Alcohol by volume)?",
    re.I,
)
_ABV_LOOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%", re.I)


def main():
    path = find_test2()
    if not path:
        print("test_2.png not found")
        return
    print(f"Loading: {path}")

    img = Image.open(path).convert("RGB")
    from src.ocr import run_ocr
    from src.extraction import extract_fields

    blocks = run_ocr(img)
    extracted = extract_fields(blocks, app_data={"alcohol_pct": "5"})

    abv_val = extracted.get("alcohol_pct", {}).get("value", "")
    print(f"\n=== Extracted alcohol_pct: '{abv_val}' ===")

    print("\n=== All blocks containing % or digits (potential ABV sources) ===")
    for i, b in enumerate(blocks):
        t = b.get("text", "")
        if "%" in t or re.search(r"\d+(?:\.\d+)?\s*%", t):
            print(f"  block {i}: '{t}' bbox={b.get('bbox')} conf={b.get('confidence')}")

    print("\n=== Blocks matching ABV regexes ===")
    for i, b in enumerate(blocks):
        t = b.get("text", "")
        for name, pat in [("STRICT", _ABV_STRICT_RE), ("QUAL", _ABV_QUAL_RE), ("LOOSE", _ABV_LOOSE_RE)]:
            m = pat.search(t)
            if m:
                print(f"  block {i} {name}: '{t}' -> captured '{m.group(1)}'")

    print("\n=== Combined text (first 500 chars) ===")
    combined = " ".join(b.get("text", "") for b in blocks)
    print(f"  {combined[:500]}...")

    # Trace _extract_abv_proof logic
    print("\n=== Simulating extraction candidates ===")
    from src.extraction import _extract_abv_proof, _ABV_STRICT_RE, _ABV_QUAL_RE

    def _abv_plausible(pct_str):
        try:
            v = float(pct_str.strip().rstrip("%"))
            return 3.0 <= v <= 75.0
        except (TypeError, ValueError):
            return False

    candidates = []
    for b in blocks:
        t = b.get("text", "")
        m = _ABV_STRICT_RE.search(t) or _ABV_QUAL_RE.search(t)
        if m:
            pct = m.group(1)
            score = 1.0 if _abv_plausible(pct) else 0.5
            if "ALC" in t.upper() or "VOL" in t.upper() or "PROOF" in t.upper():
                score += 1.0
            candidates.append((pct, b.get("text"), score))
            print(f"  candidate: pct='{pct}' score={score} from block '{t[:60]}...'")

    if candidates:
        best = max(candidates, key=lambda x: x[2])
        print(f"\n  -> Best would be: '{best[0]}' (score={best[2]})")


if __name__ == "__main__":
    main()
