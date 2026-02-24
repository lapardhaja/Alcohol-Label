"""
Debug: Trace OCR gaps for government warning — find why blocks get combined.
Run: python scripts/debug_gov_warning_gaps.py
Output: gaps between words, split points, and block boundaries.
"""
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import numpy as np
from PIL import Image

# Find test image
_candidates = [
    _root / "Test Images" / "test_2.png",
    _root / "test_2.png",
    _root.parent / "Test Images" / "test_2.png",
    Path(__file__).resolve().parent.parent.parent / "Test Images" / "test_2.png",
]


def main():
    path = None
    for p in _candidates:
        if p.resolve().exists():
            path = p.resolve()
            break
    if not path:
        print("No test_2.png found")
        return

    print(f"Loading: {path}")
    img = Image.open(path).convert("RGB")

    from src.ocr import run_ocr
    import pytesseract

    # Get raw Tesseract data
    arr = np.array(img)
    data = pytesseract.image_to_data(arr, output_type=pytesseract.Output.DICT)

    # Find lines containing GOVERNMENT WARNING
    n = len(data.get("text", []))
    for i in range(n):
        t = (data.get("text") or [])[i] or ""
        if "GOVERNMENT" in t.upper() or "WARNING" in t.upper():
            block_num = int((data.get("block_num") or [0])[i])
            par_num = int((data.get("par_num") or [0])[i])
            line_num = int((data.get("line_num") or [0])[i])
            print(f"\n=== Word '{t}' at block={block_num} par={par_num} line={line_num} ===")
            break
    else:
        print("GOVERNMENT WARNING not found in OCR")
        return

    # Collect all words in same (block, par, line) as gov warning
    from collections import defaultdict
    lines = defaultdict(list)
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        lines[key].append(i)

    # Show lines that have gov warning keywords
    for key in sorted(lines.keys()):
        indices = lines[key]
        words = [(data["text"][i], int(data["left"][i]), int(data["width"][i])) for i in indices]
        has_gov = any("GOVERNMENT" in w[0].upper() or "WARNING" in w[0].upper() for w in words)
        has_serving = any("SERVING" in w[0].upper() or "SERYING" in w[0].upper() for w in words)
        if has_gov or has_serving:
            # Sort by x
            words_sorted = sorted(words, key=lambda x: x[1])
            print(f"\n--- Line {key} (gov={has_gov} serving={has_serving}) ---")
            print("Words (x, width):", [(w[0][:20], w[1], w[2]) for w in words_sorted])
            # Compute gaps
            for j in range(1, len(words_sorted)):
                prev_end = words_sorted[j-1][1] + words_sorted[j-1][2]
                curr_start = words_sorted[j][1]
                gap = curr_start - prev_end
                print(f"  GAP {gap}px between '{words_sorted[j-1][0][:15]}' and '{words_sorted[j][0][:15]}'")
            # Would we split? (abs_split=60, gap_threshold)
            word_ends = [w[1] + w[2] for w in words_sorted]
            word_starts = [w[1] for w in words_sorted]
            gaps = [word_starts[j] - word_ends[j-1] for j in range(1, len(words_sorted))]
            median_gap = float(np.median(gaps)) if gaps else 0
            gap_threshold = max(median_gap * 2.5, 40)
            abs_split = 60
            print(f"  median_gap={median_gap:.0f} gap_threshold={max(gap_threshold, abs_split)}")
            for j, g in enumerate(gaps):
                split = "SPLIT" if g > gap_threshold or g > abs_split else "merge"
                print(f"  gap[{j}]={g} -> {split}")

    print("\n=== Running full OCR + extraction ===")
    blocks = run_ocr(img)
    from src.extraction import extract_fields
    extracted = extract_fields(blocks, warning_reference="GOVERNMENT WARNING: (1) According to the Surgeon General...")
    val = extracted.get("government_warning", {}).get("value", "")
    print(f"Extracted ({len(val)} chars): {val[:200]}...")
