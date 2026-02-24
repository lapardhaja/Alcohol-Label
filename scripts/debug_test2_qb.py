"""
Debug script: Trace how OCR produces "QB" from test_2 image (Government Warning).
Run: python scripts/debug_test2_qb.py
"""
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from PIL import Image
import pytesseract
from pytesseract import Output
import numpy as np

# Try multiple paths for test_2
_candidates = [
    _root / "Test Images" / "test_2.png",
    _root.parent / "Test Images" / "test_2.png",
    Path("Test Images/test_2.png"),
    Path("sample_data/test_2.png"),
]


def find_test2() -> Path | None:
    for p in _candidates:
        if p.resolve().exists():
            return p.resolve()
    return None


def main():
    path = find_test2()
    if not path:
        print("test_2.png not found. Tried:", [str(p) for p in _candidates])
        return
    print(f"Loading: {path}")

    img = Image.open(path).convert("RGB")
    from src.ocr import _preprocess_for_tesseract
    original, sharpened, binary = _preprocess_for_tesseract(img)

    # Run each pass separately to see which produces QB
    for name, im, psm in [
        ("original psm 3", np.array(original), 3),
        ("sharpened psm 6", np.array(sharpened), 6),
        ("binary psm 6", np.array(binary), 6),
    ]:
        print(f"\n=== {name} ===")
        data = pytesseract.image_to_data(im, output_type=Output.DICT, config=f"--psm {psm}")
        n = len(data.get("text", []))
        for i in range(n):
            text = (data.get("text") or [""])[i] or ""
            if not text.strip():
                continue
            conf = int((data.get("conf") or [0])[i] or 0)
            block = int((data.get("block_num") or [0])[i] or 0)
            line = int((data.get("line_num") or [0])[i] or 0)
            if "QB" in text.upper() or ("GOVERNMENT" in text.upper() and block < 10) or ("WARNING" in text.upper() and block < 10):
                left = int((data.get("left") or [0])[i] or 0)
                top = int((data.get("top") or [0])[i] or 0)
                w = int((data.get("width") or [0])[i] or 0)
                h = int((data.get("height") or [0])[i] or 0)
                print(f"  *** block={block} line={line}: '{text}' conf={conf} bbox=({left},{top},{w}x{h})")

    # Our pipeline: show blocks that form the warning
    print("\n=== Our OCR pipeline (all blocks with QB/GOVERNMENT/WARNING) ===")
    from src.ocr import run_ocr
    blocks = run_ocr(img)
    for i, blk in enumerate(blocks):
        t = blk.get("text", "")
        if "QB" in t.upper() or "GOVERNMENT" in t.upper() or "WARNING" in t.upper():
            print(f"  block {i}: '{t}' bbox={blk.get('bbox')} conf={blk.get('confidence')}")

    # Trace: where does "WARNING: QB" come from? Show line structure and gaps
    print("\n=== Raw words on same line as WARNING (sharpened psm 6) ===")
    arr = np.array(sharpened)
    data = pytesseract.image_to_data(arr, output_type=Output.DICT, config="--psm 6")
    n = len(data.get("text", []))
    # Find line_num for WARNING and QB
    warn_line, qb_line = None, None
    for i in range(n):
        t = (data.get("text") or [""])[i] or ""
        if "WARNING" in t.upper() and "QB" not in t:
            warn_line = (int(data["block_num"][i]), int(data["line_num"][i]))
        if "QB" in t.upper():
            qb_line = (int(data["block_num"][i]), int(data["line_num"][i]))
    print(f"  WARNING line: block,line = {warn_line}, QB line: {qb_line}")
    # Show all words on line (1,7) with gaps
    indices = [i for i in range(n) if (int(data["block_num"][i]), int(data["line_num"][i])) == (1, 7)]
    print(f"  Line (1,7) has {len(indices)} words:")
    for j, i in enumerate(indices):
        t = (data.get("text") or [""])[i] or ""
        left = int(data["left"][i])
        w = int(data["width"][i])
        end = left + w
        gap = ""
        if j > 0:
            prev_end = int(data["left"][indices[j-1]]) + int(data["width"][indices[j-1]])
            gap = f" gap_from_prev={left - prev_end}"
        print(f"    [{j}] '{t}' left={left} end={end}{gap}")


if __name__ == "__main__":
    main()
