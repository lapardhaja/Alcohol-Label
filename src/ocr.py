"""
Image preprocessing + multi-pass Tesseract OCR. Returns deduplicated text blocks
with bbox and confidence. Raises OcrUnavailableError when Tesseract is missing.
Auto-detects Tesseract in common install locations.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


class OcrUnavailableError(Exception):
    """Tesseract OCR is not installed or not on PATH."""


_TESSERACT_CANDIDATES: list[str] = []
if sys.platform == "win32":
    _TESSERACT_CANDIDATES = [
        os.environ.get("ProgramFiles", "C:\\Program Files") + "\\Tesseract-OCR\\tesseract.exe",
        os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)") + "\\Tesseract-OCR\\tesseract.exe",
        os.path.expandvars("%LOCALAPPDATA%\\Tesseract-OCR\\tesseract.exe"),
    ]
elif sys.platform == "darwin":
    _TESSERACT_CANDIDATES = ["/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"]
else:
    _TESSERACT_CANDIDATES = ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]


def _ensure_tesseract_cmd() -> None:
    """Set pytesseract.tesseract_cmd to a working path if not already in PATH."""
    try:
        import pytesseract
    except ImportError:
        return
    try:
        pytesseract.get_tesseract_version()
        return
    except Exception:
        pass
    for path in _TESSERACT_CANDIDATES:
        if path and Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = path
            return


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _resize(img: Image.Image, max_dim: int = 2000) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def _preprocess(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Return (resized_original, enhanced_grayscale) for multi-pass OCR."""
    img = _resize(img)
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return img, Image.fromarray(enhanced)


def _deskew(gray: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Correct slight rotation (up to max_angle degrees) using edge detection."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return gray
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= max_angle:
            angles.append(angle)
    if not angles:
        return gray
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return gray
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Block-level helpers
# ---------------------------------------------------------------------------

def _data_to_blocks(data: Any) -> list[dict[str, Any]]:
    """
    Convert pytesseract image_to_data dict to list of **line-level** block dicts.
    Groups words by (block_num, par_num, line_num) so text reads coherently
    even on multi-column / multi-panel label images.
    """
    from collections import defaultdict

    n = len(data.get("text", []))
    has_hierarchy = all(k in data for k in ("block_num", "par_num", "line_num"))

    if not has_hierarchy:
        # Fallback: word-level blocks (e.g. from mocked data without hierarchy)
        blocks: list[dict[str, Any]] = []
        for i in range(n):
            text = (data.get("text") or [])[i] or ""
            if not text.strip():
                continue
            conf = float((data.get("conf") or [0])[i] or 0)
            if conf < 0:
                continue
            x = int((data.get("left") or [0])[i])
            y = int((data.get("top") or [0])[i])
            w = int((data.get("width") or [0])[i])
            ht = int((data.get("height") or [0])[i])
            blocks.append({"text": text.strip(), "bbox": [x, y, x + w, y + ht], "confidence": conf})
        return blocks

    lines: dict[tuple, list[int]] = defaultdict(list)
    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf = float(data["conf"][i] or 0)
        if not text or conf < 0:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        lines[key].append(i)

    blocks = []
    for key in sorted(lines.keys()):
        indices = lines[key]
        words = [data["text"][i].strip() for i in indices]
        x1 = min(int(data["left"][i]) for i in indices)
        y1 = min(int(data["top"][i]) for i in indices)
        x2 = max(int(data["left"][i]) + int(data["width"][i]) for i in indices)
        y2 = max(int(data["top"][i]) + int(data["height"][i]) for i in indices)
        avg_conf = sum(float(data["conf"][i]) for i in indices) / len(indices)
        blocks.append({
            "text": " ".join(words),
            "bbox": [x1, y1, x2, y2],
            "confidence": avg_conf,
        })
    return blocks


def _bbox_iou(a: list[int], b: list[int]) -> float:
    """Intersection-over-union for two [x1,y1,x2,y2] boxes."""
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(area_a, area_b)


def _fuzzy_sim(a: str, b: str) -> float:
    """Quick fuzzy similarity (0-1) for dedup. Uses difflib (no external deps)."""
    a_n, b_n = a.lower().strip(), b.lower().strip()
    if a_n == b_n:
        return 1.0
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a_n, b_n) / 100.0
    except ImportError:
        import difflib
        return difflib.SequenceMatcher(None, a_n, b_n).ratio()


def _deduplicate_blocks(blocks: list[dict[str, Any]], iou_thresh: float = 0.4) -> list[dict[str, Any]]:
    """Remove near-duplicate blocks (from multi-pass), keeping higher confidence.
    Uses fuzzy string similarity when bboxes overlap to catch near-identical OCR variants
    like 'ALC.VOL.' vs 'ALC./VOL.' that differ by one char."""
    blocks = sorted(blocks, key=lambda b: b["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    for blk in blocks:
        duplicate = False
        for existing in kept:
            iou = _bbox_iou(blk["bbox"], existing["bbox"])
            if iou > iou_thresh:
                if _fuzzy_sim(blk["text"], existing["text"]) > 0.85:
                    duplicate = True
                    break
                if len(blk["text"]) <= len(existing["text"]):
                    duplicate = True
                    break
        if not duplicate:
            kept.append(blk)
    return kept


# ---------------------------------------------------------------------------
# Main OCR entry point
# ---------------------------------------------------------------------------

def run_ocr(img: Image.Image) -> list[dict[str, Any]]:
    """
    Multi-pass Tesseract OCR on label image.
    Pass 1: PSM 3 (auto page segmentation) on original — handles multi-column/region labels.
    Pass 2: PSM 6 (single block) on CLAHE-enhanced grayscale — recovers small/low-contrast text.
    Pass 3: PSM 6 on deskewed enhanced image — handles slight rotations.
    Results are deduplicated by bbox overlap.
    Raises OcrUnavailableError if Tesseract is not installed.
    """
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        raise OcrUnavailableError(
            "pytesseract is not installed. Install Tesseract and pytesseract. See README."
        )

    _ensure_tesseract_cmd()
    original, enhanced = _preprocess(img)

    def _run_pass(image: Image.Image | np.ndarray, psm: int) -> list[dict[str, Any]]:
        arr = np.array(image) if isinstance(image, Image.Image) else image
        try:
            data = pytesseract.image_to_data(arr, output_type=Output.DICT, config=f"--psm {psm}")
            return _data_to_blocks(data)
        except pytesseract.TesseractNotFoundError:
            raise OcrUnavailableError(
                "Tesseract OCR is not installed or not on your PATH. "
                "Download from https://github.com/UB-Mannheim/tesseract/wiki (Windows), "
                "then add its folder to PATH."
            )
        except Exception:
            return []

    # Pass 1: auto page segmentation on original (best for multi-region labels)
    blocks = _run_pass(original, psm=3)

    # Pass 2: single-block on enhanced grayscale (recovers small/faint text)
    blocks.extend(_run_pass(enhanced, psm=6))

    # Pass 3: deskew + single-block (handles slight rotation)
    enhanced_arr = np.array(enhanced)
    deskewed = _deskew(enhanced_arr)
    if deskewed is not enhanced_arr:
        blocks.extend(_run_pass(deskewed, psm=6))

    return _deduplicate_blocks(blocks)
