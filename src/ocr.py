"""
OCR module: Tesseract only.
Returns deduplicated text blocks with bbox and confidence.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


class OcrUnavailableError(Exception):
    """No OCR engine is available."""


_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tesseract path detection
# ---------------------------------------------------------------------------

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

_SHARPEN_KERNEL = np.array([[-1, -1, -1],
                            [-1,  9, -1],
                            [-1, -1, -1]], dtype=np.float32)


def _resize(img: Image.Image, max_dim: int = 2000, min_dim: int = 1000) -> Image.Image:
    """Resize image: upscale small images so text is readable, cap large ones."""
    w, h = img.size
    if max(w, h) < min_dim:
        scale = min_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    elif max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive thresholding for clean black-on-white text."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )


def _preprocess_for_tesseract(img: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
    """
    Return (resized_original, enhanced+sharpened, binarized) for Tesseract multi-pass.
    No GaussianBlur -- OCR needs sharp text edges.
    """
    img = _resize(img)
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = _deskew(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sharpened = cv2.filter2D(enhanced, -1, _SHARPEN_KERNEL)
    binary = _binarize(enhanced)
    return img, Image.fromarray(sharpened), Image.fromarray(binary)


def _deskew(gray: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Correct slight rotation using edge detection."""
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
# Tesseract backend
# ---------------------------------------------------------------------------

def _data_to_blocks(data: Any) -> list[dict[str, Any]]:
    """
    Convert pytesseract image_to_data dict to line-level blocks.
    Groups words by (block_num, par_num, line_num), then splits lines
    with large horizontal gaps to avoid cross-panel merging.
    """
    from collections import defaultdict

    n = len(data.get("text", []))
    has_hierarchy = all(k in data for k in ("block_num", "par_num", "line_num"))

    if not has_hierarchy:
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

    blocks: list[dict[str, Any]] = []
    for key in sorted(lines.keys()):
        indices = lines[key]
        split_blocks = _split_line_by_gaps(data, indices)
        blocks.extend(split_blocks)
    return blocks


def _split_line_by_gaps(data: Any, indices: list[int]) -> list[dict[str, Any]]:
    """
    Split a Tesseract line into multiple blocks when there are large horizontal
    gaps (> 2x median word spacing) between words -- prevents cross-panel merging.
    """
    if len(indices) <= 1:
        words = [data["text"][i].strip() for i in indices]
        x1 = int(data["left"][indices[0]])
        y1 = int(data["top"][indices[0]])
        x2 = x1 + int(data["width"][indices[0]])
        y2 = y1 + int(data["height"][indices[0]])
        conf = float(data["conf"][indices[0]])
        return [{"text": " ".join(words), "bbox": [x1, y1, x2, y2], "confidence": conf}]

    word_ends = []
    word_starts = []
    for i in indices:
        word_starts.append(int(data["left"][i]))
        word_ends.append(int(data["left"][i]) + int(data["width"][i]))

    gaps = []
    for j in range(1, len(indices)):
        gap = word_starts[j] - word_ends[j - 1]
        gaps.append(gap)

    if not gaps:
        return _build_block(data, indices)

    median_gap = float(np.median(gaps)) if gaps else 0
    gap_threshold = max(median_gap * 2.5, 40)

    split_points = [0]
    for j, gap in enumerate(gaps):
        if gap > gap_threshold:
            split_points.append(j + 1)
    split_points.append(len(indices))

    blocks: list[dict[str, Any]] = []
    for s in range(len(split_points) - 1):
        chunk = indices[split_points[s]:split_points[s + 1]]
        if chunk:
            blocks.extend(_build_block(data, chunk))
    return blocks


def _build_block(data: Any, indices: list[int]) -> list[dict[str, Any]]:
    words = [data["text"][i].strip() for i in indices]
    x1 = min(int(data["left"][i]) for i in indices)
    y1 = min(int(data["top"][i]) for i in indices)
    x2 = max(int(data["left"][i]) + int(data["width"][i]) for i in indices)
    y2 = max(int(data["top"][i]) + int(data["height"][i]) for i in indices)
    avg_conf = sum(float(data["conf"][i]) for i in indices) / len(indices)
    return [{"text": " ".join(words), "bbox": [x1, y1, x2, y2], "confidence": avg_conf}]


def _bbox_iou(a: list[int], b: list[int]) -> float:
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


def get_preprocessing_preview(img: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
    """
    Return (resized_original, sharpened, binarized) — the same three images
    that are passed to Tesseract in _run_tesseract_ocr. For debugging/UI preview.
    """
    return _preprocess_for_tesseract(img)


def _run_tesseract_ocr(img: Image.Image) -> list[dict[str, Any]]:
    """Tesseract multi-pass OCR with improved preprocessing."""
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        raise OcrUnavailableError(
            "pytesseract is not installed. Install Tesseract and pytesseract. See README."
        )

    _ensure_tesseract_cmd()
    original, sharpened, binary = _preprocess_for_tesseract(img)

    def _run_pass(image: Image.Image | np.ndarray, psm: int) -> list[dict[str, Any]]:
        arr = np.array(image) if isinstance(image, Image.Image) else image
        try:
            data = pytesseract.image_to_data(arr, output_type=Output.DICT, config=f"--psm {psm}", lang="eng")
            return _data_to_blocks(data)
        except pytesseract.TesseractNotFoundError:
            raise OcrUnavailableError(
                "Tesseract OCR is not installed or not on your PATH. "
                "Download from https://github.com/UB-Mannheim/tesseract/wiki (Windows), "
                "then add its folder to PATH."
            )
        except Exception:
            return []

    blocks = _run_pass(original, psm=3)
    blocks.extend(_run_pass(sharpened, psm=6))
    blocks.extend(_run_pass(binary, psm=6))

    return _deduplicate_blocks(blocks)


# ---------------------------------------------------------------------------
# Main entry point: Tesseract only
# ---------------------------------------------------------------------------

def run_ocr(img: Image.Image) -> list[dict[str, Any]]:
    """
    Run Tesseract multi-pass OCR with OpenCV preprocessing.
    Returns list of {text, bbox, confidence} blocks.
    """
    return _run_tesseract_ocr(img)
