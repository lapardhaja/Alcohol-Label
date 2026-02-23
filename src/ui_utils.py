"""
UI helpers: draw bbox on image for "Show on label".
"""
from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw


def draw_bbox_on_image(img: Image.Image, bbox: list[int] | None, color: str = "red", width: int = 4) -> Image.Image:
    """Return a copy of img with bbox [x1,y1,x2,y2] drawn. If bbox is None, return copy unchanged."""
    out = img.copy()
    if not bbox or len(bbox) < 4:
        return out
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    for w in range(width):
        draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=color, width=1)
    return out
