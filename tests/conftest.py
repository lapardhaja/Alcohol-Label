"""Pytest fixtures: app_data, ocr_blocks, extracted fields."""
import sys
from pathlib import Path

import pytest
from PIL import Image

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# ---------------------------------------------------------------------------
# OLD TOM DISTILLERY — classic single-line OCR blocks
# ---------------------------------------------------------------------------

@pytest.fixture
def app_data():
    return {
        "brand_name": "OLD TOM DISTILLERY",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_pct": "45",
        "proof": "90",
        "net_contents_ml": "750",
        "bottler_name": "Old Tom Distillery",
        "bottler_city": "Louisville",
        "bottler_state": "KY",
        "imported": False,
        "country_of_origin": "",
        "sulfites_required": False,
        "fd_c_yellow_5_required": False,
        "carmine_required": False,
        "wood_treatment_required": False,
        "age_statement_required": False,
        "neutral_spirits_required": False,
    }


@pytest.fixture
def app_data_imported(app_data):
    d = dict(app_data)
    d["imported"] = True
    d["country_of_origin"] = "Scotland"
    return d


@pytest.fixture
def ocr_blocks():
    return [
        {"text": "OLD TOM DISTILLERY", "bbox": [10, 10, 200, 40], "confidence": 90},
        {"text": "Kentucky Straight Bourbon Whiskey", "bbox": [10, 50, 300, 80], "confidence": 88},
        {"text": "45% Alc./Vol. (90 Proof)", "bbox": [10, 90, 180, 120], "confidence": 92},
        {"text": "750 mL", "bbox": [10, 130, 80, 160], "confidence": 95},
        {"text": "GOVERNMENT WARNING:", "bbox": [10, 200, 220, 230], "confidence": 91},
        {"text": "According to the Surgeon General...", "bbox": [10, 235, 400, 265], "confidence": 85},
        {"text": "Bottled by Old Tom Distillery, Louisville, KY", "bbox": [10, 260, 350, 290], "confidence": 85},
    ]


# ---------------------------------------------------------------------------
# ABC DISTILLERY — multi-block layout (two-panel label)
# ---------------------------------------------------------------------------

@pytest.fixture
def app_data_abc():
    return {
        "brand_name": "ABC",
        "class_type": "Straight Rye Whisky",
        "alcohol_pct": "45",
        "proof": "",
        "net_contents_ml": "750",
        "bottler_name": "ABC Distillery",
        "bottler_city": "Frederick",
        "bottler_state": "MD",
        "imported": False,
        "country_of_origin": "",
        "sulfites_required": False,
        "fd_c_yellow_5_required": False,
        "carmine_required": False,
        "wood_treatment_required": False,
        "age_statement_required": False,
        "neutral_spirits_required": False,
    }


@pytest.fixture
def ocr_blocks_abc():
    """Simulates OCR output from the two-panel ABC label test image."""
    return [
        {"text": "DISTILLED AND BOTTLED BY:", "bbox": [20, 20, 250, 45], "confidence": 88},
        {"text": "ABC DISTILLERY", "bbox": [40, 50, 210, 85], "confidence": 92},
        {"text": "FREDERICK, MD", "bbox": [50, 90, 190, 110], "confidence": 90},
        {"text": "ABC", "bbox": [60, 130, 190, 200], "confidence": 95},
        {"text": "SINGLE BARREL", "bbox": [40, 210, 210, 240], "confidence": 89},
        {"text": "STRAIGHT RYE", "bbox": [30, 250, 220, 285], "confidence": 91},
        {"text": "WHISKY", "bbox": [50, 290, 200, 325], "confidence": 93},
        {"text": "750 ML", "bbox": [30, 350, 120, 375], "confidence": 94},
        {"text": "45% ALC/VOL", "bbox": [30, 380, 150, 405], "confidence": 92},
        # Right panel (back label)
        {"text": "ABC", "bbox": [320, 30, 420, 70], "confidence": 90},
        {"text": "STRAIGHT RYE WHISKY", "bbox": [300, 80, 470, 105], "confidence": 88},
        {"text": "GOVERNMENT WARNING:", "bbox": [300, 120, 480, 145], "confidence": 91},
        {"text": "(1) According to the Surgeon General,", "bbox": [300, 150, 490, 170], "confidence": 86},
        {"text": "women should not drink alcoholic beverages", "bbox": [300, 175, 490, 195], "confidence": 85},
        {"text": "during pregnancy because of the risk of", "bbox": [300, 200, 490, 220], "confidence": 84},
        {"text": "birth defects.", "bbox": [300, 225, 390, 245], "confidence": 87},
        {"text": "(2) Consumption of alcoholic beverages", "bbox": [300, 250, 490, 270], "confidence": 85},
        {"text": "impairs your ability to drive a car or operate", "bbox": [300, 275, 490, 295], "confidence": 84},
        {"text": "machinery, and may cause health problems.", "bbox": [300, 300, 490, 320], "confidence": 86},
    ]


# ---------------------------------------------------------------------------
# Minimal / edge-case fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ocr_blocks_split_abv():
    """ABV split across separate blocks: '45' in one, '%' in another."""
    return [
        {"text": "BRAND X", "bbox": [10, 10, 150, 50], "confidence": 90},
        {"text": "Vodka", "bbox": [10, 60, 100, 85], "confidence": 88},
        {"text": "45", "bbox": [10, 100, 40, 120], "confidence": 91},
        {"text": "% ALC/VOL", "bbox": [45, 100, 140, 120], "confidence": 90},
        {"text": "750", "bbox": [10, 130, 50, 150], "confidence": 93},
        {"text": "mL", "bbox": [55, 130, 80, 150], "confidence": 92},
        {"text": "GOVERNMENT WARNING:", "bbox": [10, 180, 200, 200], "confidence": 89},
        {"text": "According to the Surgeon General...", "bbox": [10, 210, 350, 235], "confidence": 82},
    ]


@pytest.fixture
def empty_ocr_blocks():
    return []


@pytest.fixture
def white_image():
    return Image.new("RGB", (400, 300), color="white")
