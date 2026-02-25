"""Tests for batch CSV parsing and column normalization."""
import sys
from pathlib import Path

import pandas as pd
import pytest

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.app import _normalize_csv_columns, _row_to_app_data


@pytest.fixture
def sample_csv_path():
    return _root / "sample_data" / "batch_example.csv"


def test_sample_csv_exists(sample_csv_path):
    assert sample_csv_path.exists(), f"batch_example.csv not found at {sample_csv_path}"


def test_sample_csv_has_required_columns(sample_csv_path):
    df = pd.read_csv(sample_csv_path)
    required = {"label_id", "brand_name", "class_type", "alcohol_pct", "net_contents_ml",
                "bottler_name", "imported", "beverage_type"}
    assert required <= set(df.columns)


def test_sample_csv_has_7_rows(sample_csv_path):
    df = pd.read_csv(sample_csv_path)
    assert len(df) == 7


def test_normalize_csv_columns_standard():
    df = pd.DataFrame({"label_id": ["a"], "brand_name": ["b"], "class_type": ["c"]})
    result = _normalize_csv_columns(df)
    assert "label_id" in result.columns
    assert "brand_name" in result.columns


def test_normalize_csv_columns_spaces():
    df = pd.DataFrame({"Label ID": ["a"], "Brand Name": ["b"], "Class Type": ["c"]})
    result = _normalize_csv_columns(df)
    assert "label_id" in result.columns
    assert "brand_name" in result.columns


def test_row_to_app_data_spirits():
    row = pd.Series({
        "label_id": "test_1",
        "brand_name": "ABC Distillery",
        "class_type": "Bourbon",
        "alcohol_pct": "45",
        "proof": "90",
        "net_contents_ml": "750 mL",
        "bottler_name": "ABC Distillery",
        "bottler_city": "Frederick",
        "bottler_state": "MD",
        "imported": "false",
        "country_of_origin": "",
        "beverage_type": "Distilled Spirits",
    })
    app_data = _row_to_app_data(row)
    assert app_data["beverage_type"] == "spirits"
    assert app_data["brand_name"] == "ABC Distillery"
    assert app_data["imported"] is False


def test_row_to_app_data_beer():
    row = pd.Series({
        "label_id": "test_2",
        "brand_name": "Malt & Hop",
        "class_type": "Pale Ale",
        "alcohol_pct": "5",
        "net_contents_ml": "12 fl oz",
        "bottler_name": "Malt & Hop",
        "imported": "false",
        "beverage_type": "Beer / Malt Beverage",
    })
    app_data = _row_to_app_data(row)
    assert app_data["beverage_type"] == "beer"


def test_row_to_app_data_wine_imported():
    row = pd.Series({
        "label_id": "test_5",
        "brand_name": "Downunder",
        "class_type": "Red Wine",
        "alcohol_pct": "12",
        "net_contents_ml": "750 mL",
        "bottler_name": "OZ Imports",
        "imported": "true",
        "country_of_origin": "Australia",
        "beverage_type": "Wine",
        "sulfites_required": "true",
    })
    app_data = _row_to_app_data(row)
    assert app_data["beverage_type"] == "wine"
    assert app_data["imported"] is True
    assert app_data["sulfites_required"] is True


def test_full_csv_round_trip(sample_csv_path):
    """Parse CSV, normalize columns, convert each row to app_data."""
    df = pd.read_csv(sample_csv_path)
    df = _normalize_csv_columns(df)
    for _, row in df.iterrows():
        app_data = _row_to_app_data(row)
        assert "beverage_type" in app_data
        assert "brand_name" in app_data
        assert app_data["beverage_type"] in ("spirits", "wine", "beer")
