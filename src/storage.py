"""
Local JSON storage for application lists. No external APIs, works offline.
"""
import base64
import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DATA_FILE = _DATA_DIR / "applications.json"


def _entry_to_json(entry: dict) -> dict:
    """Convert entry for JSON (base64-encode image_bytes)."""
    out = dict(entry)
    if "image_bytes" in out:
        out["image_bytes"] = base64.b64encode(out["image_bytes"] or b"").decode("ascii")
    return out


def _entry_from_json(data: dict) -> dict:
    """Restore entry from JSON (base64-decode image_bytes)."""
    out = dict(data)
    if "image_bytes" in out and isinstance(out["image_bytes"], str):
        out["image_bytes"] = base64.b64decode(out["image_bytes"])
    return out


def load_applications() -> dict[str, list]:
    """Load under_review, approved, rejected from local JSON file."""
    default = {
        "applications_under_review": [],
        "applications_approved": [],
        "applications_rejected": [],
    }
    if not _DATA_FILE.exists():
        return default
    try:
        raw = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
        return {
            "applications_under_review": [_entry_from_json(e) for e in raw.get("under_review", [])],
            "applications_approved": [_entry_from_json(e) for e in raw.get("approved", [])],
            "applications_rejected": [_entry_from_json(e) for e in raw.get("rejected", [])],
        }
    except (json.JSONDecodeError, OSError):
        return default


def save_applications(
    under_review: list,
    approved: list,
    rejected: list,
) -> None:
    """Persist all lists to local JSON file."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "under_review": [_entry_to_json(e) for e in under_review],
        "approved": [_entry_to_json(e) for e in approved],
        "rejected": [_entry_to_json(e) for e in rejected],
    }
    _DATA_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
