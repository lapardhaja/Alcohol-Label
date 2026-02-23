"""
Load rules config and run all rule categories. Return list of { rule_id, category, status, message, bbox_ref }.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def _load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "rules.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_rules(extracted: dict[str, Any], app_data: dict[str, Any]) -> list[dict[str, Any]]:
    config = _load_config()
    results: list[dict[str, Any]] = []
    # Identity
    results.extend(_rules_identity(extracted, app_data, config))
    # Alcohol & contents
    results.extend(_rules_alcohol_contents(extracted, app_data, config))
    # Warning
    results.extend(_rules_warning(extracted, app_data, config))
    # Origin
    results.extend(_rules_origin(extracted, app_data, config))
    # Other (conditional)
    results.extend(_rules_other(extracted, app_data, config))
    return results


def _norm(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _net_contents_to_ml(s: str) -> int | None:
    """Parse TTB-style net contents (e.g. '750 mL', '1 L') to milliliters. Returns None if unparseable."""
    s = _norm(s)
    if not s:
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(mL|L)\s*$", s, re.I)
    if not m:
        return None
    val = float(m.group(1))
    if m.group(2).lower() == "l":
        val *= 1000
    return int(round(val))


def _similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a.lower(), b.lower()) / 100.0
    except ImportError:
        import difflib
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _rules_identity(extracted: dict, app_data: dict, config: dict) -> list[dict]:
    results = []
    sim_config = config.get("similarity", {})
    pass_thresh = sim_config.get("brand_class_pass", 0.95)
    review_thresh = sim_config.get("brand_class_review", 0.80)

    brand_app = _norm(app_data.get("brand_name", ""))
    brand_label = _norm(extracted.get("brand_name", {}).get("value", ""))
    if not brand_label:
        results.append({
            "rule_id": "Brand name present",
            "category": "Identity",
            "status": "fail",
            "message": "Brand name not found on label.",
            "bbox_ref": extracted.get("brand_name", {}).get("bbox"),
        })
    elif not brand_app:
        results.append({
            "rule_id": "Brand name present",
            "category": "Identity",
            "status": "pass",
            "message": "Brand name found on label.",
            "bbox_ref": extracted.get("brand_name", {}).get("bbox"),
        })
    else:
        sim = _similarity(brand_app, brand_label)
        if sim >= pass_thresh:
            results.append({"rule_id": "Brand name matches", "category": "Identity", "status": "pass", "message": "Brand name matches application.", "bbox_ref": extracted.get("brand_name", {}).get("bbox")})
        elif sim >= review_thresh:
            results.append({"rule_id": "Brand name matches", "category": "Identity", "status": "needs_review", "message": f"Brand name similar but not exact: label '{brand_label}' vs application '{brand_app}'.", "bbox_ref": extracted.get("brand_name", {}).get("bbox")})
        else:
            results.append({"rule_id": "Brand name matches", "category": "Identity", "status": "fail", "message": f"Brand name mismatch: label '{brand_label}' vs application '{brand_app}'.", "bbox_ref": extracted.get("brand_name", {}).get("bbox")})

    class_app = _norm(app_data.get("class_type", ""))
    class_label = _norm(extracted.get("class_type", {}).get("value", ""))
    if not class_label:
        results.append({"rule_id": "Class/type present", "category": "Identity", "status": "fail", "message": "Class/type not found on label.", "bbox_ref": extracted.get("class_type", {}).get("bbox")})
    elif not class_app:
        results.append({"rule_id": "Class/type present", "category": "Identity", "status": "pass", "message": "Class/type found on label.", "bbox_ref": extracted.get("class_type", {}).get("bbox")})
    else:
        sim = _similarity(class_app, class_label)
        if sim >= pass_thresh:
            results.append({"rule_id": "Class/type matches", "category": "Identity", "status": "pass", "message": "Class/type matches application.", "bbox_ref": extracted.get("class_type", {}).get("bbox")})
        elif sim >= review_thresh:
            results.append({"rule_id": "Class/type matches", "category": "Identity", "status": "needs_review", "message": f"Class/type similar but not exact: label '{class_label}' vs application '{class_app}'.", "bbox_ref": extracted.get("class_type", {}).get("bbox")})
        else:
            results.append({"rule_id": "Class/type matches", "category": "Identity", "status": "fail", "message": f"Class/type mismatch: label '{class_label}' vs application '{class_app}'.", "bbox_ref": extracted.get("class_type", {}).get("bbox")})

    return results


def _rules_alcohol_contents(extracted: dict, app_data: dict, config: dict) -> list[dict]:
    results = []
    app_pct = _norm(app_data.get("alcohol_pct", ""))
    app_proof = _norm(app_data.get("proof", ""))
    app_net = _norm(app_data.get("net_contents_ml", ""))

    label_pct = _norm(extracted.get("alcohol_pct", {}).get("value", ""))
    label_proof = _norm(extracted.get("proof", {}).get("value", ""))
    label_net = _norm(extracted.get("net_contents", {}).get("value", ""))

    if not label_pct:
        results.append({"rule_id": "Alcohol content present", "category": "Alcohol & contents", "status": "fail", "message": "Alcohol content (ABV) not found on label.", "bbox_ref": extracted.get("alcohol_pct", {}).get("bbox")})
    elif app_pct and label_pct != app_pct:
        results.append({"rule_id": "Alcohol content matches", "category": "Alcohol & contents", "status": "needs_review", "message": f"ABV on label ({label_pct}%) does not match application ({app_pct}%).", "bbox_ref": extracted.get("alcohol_pct", {}).get("bbox")})
    else:
        results.append({"rule_id": "Alcohol content", "category": "Alcohol & contents", "status": "pass", "message": "Alcohol content present and matches.", "bbox_ref": extracted.get("alcohol_pct", {}).get("bbox")})

    if not label_proof and app_proof:
        results.append({"rule_id": "Proof present", "category": "Alcohol & contents", "status": "needs_review", "message": "Proof not found on label but required.", "bbox_ref": extracted.get("proof", {}).get("bbox")})
    elif app_proof and label_proof != app_proof:
        results.append({"rule_id": "Proof matches", "category": "Alcohol & contents", "status": "needs_review", "message": f"Proof on label ({label_proof}) does not match application ({app_proof}).", "bbox_ref": extracted.get("proof", {}).get("bbox")})
    else:
        results.append({"rule_id": "Proof", "category": "Alcohol & contents", "status": "pass", "message": "Proof present and matches.", "bbox_ref": extracted.get("proof", {}).get("bbox")})

    if not label_net:
        results.append({"rule_id": "Net contents present", "category": "Alcohol & contents", "status": "fail", "message": "Net contents not found on label.", "bbox_ref": extracted.get("net_contents", {}).get("bbox")})
    else:
        # TTB: stated in L/mL; must match standard of fill (27 CFR 5.203) when prescribed
        label_ml = _net_contents_to_ml(label_net)
        app_ml: int | None = None
        if app_net:
            try:
                app_ml = int(round(float(app_net)))
            except (TypeError, ValueError):
                pass
        allowed_ml = config.get("net_contents", {}).get("standard_of_fill_ml") or []
        if label_ml is not None and allowed_ml and label_ml not in allowed_ml:
            results.append({"rule_id": "Net contents standard of fill", "category": "Alcohol & contents", "status": "needs_review", "message": f"Net contents '{label_net}' is not a TTB authorized standard of fill (27 CFR 5.203).", "bbox_ref": extracted.get("net_contents", {}).get("bbox")})
        if app_ml is not None and label_ml is not None and app_ml != label_ml:
            results.append({"rule_id": "Net contents matches", "category": "Alcohol & contents", "status": "needs_review", "message": f"Net contents on label ({label_net}) does not match application ({app_net} mL).", "bbox_ref": extracted.get("net_contents", {}).get("bbox")})
        if not any(r.get("rule_id") in ("Net contents standard of fill", "Net contents matches") for r in results):
            results.append({"rule_id": "Net contents", "category": "Alcohol & contents", "status": "pass", "message": f"Net contents found: {label_net}.", "bbox_ref": extracted.get("net_contents", {}).get("bbox")})

    return results


def _rules_warning(extracted: dict, app_data: dict, config: dict) -> list[dict]:
    results = []
    warning_cfg = config.get("warning", {})
    required_lead = warning_cfg.get("government_warning_lead", "GOVERNMENT WARNING:")
    full_text = (extracted.get("government_warning", {}).get("value") or "").strip()
    normalize = warning_cfg.get("normalize_whitespace", True)
    if normalize:
        full_text = " ".join(full_text.split())

    if not full_text:
        results.append({"rule_id": "Government warning present", "category": "Warning", "status": "fail", "message": "Government warning statement not found on label.", "bbox_ref": extracted.get("government_warning", {}).get("bbox")})
        return results

    if required_lead not in full_text and "GOVERNMENT WARNING" not in full_text.upper():
        results.append({"rule_id": "GOVERNMENT WARNING in caps", "category": "Warning", "status": "fail", "message": "The phrase 'GOVERNMENT WARNING' must appear in all caps.", "bbox_ref": extracted.get("government_warning", {}).get("bbox")})
    else:
        results.append({"rule_id": "GOVERNMENT WARNING in caps", "category": "Warning", "status": "pass", "message": "GOVERNMENT WARNING appears in required form.", "bbox_ref": extracted.get("government_warning", {}).get("bbox")})

    required_full = (warning_cfg.get("full_statement") or "").strip()
    if normalize and required_full:
        required_full = " ".join(required_full.split())
    if required_full and required_full not in full_text:
        # Allow minor variation
        if len(full_text) < 50:
            results.append({"rule_id": "Exact warning wording", "category": "Warning", "status": "fail", "message": "Warning text appears incomplete or incorrect.", "bbox_ref": extracted.get("government_warning", {}).get("bbox")})
        else:
            results.append({"rule_id": "Exact warning wording", "category": "Warning", "status": "needs_review", "message": "Warning text may not match required wording exactly. Please verify.", "bbox_ref": extracted.get("government_warning", {}).get("bbox")})
    else:
        results.append({"rule_id": "Exact warning wording", "category": "Warning", "status": "pass", "message": "Warning statement present and appears complete.", "bbox_ref": extracted.get("government_warning", {}).get("bbox")})

    return results


def _rules_origin(extracted: dict, app_data: dict, config: dict) -> list[dict]:
    results = []
    bottler_label = _norm(extracted.get("bottler", {}).get("value", ""))
    bottler_app = _norm(app_data.get("bottler_name", ""))

    if not bottler_label:
        results.append({"rule_id": "Bottler/producer statement", "category": "Origin", "status": "fail", "message": "Bottler/producer name and address not found on label.", "bbox_ref": extracted.get("bottler", {}).get("bbox")})
    elif bottler_app and _similarity(bottler_app, bottler_label) < 0.8:
        results.append({"rule_id": "Bottler matches", "category": "Origin", "status": "needs_review", "message": f"Bottler on label may not match application: '{bottler_label}'.", "bbox_ref": extracted.get("bottler", {}).get("bbox")})
    else:
        results.append({"rule_id": "Bottler/producer statement", "category": "Origin", "status": "pass", "message": "Bottler/producer statement found.", "bbox_ref": extracted.get("bottler", {}).get("bbox")})

    if app_data.get("imported"):
        co = _norm(extracted.get("country_of_origin", {}).get("value", ""))
        if not co:
            results.append({"rule_id": "Country of origin", "category": "Origin", "status": "fail", "message": "Imported product must show country of origin.", "bbox_ref": extracted.get("country_of_origin", {}).get("bbox")})
        else:
            results.append({"rule_id": "Country of origin", "category": "Origin", "status": "pass", "message": f"Country of origin found: {co}.", "bbox_ref": extracted.get("country_of_origin", {}).get("bbox")})

    return results


def _rules_other(extracted: dict, app_data: dict, config: dict) -> list[dict]:
    results = []
    cond = config.get("conditional_statements", {})
    all_blocks_text = " ".join(b.get("text", "") for b in extracted.get("_all_blocks", []))
    blocks_lower = all_blocks_text.lower()

    if app_data.get("sulfites_required"):
        if (cond.get("sulfites", "sulfites") or "sulfites").lower() in blocks_lower:
            results.append({"rule_id": "Sulfites statement", "category": "Other", "status": "pass", "message": "Sulfites statement found.", "bbox_ref": None})
        else:
            results.append({"rule_id": "Sulfites statement", "category": "Other", "status": "fail", "message": "Sulfites statement required but not found.", "bbox_ref": None})
    if app_data.get("fd_c_yellow_5_required"):
        if "yellow" in blocks_lower and ("5" in all_blocks_text or "no. 5" in blocks_lower):
            results.append({"rule_id": "FD&C Yellow No. 5", "category": "Other", "status": "pass", "message": "FD&C Yellow No. 5 statement found.", "bbox_ref": None})
        else:
            results.append({"rule_id": "FD&C Yellow No. 5", "category": "Other", "status": "fail", "message": "FD&C Yellow No. 5 statement required but not found.", "bbox_ref": None})
    if app_data.get("carmine_required"):
        if "carmine" in blocks_lower:
            results.append({"rule_id": "Carmine statement", "category": "Other", "status": "pass", "message": "Carmine statement found.", "bbox_ref": None})
        else:
            results.append({"rule_id": "Carmine statement", "category": "Other", "status": "fail", "message": "Carmine statement required but not found.", "bbox_ref": None})
    if app_data.get("wood_treatment_required"):
        if "treated" in blocks_lower or "wood" in blocks_lower:
            results.append({"rule_id": "Wood treatment", "category": "Other", "status": "pass", "message": "Wood treatment statement found.", "bbox_ref": None})
        else:
            results.append({"rule_id": "Wood treatment", "category": "Other", "status": "fail", "message": "Wood treatment statement required but not found.", "bbox_ref": None})
    if app_data.get("age_statement_required"):
        if "aged" in blocks_lower or "age" in blocks_lower:
            results.append({"rule_id": "Age statement", "category": "Other", "status": "pass", "message": "Age statement found.", "bbox_ref": None})
        else:
            results.append({"rule_id": "Age statement", "category": "Other", "status": "fail", "message": "Age statement required but not found.", "bbox_ref": None})
    if app_data.get("neutral_spirits_required"):
        if "neutral spirits" in blocks_lower:
            results.append({"rule_id": "Neutral spirits", "category": "Other", "status": "pass", "message": "Neutral spirits statement found.", "bbox_ref": None})
        else:
            results.append({"rule_id": "Neutral spirits", "category": "Other", "status": "fail", "message": "Neutral spirits statement required but not found.", "bbox_ref": None})

    return results
