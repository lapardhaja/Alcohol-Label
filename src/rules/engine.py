"""
Load rules config and run all rule categories. Return list of { rule_id, category, status, message, bbox_ref, extracted_value, app_value }.
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
    beverage_type = (app_data.get("beverage_type") or "spirits").lower().replace("/", "_").replace(" ", "_")
    results: list[dict[str, Any]] = []
    results.extend(_rules_identity(extracted, app_data, config))
    results.extend(_rules_alcohol_contents(extracted, app_data, config, beverage_type))
    results.extend(_rules_warning(extracted, app_data, config))
    results.extend(_rules_origin(extracted, app_data, config))
    results.extend(_rules_other(extracted, app_data, config, beverage_type))
    return results


def _norm(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _net_contents_to_ml(s: str) -> int | None:
    """Parse net contents string (metric or imperial) to milliliters."""
    s = _norm(s)
    if not s:
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(mL|ml|ML|L|l)\s*$", s, re.I)
    if m:
        val = float(m.group(1))
        if m.group(2).lower() == "l":
            val *= 1000
        return int(round(val))
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(?:fl\.?\s*oz\.?|fluid\s+ounces?)\s*$", s, re.I)
    if m:
        return int(round(float(m.group(1)) * 29.5735))
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(?:qt\.?|quart)\s*$", s, re.I)
    if m:
        return int(round(float(m.group(1)) * 946.353))
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(?:pt\.?|pint)\s*$", s, re.I)
    if m:
        return int(round(float(m.group(1)) * 473.176))
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(?:gal\.?|gallon)\s*$", s, re.I)
    if m:
        return int(round(float(m.group(1)) * 3785.41))
    try:
        return int(round(float(s)))
    except (TypeError, ValueError):
        return None


def _similarity(a: str, b: str) -> float:
    """Legacy character-level similarity (kept for backward compat)."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a.lower(), b.lower()) / 100.0
    except ImportError:
        import difflib
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _tokenize(s: str) -> list[str]:
    """Lowercase, strip punctuation edges, split into meaningful tokens."""
    tokens = _norm(s).lower().split()
    return [t.strip(".,;:!?'\"()") for t in tokens if len(t.strip(".,;:!?'\"()")) > 0]


def _fuzzy_token_ratio(a: str, b: str) -> float:
    """Best fuzzy ratio for a single token against another."""
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a, b) / 100.0
    except ImportError:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()


def _smart_match(app_val: str, label_val: str, config: dict | None = None) -> tuple[float, str]:
    """
    Multi-strategy matching that handles real-world label verification scenarios.
    Returns (score 0.0-1.0, reason_string).

    Strategies in order:
      1. Exact match (case-insensitive, whitespace-normalized)
      2. Token containment — all app tokens found in label tokens
      3. Reverse containment — all label tokens found in app tokens
      4. Substring containment — one value is a substring of the other
      5. Fuzzy token match — each app token has a close match in label tokens
      6. Fallback — max of fuzz.ratio and fuzz.token_sort_ratio
    """
    cfg = (config or {}).get("similarity", {})
    fuzzy_token_thresh = cfg.get("fuzzy_token_threshold", 0.85)

    a_norm = _norm(app_val).lower()
    b_norm = _norm(label_val).lower()

    if not a_norm or not b_norm:
        return (0.0, "empty")

    # 1. Exact match
    if a_norm == b_norm:
        return (1.0, "exact")

    a_tokens = _tokenize(app_val)
    b_tokens = _tokenize(label_val)

    if not a_tokens or not b_tokens:
        return (0.0, "empty_tokens")

    a_set = set(a_tokens)
    b_set = set(b_tokens)

    # 2. Token containment: every app token appears in label tokens
    if a_set <= b_set:
        return (0.95, "token_containment")

    # 3. Reverse containment: every label token appears in app tokens
    if b_set <= a_set:
        return (0.90, "reverse_containment")

    # 4. Substring containment
    if a_norm in b_norm or b_norm in a_norm:
        return (0.92, "substring")

    # 5. Fuzzy token match: each app token has a close match in some label token
    all_fuzzy_matched = True
    min_token_score = 1.0
    for at in a_tokens:
        best = max((_fuzzy_token_ratio(at, bt) for bt in b_tokens), default=0.0)
        if best < fuzzy_token_thresh:
            all_fuzzy_matched = False
            break
        min_token_score = min(min_token_score, best)
    if all_fuzzy_matched:
        return (max(0.88, min_token_score * 0.95), "fuzzy_token")

    # 6. Fallback: character-level ratios
    try:
        from rapidfuzz import fuzz
        ratio = fuzz.ratio(a_norm, b_norm) / 100.0
        token_sort = fuzz.token_sort_ratio(a_norm, b_norm) / 100.0
        best = max(ratio, token_sort)
    except ImportError:
        import difflib
        best = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()

    return (best, "fuzzy_ratio")


def _tokens_found_in_text(app_val: str, full_text: str) -> bool:
    """Check if all tokens from app_val appear somewhere in full_text."""
    a_tokens = _tokenize(app_val)
    if not a_tokens:
        return False
    text_lower = full_text.lower()
    return all(t in text_lower for t in a_tokens)


def _rules_identity(extracted: dict, app_data: dict, config: dict) -> list[dict]:
    results = []
    sim_config = config.get("similarity", {})
    pass_thresh = sim_config.get("brand_class_pass", 0.90)
    review_thresh = sim_config.get("brand_class_review", 0.70)

    brand_app = _norm(app_data.get("brand_name", ""))
    brand_label = _norm(extracted.get("brand_name", {}).get("value", ""))
    bbox_brand = extracted.get("brand_name", {}).get("bbox")

    if not brand_label:
        results.append({"rule_id": "Brand name present", "category": "Identity", "status": "fail",
                        "message": "Brand name not found on label.", "bbox_ref": bbox_brand,
                        "extracted_value": "", "app_value": brand_app})
    elif not brand_app:
        results.append({"rule_id": "Brand name present", "category": "Identity", "status": "pass",
                        "message": "Brand name found on label (no application value to compare).", "bbox_ref": bbox_brand,
                        "extracted_value": brand_label, "app_value": ""})
    else:
        score, reason = _smart_match(brand_app, brand_label, config)
        if score >= pass_thresh:
            results.append({"rule_id": "Brand name matches", "category": "Identity", "status": "pass",
                            "message": f"Brand name matches ({reason}).", "bbox_ref": bbox_brand,
                            "extracted_value": brand_label, "app_value": brand_app})
        elif score >= review_thresh:
            results.append({"rule_id": "Brand name matches", "category": "Identity", "status": "needs_review",
                            "message": f"Brand name similar but not exact ({reason}, {score:.0%}).", "bbox_ref": bbox_brand,
                            "extracted_value": brand_label, "app_value": brand_app})
        else:
            all_text = " ".join(b.get("text", "") for b in extracted.get("_all_blocks", []))
            if brand_app and _tokens_found_in_text(brand_app, all_text):
                results.append({"rule_id": "Brand name matches", "category": "Identity", "status": "needs_review",
                                "message": "Brand name found elsewhere on label, not in primary position.", "bbox_ref": bbox_brand,
                                "extracted_value": brand_label, "app_value": brand_app})
            else:
                results.append({"rule_id": "Brand name matches", "category": "Identity", "status": "fail",
                                "message": f"Brand name mismatch ({score:.0%}).", "bbox_ref": bbox_brand,
                                "extracted_value": brand_label, "app_value": brand_app})

    class_app = _norm(app_data.get("class_type", ""))
    class_label = _norm(extracted.get("class_type", {}).get("value", ""))
    bbox_class = extracted.get("class_type", {}).get("bbox")

    if not class_label:
        results.append({"rule_id": "Class/type present", "category": "Identity", "status": "fail",
                        "message": "Class/type not found on label.", "bbox_ref": bbox_class,
                        "extracted_value": "", "app_value": class_app})
    elif not class_app:
        results.append({"rule_id": "Class/type present", "category": "Identity", "status": "pass",
                        "message": "Class/type found on label.", "bbox_ref": bbox_class,
                        "extracted_value": class_label, "app_value": ""})
    else:
        score, reason = _smart_match(class_app, class_label, config)
        if score >= pass_thresh:
            results.append({"rule_id": "Class/type matches", "category": "Identity", "status": "pass",
                            "message": f"Class/type matches ({reason}).", "bbox_ref": bbox_class,
                            "extracted_value": class_label, "app_value": class_app})
        elif score >= review_thresh:
            results.append({"rule_id": "Class/type matches", "category": "Identity", "status": "needs_review",
                            "message": f"Class/type similar but not exact ({reason}, {score:.0%}).", "bbox_ref": bbox_class,
                            "extracted_value": class_label, "app_value": class_app})
        else:
            all_text = " ".join(b.get("text", "") for b in extracted.get("_all_blocks", []))
            if class_app and _tokens_found_in_text(class_app, all_text):
                results.append({"rule_id": "Class/type matches", "category": "Identity", "status": "needs_review",
                                "message": f"Class/type '{class_app}' found elsewhere on label, not in primary class position.", "bbox_ref": bbox_class,
                                "extracted_value": class_label, "app_value": class_app})
            else:
                results.append({"rule_id": "Class/type matches", "category": "Identity", "status": "fail",
                                "message": f"Class/type mismatch ({score:.0%}).", "bbox_ref": bbox_class,
                                "extracted_value": class_label, "app_value": class_app})

    return results


def _rules_alcohol_contents(extracted: dict, app_data: dict, config: dict, beverage_type: str = "spirits") -> list[dict]:
    results = []
    app_pct = _norm(app_data.get("alcohol_pct", ""))
    app_proof = _norm(app_data.get("proof", ""))
    app_net = _norm(app_data.get("net_contents_ml", ""))

    label_pct = _norm(extracted.get("alcohol_pct", {}).get("value", ""))
    label_proof = _norm(extracted.get("proof", {}).get("value", ""))
    label_net = _norm(extracted.get("net_contents", {}).get("value", ""))
    bbox_pct = extracted.get("alcohol_pct", {}).get("bbox")
    bbox_proof = extracted.get("proof", {}).get("bbox")
    bbox_net = extracted.get("net_contents", {}).get("bbox")

    bev_cfg = config.get("beverage_types", {}).get(beverage_type, {})
    abv_mandatory = bev_cfg.get("abv_mandatory", True)

    if not label_pct:
        if abv_mandatory:
            results.append({"rule_id": "Alcohol content present", "category": "Alcohol & contents", "status": "fail",
                            "message": "Alcohol content (ABV) not found on label.", "bbox_ref": bbox_pct,
                            "extracted_value": "", "app_value": app_pct})
        else:
            results.append({"rule_id": "Alcohol content present", "category": "Alcohol & contents", "status": "pass",
                            "message": "Alcohol content not required for this beverage type.", "bbox_ref": None,
                            "extracted_value": "", "app_value": app_pct})
    elif app_pct and label_pct != app_pct:
        results.append({"rule_id": "Alcohol content matches", "category": "Alcohol & contents", "status": "needs_review",
                        "message": f"ABV on label ({label_pct}%) does not match application ({app_pct}%).", "bbox_ref": bbox_pct,
                        "extracted_value": label_pct, "app_value": app_pct})
    else:
        results.append({"rule_id": "Alcohol content", "category": "Alcohol & contents", "status": "pass",
                        "message": "Alcohol content present and matches.", "bbox_ref": bbox_pct,
                        "extracted_value": label_pct, "app_value": app_pct})

    if beverage_type in ("beer", "beer_malt_beverage", "wine"):
        results.append({"rule_id": "Proof", "category": "Alcohol & contents", "status": "pass",
                        "message": "Proof not applicable for this beverage type.", "bbox_ref": None,
                        "extracted_value": label_proof, "app_value": app_proof})
    elif not label_proof and app_proof:
        results.append({"rule_id": "Proof present", "category": "Alcohol & contents", "status": "needs_review",
                        "message": "Proof not found on label but specified in application.", "bbox_ref": bbox_proof,
                        "extracted_value": "", "app_value": app_proof})
    elif app_proof and label_proof != app_proof:
        results.append({"rule_id": "Proof matches", "category": "Alcohol & contents", "status": "needs_review",
                        "message": f"Proof on label ({label_proof}) does not match application ({app_proof}).", "bbox_ref": bbox_proof,
                        "extracted_value": label_proof, "app_value": app_proof})
    else:
        results.append({"rule_id": "Proof", "category": "Alcohol & contents", "status": "pass",
                        "message": "Proof present and matches.", "bbox_ref": bbox_proof,
                        "extracted_value": label_proof, "app_value": app_proof})

    if not label_net:
        results.append({"rule_id": "Net contents present", "category": "Alcohol & contents", "status": "fail",
                        "message": "Net contents not found on label.", "bbox_ref": bbox_net,
                        "extracted_value": "", "app_value": app_net})
    else:
        label_ml = _net_contents_to_ml(label_net)
        app_ml = _net_contents_to_ml(app_net) if app_net else None
        allowed_ml = config.get("net_contents", {}).get("standard_of_fill_ml") or []
        if label_ml is not None and allowed_ml and label_ml not in allowed_ml:
            results.append({"rule_id": "Net contents standard of fill", "category": "Alcohol & contents", "status": "needs_review",
                            "message": f"Net contents '{label_net}' is not a TTB authorized standard of fill.", "bbox_ref": bbox_net,
                            "extracted_value": label_net, "app_value": app_net})
        if app_ml is not None and label_ml is not None and abs(app_ml - label_ml) > 5:
            results.append({"rule_id": "Net contents matches", "category": "Alcohol & contents", "status": "needs_review",
                            "message": f"Net contents on label ({label_net} ≈ {label_ml} mL) does not match application ({app_net} ≈ {app_ml} mL).", "bbox_ref": bbox_net,
                            "extracted_value": label_net, "app_value": app_net})
        if not any(r.get("rule_id") in ("Net contents standard of fill", "Net contents matches") for r in results):
            results.append({"rule_id": "Net contents", "category": "Alcohol & contents", "status": "pass",
                            "message": f"Net contents found: {label_net}.", "bbox_ref": bbox_net,
                            "extracted_value": label_net, "app_value": app_net})

    return results


def _rules_warning(extracted: dict, app_data: dict, config: dict) -> list[dict]:
    results = []
    warning_cfg = config.get("warning", {})
    full_text = (extracted.get("government_warning", {}).get("value") or "").strip()
    normalize = warning_cfg.get("normalize_whitespace", True)
    if normalize:
        full_text = " ".join(full_text.split())
    bbox_warn = extracted.get("government_warning", {}).get("bbox")

    if not full_text:
        results.append({"rule_id": "Government warning present", "category": "Warning", "status": "fail",
                        "message": "Government warning statement not found on label.", "bbox_ref": bbox_warn,
                        "extracted_value": "", "app_value": "Required"})
        return results

    if "GOVERNMENT WARNING" not in full_text.upper():
        results.append({"rule_id": "GOVERNMENT WARNING in caps", "category": "Warning", "status": "fail",
                        "message": "The phrase 'GOVERNMENT WARNING' must appear in all caps.", "bbox_ref": bbox_warn,
                        "extracted_value": full_text[:60], "app_value": "GOVERNMENT WARNING"})
    else:
        results.append({"rule_id": "GOVERNMENT WARNING in caps", "category": "Warning", "status": "pass",
                        "message": "GOVERNMENT WARNING appears in required form.", "bbox_ref": bbox_warn,
                        "extracted_value": "GOVERNMENT WARNING", "app_value": "GOVERNMENT WARNING"})

    required_full = (warning_cfg.get("full_statement") or "").strip()
    if normalize and required_full:
        required_full = " ".join(required_full.split())
    if required_full and required_full not in full_text:
        if len(full_text) < 50:
            results.append({"rule_id": "Exact warning wording", "category": "Warning", "status": "fail",
                            "message": "Warning text appears incomplete or incorrect.", "bbox_ref": bbox_warn,
                            "extracted_value": full_text[:80], "app_value": required_full[:80]})
        else:
            results.append({"rule_id": "Exact warning wording", "category": "Warning", "status": "needs_review",
                            "message": "Warning text may not match required wording exactly.", "bbox_ref": bbox_warn,
                            "extracted_value": full_text[:80], "app_value": required_full[:80]})
    else:
        results.append({"rule_id": "Exact warning wording", "category": "Warning", "status": "pass",
                        "message": "Warning statement present and appears complete.", "bbox_ref": bbox_warn,
                        "extracted_value": full_text[:80], "app_value": "Required statement"})

    return results


def _rules_origin(extracted: dict, app_data: dict, config: dict) -> list[dict]:
    results = []
    bottler_label = _norm(extracted.get("bottler", {}).get("value", ""))
    bottler_app = _norm(app_data.get("bottler_name", ""))
    bbox_bottler = extracted.get("bottler", {}).get("bbox")

    if not bottler_label:
        if bottler_app:
            all_text = " ".join(b.get("text", "") for b in extracted.get("_all_blocks", []))
            if _tokens_found_in_text(bottler_app, all_text):
                results.append({"rule_id": "Bottler/producer statement", "category": "Origin", "status": "needs_review",
                                "message": "Bottler name found in label text but not in standard bottler format.", "bbox_ref": bbox_bottler,
                                "extracted_value": "", "app_value": bottler_app})
            else:
                results.append({"rule_id": "Bottler/producer statement", "category": "Origin", "status": "fail",
                                "message": "Bottler/producer name and address not found on label.", "bbox_ref": bbox_bottler,
                                "extracted_value": "", "app_value": bottler_app})
        else:
            results.append({"rule_id": "Bottler/producer statement", "category": "Origin", "status": "fail",
                            "message": "Bottler/producer name and address not found on label.", "bbox_ref": bbox_bottler,
                            "extracted_value": "", "app_value": bottler_app})
    elif not bottler_app:
        results.append({"rule_id": "Bottler/producer statement", "category": "Origin", "status": "pass",
                        "message": "Bottler/producer statement found.", "bbox_ref": bbox_bottler,
                        "extracted_value": bottler_label, "app_value": bottler_app})
    else:
        score, reason = _smart_match(bottler_app, bottler_label, config)
        if score >= 0.80:
            results.append({"rule_id": "Bottler/producer statement", "category": "Origin", "status": "pass",
                            "message": f"Bottler/producer matches ({reason}).", "bbox_ref": bbox_bottler,
                            "extracted_value": bottler_label, "app_value": bottler_app})
        elif score >= 0.60:
            results.append({"rule_id": "Bottler matches", "category": "Origin", "status": "needs_review",
                            "message": f"Bottler on label may not match application ({reason}, {score:.0%}).", "bbox_ref": bbox_bottler,
                            "extracted_value": bottler_label, "app_value": bottler_app})
        else:
            results.append({"rule_id": "Bottler matches", "category": "Origin", "status": "needs_review",
                            "message": f"Bottler on label differs from application ({score:.0%}).", "bbox_ref": bbox_bottler,
                            "extracted_value": bottler_label, "app_value": bottler_app})

    if app_data.get("imported"):
        co = _norm(extracted.get("country_of_origin", {}).get("value", ""))
        co_app = _norm(app_data.get("country_of_origin", ""))
        bbox_co = extracted.get("country_of_origin", {}).get("bbox")
        if not co:
            results.append({"rule_id": "Country of origin", "category": "Origin", "status": "fail",
                            "message": "Imported product must show country of origin.", "bbox_ref": bbox_co,
                            "extracted_value": "", "app_value": co_app})
        else:
            results.append({"rule_id": "Country of origin", "category": "Origin", "status": "pass",
                            "message": f"Country of origin found: {co}.", "bbox_ref": bbox_co,
                            "extracted_value": co, "app_value": co_app})

    return results


def _rules_other(extracted: dict, app_data: dict, config: dict, beverage_type: str = "spirits") -> list[dict]:
    results = []
    all_blocks_text = " ".join(b.get("text", "") for b in extracted.get("_all_blocks", []))
    blocks_lower = all_blocks_text.lower()

    sulfites_required = app_data.get("sulfites_required", False)
    if beverage_type == "wine":
        bev_cfg = config.get("beverage_types", {}).get("wine", {})
        sulfites_required = sulfites_required or bev_cfg.get("sulfites_default", True)
    if sulfites_required:
        found = "sulfite" in blocks_lower or "sulfites" in blocks_lower
        results.append({"rule_id": "Sulfites statement", "category": "Other",
                        "status": "pass" if found else "fail",
                        "message": "Sulfites statement found." if found else "Sulfites declaration required but not found.",
                        "bbox_ref": None, "extracted_value": "Contains Sulfites" if found else "", "app_value": "Required"})

    if app_data.get("fd_c_yellow_5_required"):
        found = "yellow" in blocks_lower and ("5" in all_blocks_text or "no. 5" in blocks_lower)
        results.append({"rule_id": "FD&C Yellow No. 5", "category": "Other",
                        "status": "pass" if found else "fail",
                        "message": "FD&C Yellow No. 5 statement found." if found else "FD&C Yellow No. 5 statement required but not found.",
                        "bbox_ref": None, "extracted_value": "Found" if found else "", "app_value": "Required"})

    if app_data.get("carmine_required"):
        found = "carmine" in blocks_lower or "cochineal" in blocks_lower
        results.append({"rule_id": "Cochineal/Carmine statement", "category": "Other",
                        "status": "pass" if found else "fail",
                        "message": "Cochineal/Carmine declaration found." if found else "Cochineal/Carmine disclosure required but not found.",
                        "bbox_ref": None, "extracted_value": "Found" if found else "", "app_value": "Required"})

    if app_data.get("wood_treatment_required") and beverage_type in ("spirits", "distilled_spirits"):
        found = "treated" in blocks_lower or "wood" in blocks_lower
        results.append({"rule_id": "Wood treatment", "category": "Other",
                        "status": "pass" if found else "fail",
                        "message": "Wood treatment statement found." if found else "Wood treatment statement required but not found.",
                        "bbox_ref": None, "extracted_value": "Found" if found else "", "app_value": "Required"})

    if app_data.get("age_statement_required"):
        found = "aged" in blocks_lower or "age" in blocks_lower or bool(re.search(r"\d+\s*years?", blocks_lower))
        results.append({"rule_id": "Age statement", "category": "Other",
                        "status": "pass" if found else "fail",
                        "message": "Age statement found." if found else "Age statement required but not found.",
                        "bbox_ref": None, "extracted_value": "Found" if found else "", "app_value": "Required"})

    if app_data.get("neutral_spirits_required") and beverage_type in ("spirits", "distilled_spirits"):
        found = "neutral spirits" in blocks_lower or "grain spirits" in blocks_lower
        results.append({"rule_id": "Neutral spirits / commodity", "category": "Other",
                        "status": "pass" if found else "fail",
                        "message": "Neutral spirits / commodity statement found." if found else "Neutral spirits statement required but not found.",
                        "bbox_ref": None, "extracted_value": "Found" if found else "", "app_value": "Required"})

    if beverage_type in ("beer", "beer_malt_beverage") and app_data.get("aspartame_required"):
        found = "aspartame" in blocks_lower or "phenylketonurics" in blocks_lower
        results.append({"rule_id": "Aspartame statement", "category": "Other",
                        "status": "pass" if found else "fail",
                        "message": "Aspartame statement found." if found else "Aspartame statement required but not found.",
                        "bbox_ref": None, "extracted_value": "Found" if found else "", "app_value": "Required"})

    if beverage_type == "wine":
        if app_data.get("appellation_required"):
            found = "appellation" in blocks_lower or bool(re.search(r"(napa|sonoma|willamette|paso robles|american viticultural)", blocks_lower))
            results.append({"rule_id": "Appellation of origin", "category": "Other",
                            "status": "pass" if found else "needs_review",
                            "message": "Appellation of origin found." if found else "Appellation of origin may be required. Verify.",
                            "bbox_ref": None, "extracted_value": "Found" if found else "", "app_value": "Required"})
        if app_data.get("varietal_required"):
            class_label = _norm(extracted.get("class_type", {}).get("value", ""))
            found = bool(class_label)
            results.append({"rule_id": "Varietal designation", "category": "Other",
                            "status": "pass" if found else "needs_review",
                            "message": f"Varietal designation found: {class_label}." if found else "Varietal designation expected but not detected.",
                            "bbox_ref": extracted.get("class_type", {}).get("bbox") if found else None,
                            "extracted_value": class_label, "app_value": "Required"})

    return results
