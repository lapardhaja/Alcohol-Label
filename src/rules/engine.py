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


def _apply_spell_correction_to_warning(text: str, reference_words: set[str]) -> str:
    """Correct unknown words in warning text using spell checker (fixes OCR errors like 'cig' -> 'cause')."""
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker()
        spell.word_frequency.load_words(list(reference_words))
    except ImportError:
        return text

    words = re.findall(r"\b\w+\b|\W+", text)
    corrected = []
    for w in words:
        if not re.match(r"\w+", w):
            corrected.append(w)
            continue
        if w in reference_words or spell.known([w]):
            corrected.append(w)
            continue
        fix = spell.correction(w)
        corrected.append(fix if fix else w)
    return "".join(corrected)


def _filter_suspicious_from_warning(text: str, suspicious: list[str]) -> str:
    """Remove suspicious (OCR garbage) tokens from warning text for display."""
    if not text or not suspicious:
        return text
    result = text
    for tok in suspicious:
        result = re.sub(rf"\b{re.escape(tok)}\b", "", result, flags=re.I)
    return " ".join(result.split())


def _get_suspicious_warning_tokens(
    extracted_words: list[str], reference_words: set[str]
) -> list[str]:
    """Return tokens in extracted that are not in reference and not valid dictionary words."""
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker()
        spell.word_frequency.load_words(list(reference_words))
    except ImportError:
        return []

    suspicious = []
    for w in extracted_words:
        if len(w) < 2 or w.isdigit():
            continue
        if w in reference_words:
            continue
        if not spell.known([w]):
            suspicious.append(w)
    return suspicious


def _net_contents_to_ml(s: str) -> int | None:
    """Parse net contents string (metric or imperial) to milliliters."""
    s = _norm(s)
    if not s:
        return None
    # Pre-normalize OCR confusables only in numeric parts (e.g. "75O mL" -> "750 mL", preserves "fl" in "fl oz")
    s = _normalize_numeric_sequences(s)
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

    # 1. Exact match (including OCR-normalized: "T0m" vs "Tom", "Bacard1" vs "Bacardi")
    if a_norm == b_norm:
        return (1.0, "exact")
    a_ocr = _normalize_ocr_for_text(app_val)
    b_ocr = _normalize_ocr_for_text(label_val)
    if a_ocr and b_ocr and a_ocr == b_ocr:
        return (1.0, "exact_ocr_normalized")

    a_tokens = _tokenize(app_val)
    b_tokens = _tokenize(label_val)

    if not a_tokens or not b_tokens:
        return (0.0, "empty_tokens")

    a_set = set(a_tokens)
    b_set = set(b_tokens)

    # 2. Token containment: every app token appears in label tokens
    if a_set <= b_set:
        return (0.95, "token_containment")

    # 2b. OCR-normalized token containment (handles "T0m" in label vs "Tom" in app)
    a_set_ocr = {_normalize_ocr_for_text(t) for t in a_tokens}
    b_set_ocr = {_normalize_ocr_for_text(t) for t in b_tokens}
    if a_set_ocr and b_set_ocr and a_set_ocr <= b_set_ocr:
        return (0.95, "token_containment_ocr_normalized")

    # 3. Reverse containment: every label token appears in app tokens
    # Don't pass when: (a) label has fewer tokens than app, or (b) label is only generic tokens (Brewery, Distillery, etc.)
    generic = set((cfg.get("generic_brand_tokens") or ["brewery", "distillery", "winery", "inc", "co", "company", "llc", "ale", "beer", "spirits", "beverage", "beverages"]))
    if b_set <= a_set:
        if len(b_tokens) < len(a_tokens):
            return (0.88, "reverse_containment_partial")
        if len(b_tokens) == 1 and b_tokens[0] in generic:
            return (0.75, "reverse_containment_generic")
        return (0.90, "reverse_containment")
    if b_set_ocr and a_set_ocr and b_set_ocr <= a_set_ocr:
        if len(b_tokens) < len(a_tokens):
            return (0.88, "reverse_containment_partial_ocr")
        if len(b_tokens) == 1 and b_tokens[0] in generic:
            return (0.75, "reverse_containment_generic_ocr")
        return (0.90, "reverse_containment_ocr_normalized")

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


def _token_found_in_text(token: str, text_lower: str, text_words: list[str], fuzzy_thresh: float = 0.80) -> bool:
    """
    Check if token appears in OCR text, allowing for distortions.
    Tries: exact substring, OCR-normalized word match, fuzzy word match.
    """
    if token in text_lower:
        return True
    if token == "&":
        return "and" in text_lower
    token_norm = _normalize_ocr_for_text(token)
    for w in text_words:
        w_clean = w.strip(".,;:!?'\"()").lower()
        if not w_clean or len(w_clean) < 2:
            continue
        if w_clean == token:
            return True
        if _normalize_ocr_for_text(w_clean) == token_norm:
            return True
        try:
            from rapidfuzz import fuzz
            if fuzz.ratio(token, w_clean) / 100.0 >= fuzzy_thresh:
                return True
            if fuzz.ratio(token_norm, _normalize_ocr_for_text(w_clean)) / 100.0 >= fuzzy_thresh:
                return True
        except ImportError:
            import difflib
            if difflib.SequenceMatcher(None, token, w_clean).ratio() >= fuzzy_thresh:
                return True
    return False


def _app_tokens_in_full_text(app_val: str, full_text: str, fuzzy_thresh: float = 0.80) -> bool:
    """
    Check if all meaningful app tokens appear in full OCR text, allowing for OCR distortions.
    Handles: exact match, &/and, OCR confusables (1/l, 0/O), fuzzy match (Mat≈Malt).
    Used when extracted field is partial — verify app value is scattered/distorted on label.
    """
    tokens = _tokenize(app_val)
    text_lower = full_text.lower()
    text_words = text_lower.split()
    meaningful = [t for t in tokens if len(t) > 1 or t.isalnum()]
    if not meaningful:
        return False
    for t in meaningful:
        if not _token_found_in_text(t, text_lower, text_words, fuzzy_thresh):
            return False
    return True


# ---------------------------------------------------------------------------
# OCR confusable detection and normalization
# ---------------------------------------------------------------------------

_OCR_CONFUSABLE_PAIRS: list[tuple[str, str]] = [
    ("l", "1"), ("1", "l"),
    ("I", "l"), ("l", "I"),
    ("I", "1"), ("1", "I"),
    ("O", "0"), ("0", "O"),
    ("o", "0"), ("0", "o"),
    ("S", "5"), ("5", "S"),
    ("B", "8"), ("8", "B"),
    ("rn", "m"), ("m", "rn"),
    (",", "."), (".", ","),
    ("'", "'"), ("'", "'"),
    ('"', '"'), ('"', '"'),
    ("c", "e"), ("e", "c"),
]

# Canonical mappings for proactive normalization (avoids I/1, O/0 confusion in matching)
# Text context: digits that look like letters → letters (so "Bacard1" matches "Bacardi")
_OCR_TEXT_NORMALIZE: dict[str, str] = {
    "1": "i", "0": "o", "5": "s", "8": "b",
}
# Numeric context: letters that look like digits → digits (so "75O mL" parses as 750)
_OCR_NUMERIC_NORMALIZE: dict[str, str] = {
    "O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8",
}


def _normalize_ocr_for_text(s: str) -> str:
    """Normalize OCR confusables for text matching. Maps digit-like chars to letters."""
    s = _norm(s).lower()
    for char, canonical in _OCR_TEXT_NORMALIZE.items():
        s = s.replace(char, canonical)
    return s


def _normalize_ocr_for_numeric(s: str) -> str:
    """Normalize OCR confusables in numeric strings. Maps letter-like chars to digits."""
    s = _norm(s)
    for char, canonical in _OCR_NUMERIC_NORMALIZE.items():
        s = s.replace(char, canonical)
    return s


def _normalize_numeric_sequences(s: str) -> str:
    """Normalize OCR confusables only within digit sequences (e.g. '75O mL' -> '750 mL').
    Avoids corrupting units like 'fl' in '12 fl oz' (l only normalized when adjacent to digits)."""
    def _replace_in_number(match: re.Match) -> str:
        part = match.group(0)
        for char, canonical in _OCR_NUMERIC_NORMALIZE.items():
            part = part.replace(char, canonical)
        return part

    # Match sequences that contain at least one digit (avoids matching standalone 'l' in 'fl oz')
    return re.sub(r"\d[\dOolISB8.]*", _replace_in_number, s)


def _is_ocr_confusable(a: str, b: str) -> bool:
    """Check if two strings differ only by common OCR substitution characters or high fuzzy similarity (e.g. Wooprorp vs Woodford)."""
    if not a or not b:
        return False
    a_n = _norm(a).lower()
    b_n = _norm(b).lower()
    if a_n == b_n:
        return True
    if len(a_n) != len(b_n) and abs(len(a_n) - len(b_n)) > 2:
        pass  # still allow fuzzy check below
    else:
        for orig, repl in _OCR_CONFUSABLE_PAIRS:
            if a_n.replace(orig.lower(), repl.lower()) == b_n:
                return True
            if b_n.replace(orig.lower(), repl.lower()) == a_n:
                return True
        diff_count = sum(1 for ca, cb in zip(a_n, b_n) if ca != cb)
        if diff_count <= 2 and diff_count > 0:
            for i, (ca, cb) in enumerate(zip(a_n, b_n)):
                if ca != cb:
                    if not any((ca == p[0].lower() and cb == p[1].lower()) or
                               (cb == p[0].lower() and ca == p[1].lower())
                               for p in _OCR_CONFUSABLE_PAIRS):
                        return False
            return True
    try:
        from rapidfuzz import fuzz
        ratio = fuzz.ratio(a_n, b_n) / 100.0
        if ratio >= 0.65 and abs(len(a_n) - len(b_n)) <= 3:
            return True
    except ImportError:
        import difflib
        ratio = difflib.SequenceMatcher(None, a_n, b_n).ratio()
        if ratio >= 0.65 and abs(len(a_n) - len(b_n)) <= 3:
            return True
    return False


def _parse_abv_float(s: str) -> float | None:
    """Parse ABV string to float, stripping trailing % etc."""
    s = _norm(s).rstrip("%").strip()
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _rules_identity(extracted: dict, app_data: dict, config: dict) -> list[dict]:
    results = []
    sim_config = config.get("similarity", {})
    pass_thresh = sim_config.get("brand_class_pass", 0.90)
    review_thresh = sim_config.get("brand_class_review", 0.70)

    brand_app = _norm(app_data.get("brand_name", ""))
    brand_label = _norm(extracted.get("brand_name", {}).get("value", ""))
    bbox_brand = extracted.get("brand_name", {}).get("bbox")
    all_text = " ".join(b.get("text", "") for b in extracted.get("_all_blocks", []))
    if not brand_label:
        results.append({"rule_id": "Brand name present", "category": "Identity", "status": "fail",
                        "message": "Brand name not found on label.", "bbox_ref": bbox_brand,
                        "extracted_value": "", "app_value": brand_app})
    elif not brand_app:
        results.append({"rule_id": "Brand name present", "category": "Identity", "status": "needs_review",
                        "message": "Brand name found on label — add to application for verification.", "bbox_ref": bbox_brand,
                        "extracted_value": brand_label, "app_value": ""})
    else:
        score, reason = _smart_match(brand_app, brand_label, config)
        if score >= pass_thresh:
            results.append({"rule_id": "Brand name matches", "category": "Identity", "status": "pass",
                            "message": f"Brand name matches ({reason}).", "bbox_ref": bbox_brand,
                            "extracted_value": brand_label, "app_value": brand_app})
        elif score >= review_thresh:
            # Extracted brand partial (e.g. "BREWERY") — check if app tokens appear elsewhere on label (with fuzzy/OCR)
            if brand_app and _app_tokens_in_full_text(brand_app, all_text):
                results.append({"rule_id": "Brand name matches", "category": "Identity", "status": "pass",
                                "message": "Brand name found on label (tokens in full text).", "bbox_ref": bbox_brand,
                                "extracted_value": brand_label, "app_value": brand_app, "prefer_app_display": True})
            else:
                ocr_hint = " — likely OCR misread" if _is_ocr_confusable(brand_app, brand_label) else ""
                results.append({"rule_id": "Brand name matches", "category": "Identity", "status": "needs_review",
                                "message": f"Brand name similar but not exact ({reason}, {score:.0%}){ocr_hint}.", "bbox_ref": bbox_brand,
                                "extracted_value": brand_label, "app_value": brand_app})
        else:
            if brand_app and _app_tokens_in_full_text(brand_app, all_text):
                results.append({"rule_id": "Brand name matches", "category": "Identity", "status": "pass",
                                "message": "Brand name found on label (tokens in full text).", "bbox_ref": bbox_brand,
                                "extracted_value": brand_label, "app_value": brand_app, "prefer_app_display": True})
            elif _is_ocr_confusable(brand_app, brand_label):
                results.append({"rule_id": "Brand name matches", "category": "Identity", "status": "needs_review",
                                "message": f"Brand name differs — likely OCR misread, verify manually ({score:.0%}).", "bbox_ref": bbox_brand,
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
        results.append({"rule_id": "Class/type present", "category": "Identity", "status": "needs_review",
                        "message": "Class/type found on label — add to application for verification.", "bbox_ref": bbox_class,
                        "extracted_value": class_label, "app_value": ""})
    else:
        score, reason = _smart_match(class_app, class_label, config)
        if score >= pass_thresh:
            results.append({"rule_id": "Class/type matches", "category": "Identity", "status": "pass",
                            "message": f"Class/type matches ({reason}).", "bbox_ref": bbox_class,
                            "extracted_value": class_label, "app_value": class_app})
        elif score >= review_thresh:
            if class_app and _app_tokens_in_full_text(class_app, all_text):
                results.append({"rule_id": "Class/type matches", "category": "Identity", "status": "pass",
                                "message": "Class/type found on label (tokens in full text).", "bbox_ref": bbox_class,
                                "extracted_value": class_label, "app_value": class_app, "prefer_app_display": True})
            else:
                ocr_hint = " — likely OCR misread" if _is_ocr_confusable(class_app, class_label) else ""
                results.append({"rule_id": "Class/type matches", "category": "Identity", "status": "needs_review",
                                "message": f"Class/type similar but not exact ({reason}, {score:.0%}){ocr_hint}.", "bbox_ref": bbox_class,
                                "extracted_value": class_label, "app_value": class_app})
        else:
            if class_app and _app_tokens_in_full_text(class_app, all_text):
                results.append({"rule_id": "Class/type matches", "category": "Identity", "status": "needs_review",
                                "message": f"Application class '{class_app}' found on label but extracted class is '{class_label}' — verify.", "bbox_ref": bbox_class,
                                "extracted_value": class_label, "app_value": class_app})
            elif _is_ocr_confusable(class_app, class_label):
                results.append({"rule_id": "Class/type matches", "category": "Identity", "status": "needs_review",
                                "message": f"Class/type differs — likely OCR misread, verify manually ({score:.0%}).", "bbox_ref": bbox_class,
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
    elif app_pct:
        label_f = _parse_abv_float(label_pct)
        app_f = _parse_abv_float(app_pct)
        if label_f is not None and app_f is not None and abs(label_f - app_f) <= 0.3:
            results.append({"rule_id": "Alcohol content matches", "category": "Alcohol & contents", "status": "pass",
                            "message": "Alcohol content present and matches.", "bbox_ref": bbox_pct,
                            "extracted_value": label_pct, "app_value": app_pct})
        elif label_f is not None and app_f is not None:
            if _is_ocr_confusable(label_pct, app_pct):
                results.append({"rule_id": "Alcohol content matches", "category": "Alcohol & contents", "status": "needs_review",
                                "message": f"ABV on label ({label_pct}%) differs from application ({app_pct}%) — likely OCR misread, verify manually.", "bbox_ref": bbox_pct,
                                "extracted_value": label_pct, "app_value": app_pct})
            else:
                results.append({"rule_id": "Alcohol content matches", "category": "Alcohol & contents", "status": "needs_review",
                                "message": f"ABV on label ({label_pct}%) does not match application ({app_pct}%).", "bbox_ref": bbox_pct,
                                "extracted_value": label_pct, "app_value": app_pct})
        else:
            results.append({"rule_id": "Alcohol content", "category": "Alcohol & contents", "status": "pass",
                            "message": "Alcohol content present.", "bbox_ref": bbox_pct,
                            "extracted_value": label_pct, "app_value": app_pct})
    else:
        results.append({"rule_id": "Alcohol content", "category": "Alcohol & contents", "status": "needs_review",
                        "message": "Alcohol content found on label — add to application for verification.", "bbox_ref": bbox_pct,
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
        # Check OCR confusables (e.g. "8O" vs "80", "9O" vs "90")
        label_proof_norm = _normalize_ocr_for_numeric(label_proof)
        app_proof_norm = _normalize_ocr_for_numeric(app_proof)
        if label_proof_norm == app_proof_norm:
            results.append({"rule_id": "Proof matches", "category": "Alcohol & contents", "status": "pass",
                            "message": f"Proof matches after OCR normalization ({label_proof} ≈ {app_proof}).", "bbox_ref": bbox_proof,
                            "extracted_value": label_proof, "app_value": app_proof})
        elif _is_ocr_confusable(label_proof, app_proof):
            results.append({"rule_id": "Proof matches", "category": "Alcohol & contents", "status": "needs_review",
                            "message": f"Proof on label ({label_proof}) differs from application ({app_proof}) — likely OCR misread, verify manually.", "bbox_ref": bbox_proof,
                            "extracted_value": label_proof, "app_value": app_proof})
        else:
            results.append({"rule_id": "Proof matches", "category": "Alcohol & contents", "status": "needs_review",
                            "message": f"Proof on label ({label_proof}) does not match application ({app_proof}).", "bbox_ref": bbox_proof,
                            "extracted_value": label_proof, "app_value": app_proof})
    else:
        results.append({"rule_id": "Proof", "category": "Alcohol & contents", "status": "pass",
                        "message": "Proof present and matches.", "bbox_ref": bbox_proof,
                        "extracted_value": label_proof, "app_value": app_proof})

    # Proof-to-ABV consistency: proof should equal 2 × ABV
    if label_pct and label_proof:
        abv_f = _parse_abv_float(label_pct)
        proof_f = _parse_abv_float(label_proof)
        if abv_f is not None and proof_f is not None:
            expected_proof = abv_f * 2
            if abs(proof_f - expected_proof) > 1.0:
                results.append({"rule_id": "Proof/ABV consistency", "category": "Alcohol & contents", "status": "needs_review",
                                "message": f"Proof ({label_proof}) does not equal 2× ABV ({label_pct}%). Expected proof ≈ {expected_proof:.0f}.",
                                "bbox_ref": bbox_proof, "extracted_value": label_proof,
                                "app_value": f"proof = 2 × ABV"})
            else:
                results.append({"rule_id": "Proof/ABV consistency", "category": "Alcohol & contents", "status": "pass",
                                "message": "Proof and ABV are consistent (proof = 2 × ABV).",
                                "bbox_ref": bbox_proof, "extracted_value": label_proof,
                                "app_value": app_proof})

    if not label_net:
        results.append({"rule_id": "Net contents present", "category": "Alcohol & contents", "status": "fail",
                        "message": "Net contents not found on label.", "bbox_ref": bbox_net,
                        "extracted_value": "", "app_value": app_net})
    elif not app_net:
        results.append({"rule_id": "Net contents present", "category": "Alcohol & contents", "status": "needs_review",
                        "message": "Net contents found on label — add to application for verification.", "bbox_ref": bbox_net,
                        "extracted_value": label_net, "app_value": ""})
    else:
        label_ml = _net_contents_to_ml(label_net)
        app_ml = _net_contents_to_ml(app_net) if app_net else None
        # Standard of fill (27 CFR 5.203) applies to distilled spirits and wine only; malt beverages may use any size.
        malt_bev = beverage_type in ("beer", "beer_malt_beverage")
        allowed_ml = config.get("net_contents", {}).get("standard_of_fill_ml") or []
        if not malt_bev and label_ml is not None and allowed_ml and label_ml not in allowed_ml:
            results.append({"rule_id": "Net contents standard of fill", "category": "Alcohol & contents", "status": "needs_review",
                            "message": f"Net contents '{label_net}' is not a TTB authorized standard of fill.", "bbox_ref": bbox_net,
                            "extracted_value": label_net, "app_value": app_net})
        if app_ml is not None and label_ml is not None and abs(app_ml - label_ml) > 5:
            ocr_hint = " — likely OCR misread, verify manually" if _is_ocr_confusable(label_net, app_net) else ""
            results.append({"rule_id": "Net contents matches", "category": "Alcohol & contents", "status": "needs_review",
                            "message": f"Net contents on label ({label_net} ≈ {label_ml} mL) does not match application ({app_net} ≈ {app_ml} mL){ocr_hint}.", "bbox_ref": bbox_net,
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

    # Normalize for comparison: OCR errors + punctuation (OCR often drops commas)
    def _normalize_warning_ocr(s: str) -> str:
        s = s.upper()
        s = re.sub(r"\bOM\s*$", " GENERAL", s)
        # OCR misreads of "GENERAL": om, anal, geral, generai, etc.
        s = re.sub(r"SURGEON\s+OM\b", "SURGEON GENERAL", s, flags=re.I)
        s = re.sub(r"SURGEON\s+ANAL\b", "SURGEON GENERAL", s, flags=re.I)
        s = re.sub(r"SURGEON\s+GERAL\b", "SURGEON GENERAL", s, flags=re.I)
        s = re.sub(r"\bi\)\s*", "(1) ", s)
        s = re.sub(r"\bl\)\s*", "(2) ", s)
        s = re.sub(r"\(1\s*\)", "(1)", s)
        s = re.sub(r"\(2\s*\)", "(2)", s)
        # Line-break hyphens
        s = re.sub(r"PREG-\s*NANCY", "PREGNANCY", s)
        s = re.sub(r"MACHIN-\s*ERY", "MACHINERY", s, flags=re.I)
        # Compound OCR errors
        s = re.sub(r"HEALTHIPROBLEMS", "HEALTH PROBLEMS", s, flags=re.I)
        s = re.sub(r"HEALTHPROBLEMS", "HEALTH PROBLEMS", s, flags=re.I)
        # OCR artifacts: |) 7] etc
        s = re.sub(r"\|\s*\)", " ", s)
        s = re.sub(r"\d+\s*\]", " ", s)
        s = " ".join(s.split())
        # Treat "GENERAL, WOMEN" and "GENERAL WOMEN" as same (OCR often drops commas)
        s = re.sub(r",\s*", " ", s)
        s = " ".join(s.split())
        # Strip trailing 1–2 char junk (e.g. "ft")
        s = re.sub(r"\s+[A-Z]{1,2}\s*$", "", s)
        return " ".join(s.split())

    full_text_norm = _normalize_warning_ocr(full_text)
    required_norm = _normalize_warning_ocr(required_full) if required_full else ""
    full_stripped = full_text_norm.strip()
    # (1) Surgeon General / pregnancy; (2) consumption impairs / drive / machinery / health
    key_phrases_1 = ("(1)", "SURGEON GENERAL", "ACCORDING TO THE")
    key_phrases_2 = ("(2)", "IMPAIRS", "OPERATE MACHINERY", "HEALTH PROBLEMS")
    has_statement_1 = all(p in full_text_norm for p in key_phrases_1)
    has_statement_2 = all(p in full_text_norm for p in key_phrases_2)
    has_both_statements = has_statement_1 and has_statement_2

    # Word-level: all required words present, no extra words, critical phrases in order, Surgeon General capitalized
    def _warning_words(s: str) -> list[str]:
        return re.findall(r"\b\w+\b", (s or "").upper())

    from collections import Counter
    req_words = _warning_words(required_norm)
    ext_words = _warning_words(full_text_norm)
    req_counts = Counter(req_words)
    ext_counts = Counter(ext_words)
    all_required_present = all(ext_counts.get(w, 0) >= c for w, c in req_counts.items())
    extra_unique = [w for w in set(ext_words) if w not in req_counts]
    no_extra = len(extra_unique) == 0  # no extra words allowed

    # Critical phrases must appear in order with at most 2 intervening tokens (catches "BIRTH — os LG ... DEFECTS")
    def _phrase_gap_ok(text_norm: str, phrase: str, max_gap: int = 2) -> bool:
        """True if phrase words appear consecutively or with at most max_gap tokens between."""
        words = phrase.upper().split()
        tokens = _warning_words(text_norm)
        j = 0
        for w in words:
            found = False
            for _ in range(max_gap + 1):
                if j >= len(tokens):
                    return False
                if tokens[j] == w:
                    j += 1
                    found = True
                    break
                j += 1
            if not found:
                return False
        return True

    critical_phrases = ("BIRTH DEFECTS", "HEALTH PROBLEMS")
    phrases_in_order = all(_phrase_gap_ok(full_text_norm, p) for p in critical_phrases)

    # Surgeon General: S and G must be capital (Surgeon, SURGEON, General, GENERAL)
    has_surgeon_general_caps = bool(
        re.search(r"\bS(?:urgeon|URGEON)\b", full_text)
        and re.search(r"\bG(?:eneral|ENERAL)\b", full_text)
    )

    suspicious = _get_suspicious_warning_tokens(extra_unique, set(req_words))
    ref_words = set(req_words)
    filtered_text = _filter_suspicious_from_warning(full_text or "", suspicious) or full_text or ""
    display_extracted = _apply_spell_correction_to_warning(filtered_text, ref_words) or filtered_text
    required_in_label = required_norm and required_norm in full_text_norm
    label_is_prefix = required_norm and full_stripped and required_norm.startswith(full_stripped) and len(full_stripped) >= 50
    if required_full and not required_in_label and not label_is_prefix:
        if len(full_text_norm) < 50:
            results.append({"rule_id": "Exact warning wording", "category": "Warning", "status": "fail",
                            "message": "Warning text appears incomplete or incorrect.", "bbox_ref": bbox_warn,
                            "extracted_value": display_extracted, "app_value": required_full})
        elif has_both_statements and all_required_present and no_extra and phrases_in_order and has_surgeon_general_caps:
            results.append({"rule_id": "Exact warning wording", "category": "Warning", "status": "pass",
                            "message": "All required words present, no extra words, Surgeon General capitalized.", "bbox_ref": bbox_warn,
                            "extracted_value": display_extracted, "app_value": required_full})
        elif has_both_statements:
            reasons = []
            if not all_required_present:
                reasons.append("missing required words")
            if not no_extra:
                reasons.append(f"extra words: {extra_unique[:5]}")
            if suspicious:
                reasons.append(f"suspicious tokens (likely OCR errors): {', '.join(suspicious[:5])}")
            if not phrases_in_order:
                reasons.append("critical phrases (BIRTH DEFECTS, HEALTH PROBLEMS) out of order or too far apart")
            if not has_surgeon_general_caps:
                reasons.append("Surgeon General not capitalized")
            results.append({"rule_id": "Exact warning wording", "category": "Warning", "status": "needs_review",
                            "message": f"Key phrases present but: {'; '.join(reasons)}.", "bbox_ref": bbox_warn,
                            "extracted_value": display_extracted, "app_value": required_full})
        elif has_statement_1 or has_statement_2:
            missing = "statement (2)" if not has_statement_2 else "statement (1)"
            results.append({"rule_id": "Exact warning wording", "category": "Warning", "status": "needs_review",
                            "message": f"Only one statement found; both (1) and (2) required. Missing: {missing}.", "bbox_ref": bbox_warn,
                            "extracted_value": display_extracted, "app_value": required_full})
        else:
            msg = "Warning text may not match required wording exactly."
            if suspicious:
                msg += f" Suspicious tokens (likely OCR errors): {', '.join(suspicious[:5])}."
            results.append({"rule_id": "Exact warning wording", "category": "Warning", "status": "needs_review",
                            "message": msg, "bbox_ref": bbox_warn,
                            "extracted_value": display_extracted, "app_value": required_full})
    else:
        results.append({"rule_id": "Exact warning wording", "category": "Warning", "status": "pass",
                        "message": "Warning statement present and appears complete.", "bbox_ref": bbox_warn,
                        "extracted_value": display_extracted, "app_value": required_full if required_full else "Required statement"})

    return results


def _rules_origin(extracted: dict, app_data: dict, config: dict) -> list[dict]:
    results = []
    bottler_label = _norm(extracted.get("bottler", {}).get("value", ""))
    bottler_app = _norm(app_data.get("bottler_name", ""))
    bbox_bottler = extracted.get("bottler", {}).get("bbox")
    all_text = " ".join(b.get("text", "") for b in extracted.get("_all_blocks", []))

    if not bottler_label:
        if bottler_app:
            if _app_tokens_in_full_text(bottler_app, all_text):
                results.append({"rule_id": "Bottler/producer statement", "category": "Origin", "status": "pass",
                                "message": "Bottler/producer found on label (tokens in full text).", "bbox_ref": bbox_bottler,
                                "extracted_value": "", "app_value": bottler_app, "prefer_app_display": True})
            else:
                brand_label = _norm(extracted.get("brand_name", {}).get("value", ""))
                score, _ = _smart_match(bottler_app, brand_label, config)
                if score >= 0.75:
                    results.append({"rule_id": "Bottler/producer statement", "category": "Origin", "status": "needs_review",
                                    "message": "Bottler same as brand (no separate bottler line); verify on label.", "bbox_ref": bbox_bottler,
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
        results.append({"rule_id": "Bottler/producer statement", "category": "Origin", "status": "needs_review",
                        "message": "Bottler/producer found on label — add to application for verification.", "bbox_ref": bbox_bottler,
                        "extracted_value": bottler_label, "app_value": bottler_app})
    else:
        score, reason = _smart_match(bottler_app, bottler_label, config)
        if score >= 0.80:
            results.append({"rule_id": "Bottler/producer statement", "category": "Origin", "status": "pass",
                            "message": f"Bottler/producer matches ({reason}).", "bbox_ref": bbox_bottler,
                            "extracted_value": bottler_label, "app_value": bottler_app})
        elif bottler_app and _app_tokens_in_full_text(bottler_app, all_text):
            results.append({"rule_id": "Bottler/producer statement", "category": "Origin", "status": "pass",
                            "message": "Bottler/producer found on label (tokens in full text).", "bbox_ref": bbox_bottler,
                            "extracted_value": bottler_label, "app_value": bottler_app, "prefer_app_display": True})
        elif score >= 0.60:
            results.append({"rule_id": "Bottler matches", "category": "Origin", "status": "needs_review",
                            "message": f"Bottler on label may not match application ({reason}, {score:.0%}).", "bbox_ref": bbox_bottler,
                            "extracted_value": bottler_label, "app_value": bottler_app})
        else:
            ocr_hint = " — likely OCR misread" if _is_ocr_confusable(bottler_app, bottler_label) else ""
            results.append({"rule_id": "Bottler matches", "category": "Origin", "status": "needs_review",
                            "message": f"Bottler on label differs from application ({score:.0%}){ocr_hint}.", "bbox_ref": bbox_bottler,
                            "extracted_value": bottler_label, "app_value": bottler_app})

    # Bottler city/state cross-validation
    bottler_city_app = _norm(app_data.get("bottler_city", ""))
    bottler_state_app = _norm(app_data.get("bottler_state", ""))
    if bottler_label and (bottler_city_app or bottler_state_app):
        bl_lower = bottler_label.lower()
        missing_parts = []
        if bottler_city_app and bottler_city_app.lower() not in bl_lower:
            missing_parts.append(f"city '{bottler_city_app}'")
        if bottler_state_app and bottler_state_app.lower() not in bl_lower:
            missing_parts.append(f"state '{bottler_state_app}'")
        if missing_parts:
            results.append({"rule_id": "Bottler address", "category": "Origin", "status": "needs_review",
                            "message": f"Bottler {', '.join(missing_parts)} not found on label.", "bbox_ref": bbox_bottler,
                            "extracted_value": bottler_label, "app_value": f"{bottler_city_app}, {bottler_state_app}"})
        else:
            results.append({"rule_id": "Bottler address", "category": "Origin", "status": "pass",
                            "message": "Bottler city/state found on label.", "bbox_ref": bbox_bottler,
                            "extracted_value": bottler_label, "app_value": f"{bottler_city_app}, {bottler_state_app}"})

    if app_data.get("imported"):
        co = _norm(extracted.get("country_of_origin", {}).get("value", ""))
        co_app = _norm(app_data.get("country_of_origin", ""))
        bbox_co = extracted.get("country_of_origin", {}).get("bbox")
        if not co:
            results.append({"rule_id": "Country of origin", "category": "Origin", "status": "fail",
                            "message": "Imported product must show country of origin.", "bbox_ref": bbox_co,
                            "extracted_value": "", "app_value": co_app})
        elif co_app and co_app.lower() not in co.lower() and co.lower() not in co_app.lower():
            if _is_ocr_confusable(co, co_app):
                results.append({"rule_id": "Country of origin matches", "category": "Origin", "status": "needs_review",
                                "message": f"Country on label '{co}' differs from application '{co_app}' — likely OCR misread, verify manually.", "bbox_ref": bbox_co,
                                "extracted_value": co, "app_value": co_app})
            elif _normalize_ocr_for_text(co) == _normalize_ocr_for_text(co_app):
                results.append({"rule_id": "Country of origin", "category": "Origin", "status": "pass",
                                "message": f"Country of origin matches after OCR normalization: {co}.", "bbox_ref": bbox_co,
                                "extracted_value": co, "app_value": co_app})
            else:
                results.append({"rule_id": "Country of origin matches", "category": "Origin", "status": "needs_review",
                                "message": f"Country on label '{co}' may not match application '{co_app}'.", "bbox_ref": bbox_co,
                                "extracted_value": co, "app_value": co_app})
        elif not co_app:
            results.append({"rule_id": "Country of origin", "category": "Origin", "status": "needs_review",
                            "message": f"Country of origin found on label — add to application for verification.", "bbox_ref": bbox_co,
                            "extracted_value": co, "app_value": co_app})
        else:
            results.append({"rule_id": "Country of origin", "category": "Origin", "status": "pass",
                            "message": f"Country of origin found: {co}.", "bbox_ref": bbox_co,
                            "extracted_value": co, "app_value": co_app})

    return results


def _infer_conditionals_from_class(class_type: str, config: dict) -> set[str]:
    """Return set of conditional keys auto-required by spirit class (e.g. state_of_distillation)."""
    if not class_type:
        return set()
    ct_lower = class_type.lower()
    required: set[str] = set()
    for _rule_group in (config.get("spirit_class_rules") or {}).values():
        for kw in _rule_group.get("keywords", []):
            if kw.lower() in ct_lower:
                required.update(_rule_group.get("require", []))
    return required


def _is_whisky(class_type: str) -> bool:
    """True if class_type indicates whisky (27 CFR 5.40(a) applies)."""
    if not class_type:
        return False
    ct = class_type.lower()
    return "whiskey" in ct or "whisky" in ct


def _parse_age_years_from_label(blocks_lower: str) -> int | None:
    """Extract youngest age in years from label (e.g. '4 years', '12 year'). Returns None if not found."""
    matches = list(re.finditer(r"\d+\s*years?\b", blocks_lower))
    if not matches:
        return None
    ages = []
    for m in matches:
        num = re.match(r"\d+", m.group())
        if num:
            ages.append(int(num.group()))
    return min(ages) if ages else None


def _get_age_years(app_data: dict, blocks_lower: str) -> int | None:
    """Age in years (youngest for blends). From app_data or parsed from label. 0 = unknown."""
    # App data: age_years or youngest_age_years (for blends)
    for key in ("youngest_age_years", "age_years"):
        val = app_data.get(key)
        if val is not None and val != "":
            try:
                n = int(float(str(val).strip()))
                if n > 0:
                    return n
            except (ValueError, TypeError):
                pass
    return _parse_age_years_from_label(blocks_lower)


def _rules_other(extracted: dict, app_data: dict, config: dict, beverage_type: str = "spirits") -> list[dict]:
    results = []
    all_blocks_text = " ".join(b.get("text", "") for b in extracted.get("_all_blocks", []))
    blocks_lower = all_blocks_text.lower()

    class_label = _norm(extracted.get("class_type", {}).get("value", ""))
    class_app = _norm(app_data.get("class_type", ""))
    inferred = _infer_conditionals_from_class(class_app or class_label, config)

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

    _wood_required = app_data.get("wood_treatment_required") or ("wood_treatment" in inferred)
    if _wood_required and beverage_type in ("spirits", "distilled_spirits"):
        found = "treated" in blocks_lower or "wood" in blocks_lower
        results.append({"rule_id": "Wood treatment", "category": "Other",
                        "status": "pass" if found else "fail",
                        "message": "Wood treatment statement found." if found else "Wood treatment statement required but not found.",
                        "bbox_ref": None, "extracted_value": "Found" if found else "", "app_value": "Required"})

    # 27 CFR 5.40(a): Age statement required for whisky aged < 4 years; optional for 4+ years.
    class_type = class_app or class_label
    is_whisky = beverage_type in ("spirits", "distilled_spirits") and _is_whisky(class_type)
    age_cfg = config.get("age_statement") or {}
    threshold = age_cfg.get("whisky_age_threshold_years", 4)
    age_years = _get_age_years(app_data, blocks_lower)

    if is_whisky:
        # 27 CFR 5.40(a): required if age < threshold or age unknown; optional if age >= threshold
        _age_required = (
            app_data.get("age_statement_required")
            or age_years is None
            or age_years < threshold
        )
    else:
        # Non-whisky: only if explicitly required
        _age_required = bool(app_data.get("age_statement_required"))

    if _age_required:
        # Word-boundary to avoid false positives (e.g. "cooperage", "percentage")
        aged_match = re.search(r"\baged\b", blocks_lower)
        years_match = re.search(r"\d+\s*years?\b", blocks_lower)
        age_stmt_match = re.search(r"age\s+statement", blocks_lower)
        found = bool(aged_match or years_match or age_stmt_match)
        snippet = ""
        if aged_match:
            start = max(0, aged_match.start() - 10)
            snippet = blocks_lower[start : aged_match.end() + 30].strip()
        elif years_match:
            start = max(0, years_match.start() - 15)
            snippet = blocks_lower[start : years_match.end() + 5].strip()
        elif age_stmt_match:
            snippet = blocks_lower[age_stmt_match.start() : age_stmt_match.end() + 20].strip()
        if snippet:
            snippet = snippet[:50] + ("..." if len(snippet) > 50 else "")
        msg = "Age statement found." if found else "Age statement required but not found."
        if is_whisky and age_years is None:
            msg = msg + " (27 CFR 5.40(a): required for whisky aged < 4 years.)"
        results.append({"rule_id": "Age statement", "category": "Other",
                        "status": "pass" if found else "fail",
                        "message": msg,
                        "bbox_ref": None, "extracted_value": snippet or ("Found" if found else ""), "app_value": "Required"})

    _neutral_required = app_data.get("neutral_spirits_required") or ("neutral_spirits" in inferred)
    if _neutral_required and beverage_type in ("spirits", "distilled_spirits"):
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
