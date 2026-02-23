# Approach and design

## Purpose

TTB agents review ~150k label applications per year. This app reduces routine "does the label match the form?" work by running local OCR and a rule engine, then presenting a human-review-friendly checklist with Pass / Needs review / Fail and visual linking to the label.

## Architecture

- **Single codebase**: Streamlit app plus Python modules (OCR, extraction, rules, scoring). No separate backend service.
- **Local OCR**: Tesseract only; no cloud APIs (avoids firewall and latency issues; target ~5 s per label).
- **Rule engine**: Pure Python; reads `config/rules.yaml` for similarity thresholds, warning text, and patterns. Rules return status, message, and bbox_ref for "Show on label".
- **Scoring**: Any Fail → "Critical issues"; else any Needs review → "Needs review"; else "Ready to approve".

## Data flow

1. **Input**: Label image (file/bytes) + application data (form or CSV row).
2. **OCR**: Preprocess (resize, optional contrast) → Tesseract → list of text blocks with bbox and confidence.
3. **Extraction**: Heuristics and regex to find candidates for brand, class/type, ABV, proof, net contents, government warning, bottler, country of origin.
4. **Rules**: Compare extracted values to application and config; fuzzy match for brand/class; exact/semantic checks for warning and origin.
5. **Output**: Checklist by category + overall status + bbox refs for UI highlighting.

## TTB checklist coverage

- **Identity**: Brand and class/type present and match (fuzzy).
- **Alcohol & contents**: ABV and proof present and match; net contents present and metric.
- **Warning**: Full statement present; "GOVERNMENT WARNING" in caps; wording per config.
- **Origin**: Bottler/producer; if imported, country of origin.
- **Other**: Conditional statements (sulfites, colorings, wood, age, neutral spirits) when required by application flags.

## Limitations

- Standard of fill: fixed list in config, not full CFR logic.
- Emphasized warning: caps and distinct block only; no font-size detection.
- Batch: sequential processing; no persistence.
- Test labels: use generated or sourced images; no built-in dataset.

## Extensibility

- Swap or add OCR by replacing `ocr.run_ocr()`.
- Add rules or categories by extending `rules/engine.py` and `config/rules.yaml`.
- Expose `pipeline.run_pipeline()` as an API later for COLA integration.
