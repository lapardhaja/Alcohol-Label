# BottleProof — Approach, Tools, Assumptions & Limitations

## What We Are Doing

BottleProof is a TTB label compliance verification prototype. It checks whether a label image matches the application data the producer submitted. TTB agents handle around 150k label applications per year; this automates the routine comparison work.

Given a label image and application data, the system runs OCR, extracts key fields (brand, class/type, ABV, proof, net contents, government warning, bottler, country of origin), compares them against the application using configurable rules, and outputs a checklist with Pass / Needs review / Fail per field. Overall status is Ready to approve, Needs review, or Critical issues.

---

## How It Works

### Process

You upload a label image and the application data. The system:

1. Preprocesses the image — deskew, CLAHE contrast, sharpen, binarize — so OCR has a cleaner input.
2. Runs Tesseract OCR in multiple passes (PSM 3 and 6 on original, sharpened, and binary versions) to read text from the label.
3. Deduplicates overlapping or repeated text blocks.
4. Extracts fields from the OCR output using layout heuristics, regex, and common patterns (e.g. "Distilled by" before bottler name).
5. Compares extracted values to the application. Exact match = Pass, close match = Needs review, mismatch or missing = Fail.
6. Assigns overall status: any Fail → Critical issues; any Needs review → Needs review; otherwise Ready to approve.

You get a checklist per field plus the overall status.

### Single vs batch mode

Single label: upload one image, fill the form, click Check label. Batch: upload a ZIP of images and a CSV with one row per label. The CSV must have a `label_id` column matching the filename (e.g. `test_1.png` → `test_1`). You can filter by status and drill into any label.

### Approve flow

Applications can be moved between Under review, Approved, and Rejected. State is stored in `data/applications.json` (local, offline).

### Technical flow

```
Label image + Application data
    → Preprocess (resize, deskew, CLAHE, sharpen, binarize)
    → Tesseract OCR (multi-pass: PSM 3, 6 on original/sharpened/binary)
    → Deduplicate blocks (IoU + fuzzy similarity)
    → Field extraction (heuristics, regex, spatial layout)
    → Rule engine (compare extracted vs application; config-driven thresholds)
    → Scoring (any Fail → Critical; any Needs review → Needs review; else Ready to approve)
    → UI: checklist
```

---

## Innovations

The system is designed for speed and agent efficiency. It runs entirely locally (no cloud API calls), so a single label check completes in seconds. That gives agents a fast first scan on the essential fields — brand, class/type, ABV, proof, net contents, government warning, bottler, origin — before they dig into details. For high volume, that first-pass triage saves time.

Technical innovations:

- **Multi-pass OCR** — Three Tesseract passes (original, CLAHE+sharpened, binarized) with different PSM modes. Each pass catches text the others miss; results are merged and deduplicated.
- **IoU + fuzzy deduplication** — Overlapping blocks from multiple passes are merged using IoU and fuzzy similarity. The highest-confidence block is kept, so you get the best read without duplicate noise.
- **Fuzzy matching with OCR tolerance** — Brand, class/type, and warning use rapidfuzz with configurable thresholds. OCR confusables (1/l, 0/O) and typos (Mat vs Malt) are handled so minor scan errors do not cause false fails.
- **Government warning spell correction** — pyspellchecker corrects common OCR errors in the warning text before comparison, reducing false needs-review on minor misspellings.
- **Config-driven rules** — Thresholds, patterns, and beverage-type rules live in `config/rules.yaml`. Tuning does not require code changes.
- **Seven presets** — Test presets for spirits, beer, and wine match the sample images so agents can try the flow immediately.

---

## Tools Used

<div style="overflow-x: auto; max-width: 100%;">

<table style="width: 100%; table-layout: fixed; overflow-wrap: break-word;">
<thead>
<tr><th style="width: 22%;">Tool</th><th style="width: 18%;">Version / Notes</th><th style="width: 60%;">Purpose</th></tr>
</thead>
<tbody>
<tr><td>Python</td><td>3.10+</td><td>Runtime</td></tr>
<tr><td>Streamlit</td><td>≥1.28</td><td>Web UI</td></tr>
<tr><td>Tesseract OCR</td><td>5.x (system-installed)</td><td>Text recognition from label images</td></tr>
<tr><td>pytesseract</td><td>≥0.3.10</td><td>Python bindings for Tesseract</td></tr>
<tr><td>OpenCV (opencv-python-headless)</td><td>≥4.8</td><td>Image preprocessing: deskew, CLAHE, sharpen, binarization (Otsu)</td></tr>
<tr><td>Pillow</td><td>≥10.0</td><td>Image I/O, resize</td></tr>
<tr><td>PyYAML</td><td>≥6.0</td><td>Load config/rules.yaml</td></tr>
<tr><td>rapidfuzz</td><td>≥3.0</td><td>Fuzzy string matching (brand, class/type, warning)</td></tr>
<tr><td>pyspellchecker</td><td>≥0.8.0</td><td>OCR error correction in government warning text</td></tr>
<tr><td>pandas</td><td>≥2.0</td><td>CSV parsing for batch mode</td></tr>
<tr><td>numpy</td><td>≥1.24</td><td>Array ops for preprocessing</td></tr>
<tr><td>pytest</td><td>≥7.0</td><td>Unit tests</td></tr>
<tr><td>Playwright</td><td>≥1.40</td><td>E2E approve-flow test (optional)</td></tr>
</tbody>
</table>

</div>

Deployment: Streamlit Cloud uses `packages.txt` and `sources.list` to install Tesseract 5.3 on Debian (bookworm).

---

## Assumptions

Input: Label images are flat (front/back photos or scans), not 3D bottle shots. Application data is entered manually (form or CSV); there is no COLA integration. Beverage type (spirits / wine / beer) is known and drives which rules apply.

OCR and extraction: Tesseract handles typical label layouts. Text is assumed left-to-right, top-to-bottom; multi-column layouts are split on large horizontal gaps. Government warning is usually in a distinct column; Serving Facts is filtered out. Brand extraction uses domain suffixes (Distillery, Brewery, Winery) and strips corp suffixes (Inc, LLC, Co). Class/type uses a keyword list from config (27 CFR Parts 4, 5, 7). Net contents supports metric (mL, L) and imperial (fl oz, qt, pt, gal); compound values like "1 PINT 8 FL OZ" are converted. Bottler is found via header patterns (Distilled and Bottled by, Produced by, etc.) or fallback.

Rules: Brand and class/type use rapidfuzz with configurable thresholds (pass ≥90%, needs review ≥70%). Government warning must contain required wording; minor OCR variations are allowed. Net contents are validated against a standard-of-fill list in config (27 CFR 5.203). Origin: bottler required; country of origin required when imported=true. Conditional statements (sulfites, FD&C Yellow No. 5, carmine, wood treatment, age, neutral spirits) are required when application flags say so.

Data: Batch results are in-session only. Approve flow is persisted in `data/applications.json`.

---

## Limitations

OCR: No font-size detection, so emphasized warning cannot be checked for size. Standard of fill uses a fixed list in config, not full CFR logic. Handwritten or low-quality images may fail. Non-standard layouts can confuse extraction. Class/type relies on a keyword list; novel variants may not match.

Rules: Emphasized warning is checked for caps and distinct block only; no font-size enforcement. Conditional statements are regex-based; complex phrasing may not match. Age statement uses a config-driven whisky threshold (4 years), not full CFR 5.40(a).

System: No COLA integration. Batch runs sequentially. Sample images are provided; no built-in benchmark dataset.

---

## Extensibility

OCR can be swapped (e.g. for a cloud API). Rules live in `rules/engine.py` and `config/rules.yaml`. `pipeline.run_pipeline()` can be exposed for COLA or external integration.

---

## File Structure

```
app.py                 # Entry point (streamlit run app.py)
scripts/
  run.py               # Cross-platform launcher
  setup.py             # Cross-platform setup
  compare_test_images.py
src/
  app.py               # Streamlit UI
  ocr.py               # Tesseract + preprocessing
  extraction.py        # Field extraction from OCR blocks
  pipeline.py          # Orchestration: OCR → extraction → rules → scoring
  scoring.py           # Overall status from rule results
  storage.py           # Local JSON for approve flow
  ui_utils.py          # UI helpers
  rules/
    engine.py          # Rule evaluation (identity, alcohol, warning, origin, other)
config/
  rules.yaml           # Thresholds, patterns, beverage types, conditionals
sample_data/
  batch_example.csv    # Example batch CSV
  test_1.jpg...       # Sample label images
  test_images.zip     # Same images for batch ZIP upload
```

---

## References

- 27 CFR Part 4 (Wine labels)
- 27 CFR Part 5 (Distilled spirits labels)
- 27 CFR Part 7 (Malt Beverage labels)
- TTB Beverage Alcohol Manual (BAM)
