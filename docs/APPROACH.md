# BottleProof — Approach, Tools, Assumptions & Limitations

## What We Are Doing

BottleProof is a **TTB (Alcohol and Tobacco Tax and Trade Bureau) label compliance verification prototype**. It automates the routine "does the label match the application?" workflow that TTB agents perform on ~150k label applications per year.

**Core task:** Given a label image and the application data (what the producer submitted), the system:
1. Reads text from the label image via OCR
2. Extracts key fields (brand, class/type, ABV, proof, net contents, government warning, bottler, country of origin)
3. Compares extracted values to application data using configurable rules
4. Produces a human-review-friendly checklist with Pass / Needs review / Fail per field

**Output:** Overall status (Ready to approve / Needs review / Critical issues) and a per-field checklist.

---

## How It Works

### The process (plain language)

**What you do:** Upload a photo or scan of an alcohol label, plus the application data (what the producer submitted to TTB).

**What the system does:**

1. **Image cleanup** — Straightens the image, improves contrast and sharpness so text is easier to read. If the label is crooked or faded, this step helps.

2. **Text recognition (OCR)** — Reads all visible text from the label. It runs multiple passes with different settings to catch as much as possible.

3. **Merge duplicates** — Removes repeated or overlapping text blocks so each piece of text is counted once.

4. **Field extraction** — Figures out *which* text belongs to *which* field. For example: "Is this the brand name? The ABV? The government warning?" It uses layout rules (where things usually appear) and common patterns (e.g. "Distilled by" before a bottler name).

5. **Rule comparison** — Compares what it found to what the producer submitted. For each field: exact match = Pass, close match = Needs review, mismatch or missing = Fail.

6. **Overall verdict** — If any field fails → Critical issues. If any needs review → Needs review. Otherwise → Ready to approve.

**You get:** A checklist showing Pass/Needs review/Fail for each field, plus an overall status.

---

### Single vs batch mode

- **Single label:** Upload one image, fill in the application fields in the form, click "Check label". You get one result.
- **Batch:** Upload a ZIP of images plus a CSV with one row per label. The CSV needs a `label_id` column that matches the filename (e.g. `test_1.png` → `test_1`). Filter by status, drill into details for any label.

---

### Approve flow (optional)

The app supports an approve/reject workflow: applications can be moved between "Under review", "Approved", and "Rejected". State is saved in `data/applications.json` (local file, offline, no external APIs).

---

### Technical flow (for developers)

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

## Tools Used

<div style="overflow-x: auto; max-width: 100%;">

<table style="width: 100%; table-layout: fixed; overflow-wrap: break-word;">
<thead>
<tr><th style="width: 22%;">Tool</th><th style="width: 18%;">Version / Notes</th><th style="width: 60%;">Purpose</th></tr>
</thead>
<tbody>
<tr><td><strong>Python</strong></td><td>3.10+</td><td>Runtime</td></tr>
<tr><td><strong>Streamlit</strong></td><td>≥1.28</td><td>Web UI</td></tr>
<tr><td><strong>Tesseract OCR</strong></td><td>5.x (system-installed)</td><td>Text recognition from label images</td></tr>
<tr><td><strong>pytesseract</strong></td><td>≥0.3.10</td><td>Python bindings for Tesseract</td></tr>
<tr><td><strong>OpenCV</strong> (opencv-python-headless)</td><td>≥4.8</td><td>Image preprocessing: deskew, CLAHE, sharpen, binarization (Otsu)</td></tr>
<tr><td><strong>Pillow</strong></td><td>≥10.0</td><td>Image I/O, resize</td></tr>
<tr><td><strong>PyYAML</strong></td><td>≥6.0</td><td>Load <code>config/rules.yaml</code></td></tr>
<tr><td><strong>rapidfuzz</strong></td><td>≥3.0</td><td>Fuzzy string matching (brand, class/type, warning)</td></tr>
<tr><td><strong>pyspellchecker</strong></td><td>≥0.8.0</td><td>OCR error correction in government warning text</td></tr>
<tr><td><strong>pandas</strong></td><td>≥2.0</td><td>CSV parsing for batch mode</td></tr>
<tr><td><strong>numpy</strong></td><td>≥1.24</td><td>Array ops for preprocessing</td></tr>
<tr><td><strong>pytest</strong></td><td>≥7.0</td><td>Unit tests</td></tr>
<tr><td><strong>Playwright</strong></td><td>≥1.40</td><td>E2E approve-flow test (optional)</td></tr>
</tbody>
</table>

</div>

### Deployment

- **Streamlit Cloud:** `packages.txt` + `sources.list` install Tesseract 5.3 on Debian (bookworm).

---

## Assumptions Made

### Input data

- Label images are **flat** (front/back label photos or scans), not 3D bottle shots.
- Application data is provided **manually** (form or CSV) — no COLA integration.
- **Beverage type** is known (spirits / wine / beer) and drives which rules apply (e.g. proof optional for spirits, ABV optional for beer).

### OCR & extraction

- Tesseract is sufficient for typical label layout and fonts; no cloud OCR.
- Text is mostly **left-to-right, top-to-bottom**; multi-column layouts are handled by splitting on large horizontal gaps.
- **Government warning** is typically in a distinct column; Serving Facts / nutrition panel is separated by spatial filtering and keyword exclusion.
- **Brand** extraction uses domain suffixes (Distillery, Brewery, Winery, etc.) and strips corp suffixes (Inc, LLC, Co).
- **Class/type** uses a fixed keyword list (27 CFR Parts 4, 5, 7) from config; extraction stops at non-class content (ABV, bottler, etc.).
- **Net contents** supports metric (mL, L) and imperial (fl oz, qt, pt, gal); compound values (e.g. "1 PINT 8 FL OZ") are converted.
- **Bottler** is found via header patterns (Distilled and Bottled by, Produced by, etc.) or fallback (CompanyName, City, ST).

### Rules & scoring

- **Brand** and **class/type** use fuzzy matching (rapidfuzz) with configurable thresholds (pass ≥90%, needs review ≥70%).
- **Government warning** must contain required wording; semantic checks allow minor OCR variations.
- **Net contents** are validated against a fixed list of standard-of-fill values (27 CFR 5.203) in config.
- **Origin** rules: bottler required; country of origin required when `imported=true`.
- **Conditional statements** (sulfites, FD&C Yellow No. 5, carmine, wood treatment, age, neutral spirits) are required when application flags say so.

### Data & persistence

- **Batch results** are in-session only; no persistence.
- **Approve flow** is persisted in `data/applications.json` (gitignored).

---

## Limitations

### OCR & extraction

- **No font-size detection** — emphasized warning cannot be checked for font size.
- **Standard of fill** — fixed list in config, not full CFR logic.
- **Handwritten / low-quality images** — OCR may fail or produce poor output.
- **Non-standard layouts** — unusual label designs may confuse extraction heuristics.
- **Class/type** — relies on keyword list; novel or regional variants may not match.

### Rules

- **Emphasized warning** — only checks for caps and distinct block; no font-size enforcement.
- **Conditional statements** — regex-based; complex phrasing may not match.
- **Age statement** — whisky age threshold (4 years) is config-driven; no full CFR 5.40(a) logic.

### System

- **No COLA integration** — application data is manual; no lookup of approved labels.
- **Batch** — sequential processing; no parallelization.
- **Test labels** — sample images are provided; no built-in benchmark dataset.

---

## Extensibility

- **OCR:** Swap `ocr.run_ocr()` for another engine (e.g. cloud API).
- **Rules:** Extend `rules/engine.py` and `config/rules.yaml` for new categories or thresholds.
- **API:** Expose `pipeline.run_pipeline()` for COLA or external integration.

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
  test_1.jpg...        # Sample label images
  test_images.zip      # Same images for batch ZIP upload
```

---

## References

- 27 CFR Part 4 (Wine labels)
- 27 CFR Part 5 (Distilled spirits labels)
- 27 CFR Part 7 (Malt Beverage labels)
- TTB Beverage Alcohol Manual (BAM)
