# BottleProof

TTB label compliance verification prototype. Upload a label image and application data — BottleProof runs OCR, extracts key fields, and checks them against what you submitted. Pass / Needs review / Fail per field, with a checklist.

**Live demo:** [bottleproof.streamlit.app](https://bottleproof.streamlit.app/)

**Full documentation:** [docs/APPROACH.md](docs/APPROACH.md) — approach, tools, assumptions, limitations.

---

## Quick start

Requires **Python 3.10+** and **Tesseract OCR**.

### 1. Clone and setup

**Option A — Python (cross-platform):**
```bash
git clone https://github.com/lapardhaja/Alcohol-Label.git
cd Alcohol-Label
python scripts/setup.py    # or python3 on Mac/Linux
python scripts/run.py      # start the app
```

**Option B — Platform scripts:**
- **Windows:** `setup.bat` then `run.bat`
- **Mac/Linux:** `./setup.sh` then `./run.sh`

### 2. Install Tesseract OCR

Tesseract reads text from label images. Install on your system:

- **Windows:** [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) — install to `C:\Program Files\Tesseract-OCR`
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

If installed elsewhere, add it to PATH.

### 3. Optional: E2E test

For the approve-flow test (`python tests/test_approve_flow.py`):
```bash
playwright install
```

### 4. Run

```bash
python scripts/run.py
# or: streamlit run app.py (with venv activated)
```

Browser opens at `http://localhost:8501`.

---

## How to use

**Single label:** Upload an image, fill application fields (brand, class/type, ABV, proof, net contents, bottler, etc.), click **Check label**. Use the preset dropdown to load one of seven test presets (spirits, beer, wine) that match sample images in `sample_data/`. Get overall status and checklist by category.

**Batch:** Upload a ZIP of images + CSV with one row per label. CSV must have `label_id` matching filename (e.g. `test_1.png` → `test_1`). Filter by status, drill into details.

**CSV columns:** `label_id`, `brand_name`, `class_type`, `alcohol_pct`, `proof`, `net_contents_ml`, `bottler_name`, `bottler_city`, `bottler_state`, `imported`, `country_of_origin`, `beverage_type`. See `sample_data/batch_example.csv`. Sample images in `sample_data/` and `sample_data/test_images.zip`.

**Approve flow:** Move applications between Under review / Approved / Rejected. State saved in `data/applications.json` (local, offline).

---

## Tech stack

| Component | Tool |
|-----------|------|
| UI | Streamlit |
| OCR | Tesseract (local) |
| Preprocessing | OpenCV, Pillow |
| Rules | YAML config, rapidfuzz |
| Batch CSV | pandas |

**Flow:** Image → preprocess → OCR → extract fields → run rules → score (Ready to approve / Needs review / Critical issues). Brand and class/type use fuzzy matching; small differences may pass or flag needs review.

---

## Deploy (Streamlit Cloud)

Uses `packages.txt` and `sources.list` to install Tesseract 5.3. Point the app at this repo, run `streamlit run app.py` from root.

---

Label compliance verification prototype. Developed by Servet Lapardhaja.
