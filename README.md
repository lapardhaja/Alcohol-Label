# BottleProof

A prototype app for TTB label compliance agents. Upload a label image and application data—BottleProof runs OCR on the label, extracts the key fields, and checks them against what you submitted. Pass / Needs review / Fail, with a checklist and "Show on label" highlighting so you can see exactly what the system picked up.

**Try it live:** [bottleproof.streamlit.app](https://bottleproof.streamlit.app/)

---

## Quick start (run it locally)

You’ll need Python 3.10+ and Tesseract OCR installed.

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd "Alcohol Label"

python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 2. Install Tesseract OCR

The app uses Tesseract to read text from label images. Install it on your system first:

- **Windows:** Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and install to the default path (`C:\Program Files\Tesseract-OCR`). The app will find it there.
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

If you install Tesseract somewhere else, add that folder to your PATH.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

From the project root:

```bash
streamlit run app.py
```

Your browser will open (or go to the URL shown, usually `http://localhost:8501`).

---

## How to use it

**Single label mode:** Upload a label image, fill in the application fields (brand, class/type, ABV, proof, net contents, bottler, etc.), then click **Check label**. You’ll get an overall status and a checklist by category. Use **Show on label** to see where each field was found on the image.

**Batch mode:** Upload a ZIP of label images and a CSV with one row per label. The CSV needs a `label_id` column that matches the filename without extension (e.g. `test_1.png` → `test_1`). Click **Check labels**, then filter by status and use **Show details** to drill into any label.

Example CSV columns: `label_id`, `brand_name`, `class_type`, `alcohol_pct`, `proof`, `net_contents_ml`, `bottler_name`, `bottler_city`, `bottler_state`, `imported`, `country_of_origin`, `beverage_type`. See `sample_data/batch_example.csv` for a full example. Sample images are in `sample_data/` and `sample_data/test_images.zip`.

---

## Tech notes

- **Stack:** Streamlit, Tesseract (local OCR), Pillow/OpenCV for preprocessing, YAML-driven rules, rapidfuzz for fuzzy matching.
- **Flow:** Image → preprocess → OCR → extract fields → run rules → score (Ready to approve / Needs review / Critical issues).
- Brand and class/type use fuzzy matching, so small differences like "STONE'S THROW" vs "Stone's Throw" can still pass or get a needs-review flag instead of a hard fail.
- Results are in-session only; nothing is persisted. No COLA integration.

---

## Deploying to Streamlit Cloud

The app uses `packages.txt` and `sources.list` to install Tesseract 5.3 on Streamlit Community Cloud. Point the app at this repo and run `streamlit run app.py` from the root. See the [Debian 11 EOL thread](https://discuss.streamlit.io/t/debian-11-eol/80690) for context on the Tesseract setup.

---

Take-home prototype. Not for production use without TTB approval.
