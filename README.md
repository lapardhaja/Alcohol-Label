# TTB Alcohol Label Verification App

Streamlit prototype for TTB label compliance agents: verify distilled spirits labels against application data using **local OCR** (Tesseract) and a **config-driven rule engine**. Single-label and batch workflows; checklist with Pass / Needs review / Fail and "Show on label" highlighting.

## Setup

1. **Python 3.10+** and a virtual environment (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```

2. **Install Tesseract OCR** (required to read text from label images):

   - **Windows:** [Download installer](https://github.com/UB-Mannheim/tesseract/wiki). Install to the default folder (`C:\Program Files\Tesseract-OCR`); the app will find it automatically. If you install elsewhere, add that folder to your system PATH.
   - **macOS:** `brew install tesseract`
   - **Linux:** `sudo apt install tesseract-ocr` (or equivalent)

   The app looks for Tesseract in standard install locations first, so adding it to PATH is only needed if you installed to a custom location.

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app** (from project root):

   ```bash
   streamlit run app.py
   ```
   Or: `streamlit run src/app.py`

   Open the URL shown (e.g. http://localhost:8501).

## Usage

- **Single label**: Upload an image, fill application fields (brand, class/type, ABV, proof, net contents, bottler, etc.), click **Check label**. View overall status, checklist by category, and use **Show on label** to highlight the region used for each rule.
- **Batch review**: Upload a **ZIP** of label images and a **CSV** with one row per label. CSV must include `label_id` (matching filename without extension, e.g. `label_001.png` → `label_001`) and application columns. Click **Run batch checks**, then filter by status and use **View detail** to open the full checklist for a label.

Example CSV columns: `label_id`, `brand_name`, `class_type`, `alcohol_pct`, `proof`, `net_contents_ml`, `bottler_name`, `bottler_city`, `bottler_state`, `imported`, `country_of_origin`, and optional flags (`sulfites_required`, `fd_c_yellow_5_required`, etc.). See `sample_data/batch_example.csv`.

## Approach, tools, assumptions

- **Stack**: Streamlit (UI), Tesseract via `pytesseract` (local OCR), Pillow/OpenCV (preprocess), YAML config (rules/thresholds), `rapidfuzz` (fuzzy match for brand/class), pandas (batch CSV).
- **Pipeline**: Image → preprocess → OCR (blocks with bbox) → field extraction (brand, class/type, ABV, proof, net contents, warning, bottler, origin) → rule engine (Identity, Alcohol & contents, Warning, Origin, Other) → scoring (Ready to approve / Needs review / Critical issues).
- **Nuance**: Brand and class/type use similarity thresholds (e.g. ≥0.95 pass, 0.80–0.95 needs review, &lt;0.80 fail) so "STONES THROW" vs "Stones Throw" can pass or need review instead of hard fail.
- **Assumptions**: Standard of fill is a fixed list in config; "emphasized" warning is implemented as "GOVERNMENT WARNING" in caps in a distinct block; no DB—results in session only; batch runs sequentially.
- **Trade-offs**: If Tesseract is unavailable, the app shows a clear error and install instructions (no fake results); prototype does not persist results or integrate with COLA.

## Deploy (e.g. Streamlit Community Cloud)

- Add a `packages.txt` with `tesseract-ocr` if the platform supports it, or document that OCR may be mock unless Tesseract is installed.
- Run from repo root with `streamlit run src/app.py` and set working directory to the project root.

## License

Take-home prototype; no production use without TTB approval.
