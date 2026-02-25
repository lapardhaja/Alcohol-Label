#!/bin/bash
# First-time setup for Mac/Linux. Run: ./setup.sh or bash setup.sh
set -e
echo "Creating virtual environment..."
python3 -m venv .venv
echo "Installing dependencies..."
.venv/bin/pip install -r requirements.txt
echo ""
echo "Setup complete. Install Tesseract if needed:"
echo "  macOS: brew install tesseract"
echo "  Linux: sudo apt install tesseract-ocr"
echo ""
echo "Run the app with: ./run.sh (or python scripts/run.py on any OS)"
