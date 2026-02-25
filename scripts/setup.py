#!/usr/bin/env python3
"""Cross-platform setup. Run from project root: python scripts/setup.py"""
import os
import platform
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def venv_python():
    if platform.system() == "Windows":
        return os.path.join(".venv", "Scripts", "python.exe")
    return os.path.join(".venv", "bin", "python")

def main():
    os.chdir(_PROJECT_ROOT)
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)

    py = venv_python()
    print("Installing dependencies...")
    subprocess.run([py, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

    print("\nSetup complete. Install Tesseract OCR if not already:")
    if platform.system() == "Darwin":
        print("  brew install tesseract")
    elif platform.system() == "Linux":
        print("  sudo apt install tesseract-ocr")
    else:
        print("  https://github.com/UB-Mannheim/tesseract/wiki")
    print("\nRun the app with: python scripts/run.py")

if __name__ == "__main__":
    main()
