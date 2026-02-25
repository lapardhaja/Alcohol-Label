#!/usr/bin/env python3
"""Cross-platform setup. Run: python setup.py"""
import os
import platform
import subprocess
import sys

def venv_python():
    if platform.system() == "Windows":
        return os.path.join(".venv", "Scripts", "python.exe")
    return os.path.join(".venv", "bin", "python")

def main():
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
    print("\nRun the app with: python run.py")

if __name__ == "__main__":
    main()
