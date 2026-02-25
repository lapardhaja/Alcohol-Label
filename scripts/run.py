#!/usr/bin/env python3
"""Cross-platform launcher. Run from project root: python scripts/run.py"""
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
    py = venv_python()
    if not os.path.isfile(py):
        print(".venv not found. Run: python scripts/setup.py")
        sys.exit(1)
    subprocess.run([py, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()
