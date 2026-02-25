#!/usr/bin/env python3
"""Cross-platform launcher. Run: python run.py"""
import os
import platform
import subprocess
import sys

def venv_python():
    if platform.system() == "Windows":
        return os.path.join(".venv", "Scripts", "python.exe")
    return os.path.join(".venv", "bin", "python")

def main():
    py = venv_python()
    if not os.path.isfile(py):
        print(".venv not found. Run: python setup.py")
        sys.exit(1)
    subprocess.run([py, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()
