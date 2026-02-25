@echo off
REM Run BottleProof. Run setup.bat first on a new PC.
if not exist .venv\Scripts\activate.bat (
    echo .venv not found. Run setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python scripts\run.py
