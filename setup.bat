@echo off
REM First-time setup for new PC. Run this once.
echo Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo Activating and installing dependencies...
call .venv\Scripts\activate.bat
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed
    pause
    exit /b 1
)

echo.
echo Setup complete. Install Tesseract OCR if not already:
echo   https://github.com/UB-Mannheim/tesseract/wiki
echo.
echo Run the app with: run.bat (or python run.py on any OS)
pause
