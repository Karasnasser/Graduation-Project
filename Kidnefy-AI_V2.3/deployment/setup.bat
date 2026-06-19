@echo off
echo ============================================
echo  Kidney Disease Prediction System - Setup
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed! Please install Python 3.9-3.11
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate

echo [2/4] Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt

echo [3/4] Setting up environment...
if not exist .env (
    copy .env.example .env
    echo [INFO] Created .env file. Please edit it and add your GEMINI_API_KEY
    echo [INFO] Get a free key from: https://aistudio.google.com/app/apikey
) else (
    echo [INFO] .env file already exists, skipping...
)

:: Create necessary directories
if not exist generated_reports mkdir generated_reports
if not exist uploads mkdir uploads

echo [4/4] Setup complete!
echo.
echo ============================================
echo  To start the API server, run:
echo    .venv\Scripts\activate
echo    uvicorn api:app --host 0.0.0.0 --port 8000
echo.
echo  API Docs: http://localhost:8000/docs
echo ============================================
pause
