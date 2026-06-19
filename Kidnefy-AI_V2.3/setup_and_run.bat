@echo off
echo ===================================================
echo     Kidnefy-AI: Automated Setup and Run Script
echo ===================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH! Please install Python 3.9 - 3.11
    pause
    exit /b
)

:: 2. Create Virtual Environment if it doesn't exist
if not exist ".venv\Scripts\activate.bat" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
)

:: 3. Activate Virtual Environment
echo [*] Activating virtual environment...
call .venv\Scripts\activate.bat

:: 4. Install Dependencies
echo [*] Installing dependencies from requirements.txt...
pip install -r requirements.txt

:: 5. Handle .env file
if not exist ".env" (
    echo [*] .env file not found. Creating one from .env.example...
    copy .env.example .env
    echo.
    echo [WARNING] I created a .env file for you. 
    echo Please open it and add your GEMINI_API_KEY if you want the Chatbot to work!
    echo.
    pause
)

:: 6. Run the Server
echo [*] Starting FastAPI Server...
echo [*] Once the server starts, open your browser and go to: http://127.0.0.1:8000/docs
echo.
uvicorn api:app --host 127.0.0.1 --port 8000 --reload

pause
