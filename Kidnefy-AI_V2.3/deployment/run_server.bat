@echo off
echo ============================================
echo  Starting Kidney Disease Prediction API...
echo ============================================
echo.

:: Activate virtual environment
call .venv\Scripts\activate

:: Start server
echo Starting server at http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Press Ctrl+C to stop the server
echo.
uvicorn api:app --host 0.0.0.0 --port 8000
