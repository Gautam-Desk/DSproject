@echo off
echo ====================================================================
echo Starting VeritasAI - Fake News Detection Server (TensorFlow)
echo ====================================================================
echo.
echo Checking virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo Error: .venv not found. Please install Python 3.11 with uv.
    pause
    exit /b 1
)

echo Starting FastAPI & Uvicorn server on http://localhost:8000
echo Network accessibility: http://0.0.0.0:8000
echo.
.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
