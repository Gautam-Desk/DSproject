@echo off
echo ====================================================================
echo VeritasAI - Model Training & Benchmarking Pipeline
echo ====================================================================
echo.
echo 1. Generating & Cleaning News Datasets...
.venv\Scripts\python.exe prepare_data.py
if %errorlevel% neq 0 (
    echo Error in dataset preparation.
    pause
    exit /b 1
)

echo.
echo 2. Training Deep Learning Models (BiLSTM, CNN-BiLSTM, Transformer, TF-IDF)...
.venv\Scripts\python.exe model_training.py
if %errorlevel% neq 0 (
    echo Error in model training.
    pause
    exit /b 1
)

echo.
echo Training complete! Artifacts saved to models/ directory.
pause
