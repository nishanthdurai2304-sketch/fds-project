@echo off
title AI Footfall - Smart Customer Analytics System
echo ========================================================
echo    AI Footfall - Smart Customer Analytics System
echo ========================================================
echo.
echo [1/2] Checking Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    pause
    exit /b
)

echo [2/2] Launching AI Footfall Server...
echo Open your browser at: http://localhost:5000/
echo.
python app.py
pause
