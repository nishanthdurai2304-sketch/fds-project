@echo off
title Push AI Footfall Project to GitHub
cd /d "%~dp0"

echo ======================================================================
echo             Push AI Footfall Counter to GitHub
echo ======================================================================
echo.

set GIT_CMD="C:\Users\DREAMZ\AppData\Local\Programs\Git\cmd\git.exe"

set /p REPO_URL="Enter your GitHub Repository URL (e.g. https://github.com/username/ai-footfall-counter.git): "

if "%REPO_URL%"=="" (
    echo [ERROR] No URL entered. Aborting.
    pause
    exit /b
)

echo.
echo [1/3] Setting remote origin...
%GIT_CMD% remote remove origin 2>nul
%GIT_CMD% remote add origin %REPO_URL%

echo [2/3] Setting default branch to main...
%GIT_CMD% branch -M main

echo [3/3] Pushing to GitHub...
%GIT_CMD% push -u origin main

echo.
if %errorlevel% equ 0 (
    echo ======================================================================
    echo   SUCCESS! Your project has been pushed to GitHub successfully!
    echo ======================================================================
) else (
    echo [NOTE] If prompted for credentials, log in with your GitHub account or Personal Access Token.
)

pause
