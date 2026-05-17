@echo off
title Squat Posture AI
color 0A
chcp 65001 >nul 2>&1

echo ============================================================
echo   Squat Posture AI - Launcher
echo ============================================================
echo.

:: Check for Python 3.9
where py >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python launcher 'py' not found.
    echo Please install Python 3.9 from https://www.python.org/downloads/
    pause
    exit /b 1
)

py -3.9 --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python 3.9 is required but not installed.
    echo Install from: https://www.python.org/downloads/release/python-3913/
    pause
    exit /b 1
)

echo [OK] Python 3.9 found.
echo.

:: Quick dependency check (only check fast imports, not tensorflow)
echo Checking dependencies...
py -3.9 -c "import mediapipe; import cv2; import numpy" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing dependencies (first run only, this may take a few minutes)...
    py -3.9 -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
)
echo [OK] Dependencies ready.
echo.

:: Check model
if exist "models\squat_model.keras" (
    echo [OK] Trained model found.
) else (
    echo [!!] No trained model found. The app will use rules only.
    echo      Run option 4 to train a model.
)

:: Menu loop
:MENU
echo.
echo ============================================================
echo   Choose a mode:
echo ============================================================
echo.
echo   1. Live Camera (webcam)
echo   2. Live Camera (phone / custom source)
echo   3. Analyze a video file
echo   4. Train / retrain the model
echo   5. Exit
echo.
set /p "CHOICE=Enter choice (1-5): "

if "%CHOICE%"=="1" (
    echo.
    echo Starting camera... (loading AI model, please wait ~15 seconds)
    echo Press Q to quit, R to reset, C to change camera.
    echo.
    py -3.9 -u src/main.py --camera
    goto MENU
)

if "%CHOICE%"=="2" (
    echo.
    echo Enter camera source:
    echo   - Camera index: 0, 1, 2...
    echo   - IP Webcam URL: http://192.168.1.X:8080/video
    echo.
    set /p "SOURCE=Source: "
    echo.
    echo Starting camera... (loading AI model, please wait ~15 seconds)
    echo.
    call py -3.9 -u src/main.py --camera --source "%SOURCE%"
    goto MENU
)

if "%CHOICE%"=="3" (
    echo.
    set /p "VIDEO=Enter video file path: "
    echo.
    echo Analyzing video... (loading AI model, please wait ~15 seconds)
    echo.
    call py -3.9 -u src/main.py --video "%VIDEO%"
    goto MENU
)

if "%CHOICE%"=="4" (
    echo.
    echo Starting training pipeline...
    echo This processes all videos in SQUAT_VIDEOS/ and may take 15-20 minutes.
    echo.
    py -3.9 -u src/main.py --train
    goto MENU
)

if "%CHOICE%"=="5" (
    echo.
    echo Goodbye!
    timeout /t 2 >nul
    exit /b 0
)

echo Invalid choice. Try again.
goto MENU
