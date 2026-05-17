@echo off
title Squat Posture AI
color 0A

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

:: Verify Python 3.9 is available
py -3.9 --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python 3.9 is required but not installed.
    echo Please install Python 3.9 from https://www.python.org/downloads/release/python-3913/
    echo.
    echo TensorFlow does not yet support Python 3.10+.
    pause
    exit /b 1
)

echo [OK] Python 3.9 found.

:: Install dependencies if needed
echo.
echo Checking dependencies...
py -3.9 -c "import mediapipe, cv2, tensorflow, numpy" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing dependencies (first run only)...
    py -3.9 -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
)
echo [OK] All dependencies installed.

:: Check for trained model
echo.
if not exist "models\squat_model.keras" (
    echo WARNING: No trained model found.
    echo.
    echo Would you like to train the model now?
    echo This requires squat videos in the SQUAT_VIDEOS folder.
    echo.
    set /p TRAIN="Train now? (y/n): "
    if /i "%TRAIN%"=="y" (
        echo.
        echo Training model... this may take 15-20 minutes.
        py -3.9 -u src/main.py --train
    )
) else (
    echo [OK] Trained model found.
)

:: Launch menu
:MENU
echo.
echo ============================================================
echo   Choose a mode:
echo ============================================================
echo.
echo   1. Live Camera (webcam)
echo   2. Live Camera (custom source / phone)
echo   3. Analyze a video file
echo   4. Train / retrain the model
echo   5. Exit
echo.
set /p CHOICE="Enter choice (1-5): "

if "%CHOICE%"=="1" (
    echo.
    echo Starting camera analysis...
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
    set /p SOURCE="Source: "
    echo.
    py -3.9 -u src/main.py --camera --source "%SOURCE%"
    goto MENU
)

if "%CHOICE%"=="3" (
    echo.
    set /p VIDEO="Enter video file path: "
    echo.
    py -3.9 -u src/main.py --video "%VIDEO%"
    goto MENU
)

if "%CHOICE%"=="4" (
    echo.
    echo Training model...
    py -3.9 -u src/main.py --train
    goto MENU
)

if "%CHOICE%"=="5" (
    echo.
    echo Goodbye!
    exit /b 0
)

echo Invalid choice. Try again.
goto MENU
