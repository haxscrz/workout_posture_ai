@echo off
setlocal enabledelayedexpansion
title FormAI - Gym Posture Coach Setup
color 0B

echo.
echo  ============================================
echo    FormAI - Gym Posture Coach
echo    One-Click Setup ^& Launch
echo  ============================================
echo.

:: ─── Check Administrator ───────────────────────────────────────────────
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Requesting administrator privileges for firewall rules...
    echo.
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo [OK] Running as Administrator
echo.

:: ─── Get local IP ──────────────────────────────────────────────────────
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    set "LOCAL_IP=%%a"
    set "LOCAL_IP=!LOCAL_IP: =!"
    goto :gotip
)
:gotip
echo [*] Your local IP: %LOCAL_IP%
echo.

:: ─── Check Node.js ─────────────────────────────────────────────────────
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Node.js is NOT installed.
    echo.
    echo     Please install Node.js from: https://nodejs.org/
    echo     Download the LTS version, run the installer, then re-run this script.
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('node -v') do set NODE_VER=%%v
echo [OK] Node.js found: %NODE_VER%

:: ─── Check Python 3.9+ (for training server) ──────────────────────────
set PYTHON_CMD=
where python >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
) else (
    where python3 >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=python3
    )
)

if defined PYTHON_CMD (
    for /f "tokens=*" %%v in ('!PYTHON_CMD! --version 2^>^&1') do set PY_VER=%%v
    echo [OK] Python found: !PY_VER!
) else (
    echo [!] Python not found. Training server will not be available.
    echo     Install Python 3.9+ from https://www.python.org/downloads/
)
echo.

:: ─── Install npm dependencies ──────────────────────────────────────────
echo [1/4] Installing Node.js dependencies...
if not exist "node_modules" (
    call npm install
    if %errorlevel% neq 0 (
        echo [X] npm install failed. Please check your internet connection.
        pause
        exit /b 1
    )
) else (
    echo       Already installed, skipping.
)
echo.

:: ─── Install Python dependencies (if Python exists) ────────────────────
if defined PYTHON_CMD (
    echo [2/4] Installing Python dependencies for training server...
    !PYTHON_CMD! -m pip install -r training-server\requirements.txt --quiet 2>nul
    if %errorlevel% equ 0 (
        echo       Python dependencies installed.
    ) else (
        echo [!]   Some Python packages failed. Training may not work.
    )
) else (
    echo [2/4] Skipping Python setup (Python not found^)
)
echo.

:: ─── Add Windows Firewall rules ────────────────────────────────────────
echo [3/4] Configuring Windows Firewall for mobile access...

:: Remove old rules (if any)
netsh advfirewall firewall delete rule name="FormAI-Vite" >nul 2>&1
netsh advfirewall firewall delete rule name="FormAI-Training" >nul 2>&1

:: Add inbound rules
netsh advfirewall firewall add rule name="FormAI-Vite" dir=in action=allow protocol=TCP localport=5173 >nul 2>&1
if %errorlevel% equ 0 (
    echo       Firewall rule added: Port 5173 (App Server^)
) else (
    echo [!]   Failed to add firewall rule for port 5173
)

netsh advfirewall firewall add rule name="FormAI-Training" dir=in action=allow protocol=TCP localport=5000 >nul 2>&1
if %errorlevel% equ 0 (
    echo       Firewall rule added: Port 5000 (Training Server^)
) else (
    echo [!]   Failed to add firewall rule for port 5000
)
echo.

:: ─── Start Training Server in background (if Python exists) ────────────
if defined PYTHON_CMD (
    echo [4/4] Starting Training Server on port 5000...
    start "FormAI Training Server" /min cmd /c "cd /d "%~dp0training-server" && !PYTHON_CMD! server.py"
    echo       Training server starting in background...
) else (
    echo [4/4] Skipping training server (Python not found^)
)
echo.

:: ─── Start Vite Dev Server ─────────────────────────────────────────────
echo  ============================================
echo    ALL DONE! Starting FormAI...
echo  ============================================
echo.
echo  Access from THIS computer:
echo    https://localhost:5173
echo.
echo  Access from your PHONE:
echo    https://%LOCAL_IP%:5173
echo.
echo  NOTE: Your browser will show a security warning
echo  because we use a self-signed HTTPS certificate.
echo  Click "Advanced" then "Proceed" to continue.
echo.
echo  Press Ctrl+C to stop the server.
echo  ============================================
echo.

:: Open browser automatically
start "" "https://localhost:5173"

:: Start the Vite dev server (foreground, so Ctrl+C stops it)
call npx vite --host 0.0.0.0

:: ─── Cleanup on exit ───────────────────────────────────────────────────
echo.
echo Shutting down...
taskkill /fi "windowtitle eq FormAI Training Server" /f >nul 2>&1
echo Done. Goodbye!
pause
