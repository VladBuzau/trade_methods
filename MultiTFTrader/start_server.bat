@echo off
chcp 65001 >nul
title MultiTFTrader Server

echo.
echo  MultiTFTrader - Server Python
echo  ================================
echo.

cd /d "%~dp0server"

:: Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [EROARE] Python nu este instalat sau nu e in PATH.
    pause
    exit /b 1
)

echo  [>>] Instalez dependinte Python...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo  [EROARE] Instalare dependinte esuata.
    pause
    exit /b 1
)

echo.
echo  [OK] Dependinte instalate.
echo  [>>] Pornesc serverul pe http://localhost:5001
echo  [>>] Dashboard grafice: http://localhost:5001/charts
echo.
echo  Apasa Ctrl+C pentru a opri serverul.
echo  ----------------------------------------
echo.

python server.py

pause
