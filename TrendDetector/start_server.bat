@echo off
chcp 65001 >nul
title TrendDetector Server

echo.
echo  TrendDetector - Server Python
echo  ================================
echo.

cd /d "%~dp0server"

python --version >nul 2>&1
if errorlevel 1 (
    echo  [EROARE] Python nu este instalat.
    pause
    exit /b 1
)

echo  [>>] Instalez dependinte...
pip install -r requirements.txt --quiet

echo.
echo  [OK] Dependinte instalate.
echo  [>>] Pornesc serverul pe http://localhost:5002
echo  [>>] Dashboard: http://localhost:5002/signals
echo.
echo  Apasa Ctrl+C pentru a opri.
echo  ----------------------------------------
echo.

python server.py

pause
