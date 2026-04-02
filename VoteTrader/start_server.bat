@echo off
chcp 65001 >nul
title VoteTrader Server

echo.
echo  VoteTrader - 12 metode, vot majoritar
echo  =======================================
echo.

cd /d "%~dp0server"

python --version >nul 2>&1
if errorlevel 1 ( echo [EROARE] Python nu e instalat. & pause & exit /b 1 )

pip install -r requirements.txt --quiet

echo.
echo  [OK] Pornesc serverul pe http://localhost:5003
echo  [>>] Dashboard: http://localhost:5003/signals
echo.

python server.py
pause
