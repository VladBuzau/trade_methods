@echo off
chcp 65001 >nul
title ChartVisualizer

echo.
echo  ChartVisualizer - Pivoti, Trenduri, EMA, RSI
echo  ===============================================
echo.

cd /d "%~dp0"

pip install -r requirements.txt --quiet

echo.
echo  [OK] Deschide browserul la: http://localhost:5004
echo.

start http://localhost:5004
python app.py

pause
