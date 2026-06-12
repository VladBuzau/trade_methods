@echo off
title ClaudeTelegramBridge
cd /d "%~dp0"

echo.
echo   ===========================================
echo     ClaudeTelegramBridge
echo     Telegram -^> VSCode Claude Code chat
echo   ===========================================
echo.

REM Instaleaza dependentele (silentios)
pip install -r requirements.txt --quiet 2>nul

echo   Pornesc bot-ul... (Ctrl+C pentru oprire)
echo.

python bridge.py

pause
