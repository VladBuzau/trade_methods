@echo off
title AutoTrader
cd /d "%~dp0"

echo.
echo   ==========================================
echo       AutoTrader v2.0
echo   15 Strategies - Multi-TF - FTMO
echo   ==========================================
echo.

REM Instaleaza dependentele (silentios)
pip install -r requirements.txt --quiet 2>nul

REM Tunel public Cloudflare
where cloudflared >nul 2>nul
if not errorlevel 1 (
    echo   [..] Pornesc tunelul Cloudflare...
    start /b cloudflared tunnel --url http://localhost:5004 > cloudflare_tunnel.log 2>&1
    timeout /t 10 /nobreak >nul
    findstr "trycloudflare.com" cloudflare_tunnel.log > temp_url.txt 2>nul
    if exist temp_url.txt (
        for /f "tokens=*" %%i in ('findstr "https://.*trycloudflare" cloudflare_tunnel.log') do (
            echo   PUBLIC: %%i
        )
        del temp_url.txt
    )
) else (
    echo   [!] cloudflared nu e instalat - doar server local
)

echo   LOCAL:  http://localhost:5004
echo   AUTO:   http://localhost:5004/autotrader
echo.
echo   Ctrl+C = opreste serverul
echo.

start http://localhost:5004/autotrader
python app.py

taskkill /f /im cloudflared.exe >nul 2>&1
pause
