# =============================================================================
#  deploy.ps1 — Copiere + Compilare automata MultiTFTrader.mq5 in MT5
# =============================================================================
#  Rulare: click dreapta pe fisier → "Run with PowerShell"
#  sau din terminal:  powershell -ExecutionPolicy Bypass -File deploy.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

# ── Culori helper ─────────────────────────────────────────────────────────────
function OK   { param($msg) Write-Host "  [OK] $msg" -ForegroundColor Green  }
function INFO { param($msg) Write-Host "  [>>] $msg" -ForegroundColor Cyan   }
function WARN { param($msg) Write-Host "  [!!] $msg" -ForegroundColor Yellow }
function FAIL { param($msg) Write-Host "  [XX] $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║    MultiTFTrader — Deploy Automat        ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── 1. Fisier sursa ──────────────────────────────────────────────────────────
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceFile = Join-Path $ScriptDir "EA\MultiTFTrader.mq5"

if (-not (Test-Path $SourceFile)) {
    FAIL "Nu gasesc fisierul: $SourceFile"
}
OK "Fisier sursa gasit: $SourceFile"

# ── 2. MetaEditor ────────────────────────────────────────────────────────────
$MetaEditorPaths = @(
    "C:\Program Files\MetaTrader 5\metaeditor64.exe",
    "C:\Program Files (x86)\MetaTrader 5\metaeditor64.exe"
)

$MetaEditor = $null
foreach ($p in $MetaEditorPaths) {
    if (Test-Path $p) { $MetaEditor = $p; break }
}

# Cauta si in alte locatii daca nu l-a gasit
if (-not $MetaEditor) {
    $found = Get-ChildItem "C:\Program Files*" -Recurse -Filter "metaeditor64.exe" -ErrorAction SilentlyContinue |
             Select-Object -First 1 -ExpandProperty FullName
    if ($found) { $MetaEditor = $found }
}

if (-not $MetaEditor) {
    FAIL "MetaEditor64.exe negasit. Asigura-te ca MetaTrader 5 este instalat."
}
OK "MetaEditor gasit: $MetaEditor"

# ── 3. Folder MQL5\Experts ───────────────────────────────────────────────────
$TerminalBase = "$env:APPDATA\MetaQuotes\Terminal"

# Cauta toate instalatiile MT5
$ExpertsDirs = Get-ChildItem $TerminalBase -Directory -ErrorAction SilentlyContinue |
               ForEach-Object {
                   $ep = Join-Path $_.FullName "MQL5\Experts"
                   if (Test-Path $ep) { $ep }
               }

if (-not $ExpertsDirs) {
    FAIL "Nu gasesc niciun folder MQL5\Experts in $TerminalBase"
}

# Daca sunt mai multe instalatii, afiseaza si lasa user sa aleaga
if ($ExpertsDirs.Count -gt 1) {
    Write-Host ""
    WARN "Gasit mai multe instalatii MT5:"
    for ($i = 0; $i -lt $ExpertsDirs.Count; $i++) {
        Write-Host "    [$i] $($ExpertsDirs[$i])" -ForegroundColor Yellow
    }
    $choice = Read-Host "  Alege numarul instalatiei (default 0)"
    if ($choice -eq "") { $choice = 0 }
    $ExpertsDir = $ExpertsDirs[$choice]
} else {
    $ExpertsDir = $ExpertsDirs
}

OK "Folder Experts: $ExpertsDir"

# ── 4. Copiere fisier .mq5 ───────────────────────────────────────────────────
$Dest = Join-Path $ExpertsDir "MultiTFTrader.mq5"
INFO "Copiez: $SourceFile → $Dest"
Copy-Item -Path $SourceFile -Destination $Dest -Force
OK "Fisier copiat cu succes."

# ── 5. Compilare cu MetaEditor ───────────────────────────────────────────────
INFO "Compilez cu MetaEditor..."

# MetaEditor returneaza 0 la succes, altceva la eroare
$LogFile = Join-Path $ScriptDir "compile_log.txt"

$proc = Start-Process -FilePath $MetaEditor `
    -ArgumentList "/compile:`"$Dest`"", "/log:`"$LogFile`"", "/portable" `
    -Wait -PassThru -WindowStyle Hidden

# Asteapta sa apara fisierul .ex5
$Ex5File = $Dest -replace "\.mq5$", ".ex5"
Start-Sleep -Seconds 3

if (Test-Path $Ex5File) {
    OK "Compilare REUSITA! → $Ex5File"
} else {
    WARN "Fisierul .ex5 nu a aparut. Verifica log-ul de compilare."
}

# ── 6. Afiseaza log compilare ────────────────────────────────────────────────
Write-Host ""
Write-Host "  ── Log compilare ──────────────────────────────" -ForegroundColor DarkGray
if (Test-Path $LogFile) {
    Get-Content $LogFile | ForEach-Object {
        if ($_ -match "error")   { Write-Host "  $_" -ForegroundColor Red    }
        elseif ($_ -match "warn") { Write-Host "  $_" -ForegroundColor Yellow }
        else                      { Write-Host "  $_" -ForegroundColor DarkGray }
    }
} else {
    WARN "Log negasit (normal daca MetaEditor nu a fost apelat cu /log)."
}
Write-Host "  ──────────────────────────────────────────────" -ForegroundColor DarkGray

# ── 7. Instructiuni finale ───────────────────────────────────────────────────
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "  ║  Deploy complet! Urmatoarele pasi in MT5:            ║" -ForegroundColor Green
Write-Host "  ║                                                      ║" -ForegroundColor Green
Write-Host "  ║  1. Deschide Navigator (Ctrl+N) in MT5              ║" -ForegroundColor Green
Write-Host "  ║  2. Expert Advisors → MultiTFTrader                 ║" -ForegroundColor Green
Write-Host "  ║  3. Drag + drop pe orice grafic                     ║" -ForegroundColor Green
Write-Host "  ║  4. Permite WebRequest: http://localhost:5001        ║" -ForegroundColor Green
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

# Deschide folderul Experts in Explorer
$openExplorer = Read-Host "  Deschid folderul Experts in Explorer? (Y/n)"
if ($openExplorer -ne "n" -and $openExplorer -ne "N") {
    Start-Process explorer.exe -ArgumentList $ExpertsDir
}

Write-Host ""
Write-Host "  Apasa ENTER pentru a inchide..." -ForegroundColor DarkGray
Read-Host
