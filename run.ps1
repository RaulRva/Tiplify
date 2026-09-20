# Arranca Tiplify. Uso:  .\run.ps1        (o doble clic en run.bat)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Creando el entorno virtual..." -ForegroundColor Cyan
    $system = Get-Command python -ErrorAction SilentlyContinue
    if (-not $system) {
        Write-Host "No encuentro Python. Instalalo desde https://www.python.org/downloads/" -ForegroundColor Red
        exit 1
    }
    & $system.Source -m venv .venv
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements.txt
}

Write-Host ""
Write-Host "  Tiplify esta arrancando en  http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "  (la primera carga descarga los datos y entrena el modelo: ~15 segundos)"
Write-Host "  Para parar: Ctrl+C"
Write-Host ""

& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
