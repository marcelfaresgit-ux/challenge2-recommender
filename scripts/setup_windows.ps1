$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Candidates = @(
    "$env:LOCALAPPDATA\Python\pythoncore-3.12-64\python.exe",
    "$env:LOCALAPPDATA\Python\bin\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
)

$Python = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Python) {
    throw "Python 3.11+ nao encontrado. Instale pelo site python.org e marque 'Add python.exe to PATH'."
}

Set-Location $ProjectRoot
Write-Host "Usando Python: $Python"

& $Python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -e '.[dev]'
& ".\.venv\Scripts\python.exe" scripts\validate_env.py

Write-Host ""
Write-Host "Ambiente pronto. Use:"
Write-Host ".\.venv\Scripts\python.exe -m pytest"
Write-Host ".\.venv\Scripts\python.exe -m ruff check ."
Write-Host ".\.venv\Scripts\python.exe -m dvc repro"
