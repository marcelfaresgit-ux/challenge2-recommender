$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Ambiente .venv nao encontrado. Rode primeiro: .\scripts\setup_windows.ps1"
}

& $Python -m pytest
& $Python -m ruff check .
& $Python -m dvc repro
