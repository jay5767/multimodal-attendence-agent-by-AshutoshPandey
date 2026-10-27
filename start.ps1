# Multimodal Attendance & Engagement Agent - Launcher
Set-Location $PSScriptRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Multimodal Attendance & Engagement Agent - AI Classroom" -ForegroundColor Green
Write-Host "  Starting Local Production WSGI Server..." -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$py311 = "C:\Users\Ashutoshpandey\AppData\Local\Programs\Python\Python311\python.exe"

if (Test-Path $py311) {
    Write-Host "[RUN] Launching with Python 3.11 environment..." -ForegroundColor Green
    & $py311 -u app.py
} else {
    Write-Host "[RUN] Launching with default Python..." -ForegroundColor Yellow
    python -u app.py
}
