@echo off
title Multimodal Attendance & Engagement Agent
cd /d "%~dp0"

echo ======================================================================
echo   Multimodal Attendance & Engagement Agent - AI Classroom
echo   Starting Local Production Server...
echo ======================================================================
echo.

set "PY311=C:\Users\Ashutoshpandey\AppData\Local\Programs\Python\Python311\python.exe"

if exist "%PY311%" (
    echo [RUN] Launching with Python 3.11 environment...
    "%PY311%" -u app.py
) else (
    echo [RUN] Launching with default Python...
    python -u app.py
)

pause
