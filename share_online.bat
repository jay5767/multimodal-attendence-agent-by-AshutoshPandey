@echo off
title Multimodal Attendance Agent - Share Online (Public Web Access)
cd /d "%~dp0"

echo ======================================================================
echo   Multimodal Attendance Agent - Instant Global Web Access
echo ======================================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0share_online.ps1"

pause

