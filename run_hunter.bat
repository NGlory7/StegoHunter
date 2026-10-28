@echo off
chcp 65001 >nul
title Audio Stego Hunter

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_EXE=python"
) else (
    set "PYTHON_EXE=py"
)

if "%~1"=="" (
    echo [!] No file dropped. Running built-in DEMO challenge...
    "%PYTHON_EXE%" "%~dp0stego_hunter.py" --demo --open
) else (
    echo [*] Scanning target: %~1
    "%PYTHON_EXE%" "%~dp0stego_hunter.py" "%~1" --open
)

pause
