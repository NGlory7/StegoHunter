@echo off
chcp 65001 >nul
title Audio Stego Hunter

set PYTHON_EXE=C:\Users\gorke\AppData\Local\Programs\Python\Python314\python.exe

if "%~1"=="" (
    echo [!] No file dropped. Running built-in DEMO challenge...
    "%PYTHON_EXE%" "%~dp0stego_hunter.py" --demo --open
) else (
    echo [*] Scanning target: %~1
    "%PYTHON_EXE%" "%~dp0stego_hunter.py" "%~1" --open
)

pause
