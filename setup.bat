@echo off
REM ===================================================================
REM  Financial Digital Twin - one-time setup (Windows)
REM  Creates a virtual environment, installs dependencies, and runs
REM  the full validation pipeline.
REM
REM  Just double-click this file.
REM ===================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ==========================================================
echo   Financial Digital Twin - setup
echo   Folder: %CD%
echo ==========================================================
echo.

REM ---- locate a Python interpreter --------------------------------
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [ERROR] No Python found on PATH.
    echo         Install Python 3.10+ from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH" during installation.
    goto :fail
)

for /f "tokens=*" %%v in ('%PY% --version 2^>^&1') do set "PYVER=%%v"
echo [1/5] Using !PYVER!

REM ---- create the virtual environment ------------------------------
if exist ".venv\Scripts\python.exe" (
    echo [2/5] Virtual environment already exists - reusing it.
) else (
    echo [2/5] Creating virtual environment in .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        goto :fail
    )
)

set "VPY=%CD%\.venv\Scripts\python.exe"

REM ---- install dependencies ---------------------------------------
echo [3/5] Upgrading pip ...
"%VPY%" -m pip install --upgrade pip --quiet

echo [4/5] Installing dependencies ^(this takes a couple of minutes^) ...
"%VPY%" -m pip install --quiet pandas numpy scikit-learn scipy streamlit plotly requests
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Scroll up for the reason.
    goto :fail
)
echo       Core dependencies installed.
echo.
echo       NOTE: PyTorch is optional and is NOT installed here - it is a
echo             ~2.5 GB download and the project runs fine without it.
echo             The forecasting layer detects torch and simply skips the
echo             LSTM if it is absent. To add it later, run:
echo                 .venv\Scripts\python.exe -m pip install torch
echo.

REM ---- run the pipeline -------------------------------------------
echo [5/5] Running the full pipeline with validation ...
echo.
set "PYTHONPATH=%CD%\src"
"%VPY%" run_pipeline.py
if errorlevel 1 (
    echo.
    echo [ERROR] The pipeline reported a failure - see the output above.
    goto :fail
)

echo.
echo ==========================================================
echo   Setup complete.
echo   Now double-click  run_dashboard.bat  to launch the app.
echo ==========================================================
echo.
pause
exit /b 0

:fail
echo.
echo Setup did not complete. Nothing was broken - fix the error above
echo and run this file again.
echo.
pause
exit /b 1
