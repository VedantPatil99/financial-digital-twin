@echo off
REM ===================================================================
REM  Financial Digital Twin - launch the dashboard (Windows)
REM  Run setup.bat first. Then just double-click this file.
REM  The dashboard opens at http://localhost:8501
REM  Press Ctrl+C in this window to stop it.
REM ===================================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No virtual environment found.
    echo         Run setup.bat first.
    echo.
    pause
    exit /b 1
)

set "PYTHONPATH=%CD%\src"

echo.
echo ==========================================================
echo   Starting the Financial Digital Twin dashboard
echo   It will open in your browser at http://localhost:8501
echo   Press Ctrl+C here to stop it.
echo ==========================================================
echo.

".venv\Scripts\python.exe" -m streamlit run dashboard/app.py

pause
