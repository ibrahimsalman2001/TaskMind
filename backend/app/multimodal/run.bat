@echo off
REM Windows batch file to run the pipeline with manual input

cd /d %~dp0

REM Try different Python commands
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using: py
    py run_pipeline_manual.py
    goto :end
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using: python
    python run_pipeline_manual.py
    goto :end
)

where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using: python3
    python3 run_pipeline_manual.py
    goto :end
)

echo ERROR: Python not found!
echo.
echo Please install Python from https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation
echo.
pause
exit /b 1

:end
pause

