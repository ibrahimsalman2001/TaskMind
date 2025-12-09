@echo off
REM Windows batch file to run the simple pipeline

cd /d %~dp0

REM Try different Python commands
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py run_pipeline_simple.py
    goto :end
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python run_pipeline_simple.py
    goto :end
)

where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python3 run_pipeline_simple.py
    goto :end
)

echo ERROR: Python not found!
echo Please install Python from https://www.python.org/downloads/
pause
exit /b 1

:end
pause

