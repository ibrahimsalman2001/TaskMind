@echo off
REM Install dependencies for TaskMind pipeline

echo ======================================================================
echo Installing TaskMind Pipeline Dependencies
echo ======================================================================
echo.

REM Go to project root (3 levels up from multimodal)
cd /d %~dp0
cd ..\..\..

REM Try different Python commands
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using: py
    echo.
    echo Upgrading pip...
    py -m pip install --upgrade pip
    echo.
    echo Installing dependencies...
    py -m pip install -r requirements.txt
    goto :end
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using: python
    echo.
    echo Upgrading pip...
    python -m pip install --upgrade pip
    echo.
    echo Installing dependencies...
    python -m pip install -r requirements.txt
    goto :end
)

where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using: python3
    echo.
    echo Upgrading pip...
    python3 -m pip install --upgrade pip
    echo.
    echo Installing dependencies...
    python3 -m pip install -r requirements.txt
    goto :end
)

echo ERROR: Python not found!
echo Please install Python first.
pause
exit /b 1

:end
echo.
echo ======================================================================
echo Installation complete!
echo ======================================================================
echo.
echo You can now run: py run_pipeline_manual.py
echo.
pause

