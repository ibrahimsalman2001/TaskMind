# run_pipeline_manual.ps1 - PowerShell script to run the pipeline

$ErrorActionPreference = "Stop"

# Get the directory where this script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "TaskMind Pipeline Runner" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Try to find Python
$pythonCmd = $null
$pythonCommands = @("python", "python3", "py")

foreach ($cmd in $pythonCommands) {
    try {
        $result = Get-Command $cmd -ErrorAction Stop 2>$null
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0 -or $version -match "Python") {
            $pythonCmd = $cmd
            Write-Host "Using Python: $pythonCmd" -ForegroundColor Green
            Write-Host "Version: $version" -ForegroundColor Green
            Write-Host ""
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonCmd) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python:" -ForegroundColor Yellow
    Write-Host "1. Download from: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "2. During installation, check 'Add Python to PATH'" -ForegroundColor White
    Write-Host "3. Restart PowerShell after installation" -ForegroundColor White
    Write-Host ""
    Write-Host "Or run: .\check_python.ps1" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if the script exists
if (-not (Test-Path "run_pipeline_manual.py")) {
    Write-Host "ERROR: run_pipeline_manual.py not found!" -ForegroundColor Red
    Write-Host "Current directory: $scriptDir" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Run the Python script
Write-Host "Starting pipeline..." -ForegroundColor Cyan
Write-Host ""

try {
    & $pythonCmd run_pipeline_manual.py
} catch {
    Write-Host ""
    Write-Host "ERROR: Failed to run pipeline" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Read-Host "Press Enter to exit"


