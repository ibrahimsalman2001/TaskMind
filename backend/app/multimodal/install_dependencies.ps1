# install_dependencies.ps1 - Install required Python packages

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Installing TaskMind Pipeline Dependencies" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Get the project root (go up from multimodal to TaskMind root)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptDir))
$requirementsFile = Join-Path $projectRoot "requirements.txt"

Write-Host "Project root: $projectRoot" -ForegroundColor Yellow
Write-Host "Requirements file: $requirementsFile" -ForegroundColor Yellow
Write-Host ""

# Check if requirements.txt exists
if (-not (Test-Path $requirementsFile)) {
    Write-Host "ERROR: requirements.txt not found at: $requirementsFile" -ForegroundColor Red
    Write-Host "Please make sure you're in the correct directory." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Find Python
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
    Write-Host "Please install Python first." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if pip is available
Write-Host "Checking pip..." -ForegroundColor Cyan
$pipCheck = & $pythonCmd -m pip --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip not found!" -ForegroundColor Red
    Write-Host "Please install pip or reinstall Python with pip included." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "✓ pip is available" -ForegroundColor Green
Write-Host ""

# Upgrade pip first
Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $pythonCmd -m pip install --upgrade pip
Write-Host ""

# Install requirements
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
Write-Host "This may take several minutes..." -ForegroundColor Yellow
Write-Host ""

try {
    & $pythonCmd -m pip install -r $requirementsFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "=" * 70 -ForegroundColor Green
        Write-Host "✓ All dependencies installed successfully!" -ForegroundColor Green
        Write-Host "=" * 70 -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now run the pipeline:" -ForegroundColor Yellow
        Write-Host "  py run_pipeline_manual.py" -ForegroundColor White
    } else {
        Write-Host ""
        Write-Host "⚠ Some packages may have failed to install." -ForegroundColor Yellow
        Write-Host "Check the output above for errors." -ForegroundColor Yellow
    }
} catch {
    Write-Host ""
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to exit"


