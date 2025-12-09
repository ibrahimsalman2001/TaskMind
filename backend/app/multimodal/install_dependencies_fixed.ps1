# install_dependencies_fixed.ps1 - Install dependencies with fixes for Windows

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Installing TaskMind Pipeline Dependencies (Fixed for Windows)" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Get the project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptDir))

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
    exit 1
}

# Upgrade pip first
Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $pythonCmd -m pip install --upgrade pip --quiet
Write-Host "✓ pip upgraded" -ForegroundColor Green
Write-Host ""

# Install packages in order to avoid build issues
Write-Host "Installing core packages (pre-built wheels)..." -ForegroundColor Cyan

# Step 1: Install numpy first (force pre-built wheels, no source build)
Write-Host "  → Installing numpy (pre-built wheel only)..." -ForegroundColor Yellow
& $pythonCmd -m pip install numpy --only-binary :all: --prefer-binary
if ($LASTEXITCODE -ne 0) {
    Write-Host "    Warning: numpy installation had issues, continuing..." -ForegroundColor Yellow
}

# Step 2: Install other packages that might need numpy
Write-Host "  → Installing pandas, scikit-learn..." -ForegroundColor Yellow
& $pythonCmd -m pip install pandas scikit-learn joblib --only-binary :all: --prefer-binary

# Step 3: Install PyTorch (has pre-built wheels)
Write-Host "  → Installing PyTorch (this may take a while)..." -ForegroundColor Yellow
& $pythonCmd -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# If CPU-only fails, try default
if ($LASTEXITCODE -ne 0) {
    Write-Host "    Trying default PyTorch installation..." -ForegroundColor Yellow
    & $pythonCmd -m pip install torch
}

# Step 4: Install transformers and sentence-transformers
Write-Host "  → Installing transformers, sentence-transformers..." -ForegroundColor Yellow
& $pythonCmd -m pip install transformers sentence-transformers

# Step 5: Install computer vision packages
Write-Host "  → Installing opencv-python, Pillow..." -ForegroundColor Yellow
& $pythonCmd -m pip install opencv-python Pillow

# Step 6: Install audio processing
Write-Host "  → Installing openai-whisper..." -ForegroundColor Yellow
& $pythonCmd -m pip install openai-whisper

# Step 7: Install video downloading
Write-Host "  → Installing yt-dlp..." -ForegroundColor Yellow
& $pythonCmd -m pip install yt-dlp

# Step 8: Install API packages
Write-Host "  → Installing FastAPI, uvicorn..." -ForegroundColor Yellow
& $pythonCmd -m pip install fastapi uvicorn python-multipart

# Step 9: Install remaining packages
Write-Host "  → Installing remaining packages..." -ForegroundColor Yellow
& $pythonCmd -m pip install tqdm ffmpeg-python

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Green
Write-Host "✓ Installation complete!" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Green
Write-Host ""
Write-Host "You can now run the pipeline:" -ForegroundColor Yellow
Write-Host "  py run_pipeline_manual.py" -ForegroundColor White
Write-Host ""

