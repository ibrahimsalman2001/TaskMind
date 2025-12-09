# quick_install.ps1 - Quick fix for installation issues

Write-Host "Quick Install - Using Pre-built Wheels Only" -ForegroundColor Cyan
Write-Host ""

# Find Python
$pythonCmd = "py"
if (-not (Get-Command $pythonCmd -ErrorAction SilentlyContinue)) {
    $pythonCmd = "python"
}

Write-Host "Using: $pythonCmd" -ForegroundColor Green
Write-Host ""

# Navigate to project root
$projectRoot = "C:\Users\Maaz\Desktop\TaskMind\TaskMind"
if (-not (Test-Path $projectRoot)) {
    Write-Host "ERROR: Project root not found at: $projectRoot" -ForegroundColor Red
    Write-Host "Please update the path in this script." -ForegroundColor Yellow
    exit 1
}

Set-Location $projectRoot

Write-Host "Installing packages with pre-built wheels only..." -ForegroundColor Cyan
Write-Host "(This avoids building from source)" -ForegroundColor Yellow
Write-Host ""

# Install with flags to prefer pre-built wheels
& $pythonCmd -m pip install --upgrade pip --quiet

Write-Host "Installing core packages..." -ForegroundColor Yellow
& $pythonCmd -m pip install numpy pandas scikit-learn joblib --only-binary :all: --prefer-binary

Write-Host "Installing PyTorch..." -ForegroundColor Yellow
& $pythonCmd -m pip install torch --index-url https://download.pytorch.org/whl/cpu

Write-Host "Installing ML packages..." -ForegroundColor Yellow
& $pythonCmd -m pip install transformers sentence-transformers

Write-Host "Installing CV packages..." -ForegroundColor Yellow
& $pythonCmd -m pip install opencv-python Pillow

Write-Host "Installing audio packages..." -ForegroundColor Yellow
& $pythonCmd -m pip install openai-whisper

Write-Host "Installing video packages..." -ForegroundColor Yellow
& $pythonCmd -m pip install yt-dlp

Write-Host "Installing API packages..." -ForegroundColor Yellow
& $pythonCmd -m pip install fastapi uvicorn python-multipart

Write-Host "Installing utilities..." -ForegroundColor Yellow
& $pythonCmd -m pip install tqdm ffmpeg-python

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Green
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Green

