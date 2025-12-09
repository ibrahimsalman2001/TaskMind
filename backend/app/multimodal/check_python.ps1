# check_python.ps1 - Check for Python installation

Write-Host "Checking for Python installation..." -ForegroundColor Cyan

# Check common Python commands
$pythonCommands = @("python", "python3", "py", "python.exe")

foreach ($cmd in $pythonCommands) {
    try {
        $result = Get-Command $cmd -ErrorAction Stop
        Write-Host "✓ Found: $($result.Name) at $($result.Source)" -ForegroundColor Green
        
        # Test if it actually works
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0 -or $version -match "Python") {
            Write-Host "  Version: $version" -ForegroundColor Green
            Write-Host ""
            Write-Host "You can use: $cmd run_pipeline_manual.py" -ForegroundColor Yellow
            exit 0
        }
    } catch {
        # Command not found, continue
    }
}

Write-Host "✗ Python not found!" -ForegroundColor Red
Write-Host ""
Write-Host "Please install Python:" -ForegroundColor Yellow
Write-Host "1. Download from: https://www.python.org/downloads/" -ForegroundColor White
Write-Host "2. During installation, check 'Add Python to PATH'" -ForegroundColor White
Write-Host "3. Restart your terminal after installation" -ForegroundColor White
Write-Host ""
Write-Host "Or install via Microsoft Store:" -ForegroundColor Yellow
Write-Host "  Search for 'Python 3.12' in Microsoft Store" -ForegroundColor White

