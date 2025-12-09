# Python Installation Guide

## Problem: Python Not Found

If you see the error:
```
Python was not found; run without arguments to install from the Microsoft Store
```

You need to install Python first.

## Option 1: Official Python Installer (Recommended)

1. **Download Python:**
   - Go to: https://www.python.org/downloads/
   - Click "Download Python 3.12.x" (or latest version)

2. **Install Python:**
   - Run the installer
   - **IMPORTANT:** Check the box "Add Python to PATH" at the bottom
   - Click "Install Now"

3. **Verify Installation:**
   - Close and reopen PowerShell
   - Run: `python --version`
   - You should see: `Python 3.12.x`

## Option 2: Microsoft Store

1. Open Microsoft Store
2. Search for "Python 3.12"
3. Click "Install"
4. After installation, restart PowerShell
5. Run: `python --version`

## Option 3: Using py Launcher (Windows)

If Python is installed but `python` doesn't work, try:

```powershell
py run_pipeline_manual.py
```

## After Installation

1. **Restart PowerShell** (important!)
2. Navigate to the pipeline directory:
   ```powershell
   cd TaskMind\backend\app\multimodal
   ```
3. Run the pipeline:
   ```powershell
   python run_pipeline_manual.py
   ```

Or use the PowerShell script:
```powershell
.\run_pipeline_manual.ps1
```

## Verify Python is Working

Run this command:
```powershell
python --version
```

You should see something like:
```
Python 3.12.0
```

## Troubleshooting

### Still getting "Python not found" after installation?

1. **Check PATH:**
   ```powershell
   $env:PATH -split ';' | Select-String -Pattern "Python"
   ```
   Should show Python directories

2. **Restart PowerShell** - PATH changes require a new session

3. **Try full path:**
   ```powershell
   C:\Users\YourUsername\AppData\Local\Programs\Python\Python312\python.exe run_pipeline_manual.py
   ```

4. **Reinstall Python** with "Add to PATH" checked

### Check Python Installation

Run the helper script:
```powershell
.\check_python.ps1
```

This will show you what Python commands are available.

