# Fix for NumPy Installation Error

## Problem

You're getting this error:
```
ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc'], ['clang'], ['clang-cl'], ['pgcc']]
```

This happens because NumPy is trying to build from source but can't find a C compiler.

## Solution 1: Use Pre-built Wheels (Recommended)

Run this PowerShell script that installs packages with pre-built wheels:

```powershell
cd TaskMind\backend\app\multimodal
.\install_dependencies_fixed.ps1
```

## Solution 2: Manual Installation (Step by Step)

Run these commands one by one:

```powershell
# Navigate to project root
cd C:\Users\Maaz\Desktop\TaskMind\TaskMind

# Upgrade pip
py -m pip install --upgrade pip

# Install numpy with pre-built wheel (avoid building from source)
py -m pip install "numpy<2.0" --only-binary :all:

# If that fails, try:
py -m pip install numpy --only-binary :all: --prefer-binary

# Install other packages
py -m pip install pandas scikit-learn joblib --only-binary :all: --prefer-binary

# Install PyTorch (CPU version - faster)
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining packages
py -m pip install transformers sentence-transformers opencv-python Pillow openai-whisper yt-dlp fastapi uvicorn python-multipart tqdm ffmpeg-python
```

## Solution 3: Use Alternative Requirements File

Use the Windows-optimized requirements file:

```powershell
cd C:\Users\Maaz\Desktop\TaskMind\TaskMind
py -m pip install -r requirements_windows.txt --only-binary :all: --prefer-binary
```

## Solution 4: Install Visual Studio Build Tools (If you need to build from source)

If you need to build packages from source:

1. Download Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
2. Install "Desktop development with C++" workload
3. Restart PowerShell
4. Try installing again

**Note:** This is usually not necessary - pre-built wheels work fine.

## Quick Fix Command

Run this single command:

```powershell
cd C:\Users\Maaz\Desktop\TaskMind\TaskMind
py -m pip install "numpy<2.0" --only-binary :all: && py -m pip install -r requirements.txt --only-binary :all: --prefer-binary
```

## Verify Installation

After installation, test if it works:

```powershell
py -c "import torch; import numpy; import cv2; print('All packages installed!')"
```

If you see "All packages installed!", you're good to go!

## Why This Happens

- NumPy 2.2.6 might not have pre-built wheels for your Python version
- Your system doesn't have a C compiler installed
- pip is trying to build from source instead of using pre-built wheels

The fix forces pip to use pre-built wheels (binary packages) instead of building from source.

