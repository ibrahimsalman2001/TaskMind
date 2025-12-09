# Quick Fix for Installation Error

## The Problem

NumPy is trying to build from source but you don't have a C compiler installed.

## Quick Solution

Run this command in PowerShell:

```powershell
cd C:\Users\Maaz\Desktop\TaskMind\TaskMind
py -m pip install -r requirements.txt --only-binary :all: --prefer-binary
```

The `--only-binary :all:` flag forces pip to use pre-built wheels and never build from source.

## Alternative: Use the Quick Install Script

```powershell
cd TaskMind\backend\app\multimodal
.\quick_install.ps1
```

## If That Still Fails

Install packages one by one with the binary flag:

```powershell
cd C:\Users\Maaz\Desktop\TaskMind\TaskMind

# Core packages
py -m pip install numpy pandas scikit-learn joblib --only-binary :all: --prefer-binary

# PyTorch (CPU version)
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Everything else
py -m pip install transformers sentence-transformers opencv-python Pillow openai-whisper yt-dlp fastapi uvicorn python-multipart tqdm ffmpeg-python --only-binary :all: --prefer-binary
```

## What the Flags Do

- `--only-binary :all:` - Only use pre-built wheels, never build from source
- `--prefer-binary` - Prefer wheels but allow source if needed (less strict)

## Verify Installation

After installation, test:

```powershell
py -c "import torch; import numpy; print('Success!')"
```

If you see "Success!", you're ready to run the pipeline!


