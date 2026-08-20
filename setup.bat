@echo off
echo ==============================================
echo AV-SynthRestore 3D: Setup Environment
echo ==============================================
echo.

:: 1. Setup Python Virtual Environment
echo [1/4] Setting up Python Virtual Environment (.venv)...
if not exist .venv\Scripts\python.exe (
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo Error: Python is not installed or not in PATH.
        pause
        exit /b %ERRORLEVEL%
    )
) else (
    echo Virtual Environment already exists.
)

:: 2. Upgrade pip first (old pip can't resolve PyTorch dependencies)
echo.
echo [2/6] Upgrading pip...
call .venv\Scripts\python -m pip install --upgrade pip
if %ERRORLEVEL% neq 0 (
    echo Warning: Failed to upgrade pip. Continuing anyway...
)

:: 3. Setup PyTorch and Dependencies
echo.
echo [3/6] Installing PyTorch with CUDA 12.1...
call .venv\Scripts\pip install torch==2.8.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install PyTorch.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [3/5] Installing Python dependencies...
call .venv\Scripts\pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install Python packages.
    pause
    exit /b %ERRORLEVEL%
)

:: 2. Setup Real-ESRGAN Weights
echo.
echo [3/4] Checking model weights...
if not exist weights (
    mkdir weights
)
if not exist weights\RealESRGAN_x4plus.pth (
    echo Downloading RealESRGAN_x4plus.pth [67 MB]...
    curl -L "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -o weights\RealESRGAN_x4plus.pth
    if %ERRORLEVEL% neq 0 (
        echo Warning: Failed to download weights. You can download manually from:
        echo https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
        echo and place it under weights\RealESRGAN_x4plus.pth
    )
) else (
    echo Model weights already present.
)

echo Exporting PyTorch model to ONNX for TensorRT acceleration...
call .venv\Scripts\python export_onnx.py
if %ERRORLEVEL% neq 0 (
    echo Warning: Failed to export ONNX model.
)

:: 3. Setup AI Audio Model Weights (Hugging Face)
echo.
echo [3.5/4] Downloading AI Audio Model Weights...
if not exist weights\fish-speech-s2-pro-fp8 (
    echo Downloading Fish Audio S2 Pro [FP8 variant]...
    call .venv\Scripts\hf download AEmotionStudio/fish-speech-s2-pro-fp8 --local-dir weights\fish-speech-s2-pro-fp8
) else (
    echo Fish Audio S2 Pro [FP8 variant] already present.
)

if not exist weights\fireredtts (
    echo Downloading FireRedTTS3...
    call .venv\Scripts\hf download FireRedTeam/FireRedTTS --local-dir weights\fireredtts
) else (
    echo FireRedTTS3 already present.
)

:: 4. Setup FFmpeg
echo.
echo [4/4] Checking for FFmpeg...
where ffmpeg >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo FFmpeg is already installed and available on system PATH.
    goto :ffmpeg_done
)

if exist ffmpeg.exe (
    if exist ffprobe.exe (
        echo FFmpeg binaries already present in the project folder.
        goto :ffmpeg_done
    )
)

echo FFmpeg not found on PATH. Downloading static binaries for Windows...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile 'ffmpeg.zip'"
if not exist ffmpeg.zip (
    echo Error: Failed to download FFmpeg automatically.
    echo Please download FFmpeg manually, extract it, and add it to your system PATH.
    goto :ffmpeg_done
)

echo Extracting FFmpeg binaries...
powershell -Command "Expand-Archive -Path 'ffmpeg.zip' -DestinationPath 'ffmpeg_temp' -Force"
echo Copying executables to project root...
powershell -Command "Get-ChildItem -Path 'ffmpeg_temp' -Filter '*.exe' -Recurse | Copy-Item -Destination '.'"
echo Cleaning up temporary files...
powershell -Command "Remove-Item -Path 'ffmpeg.zip', 'ffmpeg_temp' -Recurse -Force"
echo FFmpeg successfully set up in project root!

:ffmpeg_done

:: 5. Setup AI Audio Codebases
echo.
echo [5/5] Cloning and setting up AI Model repositories...

if not exist libs (
    mkdir libs
)

if not exist libs\FireRedTTS2 (
    echo Cloning FireRedTTS2...
    git clone https://github.com/FireRedTeam/FireRedTTS2.git libs\FireRedTTS2
    call .venv\Scripts\pip install -e libs\FireRedTTS2
) else (
    echo FireRedTTS2 already cloned.
)

if not exist libs\fish-speech (
    echo Cloning fish-speech...
    git clone https://github.com/fishaudio/fish-speech.git libs\fish-speech
    call .venv\Scripts\pip install -e libs\fish-speech
) else (
    echo fish-speech already cloned.
)

echo.
echo ==============================================
echo Setup Complete!
echo Run 'run.bat' to start the application.
echo ==============================================
pause
