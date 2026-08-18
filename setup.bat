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

echo.
echo [2/4] Installing Python dependencies...
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

echo.
echo ==============================================
echo Setup Complete!
echo Run 'run.bat' to start the application.
echo ==============================================
pause
