@echo off
echo ==============================================
echo AV-SynthRestore 3D: Cleanup Utility
echo ==============================================
echo.
echo This script will remove:
echo   - Python Virtual Environment (.venv)
echo   - Downloaded AI Model Weights (weights/)
echo   - Downloaded FFmpeg binaries (ffmpeg.exe, ffprobe.exe, ffplay.exe)
echo   - Temporary job files (jobs/)
echo   - Restored output files (restored_output/)
echo.
set /p confirm="Are you sure you want to proceed? (Y/N): "
if /i "%confirm%" neq "Y" (
    echo Cleanup cancelled.
    pause
    exit /b
)

echo.
echo [1/4] Removing Python Virtual Environment (.venv)...
if exist .venv (
    rmdir /s /q .venv
    echo Done.
) else (
    echo .venv not found.
)

echo [2/4] Removing model weights (weights)...
if exist weights (
    rmdir /s /q weights
    echo Done.
) else (
    echo weights not found.
)

echo [3/4] Removing local FFmpeg binaries and temp files...
if exist ffmpeg.exe (
    del /f /q ffmpeg.exe
    echo Removed ffmpeg.exe.
)
if exist ffprobe.exe (
    del /f /q ffprobe.exe
    echo Removed ffprobe.exe.
)
if exist ffplay.exe (
    del /f /q ffplay.exe
    echo Removed ffplay.exe.
)
if exist ffmpeg.zip (
    del /f /q ffmpeg.zip
)
if exist ffmpeg_temp (
    rmdir /s /q ffmpeg_temp
)
echo Done.

echo [4/4] Removing runtime output directories...
if exist jobs (
    rmdir /s /q jobs
    echo Done.
)
if exist restored_output (
    rmdir /s /q restored_output
    echo Done.
)
if exist libs (
    rmdir /s /q libs
    echo Done.
)
if exist __pycache__ (
    rmdir /s /q __pycache__
    echo Done.
)

echo.
echo ==============================================
echo Cleanup Complete!
echo Run 'setup.bat' to set up the environment again.
echo ==============================================
pause
