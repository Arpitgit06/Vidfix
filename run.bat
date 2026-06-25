@echo off
echo ==============================================
echo AV-SynthRestore 3D: Launching Application
echo ==============================================
echo.
echo Starting FastAPI backend...
start "" cmd /k ".venv\Scripts\python main.py"
echo Waiting for backend to start...
timeout /t 3 /nobreak >nul
echo Opening browser...
start http://localhost:8765/ui/index.html
exit
