@echo off
setlocal EnableExtensions
title MiroFish Local Launcher

rem ============================================================
rem  MiroFish Local launcher (ZEP_BACKEND=graphiti)
rem  Visible windows only: no hidden processes, no output redirection.
rem  It never downloads models or wheels; it only checks and tells
rem  you the exact command to run when something is missing.
rem ============================================================

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "PY=%ROOT%\backend\.venv\Scripts\python.exe"
set "FRONTEND=%ROOT%\frontend"

echo ============================================================
echo   MiroFish Local Launcher
echo   Project : %ROOT%
echo ============================================================
echo.

if not exist "%PY%" (
    echo [X] Backend virtualenv not found:
    echo     %ROOT%\backend\.venv
    echo.
    echo     Run:
    echo       cd backend
    echo       uv sync --frozen --extra graphiti
    echo     See docs\DEPLOYMENT.md for details.
    echo.
    pause
    exit /b 1
)

if not exist "%FRONTEND%\node_modules\vite\bin\vite.js" (
    echo [X] Frontend dependencies not found:
    echo     %FRONTEND%\node_modules
    echo.
    echo     Run:
    echo       cd frontend
    echo       npm ci
    echo.
    pause
    exit /b 1
)

if not exist "%ROOT%\.env" (
    echo [X] .env not found.
    echo.
    echo     Copy .env.example to .env and set LLM_API_KEY:
    echo       copy .env.example .env
    echo     Local graph mode does not need ZEP_API_KEY.
    echo.
    pause
    exit /b 1
)

echo [1/3] checking local models...
"%PY%" "%ROOT%\scripts\provision-models.py" --verify >nul 2>&1
if errorlevel 1 (
    echo [X] Local models are missing or their hashes do not match.
    echo.
    echo     Fetch them first:
    echo       "%PY%" "%ROOT%\scripts\provision-models.py" --fetch
    echo     Then verify:
    echo       "%PY%" "%ROOT%\scripts\provision-models.py" --verify
    echo.
    pause
    exit /b 1
)

if not "%MIROFISH_ALLOW_CPU_RECSYS%"=="1" (
    echo [2/3] checking CUDA PyTorch...
    "%PY%" "%ROOT%\scripts\provision-torch.py" --verify >nul 2>&1
    if errorlevel 1 (
        echo [X] GPU-accelerated PyTorch is not available.
        echo     The twitter recommendation timeline would run on CPU and can be very slow.
        echo.
        echo     Install the pinned CUDA wheel:
        echo       "%PY%" "%ROOT%\scripts\provision-torch.py" --install --wheel ^<wheel dir^> --uv ^<uv.exe^>
        echo     Or, if this machine has no NVIDIA GPU, explicitly allow CPU:
        echo       set MIROFISH_ALLOW_CPU_RECSYS=1
        echo.
        pause
        exit /b 1
    )
) else (
    echo [2/3] CUDA check skipped: MIROFISH_ALLOW_CPU_RECSYS=1
)

if /I "%~1"=="--check" (
    echo [check] prerequisites OK; not starting servers.
    exit /b 0
)

netstat -ano | findstr /c:":5001 " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [!] port 5001 is already in use. Close the other backend or stop it first.
    echo     Two backends writing the same Kuzu file can stall write-back.
    echo.
)
netstat -ano | findstr /c:":3000 " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 echo [!] port 3000 is already in use; the frontend may already be running.

echo [3/3] starting backend and frontend...
set "npm_config_cache=%ROOT%\.npm-cache"
set "ZEP_BACKEND=graphiti"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "PYTHONUNBUFFERED=1"

start "MiroFish backend :5001" /D "%ROOT%\backend" cmd /k ".venv\Scripts\python.exe run.py"
start "MiroFish frontend :3000" /D "%ROOT%\frontend" cmd /k "npm run dev"

echo.
echo Two visible windows were opened.
echo   backend  : http://localhost:5001/health  (wait ~10-30 s)
echo   frontend : http://localhost:3000         (browser opens automatically)
echo.
echo Local graph: %ROOT%\backend\uploads\graphs\graphiti.kuzu
echo Stop: close both windows, or end the processes listening on 5001/3000.
echo.
ping -n 8 127.0.0.1 >nul 2>&1
exit /b 0
