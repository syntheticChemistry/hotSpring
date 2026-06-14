@echo off
setlocal enabledelayedexpansion

REM hotspring.bat — Windows launcher for the hotSpring guideStone artifact.
REM
REM Dispatch order:
REM   1. WSL2 (native Linux execution, best performance)
REM   2. Docker Desktop (container execution, GPU via --gpus all)
REM   3. Print setup instructions
REM
REM Usage:
REM   hotspring.bat validate
REM   hotspring.bat benchmark
REM   hotspring.bat help
REM
REM GPU override:
REM   set HOTSPRING_FORCE_GPU=1
REM   hotspring.bat validate

set "SCRIPT_DIR=%~dp0"
set "CMD=%~1"
if "%CMD%"=="" set "CMD=help"

echo.
echo   hotSpring guideStone v0.7.0 — Windows Launcher
echo   ================================================
echo.

REM ── Try WSL2 first ──────────────────────────────────────────────

wsl.exe --status >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo   Runtime: WSL2 ^(native Linux execution^)
    echo.
    wsl.exe sh -c "cd \"$(wslpath '%SCRIPT_DIR%')\" && sh ./hotspring %*"
    goto :eof
)

REM ── Try Docker ──────────────────────────────────────────────────

docker --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo   Runtime: Docker Desktop ^(container execution^)
    echo.

    set "IMAGE=hotspring-guidestone:v0.7.0"
    set "TAR=%SCRIPT_DIR%container\hotspring-guidestone.tar"

    docker image inspect !IMAGE! >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        if exist "!TAR!" (
            echo   Loading container image...
            docker load -i "!TAR!"
        ) else (
            echo   ERROR: Container image not found.
            echo   Build it with: scripts\build-container.sh
            exit /b 1
        )
    )

    set "GPU_FLAG="
    if "%HOTSPRING_FORCE_GPU%"=="1" set "GPU_FLAG=--gpus all"

    if not exist "%SCRIPT_DIR%results" mkdir "%SCRIPT_DIR%results"

    docker run --rm !GPU_FLAG! -v "%SCRIPT_DIR%results:/opt/validation/results" !IMAGE! %*
    goto :eof
)

REM ── Neither available ───────────────────────────────────────────

echo   No supported runtime found.
echo.
echo   The hotSpring artifact requires a Linux execution environment.
echo   On Windows, you have two options:
echo.
echo   Option 1 — WSL2 (recommended, best performance):
echo     wsl --install
echo     REM Then restart and run: hotspring.bat validate
echo.
echo   Option 2 — Docker Desktop:
echo     https://docs.docker.com/desktop/install/windows-install/
echo     REM Then run: hotspring.bat validate
echo.
echo   The artifact binaries are Linux ELFs (static musl + glibc).
echo   Native Windows binaries are not produced — the physics math
echo   lives in pure Rust compiled for Linux, and WSL2/Docker provide
echo   transparent execution on Windows with near-native performance.
echo.

exit /b 1
