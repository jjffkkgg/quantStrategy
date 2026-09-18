@echo off
setlocal
REM Load FRED configuration and run Python in the same child process.
REM Usage: setEnv.bat [LAA_MA4 | LAA_SANDBOX] [-Check]
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0utils\run_with_env.ps1" %*
exit /b %ERRORLEVEL%
