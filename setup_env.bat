@echo off
REM ================================
REM Create and activate venv, install deps (Windows)
REM ================================

SET PROJECT_DIR=%~dp0
CD /D %PROJECT_DIR%

echo.
echo Creating virtual environment in .venv ...

REM python 이 안 먹히면 "py" 로 바꿔서 한번 더 해보기
python -m venv .venv 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo "python" 명령어가 실패했습니다. "py" 로 다시 시도합니다...
    py -m venv .venv
)

echo.
echo Activating virtual environment ...

CALL .venv\Scripts\activate.bat

echo.
echo Upgrading pip ...
python -m pip install --upgrade pip

echo.
echo Installing requirements ...
pip install -r requirements.txt

echo.
echo ====================================
echo Environment setup complete!
echo.
echo 가상환경을 다시 활성화하려면:
echo   .venv\Scripts\activate
echo
echo 전략 시그널을 실행하려면:
echo   python main.py
echo ====================================
echo.
pause
