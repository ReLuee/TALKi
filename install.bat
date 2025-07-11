@echo off

:: Set Python path (adjust this if needed)
set PYTHON_EXE=python.exe

setlocal enabledelayedexpansion

:: Set current directory
cd /d %~dp0

echo Starting installation process...

:: Create and activate virtual environment
echo Creating and activating virtual environment...
%PYTHON_EXE% -m venv venv
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing torch...
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install https://raw.githubusercontent.com/KoljaB/RealtimeVoiceChat/main/wheels/deepspeed-0.16.1%%2Bunknown-cp310-cp310-win_amd64.whl

echo Installing requirements...
pip install -r requirements.txt
cmd
