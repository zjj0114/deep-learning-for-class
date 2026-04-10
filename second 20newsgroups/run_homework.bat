@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="
set "CONDA_CMD="
set "PREFERRED_ENV=ptyonTorchView"
set "FALLBACK_ENV=newsgru310"

if exist "E:\anaconda\envs\%PREFERRED_ENV%\python.exe" (
  echo Found preferred PyTorch environment: %PREFERRED_ENV%
  set "PYTHON_CMD=E:\anaconda\envs\%PREFERRED_ENV%\python.exe"
)

if not defined PYTHON_CMD (
  for %%P in (
    "%USERPROFILE%\anaconda3\condabin\conda.bat"
    "%USERPROFILE%\miniconda3\condabin\conda.bat"
    "C:\ProgramData\anaconda3\condabin\conda.bat"
    "C:\ProgramData\miniconda3\condabin\conda.bat"
    "E:\anaconda\condabin\conda.bat"
    "E:\anaconda\Scripts\conda.exe"
  ) do (
    if not defined CONDA_CMD if exist %%~P set "CONDA_CMD=%%~P"
  )
)

if not defined PYTHON_CMD if not defined CONDA_CMD (
  echo Could not find preferred Python or conda.
  exit /b 1
)

if not defined PYTHON_CMD (
  echo Using conda: %CONDA_CMD%
  call "%CONDA_CMD%" run -n %FALLBACK_ENV% python --version >nul 2>nul
  if errorlevel 1 (
    echo Creating conda environment %FALLBACK_ENV% from environment.yml ...
    call "%CONDA_CMD%" env create -f environment.yml
    if errorlevel 1 exit /b 1
  ) else (
    echo Updating conda environment %FALLBACK_ENV% from environment.yml ...
    call "%CONDA_CMD%" env update -n %FALLBACK_ENV% -f environment.yml --prune
    if errorlevel 1 exit /b 1
  )
  call "%CONDA_CMD%" run -n %FALLBACK_ENV% python src\train_gru_classifier.py --data-root ..\.tmp\20news_data
  exit /b %errorlevel%
)

echo Running training with: %PYTHON_CMD%
"%PYTHON_CMD%" src\train_gru_classifier.py --data-root ..\.tmp\20news_data
exit /b %errorlevel%
