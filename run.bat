@echo off
echo Starting Parkinson's Disease Detection API...

REM Check for virtual environment
if not exist "venv" (
    echo Creating Virtual Environment...
    python -m venv venv
)

REM Activate and install requirements
echo Installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt

REM Run the backend
echo.
echo ======================================================
echo Parkinson's AI Project is READY
echo Visit: http://localhost:8000
echo ======================================================
echo.
python backend/main.py

pause
