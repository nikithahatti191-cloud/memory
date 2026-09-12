# Parkinson's Project Startup Script
Write-Host "Starting Parkinson's Disease Detection API..." -ForegroundColor Cyan

# Check for virtual environment
if (-Not (Test-Path "venv")) {
    Write-Host "Creating Virtual Environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate and install requirements
Write-Host "Installing dependencies..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run the backend
Write-Host "Launch API..." -ForegroundColor Green
python backend/main.py
