#run this script in Powershell local with:
#.\run_local.ps1

# This script resets the virtual environment, installs dependencies, and runs FastAPI

# Remove existing virtual environment
if (Test-Path ".venv") {
    Write-Host "Removing existing .venv..."
    Remove-Item -Recurse -Force .venv
}

# Create new virtual environment
Write-Host "Creating new virtual environment..."
python -m venv .venv

# Activate virtual environment
Write-Host "Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
Write-Host "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run FastAPI app
Write-Host "Virtual environment is ready. You are now inside .venv."
Write-Host "Starting FastAPI server..."
Write-Host "You can run your Python commands here, e.g.: uvicorn main:app --host 0.0.0.0 --port 8000"
uvicorn main:app --host 0.0.0.0 --port 8000
