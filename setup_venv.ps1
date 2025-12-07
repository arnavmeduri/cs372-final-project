# PowerShell setup script for creating Python virtual environment on Windows

Write-Host "Setting up Python virtual environment for Investment Research Application..." -ForegroundColor Green

# Check if Python 3 is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python is not installed or not in PATH. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment 'venv'..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create .env file from example if it doesn't exist
if (-not (Test-Path .env)) {
    Write-Host "Creating .env file from .env.example..." -ForegroundColor Yellow
    if (Test-Path .env.example) {
        Copy-Item .env.example .env
        Write-Host "Please edit .env and add your NewsAPI key and SEC EDGAR information." -ForegroundColor Cyan
    } else {
        Write-Host "Warning: .env.example not found. Creating basic .env file..." -ForegroundColor Yellow
        @"
# NewsAPI.org API Key
NEWSAPI_KEY=your_newsapi_key_here

# SEC EDGAR User-Agent Information
SEC_EDGAR_COMPANY_NAME=YourCompanyName
SEC_EDGAR_NAME=YourName
SEC_EDGAR_EMAIL=your.email@example.com
"@ | Out-File -FilePath .env -Encoding utf8
    }
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment in the future, run:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate, run:" -ForegroundColor Cyan
Write-Host "  deactivate" -ForegroundColor White
Write-Host ""
Write-Host "Don't forget to edit .env and add your API keys!" -ForegroundColor Yellow

