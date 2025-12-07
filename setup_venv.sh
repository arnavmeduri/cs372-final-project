#!/bin/bash
# Setup script for creating Python virtual environment

echo "Setting up Python virtual environment for Investment Research Application..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment 'venv'..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Create .env file from example if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Please edit .env and add your NewsAPI key and SEC EDGAR information."
    else
        echo "Warning: .env.example not found. Creating basic .env file..."
        cat > .env << EOF
# NewsAPI.org API Key
NEWSAPI_KEY=your_newsapi_key_here

# SEC EDGAR User-Agent Information
SEC_EDGAR_COMPANY_NAME=YourCompanyName
SEC_EDGAR_NAME=YourName
SEC_EDGAR_EMAIL=your.email@example.com
EOF
    fi
fi

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "Don't forget to edit .env and add your API keys!"

