#!/bin/bash
# Helper script to run the application with the virtual environment activated
# Optimized for Apple Silicon Macs

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Suppress tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Enable MPS fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Activate virtual environment
source venv/bin/activate

# Run the application with all arguments passed through
python run.py "$@"

