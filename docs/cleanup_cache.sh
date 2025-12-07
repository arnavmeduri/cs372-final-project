#!/bin/bash
# Script to manage and cleanup HuggingFace model cache

CACHE_DIR="$HOME/.cache/huggingface/hub"

echo "=== HuggingFace Model Cache Manager ==="
echo ""

# Check if cache directory exists
if [ ! -d "$CACHE_DIR" ]; then
    echo "Cache directory not found: $CACHE_DIR"
    exit 0
fi

# Show current cache size
echo "Current cache size:"
du -sh "$CACHE_DIR"
echo ""

# List cached models
echo "Cached models:"
echo "--------------"
for dir in "$CACHE_DIR"/models--*; do
    if [ -d "$dir" ]; then
        model_name=$(basename "$dir" | sed 's/models--//' | sed 's/--/\//g')
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  $size  $model_name"
    fi
done
echo ""

# Ask for confirmation before cleanup
if [ "$1" == "--clean" ]; then
    echo "WARNING: This will delete ALL cached models!"
    read -p "Are you sure? (y/N): " confirm
    if [ "$confirm" == "y" ] || [ "$confirm" == "Y" ]; then
        echo "Cleaning cache..."
        rm -rf "$CACHE_DIR"/*
        echo "Cache cleaned!"
    else
        echo "Cancelled."
    fi
elif [ "$1" == "--clean-unused" ]; then
    echo "Cleaning unused cache files (locks, incomplete downloads)..."
    find "$CACHE_DIR" -name "*.lock" -delete 2>/dev/null
    find "$CACHE_DIR" -name ".no_exist" -delete 2>/dev/null
    echo "Cleaned unused files."
else
    echo "Usage:"
    echo "  ./cleanup_cache.sh           # Show cache info"
    echo "  ./cleanup_cache.sh --clean   # Delete ALL cached models"
    echo "  ./cleanup_cache.sh --clean-unused  # Clean lock files only"
fi
