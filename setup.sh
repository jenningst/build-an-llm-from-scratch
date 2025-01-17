#!/bin/bash

# Exit on error
set -e

echo "Starting data download setup..."

# Define variables
DATA_DIR="data"
FILE_NAME="the-verdict.txt"
FILE_PATH="${DATA_DIR}/${FILE_NAME}"
URL="https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

# Create data directory if it doesn't exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory..."
    mkdir -p "$DATA_DIR"
fi

# Check if file already exists
if [ -f "$FILE_PATH" ]; then
    echo "File $FILE_NAME already exists in $DATA_DIR directory. Skipping download."
else
    echo "Downloading $FILE_NAME..."
    # Try to download the file using curl if available
    if command -v curl &> /dev/null; then
        curl -L "$URL" -o "$FILE_PATH"
    # Fall back to wget if curl is not available
    elif command -v wget &> /dev/null; then
        wget "$URL" -O "$FILE_PATH"
    else
        echo "Error: Neither curl nor wget is installed. Please install one of them and try again."
        exit 1
    fi
    
    # Verify the download was successful
    if [ -f "$FILE_PATH" ]; then
        echo "Download completed successfully!"
    else
        echo "Error: Download failed!"
        exit 1
    fi
fi

echo "Setup completed successfully!" 