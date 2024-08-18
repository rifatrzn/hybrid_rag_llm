#!/bin/bash

set -e

# Define the repository URL and local paths
REPO_URL="https://github.com/mieweb/docs.git"
DOCS_DIR="./docs"

# Clone or update the repository
if [ -d "$DOCS_DIR" ]; then
    echo "Updating existing repository..."
    cd "$DOCS_DIR"
    git pull
else
    echo "Cloning repository..."
    git clone "$REPO_URL" "$DOCS_DIR"
fi

# Go back to the project directory
cd ..

# Run the vector embedding script from the project directory
echo "Processing markdown files and building/refreshing vector database..."
python3 ./vector_embed.py

echo "Vector database update completed successfully."