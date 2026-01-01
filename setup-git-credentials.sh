#!/bin/bash
# Setup script for git credentials in Docker container
# This ensures git credentials are configured on container startup
# Run this script in your Docker entrypoint or on container start

set -e

CREDENTIALS_FILE="/workspace/lowlight/.git-credentials"
WORKSPACE_DIR="/workspace/lowlight"

# Check if credentials file exists
if [ ! -f "$CREDENTIALS_FILE" ]; then
    echo "Warning: .git-credentials file not found at $CREDENTIALS_FILE"
    echo "Please create it with: echo 'https://YOUR_TOKEN@github.com' > $CREDENTIALS_FILE"
    exit 1
fi

# Configure git to use the credentials file in the mounted directory
git config --global credential.helper "store --file=$CREDENTIALS_FILE"

# Verify configuration
echo "Git credentials configured:"
echo "  Credentials file: $CREDENTIALS_FILE"
echo "  Credential helper: $(git config --global --get credential.helper)"

# Test that credentials file is readable
if [ -r "$CREDENTIALS_FILE" ]; then
    echo "  Credentials file is readable"
else
    echo "  Warning: Credentials file is not readable"
fi

echo "Git credentials setup complete!"
