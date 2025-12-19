#!/bin/bash
# Script to fix write protection on runs and .vscode folders

echo "Fixing permissions for runs and .vscode folders..."

# Fix runs folder ownership and permissions
if [ -d "runs" ]; then
    echo "Fixing runs folder..."
    sudo chown -R yona:yona runs
    chmod -R u+w runs
    echo "✓ runs folder fixed"
else
    echo "⚠ runs folder not found"
fi

# Fix .vscode folder ownership and permissions
if [ -d ".vscode" ]; then
    echo "Fixing .vscode folder..."
    sudo chown -R yona:yona .vscode
    chmod -R u+w .vscode
    echo "✓ .vscode folder fixed"
else
    echo "⚠ .vscode folder not found"
fi

# Also fix denoise_last_ckpt if it exists and is owned by root
if [ -d "denoise_last_ckpt" ]; then
    OWNER=$(stat -c '%U' denoise_last_ckpt)
    if [ "$OWNER" = "root" ]; then
        echo "Fixing denoise_last_ckpt folder..."
        sudo chown -R yona:yona denoise_last_ckpt
        chmod -R u+w denoise_last_ckpt
        echo "✓ denoise_last_ckpt folder fixed"
    fi
fi

echo ""
echo "Done! All folders should now be writable."


