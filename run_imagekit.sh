#!/bin/bash
# Auto-restart wrapper for Zaneva ImageKit
# Usage: bash run_imagekit.sh

cd "E:/Github/zaneva-imagekit"
source .venv/Scripts/activate

while true; do
    echo "[$(date)] Starting ImageKit..."
    PYTHONUNBUFFERED=1 python app.py
    EXIT_CODE=$?
    echo "[$(date)] ImageKit exited (code $EXIT_CODE). Restarting in 3s..."
    sleep 3
done
