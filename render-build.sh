#!/usr/bin/env bash
set -o errexit
set -o pipefail
set -o nounset

echo "⚙️ Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Build complete."
