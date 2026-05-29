#!/bin/bash
# Download and unzip the LSUI dataset from Kaggle.
#
# Usage:
#   chmod +x download_dataset.sh
#   ./download_dataset.sh
#
# Requires a Kaggle API token (~/.kaggle/kaggle.json).
# Get one at https://www.kaggle.com/settings → API → Create New Token.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="$SCRIPT_DIR/train-dataset"
ZIP="$SCRIPT_DIR/lsui.zip"
URL="https://www.kaggle.com/api/v1/datasets/download/noureldin199/lsui-large-scale-underwater-image-dataset"

# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
echo "Downloading LSUI dataset …"
curl -L \
  -u "$(python3 -c "import json,os; k=json.load(open(os.path.expanduser('~/.kaggle/kaggle.json'))); print(k['username']+':'+k['key'])" 2>/dev/null || echo ":")" \
  -o "$ZIP" \
  "$URL"

echo "Saved to: $ZIP"

# ---------------------------------------------------------------------------
# 2. Unzip
# ---------------------------------------------------------------------------
mkdir -p "$DEST"
echo "Unzipping into: $DEST"
unzip -q -o "$ZIP" -d "$DEST"
rm "$ZIP"

# ---------------------------------------------------------------------------
# 3. Quick sanity check
# ---------------------------------------------------------------------------
INPUT_COUNT=$(find "$DEST/input" -type f 2>/dev/null | wc -l | tr -d ' ')
GT_COUNT=$(find "$DEST/GT"    -type f 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "Done."
echo "  input/ : $INPUT_COUNT images"
echo "  GT/    : $GT_COUNT images"
echo ""
echo "Train with:"
echo "  source .venv/bin/activate"
echo "  python train.py --data-dir \"$DEST\" --resume LightUNet_170.pth --lr 1e-5"

