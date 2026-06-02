#!/usr/bin/env bash
# Setup FlowFormer: clone repo and download checkpoint.
#
# Usage:
#   bash scripts/setup_flowformer.sh              # downloads things.pth (default)
#   bash scripts/setup_flowformer.sh sintel
#   bash scripts/setup_flowformer.sh kitti
#
# Checkpoints (Google Drive folder):
#   https://drive.google.com/drive/folders/1K2dcWxaqOLiQ3NtqB9bZRjJw2fRGWUxI
#   things -> 1ftCNSFLMegbU39WkDmD6eGNpkuLPtC65
#   sintel -> 1-Vjm5RBpfQ0F-MikRuMIjMuoBRidxGFg
#   kitti  -> 1NMaA_i0ElFQIHm_SryU2mgxYRPBT6Nxz

set -e

CHECKPOINT="${1:-things}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FF_DIR="$PROJECT_ROOT/models/FlowFormer"
CKPT_DIR="$FF_DIR/checkpoints"

declare -A FILE_IDS=(
    [things]="1ftCNSFLMegbU39WkDmD6eGNpkuLPtC65"
    [sintel]="1-Vjm5RBpfQ0F-MikRuMIjMuoBRidxGFg"
    [kitti]="1NMaA_i0ElFQIHm_SryU2mgxYRPBT6Nxz"
)

if [[ -z "${FILE_IDS[$CHECKPOINT]+_}" ]]; then
    echo "Unknown checkpoint '$CHECKPOINT'. Choose from: things, sintel, kitti"
    exit 1
fi

# 1. Clone repo
if [ ! -d "$FF_DIR/.git" ]; then
    echo "[setup] Cloning FlowFormer-Official..."
    git clone https://github.com/drinkingcoder/FlowFormer-Official.git "$FF_DIR"
else
    echo "[setup] FlowFormer repo already present."
fi

# 2. Create checkpoints dir
mkdir -p "$CKPT_DIR"

# 3. Download checkpoint
CKPT_PATH="$CKPT_DIR/$CHECKPOINT.pth"
if [ ! -f "$CKPT_PATH" ]; then
    FILE_ID="${FILE_IDS[$CHECKPOINT]}"
    echo "[setup] Downloading $CHECKPOINT.pth via gdown..."
    gdown "$FILE_ID" -O "$CKPT_PATH" || {
        echo "[setup] gdown failed. Download manually:"
        echo "  https://drive.google.com/file/d/$FILE_ID"
        echo "  and place at: $CKPT_PATH"
    }
else
    echo "[setup] Checkpoint $CHECKPOINT.pth already present."
fi

echo ""
echo "[setup] Done. FlowFormer is ready at:"
echo "  Repo : $FF_DIR"
echo "  Ckpt : $CKPT_PATH"
echo ""
echo "Run inference with:"
echo "  py -3.10 scripts/run_flowformer.py --checkpoint $CHECKPOINT"
