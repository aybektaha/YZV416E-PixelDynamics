#!/bin/bash
# Script to clone RAFT and download its model checkpoints
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Navigating to models directory..."
mkdir -p "$DIR/../models"
cd "$DIR/../models"

if [ ! -d "RAFT" ]; then
    echo "Cloning RAFT repository..."
    git clone https://github.com/princeton-vl/RAFT.git
fi

cd RAFT
echo "Downloading RAFT checkpoints..."
bash download_models.sh

echo "RAFT setup complete."
