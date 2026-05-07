#!/bin/bash
# Script to clone RAFT and download its model checkpoints
echo "Navigating to models directory..."
mkdir -p ../models
cd ../models

if [ ! -d "RAFT" ]; then
    echo "Cloning RAFT repository..."
    git clone https://github.com/princeton-vl/RAFT.git
fi

cd RAFT
echo "Downloading RAFT checkpoints..."
bash download_models.sh

echo "RAFT setup complete."
