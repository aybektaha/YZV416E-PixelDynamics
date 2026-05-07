#!/bin/bash
# Script to download DAVIS 2020 Unsupervised Challenge Dataset
echo "Navigating to data directory..."
mkdir -p ../data
cd ../data

echo "Downloading DAVIS 2017 Unsupervised Train/Val 480p (used for 2020)..."
wget -c https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip

echo "Extracting..."
unzip -q DAVIS-2017-Unsupervised-trainval-480p.zip
rm DAVIS-2017-Unsupervised-trainval-480p.zip

echo "DAVIS Dataset downloaded successfully."
