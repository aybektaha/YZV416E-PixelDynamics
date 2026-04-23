# YZV416E-PixelDynamics

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red.svg)

A Computer Vision project focused on Video Object Segmentation leveraging motion cues and region growing techniques.

## Overview

This project aims to perform accurate motion-based segmentation on video sequences, specifically targeting the **DAVIS (Densely-Annotated Video Segmentation) dataset**. By integrating optical flow features with classical computer vision algorithms, we aim to robustly segment moving objects from their backgrounds.

The core pipeline utilizes **FlowFormer** for high-quality optical flow extraction. These extracted motion cues are then used to seed and guide a **Region Growing** algorithm to produce precise object masks.

## Features

- **High-Fidelity Optical Flow:** Integration with FlowFormer to extract dense, accurate motion fields between video frames.
- **Motion-Based Region Growing:** Custom implementation of a region growing algorithm that propagates based on optical flow similarities rather than just color or intensity.
- **DAVIS Dataset Support:** Data loaders and evaluation scripts specifically tailored for the DAVIS dataset benchmarks.

## Project Structure

```
YZV416E-PixelDynamics/
├── data/               # Directory for storing the DAVIS dataset and any pre-processed files
├── models/             # Directory for FlowFormer checkpoints and other model weights
├── notebooks/          # Jupyter notebooks for experimentation, visualization, and EDA
├── scripts/            # Source code for the project
│   ├── flow_extraction.py  # Script to run FlowFormer and extract optical flow
│   └── region_growing.py   # Implementation of the motion-based region growing algorithm
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd YZV416E-PixelDynamics
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Roadmap

- [ ] Setup project scaffolding and basic folder structure.
- [ ] Implement initial 2D Region Growing algorithm.
- [ ] Integrate FlowFormer for optical flow extraction.
- [ ] Download and prepare the DAVIS dataset.
- [ ] Adapt region growing to utilize multi-dimensional optical flow vectors.
