# Sign Language Detection System

## Project Overview
A real-time sign language detection system developed as a Capstone Project. This system uses computer vision and deep learning to recognize and interpret sign language gestures.

## Prerequisites
- GPU with CUDA support (T4 recommended)
- Python 3.7+
- NVIDIA drivers installed

## System Requirements
Before running the project:
1. Ensure runtime CPU is set to T4 GPU
2. Verify GPU availability by running:
   ```bash
   nvidia-smi
   ```

## Supported Signs
- Hello
- I Love You
- No
- Thank You
- Yes

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download the dataset from Roboflow
3. Run training using SLD.ipynb