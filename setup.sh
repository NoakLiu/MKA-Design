#!/bin/bash

# Create conda environment
conda create -n mka python=3.10 -y
conda activate mka

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
pip install transformers>=4.30.0
pip install datasets>=2.12.0
pip install tqdm>=4.65.0
pip install numpy<2.0.0  # Ensure numpy 1.x for compatibility
pip install pytest>=7.3.1
pip install accelerate>=0.20.0
pip install deepspeed>=0.9.0

# Install the package in development mode
pip install -e .

echo "MKA environment setup complete!"
echo "To activate the environment, run: conda activate mka" 