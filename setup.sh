#!/bin/bash

# Create conda environment
conda create -n mka python=3.10 -y
conda activate mka

# Install PyTorch and torchvision with specific versions
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install CUDA toolkit
conda install -c nvidia cuda-toolkit=11.8 -y

# Install other dependencies (fix command syntax)
pip install "transformers>=4.30.0"
pip install "datasets>=2.12.0"
pip install "tqdm>=4.65.0"
pip install "numpy<2.0.0"  # Ensure numpy 1.x for compatibility
pip install "pytest>=7.3.1"
pip install "accelerate>=0.20.0"
pip install "deepspeed>=0.9.0"

# Create src directory if it doesn't exist
mkdir -p src

# Create __init__.py files for proper package structure
touch src/__init__.py
touch src/attention/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py
touch src/experiments/__init__.py

# Install the package in development mode
pip install -e .

# Add PYTHONPATH to environment
echo 'export PYTHONPATH=$PYTHONPATH:'$(pwd) >> ~/.bashrc
source ~/.bashrc

echo "MKA environment setup complete!"
echo "To activate the environment, run: conda activate mka" 