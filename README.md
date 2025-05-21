# MKA - An Efficient Memorized Key Attention Design

## Overview
MKA (Memorized Key Attention) is an efficient attention mechanism designed to improve the performance of transformer-based models. This project implements and compares different attention mechanisms including MHA (Multi-Head Attention), MLA (Multi-Query Attention), MQA (Multi-Query Attention), GQA (Grouped-Query Attention), and our proposed MKA.

## Environment Setup

### Prerequisites
- CUDA-capable GPU (NVIDIA)
- Conda package manager
- Python 3.10

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/MKA-Design.git
cd MKA-Design
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Activate the environment:
```bash
conda activate mka
```

4. Test cuda environment:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'Not available')"
```

## Running Experiments

Setup root path before runing the code
```bash
# Add PYTHONPATH to environment
echo 'export PYTHONPATH=$PYTHONPATH:'$(pwd) >> ~/.bashrc
source ~/.bashrc
```

For numpy issue (<=2.00) often met after setup.sh then use follows to solve
```bash
(mka) root@dsw-238653-84cd4c6f57-5cdq7:/mnt/workspace/MKA-Design# pip uninstall -y numpy
pip install "numpy<2.0.0"
Found existing installation: numpy 2.0.1
Uninstalling numpy-2.0.1:
  Successfully uninstalled numpy-2.0.1
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
Looking in indexes: https://mirrors.aliyun.com/pypi/simple/
Collecting numpy<2.0.0
  Downloading https://mirrors.aliyun.com/pypi/packages/4b/d7/ecf66c1cd12dc28b4040b15ab4d17b773b87fa9d29ca16125de01adb36cd/numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 267.4 kB/s eta 0:00:00
Installing collected packages: numpy
Successfully installed numpy-1.26.4
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
```

### Single GPU Training
```bash
# Basic training with default settings
python src/experiments/benchmark.py

# Training with specific attention mechanism
python src/experiments/benchmark.py --attention_type mka

# Training with custom batch size and epochs
python src/experiments/benchmark.py --batch_size 8 --epochs 3
```

### Multi-GPU Training (Single Machine)
```bash
# Training on all available GPUs
torchrun --nproc_per_node=N src/experiments/benchmark.py

# Example: Training on 4 GPUs
torchrun --nproc_per_node=4 src/experiments/benchmark.py

# Training with specific attention mechanism
torchrun --nproc_per_node=8 src/experiments/benchmark.py --attention_type mka
```

### Multi-Node Training
```bash
# On the first node (master)
torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=N \
    --master_addr="IP_OF_NODE1" \
    --master_port=29500 \
    src/experiments/benchmark.py

# On the second node
torchrun \
    --nnodes=2 \
    --node_rank=1 \
    --nproc_per_node=N \
    --master_addr="IP_OF_NODE1" \
    --master_port=29500 \
    src/experiments/benchmark.py
```

### Advanced Training Options
```bash
# Training with mixed precision
torchrun --nproc_per_node=N src/experiments/benchmark.py --fp16

# Training with gradient accumulation
torchrun --nproc_per_node=N src/experiments/benchmark.py --gradient_accumulation_steps 4

# Training with specific learning rate
torchrun --nproc_per_node=N src/experiments/benchmark.py --learning_rate 1e-4

# Training with specific attention mechanism and custom parameters
torchrun --nproc_per_node=N src/experiments/benchmark.py \
    --attention_type gqa \
    --num_groups 8 \
    --batch_size 16 \
    --epochs 5
```

### Training with DeepSpeed (Optional)
```bash
# Basic DeepSpeed training
deepspeed src/experiments/benchmark.py --deepspeed

# DeepSpeed with specific configuration
deepspeed src/experiments/benchmark.py \
    --deepspeed \
    --deepspeed_config ds_config.json
```

### Training with Fast-MKA
```bash
# 单GPU测试
python src/experiments/benchmark.py --attention_type fmka

# 分布式测试（使用所有可用GPU）
torchrun --nproc_per_node=8 src/experiments/benchmark.py --attention_type fmka

# 开启混合精度加速
torchrun --nproc_per_node=8 src/experiments/benchmark.py --attention_type fmka --fp16
```

```bash
# 单GPU示例
python examples/fmka_example.py

# 分布式示例
python examples/distributed_fmka_example.py
```

## Project Structure
```
mka-design/
├── src/
│   ├── attention/
│   │   ├── mha.py
│   │   ├── mka.py
│   │   ├── mqa.py
│   │   ├── gqa.py
│   │   └── mla.py
│   ├── models/
│   │   └── gpt2.py
│   ├── utils/
│   │   ├── data.py
│   │   ├── metrics.py
│   │   └── distributed.py
│   └── experiments/
│       └── benchmark.py
├── tests/
│   ├── test_attention.py
│   ├── test_model.py
│   ├── test_distributed.py
│   └── test_benchmark.py
├── setup.sh
├── pytest.ini
├── requirements.txt
└── README.md
```

## Requirements
- Python 3.10+
- PyTorch 2.2.1+ with CUDA support
- Transformers 4.30.0+
- Datasets 2.12.0+
- Tqdm 4.65.0+
- NumPy < 2.0.0
- Pytest 7.3.1+
- Accelerate 0.20.0+ (for distributed training)
- DeepSpeed 0.9.0+ (optional, for advanced distributed training)

## Result

MKA
```bash
(mka) root@dsw-238653-84cd4c6f57-5cdq7:/mnt/workspace/MKA-Design# torchrun --nproc_per_node=8 src/experiments/benchmark.py --attention_type mka
[2025-05-18 22:34:42,582] torch.distributed.run: [WARNING] 
[2025-05-18 22:34:42,582] torch.distributed.run: [WARNING] *****************************************
[2025-05-18 22:34:42,582] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
[2025-05-18 22:34:42,582] torch.distributed.run: [WARNING] *****************************************
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
[2025-05-18 22:34:46,400] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-18 22:34:46,429] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-18 22:34:46,440] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-18 22:34:46,474] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-18 22:34:46,508] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-18 22:34:46,521] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-18 22:34:46,525] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-18 22:34:46,527] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
🚀 Running distributed training with 8 GPUs
🔁 Loading preprocessed dataset from cache...
🔁 Loading preprocessed dataset from cache...
🔁 Loading preprocessed dataset from cache...

🏁 Testing MKA...
🔁 Loading preprocessed dataset from cache...
🔁 Loading preprocessed dataset from cache...
🔁 Loading preprocessed dataset from cache...
🔁 Loading preprocessed dataset from cache...
🔁 Loading preprocessed dataset from cache...
Using Memorized Key Attention
Using Memorized Key Attention
Using Memorized Key Attention
Using Memorized Key Attention
Using Memorized Key Attention
Using Memorized Key Attention
Using Memorized Key Attention
Using Memorized Key Attention
Epoch 0:   0%|                                                                                                        | 0/1148 [00:00<?, ?it/s]`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [04:06<00:00,  4.66it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [04:06<00:00,  4.66it/s]

Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [04:06<00:00,  4.66it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [04:06<00:00,  4.66it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [04:06<00:00,  4.66it/s]
✅ Epoch 0 Loss: 1.0702
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [04:06<00:00,  4.66it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [04:06<00:00,  4.66it/s]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [01:24<00:00, 13.63it/s]

Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [01:24<00:00, 13.63it/s]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [01:24<00:00, 13.63it/s]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [01:24<00:00, 13.63it/s]



📊 Results for MKA:
  Train Loss: 1.0702
  Eval Loss: 0.8805
  Perplexity: 2.41
  Train Time: 248.44s
  Eval Time: 84.23s
  Memory Allocated: 9134.62MB
  Memory Reserved: 9728.00MB
```

Fast MKA
```
(mka) root@dsw-238653-84cd4c6f57-5cdq7:/mnt/workspace/MKA-Design# torchrun --nproc_per_node=8 src/experiments/benchmark.py --attention_type fmka
[2025-05-21 21:14:22,743] torch.distributed.run: [WARNING] 
[2025-05-21 21:14:22,743] torch.distributed.run: [WARNING] *****************************************
[2025-05-21 21:14:22,743] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
[2025-05-21 21:14:22,743] torch.distributed.run: [WARNING] *****************************************
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
[2025-05-21 21:14:26,426] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-21 21:14:26,448] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-21 21:14:26,468] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-21 21:14:26,490] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-21 21:14:26,497] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-21 21:14:26,514] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-21 21:14:26,518] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-21 21:14:26,521] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
🚀 Running distributed training with 8 GPUs
💾 Processing and caching dataset for the first time...
💾 Processing and caching dataset for the first time...
💾 Processing and caching dataset for the first time...
💾 Processing and caching dataset for the first time...
💾 Processing and caching dataset for the first time...
💾 Processing and caching dataset for the first time...
💾 Processing and caching dataset for the first time...
💾 Processing and caching dataset for the first time...
Generating test split: 100%|████████████████████████████████████████████████████████████████████| 4358/4358 [00:00<00:00, 611534.86 examples/s]
Generating train split: 100%|████████████████████████████████████████████████████████████████| 36718/36718 [00:00<00:00, 1257647.27 examples/s]
Generating validation split: 100%|██████████████████████████████████████████████████████████████| 3760/3760 [00:00<00:00, 987389.37 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 36718/36718 [00:17<00:00, 2127.22 examples/s]

🏁 Testing FMKA...
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 36718/36718 [00:17<00:00, 2125.67 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 36718/36718 [00:17<00:00, 2092.54 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 36718/36718 [00:17<00:00, 2109.22 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 36718/36718 [00:17<00:00, 2075.04 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 36718/36718 [00:17<00:00, 2116.25 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 36718/36718 [00:17<00:00, 2119.63 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 36718/36718 [00:17<00:00, 2111.10 examples/s]
Using Fast Memorized Key Attention
Using Fast Memorized Key Attention
Using Fast Memorized Key Attention
Using Fast Memorized Key Attention
Using Fast Memorized Key Attention
Using Fast Memorized Key Attention
Using Fast Memorized Key Attention
Using Fast Memorized Key Attention
Epoch 0:   0%|                                                                                                        | 0/1148 [00:00<?, ?it/s]`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [02:42<00:00,  7.08it/s]


Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [02:42<00:00,  7.08it/s]✅ Epoch 0 Loss: 1.0880

Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [02:42<00:00,  7.08it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [02:42<00:00,  7.08it/s]

Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [02:42<00:00,  7.08it/s]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [00:52<00:00, 21.93it/s]


Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [00:52<00:00, 21.93it/s]


Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [00:52<00:00, 21.93it/s]

📊 Results for FMKA:
  Train Loss: 1.0880
  Eval Loss: 0.8900
  Perplexity: 2.44
  Train Time: 165.69s
  Eval Time: 52.36s
  Memory Allocated: 7257.48MB
  Memory Reserved: 7804.00MB
```

Full experiment
```bash
(mka) root@dsw-238653-84cd4c6f57-5cdq7:/mnt/workspace/MKA-Design# torchrun --nproc_per_node=8 src/experiments/benchmark.py
[2025-05-21 22:21:34,202] torch.distributed.run: [WARNING] 
[2025-05-21 22:21:34,202] torch.distributed.run: [WARNING] *****************************************
[2025-05-21 22:21:34,202] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
[2025-05-21 22:21:34,202] torch.distributed.run: [WARNING] *****************************************
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
/root/miniconda3/envs/mka/lib/python3.10/site-packages/_distutils_hack/__init__.py:53: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
[2025-05-21 22:21:37,883] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-21 22:21:37,909] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-21 22:21:37,922] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-21 22:21:37,962] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-21 22:21:37,978] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-21 22:21:37,985] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-21 22:21:37,990] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-21 22:21:37,995] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
🚀 Running distributed training with 8 GPUs
🔁 Loading preprocessed dataset from cache...
🔁 Loading preprocessed dataset from cache...
🔁 Loading preprocessed dataset from cache...
🔁 Loading preprocessed dataset from cache...
🔁 Loading preprocessed dataset from cache...
🔁 Loading preprocessed dataset from cache...

🏁 Testing MHA...
🔁 Loading preprocessed dataset from cache...
🔁 Loading preprocessed dataset from cache...
Using Multi-Head Attention
Using Multi-Head Attention
Using Multi-Head Attention
Using Multi-Head Attention
Using Multi-Head Attention
Using Multi-Head Attention
Using Multi-Head Attention
Using Multi-Head Attention
Epoch 0:   0%|                                                                                                        | 0/1148 [00:00<?, ?it/s]`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [03:43<00:00,  5.13it/s]

Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [03:43<00:00,  5.13it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [03:43<00:00,  5.13it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [03:43<00:00,  5.13it/s]
✅ Epoch 0 Loss: 1.1807

Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [03:43<00:00,  5.13it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [03:43<00:00,  5.13it/s]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [00:48<00:00, 23.82it/s]

Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [00:48<00:00, 23.82it/s]

Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [00:48<00:00, 23.82it/s]



📊 Results for MHA:
  Train Loss: 1.1807
  Eval Loss: 0.9324
  Perplexity: 2.54
  Train Time: 226.09s
  Eval Time: 48.19s
  Memory Allocated: 6763.91MB
  Memory Reserved: 7380.00MB

🏁 Testing MKA...
Using Memorized Key Attention
Using Memorized Key Attention
Using Memorized Key Attention
Using Memorized Key Attention
Using Memorized Key Attention
Using Memorized Key Attention
Using Memorized Key Attention
Using Memorized Key Attention
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [10:59<00:00,  1.74it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [10:59<00:00,  1.74it/s]

Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [10:59<00:00,  1.74it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [10:59<00:00,  1.74it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [10:59<00:00,  1.74it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [10:59<00:00,  1.74it/s]
✅ Epoch 0 Loss: 1.0708
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [10:59<00:00,  1.74it/s]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [03:18<00:00,  5.79it/s]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [03:18<00:00,  5.79it/s]






📊 Results for MKA:
  Train Loss: 1.0708
  Eval Loss: 0.8844
  Perplexity: 2.42
  Train Time: 659.18s
  Eval Time: 198.15s
  Memory Allocated: 9134.67MB
  Memory Reserved: 9852.00MB

🏁 Testing MQA...
Using Multi-Query Attention
Using Multi-Query Attention
Using Multi-Query Attention
Using Multi-Query Attention
Using Multi-Query Attention
Using Multi-Query Attention
Using Multi-Query Attention
Using Multi-Query Attention
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [06:35<00:00,  2.90it/s]

Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [06:35<00:00,  2.90it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [06:35<00:00,  2.90it/s]

Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [06:35<00:00,  2.90it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [06:35<00:00,  2.90it/s]
✅ Epoch 0 Loss: 1.1462
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [06:35<00:00,  2.90it/s]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [04:04<00:00,  4.69it/s]







📊 Results for MQA:
  Train Loss: 1.1462
  Eval Loss: 0.9181
  Perplexity: 2.50
  Train Time: 395.30s
  Eval Time: 244.68s
  Memory Allocated: 9134.67MB
  Memory Reserved: 9886.00MB

🏁 Testing GQA...
Using Grouped-Query Attention with 4 groups
Using Grouped-Query Attention with 4 groups
Using Grouped-Query Attention with 4 groups
Using Grouped-Query Attention with 4 groups
Using Grouped-Query Attention with 4 groups
Using Grouped-Query Attention with 4 groups
Using Grouped-Query Attention with 4 groups
Using Grouped-Query Attention with 4 groups
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [12:58<00:00,  1.48it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [12:58<00:00,  1.48it/s]

Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [12:58<00:00,  1.48it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [12:58<00:00,  1.48it/s]✅ Epoch 0 Loss: 1.1223

Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [12:58<00:00,  1.48it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [12:58<00:00,  1.48it/s]

Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [04:54<00:00,  3.90it/s]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [04:54<00:00,  3.90it/s]






📊 Results for GQA:
  Train Loss: 1.1223
  Eval Loss: 0.9113
  Perplexity: 2.49
  Train Time: 778.62s
  Eval Time: 294.46s
  Memory Allocated: 9134.67MB
  Memory Reserved: 9956.00MB

🏁 Testing MLA...
Using Multi-Latent Attention
Using Multi-Latent Attention
Using Multi-Latent Attention
Using Multi-Latent Attention
Using Multi-Latent Attention
Using Multi-Latent Attention
Using Multi-Latent Attention
Using Multi-Latent Attention
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [13:01<00:00,  1.47it/s]

Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [13:01<00:00,  1.47it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [13:01<00:00,  1.47it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [13:01<00:00,  1.47it/s]Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [13:01<00:00,  1.47it/s]



Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [13:01<00:00,  1.47it/s]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [01:27<00:00, 13.17it/s]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [01:27<00:00, 13.17it/s]






📊 Results for MLA:
  Train Loss: 0.9988
  Eval Loss: 0.8858
  Perplexity: 2.43
  Train Time: 782.43s
  Eval Time: 87.20s
  Memory Allocated: 9134.67MB
  Memory Reserved: 10540.00MB

🏁 Testing FMKA...
Using Fast Memorized Key Attention
Using Fast Memorized Key Attention
Using Fast Memorized Key Attention
Using Fast Memorized Key Attention
Using Fast Memorized Key Attention
Using Fast Memorized Key Attention
Using Fast Memorized Key Attention
Using Fast Memorized Key Attention
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [02:41<00:00,  7.10it/s]

Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [02:41<00:00,  7.10it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [02:41<00:00,  7.10it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [02:41<00:00,  7.10it/s]

✅ Epoch 0 Loss: 1.1088
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [02:41<00:00,  7.10it/s]
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [02:41<00:00,  7.10it/s]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [00:52<00:00, 21.97it/s]
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1148/1148 [00:52<00:00, 21.97it/s]






📊 Results for FMKA:
  Train Loss: 1.1088
  Eval Loss: 0.9204
  Perplexity: 2.51
  Train Time: 161.77s
  Eval Time: 52.25s
  Memory Allocated: 9134.67MB
  Memory Reserved: 10540.00MB
```

## License
MIT License