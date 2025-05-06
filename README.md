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

## Running Experiments

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
torchrun --nproc_per_node=N src/experiments/benchmark.py --attention_type mka
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

## License
MIT License