# MKA - An Efficient Memorized Key Attention Design

## Overview
MKA (Memorized Key Attention) is an efficient attention mechanism designed to improve the performance of transformer-based models. This project implements and compares different attention mechanisms including MHA (Multi-Head Attention), MLA (Multi-Query Attention), MQA (Multi-Query Attention), GQA (Grouped-Query Attention), and our proposed MKA.

## Attention Mechanisms Comparison

### MHA (Multi-Head Attention)
- Traditional attention mechanism used in original transformers
- Each head has its own Q, K, V projections
- High memory usage and computational cost
- Best for tasks requiring high precision

### MQA (Multi-Query Attention)
- Shares K and V projections across all heads
- Reduces memory usage significantly
- Slightly lower performance than MHA
- Good balance between performance and efficiency

### GQA (Grouped-Query Attention)
- Groups heads and shares K, V projections within groups
- Better performance than MQA
- More flexible than MQA
- Good for large models

### MLA (Multi-Layer Attention)
- Uses multiple attention layers
- Higher computational cost
- Better for complex tasks
- Not commonly used due to efficiency concerns

### MKA (Memorized Key Attention)
- Introduces memory-based key routing
- Adaptive key selection based on input
- Better efficiency than MHA
- Maintains or improves performance
- Particularly effective for long sequences

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
├── pytest.ini
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Training Scripts

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

## Usage Examples

### Basic Usage
```python
from src import get_model, run_benchmark

# Get a model with specific attention mechanism
model = get_model("mka")

# Run benchmark comparison
results = run_benchmark()
```

### Custom Training
```python
from src import get_model, run_benchmark
from src.utils.data import get_dataloader
from src.utils.metrics import train_model, evaluate_model

# Get model and data
model = get_model("mka")
data_loader = get_dataloader(batch_size=8)

# Custom training loop
train_model(model, data_loader, epochs=3, learning_rate=1e-4)

# Evaluation
metrics = evaluate_model(model, data_loader)
print(f"Perplexity: {metrics['perplexity']:.2f}")
```

### Distributed Training
```python
from src import get_model, run_benchmark

# Run distributed training
results = run_benchmark(distributed=True)
```

## Requirements
- Python 3.8+
- PyTorch
- Transformers
- Datasets
- Tqdm
- Accelerate (for distributed training)
- DeepSpeed (optional, for advanced distributed training)

## License
MIT License