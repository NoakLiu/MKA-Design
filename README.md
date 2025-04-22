# MKA - An Efficient Memorized Key Attention Design

## Overview
MKA (Memorized Key Attention) is an efficient attention mechanism designed to improve the performance of transformer-based models. This project implements and compares different attention mechanisms including MHA (Multi-Head Attention), MLA (Multi-Query Attention), MQA (Multi-Query Attention), GQA (Grouped-Query Attention), and our proposed MKA.

## Attention Mechanisms Comparison

### MHA (Multi-Head Attention)
The traditional multi-head attention mechanism computes attention scores for each head independently:

\[
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
\]

where \(Q, K, V\) are the query, key, and value matrices for each head, and \(d_k\) is the dimension of the key vectors. Each head has its own set of projection matrices:

\[
Q_i = XW_i^Q, \quad K_i = XW_i^K, \quad V_i = XW_i^V
\]

where \(X\) is the input sequence and \(W_i^Q, W_i^K, W_i^V\) are learnable projection matrices for head \(i\).

### MQA (Multi-Query Attention)
Multi-query attention shares key and value projections across all heads while maintaining separate query projections:

\[
Q_i = XW_i^Q, \quad K = XW^K, \quad V = XW^V
\]

This reduces memory usage while maintaining reasonable performance.

### GQA (Grouped-Query Attention)
Grouped-query attention divides heads into groups and shares key-value projections within each group:

\[
Q_i = XW_i^Q, \quad K_g = XW_g^K, \quad V_g = XW_g^V
\]

where \(g\) represents the group index, and \(i\) belongs to group \(g\). This provides a balance between MHA and MQA.

### MLA (Multi-Latent Attention)
Multi-latent attention introduces a latent space for attention computation, allowing for more flexible and efficient attention patterns:

\[
Z = \text{MLP}(X)
\]
\[
Q = ZW^Q, \quad K = ZW^K, \quad V = ZW^V
\]
\[
A = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})
\]
\[
\text{Attention}(X) = \text{LayerNorm}(X + AV)
\]

where \(Z\) is the latent representation learned by the MLP, and \(W^Q, W^K, W^V\) are shared projection matrices. This approach reduces the computational complexity while maintaining the ability to capture complex attention patterns through the learned latent space.

### MKA (Memorized Key Attention)
Memorized key attention introduces a memory-based key routing mechanism:

\[
\lambda = \text{softmax}(\text{MLP}(X))
\]
\[
K_{\text{mem}} = \sum_{i=1}^{3} \lambda_i K_i
\]
\[
\text{Attention}(Q, K_{\text{mem}}, V) = \text{softmax}(\frac{QK_{\text{mem}}^T}{\sqrt{d_k}})V
\]

where \(\lambda\) represents the routing weights learned by the MLP, and \(K_i\) are different memory sources. This adaptive key selection improves efficiency while maintaining or improving performance.

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