import torch
import time
import os
import argparse
from tqdm import tqdm
from src.models.gpt2 import get_model
from src.utils.data import get_dataloader
from src.utils.metrics import evaluate_model, train_model
from src.utils.distributed import setup_distributed, cleanup_distributed

def parse_args():
    parser = argparse.ArgumentParser(description='Run attention mechanism benchmarks')
    parser.add_argument('--attention_type', type=str, default=None,
                      help='Specific attention mechanism to test (mha, mka, mqa, gqa, mla)')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='Learning rate for training')
    parser.add_argument('--num_groups', type=int, default=4,
                      help='Number of groups for GQA')
    parser.add_argument('--fp16', action='store_true',
                      help='Use mixed precision training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                      help='Number of gradient accumulation steps')
    parser.add_argument('--deepspeed', action='store_true',
                      help='Use DeepSpeed for training')
    parser.add_argument('--deepspeed_config', type=str, default=None,
                      help='Path to DeepSpeed configuration file')
    return parser.parse_args()

def run_benchmark(
    attention_types=["mha", "mka", "mqa", "gqa", "mla"],
    batch_size=4,
    epochs=1,
    learning_rate=5e-5,
    num_groups=4,
    fp16=False,
    gradient_accumulation_steps=4,
    distributed=False
):
    if distributed:
        rank, world_size, gpu = setup_distributed()
        if rank == 0:
            print(f"🚀 Running distributed training with {world_size} GPUs")
    else:
        rank, world_size, gpu = 0, 1, 0

    results = {}
    data_loader = get_dataloader(batch_size=batch_size, distributed=distributed)
    
    # If specific attention type is provided, only test that one
    if attention_types is None:
        attention_types = ["mha", "mka", "mqa", "gqa", "mla"]
    
    for attn_type in attention_types:
        if rank == 0:
            print(f"\n🏁 Testing {attn_type.upper()}...")
        
        # Initialize model with specific parameters
        model = get_model(
            attn_type,
            num_groups=num_groups
        )
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Enable mixed precision if requested
        if fp16:
            model = model.half()
        
        # Training
        train_start = time.time()
        train_loss = train_model(
            model,
            data_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            distributed=distributed,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        train_time = time.time() - train_start
        
        # Evaluation
        eval_start = time.time()
        metrics = evaluate_model(model, data_loader, distributed=distributed)
        eval_time = time.time() - eval_start
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB
        else:
            memory_allocated = memory_reserved = 0
        
        if rank == 0:
            results[attn_type] = {
                "train_loss": train_loss,
                "eval_loss": metrics["loss"],
                "perplexity": metrics["perplexity"],
                "train_time": train_time,
                "eval_time": eval_time,
                "memory_allocated": memory_allocated,
                "memory_reserved": memory_reserved
            }
            
            print(f"📊 Results for {attn_type.upper()}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Eval Loss: {metrics['loss']:.4f}")
            print(f"  Perplexity: {metrics['perplexity']:.2f}")
            print(f"  Train Time: {train_time:.2f}s")
            print(f"  Eval Time: {eval_time:.2f}s")
            if torch.cuda.is_available():
                print(f"  Memory Allocated: {memory_allocated:.2f}MB")
                print(f"  Memory Reserved: {memory_reserved:.2f}MB")
    
    if distributed:
        cleanup_distributed()
    
    return results

if __name__ == "__main__":
    args = parse_args()
    distributed = os.environ.get('WORLD_SIZE', '1') != '1'
    
    # If specific attention type is provided, only test that one
    attention_types = [args.attention_type] if args.attention_type else None
    
    results = run_benchmark(
        attention_types=attention_types,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_groups=args.num_groups,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        distributed=distributed
    ) 