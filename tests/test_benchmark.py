import pytest
from src.experiments.benchmark import run_benchmark, parse_args

def test_parse_args():
    # Test default arguments
    args = parse_args()
    assert args.attention_type is None
    assert args.batch_size == 4
    assert args.epochs == 1
    assert args.learning_rate == 5e-5
    assert args.num_groups == 4
    assert args.num_layers == 2
    assert not args.fp16
    assert args.gradient_accumulation_steps == 1
    assert not args.deepspeed
    assert args.deepspeed_config is None

def test_benchmark_single_attention():
    # Test benchmarking a single attention mechanism
    results = run_benchmark(
        attention_types=["mha"],
        batch_size=2,
        epochs=1
    )
    
    assert "mha" in results
    assert isinstance(results["mha"]["train_loss"], float)
    assert isinstance(results["mha"]["eval_loss"], float)
    assert isinstance(results["mha"]["perplexity"], float)
    assert isinstance(results["mha"]["train_time"], float)
    assert isinstance(results["mha"]["eval_time"], float)

def test_benchmark_all_attentions():
    # Test benchmarking all attention mechanisms
    results = run_benchmark(
        batch_size=2,
        epochs=1
    )
    
    expected_types = ["mha", "mka", "mqa", "gqa", "mla"]
    for attn_type in expected_types:
        assert attn_type in results
        assert isinstance(results[attn_type]["train_loss"], float)
        assert isinstance(results[attn_type]["eval_loss"], float)
        assert isinstance(results[attn_type]["perplexity"], float)

def test_benchmark_with_fp16():
    # Test benchmarking with mixed precision
    results = run_benchmark(
        attention_types=["mha"],
        batch_size=2,
        epochs=1,
        fp16=True
    )
    
    assert "mha" in results
    assert isinstance(results["mha"]["train_loss"], float)

def test_benchmark_with_gradient_accumulation():
    # Test benchmarking with gradient accumulation
    results = run_benchmark(
        attention_types=["mha"],
        batch_size=2,
        epochs=1,
        gradient_accumulation_steps=2
    )
    
    assert "mha" in results
    assert isinstance(results["mha"]["train_loss"], float)

def test_benchmark_with_custom_parameters():
    # Test benchmarking with custom parameters
    results = run_benchmark(
        attention_types=["gqa"],
        batch_size=2,
        epochs=1,
        num_groups=8
    )
    
    assert "gqa" in results
    assert isinstance(results["gqa"]["train_loss"], float)

def test_benchmark_distributed():
    # Test distributed benchmarking
    results = run_benchmark(
        attention_types=["mha"],
        batch_size=2,
        epochs=1,
        distributed=True
    )
    
    assert "mha" in results
    assert isinstance(results["mha"]["train_loss"], float) 