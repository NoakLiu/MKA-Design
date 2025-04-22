import torch
import pytest
from src.models.gpt2 import get_model
from transformers import GPT2Config

@pytest.fixture
def config():
    return GPT2Config(
        hidden_size=768,
        num_attention_heads=12,
        max_position_embeddings=512,
        num_hidden_layers=2
    )

def test_get_model_mha(config):
    model = get_model("mha")
    assert model is not None
    assert isinstance(model, torch.nn.Module)

def test_get_model_mka(config):
    model = get_model("mka")
    assert model is not None
    assert isinstance(model, torch.nn.Module)

def test_get_model_mqa(config):
    model = get_model("mqa")
    assert model is not None
    assert isinstance(model, torch.nn.Module)

def test_get_model_gqa(config):
    model = get_model("gqa", num_groups=4)
    assert model is not None
    assert isinstance(model, torch.nn.Module)

def test_get_model_mla(config):
    model = get_model("mla", num_layers=2)
    assert model is not None
    assert isinstance(model, torch.nn.Module)

def test_model_forward(config):
    attention_types = ["mha", "mka", "mqa", "gqa", "mla"]
    batch_size = 2
    seq_length = 32
    
    for attn_type in attention_types:
        model = get_model(attn_type)
        input_ids = torch.randint(0, 50257, (batch_size, seq_length))
        outputs = model(input_ids)
        
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2  # logits, past_key_values
        assert outputs[0].shape == (batch_size, seq_length, 50257)

def test_model_generation(config):
    model = get_model("mka")
    input_ids = torch.randint(0, 50257, (1, 10))
    
    # Test greedy decoding
    outputs = model.generate(
        input_ids,
        max_length=20,
        num_return_sequences=1,
        do_sample=False
    )
    assert outputs.shape[0] == 1
    assert outputs.shape[1] <= 20

    # Test sampling
    outputs = model.generate(
        input_ids,
        max_length=20,
        num_return_sequences=2,
        do_sample=True,
        temperature=0.7
    )
    assert outputs.shape[0] == 2
    assert outputs.shape[1] <= 20 