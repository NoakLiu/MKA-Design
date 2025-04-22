import torch
import pytest
from src.attention.mha import MultiHeadAttention
from src.attention.mka import MKAForGPT2Attention
from src.attention.mqa import MultiQueryAttention
from src.attention.gqa import GroupedQueryAttention
from src.attention.mla import MultiLatentAttention
from transformers import GPT2Config

@pytest.fixture
def config():
    return GPT2Config(
        hidden_size=768,
        num_attention_heads=12,
        max_position_embeddings=512
    )

@pytest.fixture
def sample_input():
    batch_size = 2
    seq_length = 32
    hidden_size = 768
    return torch.randn(batch_size, seq_length, hidden_size)

def test_mha_forward(config, sample_input):
    mha = MultiHeadAttention(config)
    output = mha(sample_input)
    assert isinstance(output, tuple)
    assert len(output) == 2  # output, present
    assert output[0].shape == sample_input.shape

def test_mka_forward(config, sample_input):
    mka = MKAForGPT2Attention(config)
    output = mka(sample_input)
    assert isinstance(output, tuple)
    assert len(output) == 2  # output, present
    assert output[0].shape == sample_input.shape

def test_mqa_forward(config, sample_input):
    mqa = MultiQueryAttention(config)
    output = mqa(sample_input)
    assert isinstance(output, tuple)
    assert len(output) == 2  # output, present
    assert output[0].shape == sample_input.shape

def test_gqa_forward(config, sample_input):
    gqa = GroupedQueryAttention(config, num_groups=4)
    output = gqa(sample_input)
    assert isinstance(output, tuple)
    assert len(output) == 2  # output, present
    assert output[0].shape == sample_input.shape

def test_mla_forward(config, sample_input):
    mla = MultiLatentAttention(config)
    output = mla(sample_input)
    assert isinstance(output, tuple)
    assert len(output) == 2  # output, present
    assert output[0].shape == sample_input.shape

def test_attention_mask(config, sample_input):
    attention_types = [
        MultiHeadAttention(config),
        MKAForGPT2Attention(config),
        MultiQueryAttention(config),
        GroupedQueryAttention(config),
        MultiLatentAttention(config)
    ]
    
    attention_mask = torch.ones(sample_input.shape[0], sample_input.shape[1])
    attention_mask[:, -5:] = 0  # Mask last 5 tokens
    
    for attn in attention_types:
        output = attn(sample_input, attention_mask=attention_mask)
        assert output[0].shape == sample_input.shape

def test_past_key_values(config, sample_input):
    attention_types = [
        MultiHeadAttention(config),
        MKAForGPT2Attention(config),
        MultiQueryAttention(config),
        GroupedQueryAttention(config),
        MultiLatentAttention(config)
    ]
    
    past_key_values = (
        torch.randn(2, 12, 16, 64),  # past_key
        torch.randn(2, 12, 16, 64)   # past_value
    )
    
    for attn in attention_types:
        output = attn(sample_input, layer_past=past_key_values)
        assert output[0].shape == sample_input.shape
        assert isinstance(output[1], tuple)  # present
        assert len(output[1]) == 2  # key, value 