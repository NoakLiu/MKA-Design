import torch
from transformers import GPT2Config, GPT2LMHeadModel
from ..attention.mha import MultiHeadAttention
from ..attention.mka import MKAForGPT2Attention
from ..attention.mqa import MultiQueryAttention
from ..attention.gqa import GroupedQueryAttention
from ..attention.mla import MultiLayerAttention

def get_model(attn_type="mha", **kwargs):
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config)
    
    if attn_type == "mha":
        print("Using Multi-Head Attention")
        for block in model.transformer.h:
            block.attn = MultiHeadAttention(config)
    elif attn_type == "mka":
        print("Using Memorized Key Attention")
        for block in model.transformer.h:
            block.attn = MKAForGPT2Attention(config)
    elif attn_type == "mqa":
        print("Using Multi-Query Attention")
        for block in model.transformer.h:
            block.attn = MultiQueryAttention(config)
    elif attn_type == "gqa":
        num_groups = kwargs.get("num_groups", 4)
        print(f"Using Grouped-Query Attention with {num_groups} groups")
        for block in model.transformer.h:
            block.attn = GroupedQueryAttention(config, num_groups=num_groups)
    elif attn_type == "mla":
        num_layers = kwargs.get("num_layers", 2)
        print(f"Using Multi-Layer Attention with {num_layers} layers")
        for block in model.transformer.h:
            block.attn = MultiLayerAttention(config, num_layers=num_layers)
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")
    
    return model 