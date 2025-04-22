import torch
from transformers import GPT2Config, GPT2LMHeadModel
from ..attention.mha import MultiHeadAttention
from ..attention.mka import MKAForGPT2Attention
from ..attention.mqa import MultiQueryAttention
from ..attention.gqa import GroupedQueryAttention
from ..attention.mla import MultiLatentAttention

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
        print("Using Multi-Latent Attention")
        for block in model.transformer.h:
            block.attn = MultiLatentAttention(config)
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")
    
    return model 