import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Latent space transformation
        self.latent_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )

        # Shared projection matrices
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self, hidden_states, layer_past=None, attention_mask=None,
        head_mask=None, use_cache=False, output_attentions=False
    ):
        B, T, C = hidden_states.size()

        # Transform input to latent space
        latent = self.latent_mlp(hidden_states)
        
        # Project to query, key, value
        query = self.q_proj(latent)
        key = self.k_proj(latent)
        value = self.v_proj(latent)

        # Reshape for multi-head attention
        query = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        # Compute attention
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = nn.Softmax(dim=-1)(attn_weights)
        if output_attentions:
            all_attentions = [attn_probs]

        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)

        # Add & Norm
        output = self.layer_norm(hidden_states + attn_output)

        outputs = (output, present)
        if output_attentions:
            outputs += (all_attentions,)

        return outputs 