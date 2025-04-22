import torch
import torch.nn as nn
import torch.nn.functional as F

class MKAForGPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.routing_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Tanh(),
            nn.Linear(self.embed_dim, 3)  # 3 memory sources: L1, L2, L3
        )

    def forward(
        self, hidden_states, layer_past=None, attention_mask=None,
        head_mask=None, use_cache=False, output_attentions=False
    ):
        B, T, C = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        # routing weights
        gate_logits = self.routing_mlp(hidden_states)
        lambdas = F.softmax(gate_logits, dim=-1)  # (B, T, 3)

        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = nn.Softmax(dim=-1)(attn_weights)
        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_probs,)

        return outputs 