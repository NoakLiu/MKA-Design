import torch
import torch.nn as nn
import torch.nn.functional as F

class FastMKA(nn.Module):
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
            nn.Linear(self.embed_dim, 3)
        )

    def forward(
        self, hidden_states, layer_past=None, attention_mask=None,
        head_mask=None, use_cache=False, output_attentions=False
    ):
        B, T, C = hidden_states.size()

        # Project Q once
        query = self.q_proj(hidden_states)
        query = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Routing logits for soft selection of memory
        routing_logits = self.routing_mlp(hidden_states)  # (B, T, 3)
        routing_weights = F.softmax(routing_logits, dim=-1)  # (B, T, 3)

        # Prepare memories
        L1 = hidden_states  # Local
        L2 = hidden_states.mean(dim=1, keepdim=True).expand(-1, T, -1)  # Global
        L3 = torch.zeros_like(hidden_states)  # Long-term (placeholder)

        # Concatenate all memories: [B, T, 3, C]
        stacked_memory = torch.stack([L1, L2, L3], dim=2)

        # Merge memory with routing weights
        # (B, T, 3, C) × (B, T, 3, 1) → (B, T, C)
        routed_memory = (stacked_memory * routing_weights.unsqueeze(-1)).sum(dim=2)

        # Compute K/V from routed memory once
        key = self.k_proj(routed_memory)
        value = self.v_proj(routed_memory)

        key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Append cache
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights += attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_probs,)

        return outputs
