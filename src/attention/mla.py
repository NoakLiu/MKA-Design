import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerAttention(nn.Module):
    def __init__(self, config, num_layers=2):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_layers = num_layers
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(self.embed_dim, self.embed_dim),
                'k_proj': nn.Linear(self.embed_dim, self.embed_dim),
                'v_proj': nn.Linear(self.embed_dim, self.embed_dim),
                'out_proj': nn.Linear(self.embed_dim, self.embed_dim),
                'layer_norm': nn.LayerNorm(self.embed_dim)
            }) for _ in range(num_layers)
        ])

    def forward(
        self, hidden_states, layer_past=None, attention_mask=None,
        head_mask=None, use_cache=False, output_attentions=False
    ):
        B, T, C = hidden_states.size()
        all_attentions = [] if output_attentions else None

        for i, layer in enumerate(self.attention_layers):
            residual = hidden_states
            hidden_states = layer['layer_norm'](hidden_states)

            query = layer['q_proj'](hidden_states)
            key = layer['k_proj'](hidden_states)
            value = layer['v_proj'](hidden_states)

            query = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            if layer_past is not None:
                past_key, past_value = layer_past
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

            present = (key, value) if use_cache else None

            attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_probs = nn.Softmax(dim=-1)(attn_weights)
            if output_attentions:
                all_attentions.append(attn_probs)

            attn_output = torch.matmul(attn_probs, value)
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
            attn_output = layer['out_proj'](attn_output)

            hidden_states = residual + attn_output

        outputs = (hidden_states, present)
        if output_attentions:
            outputs += (all_attentions,)

        return outputs 