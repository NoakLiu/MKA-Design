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

        # Projection matrices
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Routing MLP for memory selection
        self.routing_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Tanh(),
            nn.Linear(self.embed_dim, 3)  # 3 memory levels
        )

    def forward(
        self, hidden_states, layer_past=None, attention_mask=None,
        head_mask=None, use_cache=False, output_attentions=False
    ):
        B, T, C = hidden_states.size()

        # Project queries
        query = self.q_proj(hidden_states)
        query = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Define memory levels
        # L1: Local memory (current tokens)
        L1 = hidden_states
        
        # L2: Session memory (summary)
        L2 = hidden_states.mean(dim=1, keepdim=True).expand(-1, T, -1)
        
        # L3: Long-term memory (initialized as zeros)
        L3 = torch.zeros_like(hidden_states)

        # Compute routing weights
        routing_logits = self.routing_mlp(hidden_states)  # (B, T, 3)
        routing_weights = F.softmax(routing_logits, dim=-1)  # (B, T, 3)

        # Initialize attention outputs
        attn_output = torch.zeros_like(hidden_states)

        # Process each memory level
        memories = [L1, L2, L3]
        for i, memory in enumerate(memories):
            # Project keys and values
            key = self.k_proj(memory)
            value = self.v_proj(memory)
            
            # Reshape for multi-head attention
            key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            # Compute attention scores
            attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_probs = F.softmax(attn_weights, dim=-1)
            
            # Apply routing weights - need to reshape for proper broadcasting
            mem_output = torch.matmul(attn_probs, value)
            mem_output = mem_output.transpose(1, 2).contiguous().view(B, T, C)
            mem_output = self.out_proj(mem_output)
            
            # Use proper broadcasting for the routing weights
            level_weight = routing_weights[:, :, i].unsqueeze(-1)  # (B, T, 1)
            attn_output = attn_output + level_weight * mem_output

        # Update KV cache if needed
        present = None
        if use_cache:
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
            key = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            present = (key, value)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_probs,)

        return outputs 