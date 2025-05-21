import os
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from math import exp

# ---------------------------
# CONFIGURATION & CACHING
# ---------------------------

CACHE_DIR = "./.cache/wikitext2"
ENCODED_PATH = os.path.join(CACHE_DIR, "encoded_wikitext.pt")
os.makedirs(CACHE_DIR, exist_ok=True)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def encode(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

if os.path.exists(ENCODED_PATH):
    print("🔁 Loading preprocessed dataset from cache...")
    encoded_dataset = torch.load(ENCODED_PATH)
else:
    print("💾 Processing and caching dataset for the first time...")
    raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=CACHE_DIR)
    encoded_dataset = raw_dataset['train'].map(encode, batched=True)
    encoded_dataset.set_format(type='torch', columns=['input_ids'])
    torch.save(encoded_dataset, ENCODED_PATH)

data_loader = DataLoader(encoded_dataset, batch_size=4, shuffle=True)

# ---------------------------
# FMKA ATTENTION MODULE
# ---------------------------

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

# ---------------------------
# MODEL SETUP
# ---------------------------

def get_model(attn_type="mha"):
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config)
    if attn_type == "fmka":
        print("🔁 Replacing GPT2Attention with FastMKA...")
        for block in model.transformer.h:
            block.attn = FastMKA(config)
    return model

# ---------------------------
# TRAINING LOOP
# ---------------------------

def train_model(model, data_loader, epochs=1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"✅ Epoch {epoch} Loss: {total_loss / len(data_loader):.4f}")

# ---------------------------
# EVALUATION
# ---------------------------

def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(model.device)
            loss = model(input_ids, labels=input_ids).loss
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    print(f"📊 Perplexity: {exp(avg_loss):.2f}")

# ---------------------------
# RUN EXPERIMENT
# ---------------------------

if __name__ == "__main__":
    print("\n🚀 Training GPT2 with FastMKA...")
    model_fmka = get_model("fmka")
    train_model(model_fmka, data_loader, epochs=1)
    evaluate_model(model_fmka, data_loader) 