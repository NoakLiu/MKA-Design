import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from tqdm import tqdm
from math import exp
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# CONFIGURATION
# ---------------------------
CACHE_DIR = "./.cache/wikitext2"
ENCODED_PATH = os.path.join(CACHE_DIR, "encoded_wikitext.pt")
os.makedirs(CACHE_DIR, exist_ok=True)
MAX_LEN = 512

def encode(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=MAX_LEN)

# ---------------------------
# FMKA ATTENTION MODULE
# ---------------------------
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
            nn.Linear(self.embed_dim, 3)  # 3 memory sources
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
def get_model(attn_type="fmka"):
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config)
    if attn_type == "fmka":
        for block in model.transformer.h:
            block.attn = FastMKA(config)
    return model

# ---------------------------
# TRAIN/EVAL FUNCTIONS
# ---------------------------
def train_epoch(model, optimizer, dataloader, device, rank):
    model.train()
    sampler = dataloader.sampler
    sampler.set_epoch(epoch)
    total_loss = 0.0
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"[Rank {rank}] Training")
    else:
        pbar = dataloader
    for batch in pbar:
        inputs = batch['input_ids'].to(device, non_blocking=True)
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if rank == 0:
        print(f"[Rank {rank}] Avg loss: {total_loss/len(dataloader):.4f}")

def evaluate(model, dataloader, device, rank):
    model.eval()
    total_loss = 0.0
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"[Rank {rank}] Evaluating")
    else:
        pbar = dataloader
    with torch.no_grad():
        for batch in pbar:
            inputs = batch['input_ids'].to(device, non_blocking=True)
            loss = model(inputs, labels=inputs).loss
            total_loss += loss.item()
    if rank == 0:
        ppl = exp(total_loss/len(dataloader))
        print(f"[Rank {rank}] Perplexity: {ppl:.2f}")

# ---------------------------
# MAIN WORKER
# ---------------------------
def main_worker(rank, world_size):
    # 1) 初始化进程组
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # 2) Tokenizer & Dataset（只在 rank0 做一次缓存）
    global tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    if rank == 0 and not os.path.exists(ENCODED_PATH):
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=CACHE_DIR)
        ds = raw['train'].map(encode, batched=True)
        ds.set_format(type='torch', columns=['input_ids'])
        torch.save(ds, ENCODED_PATH)
    # 等待 rank0 完成缓存
    dist.barrier()

    # 所有进程加载已经缓存好的
    encoded = torch.load(ENCODED_PATH)
    sampler = DistributedSampler(encoded, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(encoded, batch_size=4, sampler=sampler, num_workers=2, pin_memory=True)

    # 3) 构建模型 & DDP 包装
    model = get_model(attn_type="fmka")
    model.cuda(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 4) 训练 & 评估
    global epoch
    epochs = 1
    for epoch in range(epochs):
        train_epoch(model, optimizer, dataloader, device, rank)
        if rank == 0:
            evaluate(model, dataloader, device, rank)

    # 5) 清理
    dist.destroy_process_group()

# ---------------------------
# ENTRY POINT
# ---------------------------
if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # 使用所有可用 GPU
    mp.spawn(main_worker,
             args=(world_size,),
             nprocs=world_size,
             join=True) 