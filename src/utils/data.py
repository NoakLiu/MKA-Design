import os
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader

CACHE_DIR = "./.cache/wikitext2"
ENCODED_PATH = os.path.join(CACHE_DIR, "encoded_wikitext.pt")

def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def encode(example, tokenizer):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

def get_dataloader(batch_size=4):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tokenizer = get_tokenizer()

    if os.path.exists(ENCODED_PATH):
        print("🔁 Loading preprocessed dataset from cache...")
        encoded_dataset = torch.load(ENCODED_PATH)
    else:
        print("💾 Processing and caching dataset for the first time...")
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=CACHE_DIR)
        encoded_dataset = raw_dataset['train'].map(
            lambda x: encode(x, tokenizer),
            batched=True
        )
        encoded_dataset.set_format(type='torch', columns=['input_ids'])
        torch.save(encoded_dataset, ENCODED_PATH)

    return DataLoader(encoded_dataset, batch_size=batch_size, shuffle=True) 