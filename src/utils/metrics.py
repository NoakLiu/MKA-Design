import torch
from math import exp
from tqdm import tqdm
from .distributed import reduce_tensor, wrap_model

def evaluate_model(model, data_loader, distributed=False):
    model.eval()
    total_loss = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            loss = model(input_ids, labels=input_ids).loss
            
            if distributed:
                loss = reduce_tensor(loss, torch.distributed.get_world_size())
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    perplexity = exp(avg_loss)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity
    }

def train_model(model, data_loader, epochs=1, learning_rate=5e-5, distributed=False, gradient_accumulation_steps=1):
    if distributed:
        model = wrap_model(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    device = next(model.parameters()).device
    
    for epoch in range(epochs):
        if distributed:
            data_loader.sampler.set_epoch(epoch)
            
        total_loss = 0
        optimizer.zero_grad()  # Reset gradients at the start of epoch
        
        for i, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
            input_ids = batch['input_ids'].to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            if distributed:
                loss = reduce_tensor(loss, torch.distributed.get_world_size())
            
            # Normalize loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Update weights if we've accumulated enough gradients
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
        
        avg_loss = total_loss / len(data_loader)
        if not distributed or torch.distributed.get_rank() == 0:
            print(f"✅ Epoch {epoch} Loss: {avg_loss:.4f}")
    
    return avg_loss 