import os
import torch
import pytest
from src.utils.distributed import setup_distributed, cleanup_distributed, get_distributed_sampler, wrap_model, reduce_tensor
from src.models.gpt2 import get_model
from src.utils.data import get_dataloader

@pytest.fixture
def mock_distributed_env(monkeypatch):
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('MASTER_ADDR', 'localhost')
    monkeypatch.setenv('MASTER_PORT', '29500')

def test_setup_distributed(mock_distributed_env):
    rank, world_size, gpu = setup_distributed()
    assert rank == 0
    assert world_size == 2
    assert gpu == 0
    cleanup_distributed()

def test_distributed_sampler():
    # Create a dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.data = torch.randn(size, 10)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = DummyDataset()
    sampler = get_distributed_sampler(dataset)
    assert isinstance(sampler, torch.utils.data.distributed.DistributedSampler)

def test_wrap_model():
    model = get_model("mha")
    wrapped_model = wrap_model(model)
    assert isinstance(wrapped_model, torch.nn.parallel.DistributedDataParallel)

def test_reduce_tensor():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    reduced = reduce_tensor(tensor, world_size=2)
    assert torch.allclose(reduced, tensor / 2)

def test_distributed_training(mock_distributed_env):
    # Setup distributed environment
    rank, world_size, gpu = setup_distributed()
    
    try:
        # Get model and data
        model = get_model("mha")
        data_loader = get_dataloader(batch_size=4, distributed=True)
        
        # Wrap model with DDP
        model = wrap_model(model)
        model.to(f"cuda:{gpu}")
        
        # Test forward pass
        for batch in data_loader:
            input_ids = batch['input_ids'].to(f"cuda:{gpu}")
            outputs = model(input_ids)
            assert outputs[0].shape[0] == input_ids.shape[0]
            break
    
    finally:
        cleanup_distributed()

def test_distributed_evaluation(mock_distributed_env):
    # Setup distributed environment
    rank, world_size, gpu = setup_distributed()
    
    try:
        # Get model and data
        model = get_model("mha")
        data_loader = get_dataloader(batch_size=4, distributed=True)
        
        # Wrap model with DDP
        model = wrap_model(model)
        model.to(f"cuda:{gpu}")
        
        # Test evaluation
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(f"cuda:{gpu}")
                outputs = model(input_ids)
                assert outputs[0].shape[0] == input_ids.shape[0]
                break
    
    finally:
        cleanup_distributed() 