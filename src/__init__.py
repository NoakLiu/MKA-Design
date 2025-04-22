from .models.gpt2 import get_model
from .experiments.benchmark import run_benchmark

__version__ = "0.1.0"
__all__ = ["get_model", "run_benchmark"] 