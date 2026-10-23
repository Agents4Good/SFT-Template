"""Backwards-compatible import path for model handlers.

New code should use `from src.model.factory import create_model_handler`.
"""

from src.model.factory import create_model_handler
from src.model.hf_loader import HFModel, LoaderError
from src.model.unsloth_loader import UnslothModel

__all__ = [
    "UnslothModel",
    "HFModel",
    "LoaderError",
    "create_model_handler",
]
