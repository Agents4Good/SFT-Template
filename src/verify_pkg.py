import sys
import os
from unittest.mock import MagicMock

# Mock external ML libraries
sys.modules["unsloth"] = MagicMock()
sys.modules["trl"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["prompts"] = MagicMock()
sys.modules["prompts.train_prompts"] = MagicMock()

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.base import AbstractDatasetProcessor, AbstractSFTTrain
    from model.handler import Model
    from dataset.processor import DatasetProcessor
    from train.sft import SFTTrain
    print("Imports from packages successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def verify_structure():
    assert issubclass(DatasetProcessor, AbstractDatasetProcessor)
    assert issubclass(SFTTrain, AbstractSFTTrain)
    print("Class hierarchy in packages verified!")

if __name__ == "__main__":
    verify_structure()
