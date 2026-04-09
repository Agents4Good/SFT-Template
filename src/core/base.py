from abc import ABC, abstractmethod

class AbstractDatasetProcessor(ABC):
    """
    Abstract base class for dataset processing.
    Ensures consistent interface for loading, formatting, and splitting datasets.
    """

    @abstractmethod
    def load_dataset(self, path: str):
        """
        Loads the raw dataset from the specified path.
        
        Args:
            path (str): Path to the dataset file or directory.
        """
        pass

    @abstractmethod
    def format_dataset(self, dataset):
        """
        Formats the raw dataset into the structure required for SFT.
        
        Args:
            dataset: The raw dataset object.
        """
        pass
    
    @abstractmethod
    def split_dataset(self, dataset):
        """
        Splits the dataset into training and evaluation sets.
        
        Args:
            dataset: The formatted dataset object.
        """
        pass

    @abstractmethod
    def save_dataset(self, dataset, path: str):
        """
        Saves the processed dataset to disk.
        
        Args:
            dataset: The processed dataset object.
            path (str): Destination path for saving.
        """
        pass

class AbstractSFTTrain(ABC):
    """
    Abstract base class for Supervised Fine-Tuning.
    Defines the standard workflow for training and saving models.
    """

    @abstractmethod
    def train(self):
        """
        Executes the training loop.
        """
        pass
    
    @abstractmethod
    def save_model(self):
        """
        Saves the fine-tuned model and configuration.
        """
        pass

class AbstractModel(ABC):
    """
    Handles model loading and configuration using Unsloth.
    
    This class is responsible for initializing the pre-trained model and tokenizer,
    applying PEFT/LoRA configurations, and providing access to the loaded model.
    """
    
    @abstractmethod
    def load_model(self):
        """
        Loads the pre-trained model and applies PEFT configurations.
        Uses FastLanguageModel for efficient loading and training.
        """
        pass

    @abstractmethod
    def get_model(self):
        """Returns the loaded model instance."""
        pass

    @abstractmethod
    def get_tokenizer(self):
        """Returns the loaded tokenizer instance."""
        pass
