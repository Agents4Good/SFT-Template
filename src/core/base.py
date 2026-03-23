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
