"""
SFT Template Main Entry Point
-----------------------------
This script orchestrates the Supervised Fine-Tuning (SFT) process by:
1. Loading the configuration from a YAML file.
2. Initializing the model and tokenizer using the Model handler.
3. Loading, formatting, and splitting the dataset using the DatasetProcessor.
4. Starting the training loop using the SFTTrain handler.
5. Saving the final fine-tuned model.
"""

import yaml
import argparse
from model.handler import Model
from dataset.processor import DatasetProcessor
from train.sft import SFTTrain

def load_config(path):
    """
    Loads configuration from a YAML file.
    
    Args:
        path (str): Path to the configuration file.
    
    Returns:
        dict: The loaded configuration.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """
    Main execution logic for the SFT training pipeline.
    Parses command line arguments and initializes the training workflow.
    """
    parser = argparse.ArgumentParser(description="Run SFT Training")
    parser.add_argument("--config", type=str, default="src/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    # 1. Model Initialization
    print("Loading Model...")
    model_handler = Model(config)
    model_handler.load_model()
    model = model_handler.get_model()
    tokenizer = model_handler.get_tokenizer()

    # 2. Dataset Processing
    print("Processing Dataset...")
    dataset_processor = DatasetProcessor(config, tokenizer)
    raw_dataset = dataset_processor.load_dataset()
    formatted_dataset = dataset_processor.format_dataset(raw_dataset)
    split_dataset = dataset_processor.split_dataset(formatted_dataset)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # 3. Training Execution
    print("Starting Training...")
    trainer = SFTTrain(config, model, tokenizer, train_dataset, eval_dataset)
    trainer.train()
    
    # 4. Persistence
    trainer.save_model()

if __name__ == "__main__":
    main()
