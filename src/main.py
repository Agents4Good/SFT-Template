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
from src.model.handler import Model
from src.dataset.processor import DatasetProcessor
from src.train.sft import SFTTrain

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
    
    # 1. Dataset Processing
    print("-" * 10 + "Processing Dataset" + "-" * 10)
    dataset_processor = DatasetProcessor(config)
    raw_dataset = dataset_processor.load_dataset()
    formatted_dataset = dataset_processor.format_dataset(raw_dataset)
    split_dataset = dataset_processor.split_dataset(formatted_dataset)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    if config["model"]["max_seq_length"] is None:
        print("Calculating max sequence length from dataset...")
        max_seq_length = dataset_processor.calculate_tokens_usage(formatted_dataset)
        config["model"]["max_seq_length"] = max(max_seq_length)
        print(f"Max sequence length set to: {config['model']['max_seq_length']}")

    # 2. Model Initialization
    print("-" * 10 + "Loading model" + "-" * 10)
    model_handler = Model(config)
    model_handler.load_model()
    model = model_handler.get_model()
    tokenizer = model_handler.get_tokenizer()
    
    # 3. Training Execution
    print("-" * 10 + "Starting training" + "-" * 10)
    trainer = SFTTrain(config, model, tokenizer, train_dataset, eval_dataset)
    trainer.train()
    
    # 4. Persistence
    print("-" * 10 + "Saving model" + "-" * 10)
    trainer.save_model()

    print("Training completed successfully!")

if __name__ == "__main__":
    main()
