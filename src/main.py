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
from src.utils.vram import estimate_vram_usage

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
    
    if config["model"].get("max_seq_length") is None:
        print("Calculating max sequence length from dataset...")
        max_seq_lengths = dataset_processor.calculate_tokens_usage(formatted_dataset)
        config["model"]["max_seq_length"] = max(max_seq_lengths)
        print(f"Max sequence length set to: {config['model']['max_seq_length']}")
    else:
        print(f"Using max sequence length from config: {config['model']['max_seq_length']}")

    # 2. Model Initialization
    print("-" * 10 + "Loading model" + "-" * 10)
    model_handler = Model(config)
    model_handler.load_model()
    model = model_handler.get_model()
    tokenizer = model_handler.get_tokenizer()
    
    # Optimize training parameters
    from src.utils.vram import estimate_vram_usage, optimize_training_params
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    vram_metrics = estimate_vram_usage(
        model_name=config["model"]["name"],
        total_params=total_params,
        trainable_params=trainable_params,
        dtype=config["model"]["dtype"],
        load_in_4bit=config["model"]["load_in_4bit"],
        max_seq_length=config["model"]["max_seq_length"],
        optimizer=config["training"]["optim"]
    )
    
    # Calculate optimal batch size and accumulation steps
    per_device_batch, grad_accum = optimize_training_params(
        vram_metrics=vram_metrics,
        target_vram_gb=config["model"]["target_total_vram_gb"],
        global_batch_size=config["training"]["global_batch_size"]
    )
    
    # Update config with optimized values
    config["training"]["per_device_train_batch_size"] = per_device_batch
    config["training"]["gradient_accumulation_steps"] = grad_accum
    
    print("-" * 10 + "Memory & Batch Optimization" + "-" * 10)
    print(f"Total Params: {total_params:,}")
    print(f"Trainable Params: {trainable_params:,}")
    print(f"Fixed Memory: {vram_metrics['total_static_gb']} GB")
    print(f"Activation Memory (per sample): {vram_metrics['activations_per_sample_gb']} GB")
    print(f"Target VRAM: {config['model']['target_total_vram_gb']} GB")
    print(f"Optimized per_device_train_batch_size: {per_device_batch}")
    print(f"Optimized gradient_accumulation_steps: {grad_accum}")
    print(f"Effective Global Batch Size: {per_device_batch * grad_accum}")
    print("-" * 30)

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
