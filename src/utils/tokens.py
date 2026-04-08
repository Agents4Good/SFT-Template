from src.model.handler import Model
from src.dataset.processor import DatasetProcessor
import yaml
import argparse
from src.model.handler import Model

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

def calculate_tokens_usage(model:Model, dataset) -> int:
    tokenizer = model.get_tokenizer()

    tokens_length = list()
    for example in dataset:
        text = example['text']
        tokens = tokenizer.tokenize(text)
        tokens_length.append(len(tokens))

    return max(tokens_length)

def main():
    parser = argparse.ArgumentParser(description="Run SFT Training")
    parser.add_argument("--config", type=str, default="src/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    print("Loading tokenizer")
    model_handler = Model(config)
    model_handler.load_model()
    tokenizer = model_handler.get_tokenizer()

    print("Processing dataset")
    dataset_processor = DatasetProcessor(config, tokenizer)
    raw_dataset = dataset_processor.load_dataset()
    formatted_dataset = dataset_processor.format_dataset(raw_dataset)
    max_tokens = calculate_tokens_usage(model_handler, formatted_dataset)

    print(f"Max tokens used in dataset: {max_tokens}")

if __name__ == "__main__":
    main()
