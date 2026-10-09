"""
SFT Template Main Entry Point
-----------------------------
1. Dataset processing + robust max_seq_length (p95, not max).
2. Model load.
3. VRAM calibration (see src/sizing.py): real static tensors +
   forward/backward probing -> optimal batch. GPU-only by design.
4. Train + save. --dry-run prints the batch decision and exits.
"""

import argparse

import yaml

from src.dataset.processor import DatasetProcessor
from src.sizing import calibrate, print_sizing_report
from src.train.sft import SFTTrain


def _lookup(config, dotted):
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None, False
        node = node[part]
    return node, True


_REQUIRED_KEYS = (
    "model.name",
    "dataset.path",
    "dataset.train_test_split.test_size",
    "training.global_batch_size",
    "output.output_dir",
)


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    missing = [k for k in _REQUIRED_KEYS if not _lookup(config, k)[1]]
    if missing:
        raise KeyError(
            f"{path} missing required keys: {', '.join(missing)}")
    return config


def main():
    parser = argparse.ArgumentParser(description="Run SFT Training")
    parser.add_argument("--config", type=str, default="src/config.yaml",
                        help="Path to config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Probe/select batch, print, exit before training")
    args = parser.parse_args()

    config = load_config(args.config)

    # Budget resolved FIRST, before anything allocates GPU memory.
    # Post-load "free" already excludes resident weights (double-counts
    # static), so the total budget must come from pre-load state.
    from src.utils.vram import get_free_vram_gb, resolve_budget

    free_before_load = get_free_vram_gb()
    budget_gb, budget_notes = resolve_budget(
        config["model"].get("target_total_vram_gb"), free_before_load
    )
    print(f"VRAM budget: {budget_gb} GB (free before load: {free_before_load})")
    for note in budget_notes:
        print(f"NOTE: {note}")

    # 1. Dataset Processing
    print("-" * 10 + "Processing Dataset" + "-" * 10)
    dataset_processor = DatasetProcessor(config)
    raw_dataset = dataset_processor.load_dataset()
    formatted_dataset = dataset_processor.format_dataset(raw_dataset)
    split_dataset = dataset_processor.split_dataset(formatted_dataset)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    if config["model"].get("max_seq_length") is None:
        print("Calculating sequence length stats from dataset...")
        lengths = dataset_processor.calculate_tokens_usage(formatted_dataset)
        stats = dataset_processor.suggest_seq_length(lengths)
        config["model"]["max_seq_length"] = stats["suggested"]
        print(f"Token stats: {stats}")
    else:
        lengths = None
        print(f"Using max sequence length from config: "
              f"{config['model']['max_seq_length']}")

    # 2. Model load (measurement needs real tensors on GPU).
    print("-" * 10 + "Loading model" + "-" * 10)
    from src.model.factory import create_model_handler

    model_handler = create_model_handler(config)
    print(f"Model backend: {model_handler.backend_name}")
    model = model_handler.get_model()
    tokenizer = model_handler.get_tokenizer()
    dataset_processor.tokenizer = tokenizer

    # 3. Batch selection: GPU measurement.
    print("-" * 10 + "VRAM calibration" + "-" * 10)
    train_texts = list(train_dataset["text"])
    # Train-split lengths align row-for-row for worst-case probe selection.
    # (Formatted-set lengths above were only needed for the p95 stats.)
    train_lengths = dataset_processor.calculate_tokens_usage(train_dataset)
    eval_texts = list(eval_dataset["text"])

    result = calibrate(
        config, model, tokenizer, train_texts, train_lengths, eval_texts,
        budget_gb, free_before_load=free_before_load,
    )

    config["training"]["per_device_train_batch_size"] = result.micro
    config["training"]["gradient_accumulation_steps"] = result.accum
    result.report["backend"] = model_handler.backend_name
    result.report["free_before_load_gb"] = free_before_load
    config["_vram_report"] = result.report

    print("-" * 10 + "Memory & Batch Optimization" + "-" * 10)
    print_sizing_report(
        result, free_before_load,
        requested_global_batch=config["training"]["global_batch_size"],
    )

    if args.dry_run:
        print("DRY-RUN: batch decision above, exiting before training.")
        return

    # 4. Training + persistence.
    print("-" * 10 + "Starting training" + "-" * 10)
    trainer = SFTTrain(config, model, tokenizer, train_dataset, eval_dataset)
    trainer.train()

    print("-" * 10 + "Saving model" + "-" * 10)
    trainer.save_model()

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
