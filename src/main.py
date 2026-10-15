"""
SFT Template Main Entry Point
-----------------------------
1. Dataset processing + exact max_seq_length (real longest sample).
2. Model load.
3. Micro batch: binary search over [1, global] when
   training.global_batch_size is set, else pinned micro/accum --
   always gated by a fit check at real data lengths.
4. Train + save. --dry-run resolves the batch, runs the gate, exits.
"""

import argparse

import yaml

from src.dataset.processor import DatasetProcessor
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
    "training",
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
                        help="Fit check at real lengths, print, exit")
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

    # Exact sequence length: tokenize the final texts and take the real
    # max. No percentile, no truncation surprises -- the fit check below
    # tests this exact length, which (under length-grouped batching) is
    # also the true worst case. An explicit config value is respected.
    if config["model"].get("max_seq_length") is None:
        print("Measuring exact sequence lengths from dataset...")
        lengths = dataset_processor.calculate_tokens_usage(
            formatted_dataset)
        config["model"]["max_seq_length"] = max(lengths) if lengths else 0
        print(f"Token count: {len(lengths)}, "
              f"max length: {config['model']['max_seq_length']}")
    else:
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

    # 3. Micro batch: config-key dispatch. global_batch_size present ->
    # binary-search the largest fitting micro in [1, global] (accum
    # derived, effective batch unchanged). Absent -> pinned micro/accum
    # (no search). Both/neither is a config error.
    print("-" * 10 + "Micro batch" + "-" * 10)
    from src.train.fit_check import (
        find_best_micro, resolve_batch_mode, verify_fit)

    training = config["training"]
    mode, value = resolve_batch_mode(training)

    train_texts = list(train_dataset["text"])
    train_lengths = dataset_processor.calculate_tokens_usage(train_dataset)
    order = sorted(range(len(train_texts)),
                   key=lambda i: train_lengths[i], reverse=True)
    eval_texts = list(eval_dataset["text"])
    eval_lengths = dataset_processor.calculate_tokens_usage(eval_dataset)
    longest_eval = eval_texts[max(range(len(eval_texts)),
                                  key=lambda i: eval_lengths[i])]
    max_len = config["model"]["max_seq_length"]
    dtype = config["model"].get("dtype", "bfloat16")

    if mode == "search":
        window = min(value, len(train_texts))
        longest_train = [train_texts[i] for i in order[:window]]
        micro, accum, longest, tried, probed_max = find_best_micro(
            model, tokenizer, longest_train, longest_eval, max_len,
            value, dtype=dtype)
        trace = " -> ".join(
            f"{m} {'OK' if state == 'OK' else 'OOM'}"
            for m, state in tried)
        extra = (f" (backed off from probed max {probed_max})"
                 if probed_max != micro else "")
        print(f"Micro search [{trace}] => micro={micro}, accum={accum} "
              f"(global={micro * accum}) at real max length "
              f"{longest}{extra}.")
        training["per_device_train_batch_size"] = micro
        training["gradient_accumulation_steps"] = accum
        config["_micro_search"] = {
            "global_batch_size": training["global_batch_size"],
            "searched_micro": micro,
            "probed_max": probed_max,
            "candidates_tried": [list(t) for t in tried],
        }
    else:
        micro = training["per_device_train_batch_size"]
        longest_train = [train_texts[i] for i in order[:max(1, micro)]]
        longest = verify_fit(
            model, tokenizer, longest_train, longest_eval, micro,
            max_len, dtype=dtype)
        print(f"Fit OK: micro={micro} at real max length {longest} tokens.")

    if args.dry_run:
        print("DRY-RUN: batch resolved and gated above, exiting.")
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
