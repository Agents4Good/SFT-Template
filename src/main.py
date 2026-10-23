"""
SFT Template Main Entry Point
-----------------------------
1. Dataset processing + exact max_seq_length (real longest sample).
2. Model load.
3. Micro batch: every candidate runs real Trainer steps at the longest
   training shapes and is gated against measured usable VRAM. With
   training.global_batch_size set the largest fitting micro is selected;
   without it a pinned micro/accum pair is fit-checked once.
4. Train + save. --dry-run resolves the batch, writes preflight.json,
   and exits.
"""

import argparse
import datetime

import yaml

from src.dataset.processor import DatasetProcessor
from src.train.fit_check import (
    check_pinned, find_best_micro, resolve_batch_mode)
from src.train.probe import ProbeSession, usable_memory_gb
from src.train.sft import SFTTrain
from src.train.telemetry import write_preflight
from src.train.trainer import (
    build_args, build_callbacks, build_trainer, close_trainer, promote_trainer)


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


def _base_preflight(config, backend_name, device_total_gb, probe_steps,
                    created):
    return {
        "created": created,
        "model": config["model"]["name"],
        "backend": backend_name,
        "device_total_gb": device_total_gb,
        "probe_steps": probe_steps,
    }


def _search_preflight(config, backend_name, device_total_gb, probe_steps,
                      created, result):
    report = _base_preflight(
        config, backend_name, device_total_gb, probe_steps, created)
    report["mode"] = "search"
    report.update(result)
    return report


def _fixed_preflight(config, backend_name, device_total_gb, probe_steps,
                     created, micro, accum, check):
    report = _base_preflight(
        config, backend_name, device_total_gb, probe_steps, created)
    report.update({
        "mode": "fixed",
        "global_batch_size": None,
        "max_micro": micro,
        "selected_micro": micro,
        "gradient_accumulation_steps": accum,
        "effective_batch_size": micro * accum,
        "usable_gb": check["usable_gb"],
        "reserve_fraction": check["reserve_fraction"],
        "limit_gb": check["limit_gb"],
        "candidates": [{
            "micro": micro,
            "result": "pass",
            "peak_reserved_gb": check["peak_reserved_gb"],
            "duration_s": check["duration_s"],
            "steps": probe_steps,
        }],
        "confirm": None,
    })
    return report


def main():
    parser = argparse.ArgumentParser(description="Run SFT Training")
    parser.add_argument("--config", type=str, default="src/config.yaml",
                        help="Path to config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the measured fit check, write "
                             "preflight.json, and exit")
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
    # max. The Trainer probes below run at this exact length, which
    # (under length-grouped batching) is also the true worst case. An
    # explicit config value is respected.
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

    # 3. Micro batch: probe the real Trainer path. Env overrides are
    # applied before mode dispatch so GLOBAL_BATCH_SIZE etc. participate.
    print("-" * 10 + "Micro batch" + "-" * 10)
    sft = SFTTrain(config, model, tokenizer, train_dataset, eval_dataset)
    training = sft.apply_env_overrides()
    mode, value = resolve_batch_mode(training)
    usable_gb, device_total_gb = usable_memory_gb()
    reserve = float(training.get("vram_reserve_fraction", 0.10))
    probe_steps = int(training.get("probe_steps", 3))
    created = datetime.datetime.now().isoformat(timespec="seconds")

    def probe_trainer(tmp_dir, steps):
        probe_args = build_args(
            config, 1, 1, tmp_dir, probe=True, probe_steps=steps)
        return build_trainer(
            model, tokenizer, train_dataset, eval_dataset, probe_args)

    # Build the real config before any Trainer exists: every TrainingArguments
    # construction resets Accelerate's global state, which would invalidate
    # the probe Trainer this run is promoted onto.
    real_args = build_args(config, 1, 1, sft.final_path)

    session = None
    trainer = None
    try:
        session = ProbeSession.open(probe_trainer, steps=probe_steps)
        if mode == "search":
            max_micro = min(value, len(train_dataset))
            result = find_best_micro(
                session.run, usable_gb, value, reserve_fraction=reserve,
                max_micro=max_micro)
            micro = result["micro"]
            accum = result["accum"]
            training["per_device_train_batch_size"] = micro
            training["gradient_accumulation_steps"] = accum
            # Persist a single reproducible batch mode.
            training.pop("global_batch_size", None)
            trace = " -> ".join(
                f"{c['micro']} {c['result']}"
                + (f" ({c['peak_reserved_gb']:.2f} GB)"
                   if c.get("peak_reserved_gb") is not None else "")
                for c in result["candidates"])
            print(f"Micro probe [{trace}] => micro={micro}, accum={accum} "
                  f"(effective={micro * accum} of {value}) "
                  f"[limit {result['limit_gb']:.2f} GB = usable "
                  f"{result['usable_gb']:.2f} GB - {reserve:.0%} reserve].")
            preflight = _search_preflight(
                config, model_handler.backend_name, device_total_gb,
                probe_steps, created, result)
        else:
            micro = training["per_device_train_batch_size"]
            accum = training["gradient_accumulation_steps"]
            check = check_pinned(
                session.run, micro, usable_gb, reserve_fraction=reserve)
            print(f"Fit OK: pinned micro={micro}, accum={accum} at real max "
                  f"length {config['model']['max_seq_length']} tokens "
                  f"(probe peak {check['peak_reserved_gb']:.2f} GB, limit "
                  f"{check['limit_gb']:.2f} GB).")
            preflight = _fixed_preflight(
                config, model_handler.backend_name, device_total_gb,
                probe_steps, created, micro, accum, check)

        preflight_path = write_preflight(sft.final_path, preflight)
        print(f"Preflight report: {preflight_path}")

        if args.dry_run:
            print("DRY-RUN: batch resolved and gated above, exiting.")
            return

        # 4. Training + persistence. Promotion happens while the probe
        # session still owns the Trainer; the real run then takes over.
        real_args.per_device_train_batch_size = micro
        real_args.gradient_accumulation_steps = accum
        promote_trainer(session.trainer, real_args, build_callbacks(config))
        trainer = session.release_trainer()

        print("-" * 10 + "Starting training" + "-" * 10)
        sft.train(trainer, preflight=preflight)

        print("-" * 10 + "Saving model" + "-" * 10)
        sft.save_model()

        print("Training completed successfully!")
    finally:
        if session is not None:
            session.close()
        close_trainer(trainer)


if __name__ == "__main__":
    main()
