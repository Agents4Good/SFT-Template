"""Run telemetry: preflight and post-run VRAM reports."""

import json
import os


def write_preflight(final_path, report):
    """Persist the measured pre-flight report; failure is fatal."""
    os.makedirs(final_path, exist_ok=True)
    path = os.path.join(final_path, "preflight.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    return path


def _arch_fingerprint(model, config):
    fingerprint = {}
    model_config = getattr(model, "config", None)
    text_config = getattr(model_config, "text_config", None) or model_config
    for name in (
        "model_type", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads",
        "vocab_size", "sliding_window",
    ):
        for obj in (model_config, text_config):
            if obj is not None and hasattr(obj, name):
                try:
                    fingerprint[name] = getattr(obj, name)
                    break
                except Exception:
                    continue
    fingerprint["attn_implementation"] = (
        getattr(model_config, "attn_implementation", None)
        if model_config is not None else None
    )
    fingerprint["gradient_checkpointing"] = config.get(
        "peft", {}).get("use_gradient_checkpointing")
    try:
        import cut_cross_entropy  # noqa: F401

        fingerprint["chunked_loss"] = True
    except Exception:
        fingerprint["chunked_loss"] = False
    return fingerprint


def _write_actuals(final_path, model, config, preflight):
    import torch

    training = config["training"]
    actuals = {
        "backend": type(model).__name__,
        "per_device_train_batch_size": training[
            "per_device_train_batch_size"],
        "gradient_accumulation_steps": training[
            "gradient_accumulation_steps"],
    }
    if torch.cuda.is_available():
        actuals["actual_peak_reserved_gb"] = round(
            torch.cuda.max_memory_reserved() / 1024**3, 3)
        actuals["actual_peak_allocated_gb"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 3)
    actuals["arch_fingerprint"] = _arch_fingerprint(model, config)
    if preflight:
        actuals["preflight"] = preflight
    os.makedirs(final_path, exist_ok=True)
    with open(os.path.join(final_path, "peak_memory.json"), "w") as handle:
        json.dump(actuals, handle, indent=2, default=str)


def record_training_actuals(final_path, model, config, preflight=None):
    """Write peak_memory.json. Observability only: never raises."""
    try:
        _write_actuals(final_path, model, config, preflight)
    except Exception as exc:
        print(f"WARNING: post-run VRAM telemetry failed ({exc}); "
              "continuing without actuals.")
