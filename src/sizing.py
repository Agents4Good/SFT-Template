"""
VRAM calibration orchestration.
------------------------------
Measure -> optimize -> report pipeline for batch-size selection.

- ``calibrate``: probe the GPU, then pick (micro_batch, grad_accum)
  for the requested global batch. GPU-only by design.
- ``print_sizing_report``: human-readable rendering of the decision.

Primitives live in ``src/utils/vram.py``; this module only orchestrates
them. ``main.py`` stays a thin pipeline (parse -> data -> model ->
``calibrate`` -> train).
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SizingResult:
    """Outcome of batch-size calibration."""

    micro: int
    accum: int
    batch_info: Dict[str, Any] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)


def _longest_texts(texts, lengths, k):
    """Return the k longest texts (worst-case probe material)."""
    order = sorted(range(len(texts)), key=lambda i: lengths[i], reverse=True)
    return [texts[i] for i in order[:max(1, min(k, len(texts)))]]


def calibrate_measured(config, model, tokenizer, train_texts, train_lengths,
                       eval_texts, budget_gb):
    """Full measurement path: real static tensors + forward/backward probing.

    No heuristic formulas on this path.
    """
    from src.utils.vram import (
        measure_static_gb,
        training_baseline_gb,
        predicted_peak_gb,
        probe_per_sample_bytes,
        build_probe_optimizer,
        optimize_training_params,
    )

    model_cfg = config["model"]
    train_cfg = config["training"]
    probe_cfg = config.get("probe", {})
    micros = tuple(probe_cfg.get("micros", [1, 2, 4]))
    repeats = probe_cfg.get("repeats", 2)
    safety = probe_cfg.get("safety", 0.95)
    seq_len = config["model"]["max_seq_length"]

    probe_opt = build_probe_optimizer(
        train_cfg.get("optim", "adamw_torch"), model.parameters()
    )
    probe_texts = _longest_texts(train_texts, train_lengths, max(micros))
    eval_probe_texts = (
        _longest_texts(eval_texts, [len(t) for t in eval_texts], 2)
        if probe_cfg.get("measure_eval", True) and eval_texts
        else None
    )
    probe = probe_per_sample_bytes(
        model,
        tokenizer,
        probe_opt,
        probe_texts,
        seq_len,
        micros=micros,
        repeats=repeats,
        dtype=model_cfg.get("dtype", "bfloat16"),
        eval_texts=eval_probe_texts,
        use_gradient_checkpointing=config["peft"].get(
            "use_gradient_checkpointing", False),
    )
    # Warmup step materialized optimizer state; measure it exactly.
    static = measure_static_gb(model, probe_opt)
    # Sizing baseline is the probe-derived resident floor (weights, context,
    # grads, probe optimizer state, probe-produced fragmentation) plus the
    # trainer's own optimizer instance, grown by the fragmentation factor.
    # Tensor sums under-read reserved by ~20%; post-load snapshots miss the
    # ~1 GB floor growth -- both are inside this baseline, so reserve_gb=0.
    sizing_static = training_baseline_gb(static, probe)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    metrics = {
        "total_static_gb": sizing_static,
        "activations_per_sample_gb": probe["per_sample_reserved_gb"],
    }
    common = dict(
        global_batch_size=train_cfg["global_batch_size"],
        safety=safety,
        reserve_gb=0.0,
        allow_approximate=train_cfg.get("allow_approximate_batch", True),
        max_deviation=train_cfg.get("max_batch_deviation", 0.05),
        max_accum_steps=train_cfg.get("max_accum_steps", 64),
    )
    # Pass 1 ignores eval to get a candidate micro; pass 2 charges
    # measured eval slope scaled by that micro (monotonic: micro can only
    # stay or shrink, so the charge is conservative).
    micro1, _, _ = optimize_training_params(
        metrics, target_vram_gb=budget_gb, eval_overhead_gb=0.0, **common
    )
    eval_overhead = probe["eval_per_sample_reserved_gb"] * micro1
    micro, accum, batch_info = optimize_training_params(
        metrics, target_vram_gb=budget_gb,
        eval_overhead_gb=eval_overhead, **common
    )
    predicted_peak_gb_val = predicted_peak_gb(
        sizing_static, probe["per_sample_reserved_gb"], micro,
        probe["eval_per_sample_reserved_gb"],
    )
    probe_warnings = []
    if probe["oom_at"] is not None:
        probe_warnings.append(
            f"Probe OOM at micro={probe['oom_at']}; slope from smaller micros."
        )
    if probe.get("low_snr"):
        probe_warnings.append(
            "Probe peaks non-monotonic (allocator noise dominates at these "
            "sizes); 2x conservative margin applied to per-sample slope."
        )
    if probe.get("clamped_eval"):
        probe_warnings.append(
            "Eval slope exceeded train slope (probe noise); clamped to train "
            "slope (eval is forward-only)."
        )
    batch_info["warnings"] = probe_warnings + batch_info.get("warnings", [])

    report = {
        "method": "measured",
        "budget_gb": budget_gb,
        "static": static,
        "sizing_static_gb": sizing_static,
        "per_sample_gb": probe["per_sample_gb"],
        "per_sample_reserved_gb": probe["per_sample_reserved_gb"],
        "eval_per_sample_gb": probe["eval_per_sample_gb"],
        "eval_per_sample_reserved_gb": probe["eval_per_sample_reserved_gb"],
        "eval_overhead_gb": round(eval_overhead, 3),
        "predicted_peak_reserved_gb": predicted_peak_gb_val,
        "peaks_bytes": probe["peaks"],
        "reserved_peaks_bytes": probe["reserved_peaks"],
        "oom_at": probe["oom_at"],
        "batch": batch_info,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    return SizingResult(
        micro=micro, accum=accum, batch_info=batch_info, report=report
    )


def calibrate(config, model, tokenizer, train_texts, train_lengths,
              eval_texts, budget_gb, free_before_load=None):
    """Pick (micro_batch, grad_accum) via GPU measurement.

    GPU-only by design: without CUDA the probe raises. Appends the
    budget-vs-device best-effort note when the batch fell back but still
    fits the physical device.
    """
    result = calibrate_measured(
        config, model, tokenizer, train_texts, train_lengths, eval_texts,
        budget_gb,
    )

    # Budget-vs-device distinction: static can exceed the (possibly stale)
    # budget cap while still fitting the physical device. Say so explicitly
    # instead of a blanket "expect OOM".
    batch_info = result.batch_info
    if batch_info.get("fell_back") and free_before_load is not None:
        worst_case = result.report.get("predicted_peak_reserved_gb")
        if worst_case is not None and worst_case <= free_before_load * 0.95:
            batch_info["warnings"] = batch_info.get("warnings", []) + [
                f"Static exceeds the {budget_gb} GB budget cap but the "
                f"~{worst_case:.2f} GB worst case fits this "
                f"{free_before_load} GB device; proceeding best-effort. "
                f"Raise model.target_total_vram_gb to silence this."
            ]
    return result


def print_sizing_report(result, free_before_load=None,
                        requested_global_batch=None):
    """Render the calibration decision to stdout."""
    report = result.report
    batch_info = result.batch_info
    micro, accum = result.micro, result.accum
    static = report["static"]

    print(f"Method: {report['method']}")
    print(f"Static (measured tensors): {static['total_static_gb']} GB "
          f"(weights {static['weights_gb']}, grads {static['gradients_gb']}, "
          f"optim {static['optimizer_gb']} x1 instance)")
    print(f"Static (sizing baseline, reserved floor): "
          f"{report['sizing_static_gb']} GB "
          f"(probe floor + trainer optimizer, x fragmentation growth)")
    print(f"Per-sample reserved (measured): "
          f"{report['per_sample_reserved_gb']:.4f} GB, "
          f"reserved peaks {report['reserved_peaks_bytes']}")
    print(f"Predicted peak (reserved): "
          f"{report['predicted_peak_reserved_gb']} GB")
    print(f"Budget: {report.get('budget_gb')} GB "
          f"(free before load: {free_before_load}, "
          f"free now: {batch_info.get('free_gb')})")
    print(f"Optimized per_device_train_batch_size: {micro}")
    print(f"Optimized gradient_accumulation_steps: {accum}")
    print(f"Effective Global Batch Size: {micro * accum} "
          f"(requested {requested_global_batch}, "
          f"approximate={batch_info.get('approximate_used')})")
    for w in batch_info.get("warnings", []):
        print(f"WARNING: {w}")
    print("-" * 30)
