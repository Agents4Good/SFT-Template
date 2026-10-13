"""VRAM estimation and batch-size optimization for SFT training."""

import warnings
from contextlib import nullcontext

import torch

# Resident floor growth from probe to steady training: reserved runs
# slightly above the probe floor under sustained alloc/free cycles. Was
# 1.1 (Oct run: 8.78 -> 9.71 GB) until per-run actuals showed that rise
# INCLUDED eval peaks -- with eval now charged explicitly at its own
# slope, the floor stays train-only (micro-1 no-eval actuals: 9.123 vs
# uninflated 8.866; the rest is covered by slope x micro + safety).
# Validated per-run via actual_peak_* in vram_report.json.
_FRAGMENTATION_FACTOR = 1.0
_BYTES_PER_DTYPE = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}


_GB_BYTES = 1024**3


def get_free_vram_gb():
    """Return free VRAM in GB via CUDA, or None if no GPU is available."""
    try:
        if not torch.cuda.is_available():
            return None
        free_bytes, _ = torch.cuda.mem_get_info()
        return round(free_bytes / _GB_BYTES, 3)
    except Exception:
        return None


def resolve_budget(target_gb, free_before_load):
    """Resolve the TOTAL VRAM budget before anything is allocated.

    min(target, detected free); either may be None (the other wins).
    Must run pre-load: post-load "free" already excludes resident weights
    and would double-count static. Returns (budget, notes).
    """
    notes = []
    if target_gb is None and free_before_load is None:
        raise ValueError(
            "No VRAM budget: set a target or run where CUDA is available."
        )
    if target_gb is None:
        budget = free_before_load
        notes.append(f"No configured target; using detected {budget} GB.")
    elif free_before_load is None:
        budget = target_gb
    else:
        budget = min(target_gb, free_before_load)
        if free_before_load < target_gb:
            notes.append(
                f"GPU has {free_before_load} GB free < target {target_gb} "
                f"GB; budgeting {budget} GB total."
            )
    return budget, notes


def _to_gb(n_bytes):
    return n_bytes / _GB_BYTES


def probe_low_snr(peaks, baseline_bytes):
    """True when probe peaks look allocator-noise-dominated. Pure.

    Non-monotonic peaks (a larger micro peaking lower) or a non-positive
    max slope mean per-micro noise is comparable to the signal; the slope is
    then a lower bound, not a measurement. Seen on real runs with tiny
    per-sample activations (GQA + sliding window + ckpt + chunked loss).
    """
    ordered = sorted(peaks.items())
    if len(ordered) < 2:
        return len(ordered) == 1 and ordered[0][1] <= baseline_bytes
    return any(b2 <= b1 for (_, b1), (_, b2) in zip(ordered, ordered[1:]))


def per_sample_gb_from_peaks(peaks, baseline_bytes):
    """Derive per-sample activation GB from measured peak-allocated bytes.

    Pure function (no GPU): takes the max consecutive slope, which is
    conservative under allocator noise. Single point -> peak minus baseline.
    Non-positive results return 0.0 so the optimizer's best-effort fallback
    triggers instead of a divide-by-zero.
    """
    if not peaks:
        return 0.0
    ordered = sorted(peaks.items())
    if len(ordered) == 1:
        return max(0.0, (ordered[0][1] - baseline_bytes) / _GB_BYTES)
    slopes = [
        (b2 - b1) / (m2 - m1)
        for (m1, b1), (m2, b2) in zip(ordered, ordered[1:])
        if m2 > m1
    ]
    if not slopes:
        return 0.0
    return max(0.0, max(slopes) / _GB_BYTES)


def measure_static_gb(model, optimizer=None):
    """Measure static (non-activation) VRAM from real tensors. No heuristics.

    Sums parameter storage, grad storage (requires_grad params) and the
    materialized state of ONE optimizer instance via element_size(). This is
    exact reporting, not sizing: see sizing_static_gb() for the
    training-time baseline (reserved snapshot + second optimizer instance).
    Works with duck-typed fakes (no torch import needed) for CPU tests.
    """
    params = list(model.parameters())
    weight_B = sum(p.numel() * p.element_size() for p in params)
    grad_B = sum(
        p.numel() * p.element_size()
        for p in params
        if getattr(p, "requires_grad", False)
    )
    opt_B = 0
    state = getattr(optimizer, "state", {}) or {}
    for st in state.values():
        if not isinstance(st, dict):
            continue
        for t in st.values():
            if hasattr(t, "numel") and hasattr(t, "element_size"):
                try:
                    opt_B += t.numel() * t.element_size()
                except Exception:
                    continue
    total_B = weight_B + grad_B + opt_B
    return {
        "weights_gb": round(_to_gb(weight_B), 3),
        "gradients_gb": round(_to_gb(grad_B), 3),
        "optimizer_gb": round(_to_gb(opt_B), 3),
        "total_static_gb": round(_to_gb(total_B), 3),
        "method": "measured",
    }


def sizing_static_gb(measured):
    """Training-time static baseline in GB for the batch optimizer. Pure.

    Tensor sums plus the trainer's OWN optimizer instance: the trainer
    builds it next to the probe's, so one extra optimizer state is always
    charged.
    """
    return round(measured["total_static_gb"] + measured["optimizer_gb"], 3)


def training_baseline_gb(static_tensors, probe):
    """Resident floor for prediction, derived from the probe itself. Pure.

    floor = smallest-micro reserved peak minus its activation share. This
    captures weights, context, grads, the probe optimizer state AND the
    fragmentation the probe itself produced -- everything a post-load
    snapshot misses (~1 GB on Gemma-4). The trainer's own optimizer
    instance is added on top. The growth factor stays 1.0: steady-state
    growth is covered by slope x micro plus the explicit eval charge
    (the old 1.1 was fit against eval-inclusive peaks, double-counting
    eval). Falls back to sizing_static_gb() when the probe produced no
    peaks (warmup OOM).
    """
    peaks = probe.get("reserved_peaks") or {}
    slope = probe.get("per_sample_reserved_gb", 0.0) or 0.0
    if peaks:
        m0 = min(peaks)
        floor = peaks[m0] / _GB_BYTES - slope * m0
        resident = max(0.0, floor) + static_tensors["optimizer_gb"]
        return round(resident * _FRAGMENTATION_FACTOR, 3)
    return sizing_static_gb(static_tensors)


def predicted_peak_gb(baseline_gb, slope_gb, micro, eval_slope_gb,
                      eval_micro):
    """Predicted steady-state reserved peak in GB. Pure.

    Baseline already includes fragmentation growth; transient activation
    and eval peaks reuse blocks and are added linearly. Eval is charged at
    its own (forward-only, un-checkpointed) slope times the eval micro,
    which the caller pins below the train micro (eval transients).
    """
    return round(
        baseline_gb + slope_gb * micro + eval_slope_gb * eval_micro,
        3,
    )


def build_probe_optimizer(optim_name, parameters, lr=3e-4):
    """Build the same optimizer class training will use, for state measurement.

    Probe is GPU-only by design (torch is a hard dependency of this module).
    Falls back to torch AdamW with a warning if bitsandbytes is missing.
    """
    name = (optim_name or "adamw_torch").lower()
    trainable = [p for p in parameters if getattr(p, "requires_grad", False)]
    params = trainable or list(parameters)
    if "8bit" in name:
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=lr)
        except Exception as e:
            warnings.warn(
                f"bitsandbytes unavailable ({e}); probing optimizer state with "
                "torch AdamW, which differs from 8-bit training state.",
                UserWarning,
                stacklevel=2,
            )
    return torch.optim.AdamW(params, lr=lr)


def probe_per_sample_bytes(
    model,
    tokenizer,
    optimizer,
    sample_texts,
    max_seq_length,
    micros=(1, 2, 4),
    repeats=2,
    dtype="bfloat16",
    eval_texts=None,
    _cuda=None,
    _batch_fn=None,
    use_gradient_checkpointing=None,
):
    """Measure activation bytes/sample with real forward+backward passes.

    Worst-case padding (padding="max_length") at max_seq_length. One warmup
    micro=1 step materializes optimizer state, then per-micro peaks are
    recorded via max_memory_allocated AND max_memory_reserved. Sizing uses
    the reserved slope (OOMs happen on reserved, which runs ~20% above
    allocated); allocated is kept for reporting. Returns peaks in bytes plus
    derived per-sample slopes (full precision; rounding would risk zeroing
    small values). Real OOMs are caught -> oom_at set, best-effort
    downstream. _cuda/_batch_fn are test hooks (stubs); production passes
    None.
    """
    cuda = _cuda if _cuda is not None else torch.cuda
    if not cuda.is_available():
        raise RuntimeError("No CUDA device for VRAM probing.")

    device = next(model.parameters()).device
    use_amp = dtype in ("float16", "bfloat16")
    amp_dtype = getattr(torch, dtype, None) if use_amp else None

    def _tokenize(texts):
        # NOTE: text= as keyword (not positional): Unsloth multimodal models
        # return a Processor whose first positional arg is `images` -- passing
        # positionally makes it try to decode our strings as image files.
        # Plain HF tokenizers treat text= identically to positional.
        return tokenizer(
            text=texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

    def _make_batch(n):
        if _batch_fn is not None:
            return _batch_fn(n)
        enc = _tokenize(sample_texts[:n])
        input_ids = enc["input_ids"].to(device)
        mask = enc.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)
        labels = input_ids.clone()
        if mask is not None:
            labels[mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}

    def _actx():
        if use_amp and amp_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    def _is_oom(e):
        return "out of memory" in str(e).lower()

    model.train()
    baseline = cuda.memory_allocated()
    try:
        reserved_baseline = cuda.memory_reserved()
    except Exception:
        reserved_baseline = baseline

    # Warmup micro=1 with a real optimizer step: materializes optimizer state
    # so measure_static_gb() afterwards sees the true state tensors.
    warmed_up = False
    try:
        b = _make_batch(1)
        with _actx():
            loss = model(**{k: v for k, v in b.items() if v is not None}).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        warmed_up = True
    except RuntimeError as e:
        if _is_oom(e):
            cuda.empty_cache()
            return {
                "peaks": {},
                "reserved_peaks": {},
                "baseline_bytes": baseline,
                "reserved_baseline_bytes": reserved_baseline,
                "per_sample_gb": 0.0,
                "per_sample_reserved_gb": 0.0,
                "eval_per_sample_gb": 0.0,
                "eval_per_sample_reserved_gb": 0.0,
                "oom_at": 1,
                "warmed_up": False,
            }
        raise
    finally:
        try:
            cuda.empty_cache()
        except Exception:
            pass

    peaks = {}
    reserved_peaks = {}
    oom_at = None
    for micro in micros:
        try:
            best = None
            best_reserved = None
            for _ in range(max(1, repeats)):
                optimizer.zero_grad(set_to_none=True)
                cuda.reset_peak_memory_stats()
                b = _make_batch(micro)
                with _actx():
                    loss = model(
                        **{k: v for k, v in b.items() if v is not None}
                    ).loss
                loss.backward()
                cuda.synchronize()
                peak = cuda.max_memory_allocated()
                best = peak if best is None else max(best, peak)
                try:
                    rpeak = cuda.max_memory_reserved()
                except Exception:
                    rpeak = peak
                best_reserved = (
                    rpeak if best_reserved is None else max(best_reserved, rpeak)
                )
                optimizer.zero_grad(set_to_none=True)
            peaks[micro] = best
            reserved_peaks[micro] = best_reserved
        except RuntimeError as e:
            if _is_oom(e):
                try:
                    cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass
                oom_at = micro
                break
            raise
    try:
        cuda.empty_cache()
    except Exception:
        pass

    per_sample_gb = per_sample_gb_from_peaks(peaks, baseline)
    per_sample_reserved_gb = per_sample_gb_from_peaks(
        reserved_peaks, reserved_baseline
    )
    eval_per_sample_gb = 0.0
    eval_per_sample_reserved_gb = 0.0
    eval_peaks = {}
    eval_reserved_peaks = {}
    if eval_texts:
        model.eval()
        # Multi-point, repeated sampling like the train probe: a 2-point
        # single-shot eval slope once read a lone allocator block step as
        # 0.76 GB/sample against a true ~0.37 (measured 2026-10).
        for micro in (1, 2, 4):
            try:
                eval_best = None
                eval_best_reserved = None
                for _ in range(max(1, repeats)):
                    cuda.reset_peak_memory_stats()
                    enc = (
                        _batch_fn(micro)
                        if _batch_fn is not None
                        else _tokenize(eval_texts[:micro])
                    )
                    if not isinstance(enc, dict):
                        ids = enc["input_ids"].to(device)
                        m = enc.get("attention_mask")
                        enc = {
                            "input_ids": ids,
                            "attention_mask": m.to(device) if m is not None else None,
                        }
                    else:
                        enc = {
                            k: (v.to(device) if hasattr(v, "to") else v)
                            for k, v in enc.items()
                        }
                    with torch.no_grad(), _actx():
                        model(
                            **{k: v for k, v in enc.items() if v is not None}
                        )
                    cuda.synchronize()
                    a = cuda.max_memory_allocated()
                    try:
                        r = cuda.max_memory_reserved()
                    except Exception:
                        r = a
                    eval_best = a if eval_best is None else max(eval_best, a)
                    eval_best_reserved = (
                        r if eval_best_reserved is None
                        else max(eval_best_reserved, r)
                    )
                eval_peaks[micro] = eval_best
                eval_reserved_peaks[micro] = eval_best_reserved
            except RuntimeError as e:
                if _is_oom(e):
                    break
                raise
            finally:
                try:
                    cuda.empty_cache()
                except Exception:
                    pass
        model.train()
        eval_per_sample_gb = per_sample_gb_from_peaks(eval_peaks, baseline)
        eval_per_sample_reserved_gb = per_sample_gb_from_peaks(
            eval_reserved_peaks, reserved_baseline
        )

    # Low-SNR guard: non-monotonic peaks mean allocator noise dominates the
    # signal, so the slope is a lower bound. Apply a 2x conservative margin
    # rather than trusting it (seen on GQA + sliding-window + ckpt + CCE,
    # where per-sample activations are ~50MB against ~300MB fluctuations).
    low_snr = probe_low_snr(peaks, baseline)
    if low_snr and per_sample_gb > 0:
        per_sample_gb *= 2.0
    if low_snr and per_sample_reserved_gb > 0:
        per_sample_reserved_gb *= 2.0
    # Eval gets its own SNR verdict: its peaks are usually clean and
    # linear even when train peaks are noise-flat (checkpointing shrinks
    # the train footprint but not eval's). Margining a clean 0.45 slope
    # to 0.90 halved the feasible micro on a real run (2026-10).
    eval_low_snr = probe_low_snr(eval_reserved_peaks, reserved_baseline)
    if eval_low_snr and eval_per_sample_gb > 0:
        eval_per_sample_gb *= 2.0
    if eval_low_snr and eval_per_sample_reserved_gb > 0:
        eval_per_sample_reserved_gb *= 2.0

    # Physical invariant (checkpointing OFF only): eval is forward-only, so
    # it cannot exceed the train (forward+backward) slope. Raw eval slopes
    # above that are probe noise (seen: 0.7 GB eval vs 0.1 GB train); clamp
    # to train. With checkpointing ON the invariant is FALSE: train runs
    # recompute (tiny activation footprint) while eval stores full layer
    # activations (HF gates checkpoint on self.training) -- measured +0.35
    # GB eval vs +0.02 GB train slope on Gemma-4. Then the raw eval slope
    # stands (low-SNR margin already applied).
    clamped_eval = False
    if use_gradient_checkpointing is False:
        if eval_per_sample_reserved_gb > per_sample_reserved_gb:
            eval_per_sample_reserved_gb = per_sample_reserved_gb
            clamped_eval = True
        if eval_per_sample_gb > per_sample_gb:
            eval_per_sample_gb = per_sample_gb
            clamped_eval = True

    return {
        "peaks": peaks,
        "reserved_peaks": reserved_peaks,
        "eval_peaks": eval_peaks,
        "eval_reserved_peaks": eval_reserved_peaks,
        "baseline_bytes": baseline,
        "reserved_baseline_bytes": reserved_baseline,
        "per_sample_gb": per_sample_gb,
        "per_sample_reserved_gb": per_sample_reserved_gb,
        "eval_per_sample_gb": eval_per_sample_gb,
        "eval_per_sample_reserved_gb": eval_per_sample_reserved_gb,
        "clamped_eval": clamped_eval,
        "oom_at": oom_at,
        "warmed_up": warmed_up,
        "low_snr": low_snr,
    }


def optimize_training_params(
    vram_metrics,
    target_vram_gb=None,
    global_batch_size=64,
    safety=0.9,
    reserve_gb=0.8,
    eval_overhead_gb=0.3,
    allow_approximate=True,
    max_deviation=0.05,
    max_accum_steps=64,
    max_micro_cap=None,
):
    """Pick micro-batch size and gradient accumulation steps.

    Budget semantics (fixed after a real-run bug): target_vram_gb is a TOTAL
    device budget, and static (already-resident weights/grads/optimizer) is
    subtracted from it. Never pass post-load "free" as the target -- static
    would be double-counted (it is already excluded from free) and every
    large model would falsely fall back to (1, N). Callers: resolve the
    budget BEFORE loading via min(config_target, free_before_load).
    target_vram_gb=None uses current free as the total (only valid pre-load).
    max_micro_cap bounds the result (sizing caps at the eval-feasible
    micro so the eval charge stays self-consistent; None = uncapped).

    Returns:
        (per_device_train_batch_size, gradient_accumulation_steps, info_dict).
        Best-effort policy: never raises on OOM, returns (1, global) with a
        warning entry in info_dict instead.
    """
    if global_batch_size is None or global_batch_size <= 0:
        raise ValueError(f"global_batch_size must be positive, got {global_batch_size}")
    if max_deviation < 0 or max_deviation > 0.5:
        raise ValueError(f"max_deviation must be in [0, 0.5], got {max_deviation}")

    warnings_list = []

    free_gb = get_free_vram_gb()
    if target_vram_gb is None:
        effective_target = free_gb
        if effective_target is None:
            raise ValueError(
                "No CUDA GPU detected and target_vram_gb is None; "
                "pass target_vram_gb explicitly."
            )
        warnings_list.append(
            f"Auto-detected VRAM budget: {free_gb} GB (valid only if "
            "measured before model load)."
        )
    else:
        effective_target = target_vram_gb

    static_gb = float(vram_metrics["total_static_gb"])
    per_sample_gb = float(vram_metrics["activations_per_sample_gb"])

    usable_gb = (effective_target - reserve_gb - eval_overhead_gb) * safety
    available_gb = usable_gb - static_gb

    def _info(micro, accum, approximate, extra_warnings=(), fell_back=False):
        effective = micro * accum
        return {
            "per_device_train_batch_size": micro,
            "gradient_accumulation_steps": accum,
            "requested_global_batch_size": global_batch_size,
            "effective_global_batch_size": effective,
            "deviation": round(abs(effective - global_batch_size) / global_batch_size, 4),
            "approximate_used": approximate,
            "fell_back": fell_back,
            "usable_gb": round(usable_gb, 3),
            "static_gb": static_gb,
            "free_gb": free_gb,
            "target_gb": effective_target,
            "eval_overhead_gb": round(eval_overhead_gb, 3),
            "warnings": warnings_list + list(extra_warnings),
        }

    # Best-effort fallbacks: static doesn't fit, or degenerate per-sample size.
    if per_sample_gb <= 0:
        w = (
            "activations_per_sample_gb <= 0; cannot compute micro batch. "
            f"Falling back to (1, {global_batch_size})."
        )
        warnings.warn(w, UserWarning, stacklevel=2)
        return 1, global_batch_size, _info(1, global_batch_size, False, [w],
                                          fell_back=True)

    if available_gb <= 0:
        w = (
            f"Static {static_gb} GB exceeds usable {round(usable_gb, 3)} GB. "
            f"Best-effort fallback to (1, {global_batch_size}); expect OOM."
        )
        warnings.warn(w, UserWarning, stacklevel=2)
        return 1, global_batch_size, _info(1, global_batch_size, False, [w],
                                          fell_back=True)

    max_micro = int(available_gb // per_sample_gb)
    if max_micro_cap is not None:
        # Before the early return: uncapped max_micro >= global silently
        # bypassed the cap (real run: predicted 67 GB yet chose micro 64).
        max_micro = min(max_micro, max(1, max_micro_cap))
    if max_micro >= global_batch_size:
        return (
            global_batch_size,
            1,
            _info(global_batch_size, 1, False),
        )

    max_micro = max(1, max_micro)

    # Strict pass: largest divisor of global <= max_micro (deviation 0).
    strict_micro = 1
    for d in range(max_micro, 0, -1):
        if global_batch_size % d == 0:
            strict_micro = d
            break
    strict_accum = min(global_batch_size // strict_micro, max_accum_steps)

    # If strict already uses memory well, keep it.
    strict_efficient = strict_micro >= max_micro * 0.75 and strict_accum <= 16
    if not allow_approximate or strict_efficient:
        extra = []
        if strict_accum > 16:
            extra.append(
                f"gradient_accumulation_steps={strict_accum} > 16: "
                "slow with stale grads; consider smaller global_batch_size."
            )
        return strict_micro, strict_accum, _info(strict_micro, strict_accum, False, extra)

    # Approximate pass: only consider micro batches LARGER than the strict
    # choice (better VRAM utilization) with effective batch within
    # max_deviation. The strict solution itself has dev 0 and would always
    # win an open search, so it is excluded here.
    best = None  # (dev, -micro, micro, accum)
    for micro in range(max_micro, strict_micro, -1):
        accum = round(global_batch_size / micro)
        accum = max(1, min(accum, max_accum_steps))
        effective = micro * accum
        dev = abs(effective - global_batch_size) / global_batch_size
        if dev <= max_deviation:
            key = (dev, -micro)
            if best is None or key < (best[0], best[1]):
                best = (dev, -micro, micro, accum)

    if best is not None and best[2] > strict_micro:
        _, _, micro, accum = best
        w = (
            f"Approximate batch: requested {global_batch_size}, effective "
            f"{micro * accum} (dev {round(best[0]*100, 2)}%). LR schedule sees "
            f"{micro * accum}, not {global_batch_size}."
        )
        return micro, accum, _info(micro, accum, True, [w])

    extra = []
    if strict_accum > 16:
        extra.append(
            f"gradient_accumulation_steps={strict_accum} > 16: "
            "slow with stale grads; consider smaller global_batch_size."
        )
    return strict_micro, strict_accum, _info(strict_micro, strict_accum, False, extra)
