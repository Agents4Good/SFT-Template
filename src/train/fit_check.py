"""Pre-flight micro-batch sizing measured on the real training path.

No predicted headroom and no memory model: every candidate micro runs a
few real SFTTrainer steps at the longest training shapes (the length-
grouped sampler puts the longest batch first), then one eval forward.
The measured peak reserved VRAM is compared with the memory currently
available to this process. Pass or raise FitError with the measurements
attached so the caller can persist an auditable report.
"""


class FitError(RuntimeError):
    """Raised when no micro batch fits the available GPU memory."""


def _limit_gb(usable_gb, reserve_fraction):
    if usable_gb is None or usable_gb <= 0:
        raise ValueError(f"usable_gb must be positive, got {usable_gb!r}.")
    if not isinstance(reserve_fraction, (int, float)) or not (
            0.0 <= reserve_fraction < 1.0):
        raise ValueError(
            "reserve_fraction must be in [0, 1), got "
            f"{reserve_fraction!r}.")
    return usable_gb * (1.0 - reserve_fraction)


def resolve_batch_mode(training):
    """Config-key dispatch: 'search' or 'fixed'.

    global_batch_size present -> ("search", global): probe candidates
    and pick the largest fitting micro. Absent -> ("fixed", micro): use
    the pinned per_device_train_batch_size. Both/neither (a fixed
    half-pair counts as neither) raises ValueError.
    """
    want_search = training.get("global_batch_size") is not None
    has_fixed = ("per_device_train_batch_size" in training
                 and "gradient_accumulation_steps" in training)
    if want_search == has_fixed:
        raise ValueError(
            "training needs exactly one batch mode: set global_batch_size "
            "(probe candidates and pick the largest fitting micro) OR "
            "per_device_train_batch_size + gradient_accumulation_steps "
            "(pinned, one fit check).")
    if want_search:
        return "search", training["global_batch_size"]
    return "fixed", training["per_device_train_batch_size"]


def _record(micro, result, probe_result=None):
    out = {"micro": micro, "result": result}
    if probe_result:
        out["peak_reserved_gb"] = probe_result.get("peak_gb")
        out["duration_s"] = probe_result.get("duration_s")
        if "steps" in probe_result:
            out["steps"] = probe_result["steps"]
    return out


def find_best_micro(probe, usable_gb, global_batch, reserve_fraction=0.10,
                    max_micro=None):
    """Binary-search the largest micro whose measured peak fits.

    probe(micro) must run the real training path and return
    {"peak_gb": float, "duration_s": float, ...} or raise FitError on
    OOM. A candidate passes when its peak is at or below
    usable_gb * (1 - reserve_fraction). The probe session (trainer,
    optimizer, compiled shapes) is reused by the caller across calls.

    After the search, the winning micro is re-probed once if it was not
    the final probe: later probes can leave optimizer state and
    compiled workspaces behind, so the confirm measures the same
    warm-cache conditions real training will see. Returns a report dict.
    """
    if not isinstance(global_batch, int) or global_batch <= 0:
        raise ValueError(
            f"global_batch must be a positive int, got {global_batch!r}.")
    if max_micro is None:
        max_micro = global_batch
    max_micro = min(int(max_micro), global_batch)
    if max_micro < 1:
        raise ValueError(
            f"max_micro must be at least 1, got {max_micro!r}.")
    limit = _limit_gb(usable_gb, reserve_fraction)

    tried = []
    best = None
    last_probed = None
    lo, hi = 1, max_micro
    while lo <= hi:
        micro = (lo + hi) // 2
        last_probed = micro
        try:
            result = probe(micro)
        except FitError as exc:
            longest = getattr(exc, "longest_tokens", None)
            item = _record(micro, "oom")
            if longest is not None:
                item["longest_tokens"] = longest
            tried.append(item)
            hi = micro - 1
            continue
        peak = float(result["peak_gb"])
        if peak <= limit:
            tried.append(_record(micro, "pass", result))
            best = micro
            lo = micro + 1
        else:
            tried.append(_record(micro, "over", result))
            hi = micro - 1

    if best is None:
        raise FitError(
            f"Micro search FAILED: no candidate at or below "
            f"{limit:.2f} GB (usable {usable_gb:.2f} GB, reserve "
            f"{reserve_fraction:.0%}, max {max_micro}). Tried: "
            f"{tried}. Lower model.max_seq_length, use a smaller model, "
            f"or raise training.vram_reserve_fraction deliberately.")

    confirm = None
    if last_probed != best:
        try:
            result = probe(best)
        except FitError as exc:
            raise FitError(
                f"Micro search FAILED: warm-cache confirm OOM at the "
                f"selected micro={best} (usable {usable_gb:.2f} GB, "
                f"reserve {reserve_fraction:.0%}). Tried: {tried}."
            ) from exc
        peak = float(result["peak_gb"])
        confirm = _record(best, "confirm-pass" if peak <= limit
                          else "confirm-over", result)
        tried.append(confirm)
        if peak > limit:
            raise FitError(
                f"Micro search FAILED: micro={best} passed one probe but "
                f"peaked at {peak:.2f} GB on the warm-cache confirm "
                f"(limit {limit:.2f} GB). Tried: {tried}. Lower "
                f"model.max_seq_length or raise training.vram_reserve_"
                f"fraction deliberately.")

    accum = max(1, global_batch // best)
    return {
        "micro": best,
        "accum": accum,
        "effective_batch_size": best * accum,
        "global_batch_size": global_batch,
        "max_micro": max_micro,
        "usable_gb": round(usable_gb, 3),
        "reserve_fraction": reserve_fraction,
        "limit_gb": round(limit, 3),
        "candidates": tried,
        "confirm": confirm,
    }


def check_pinned(probe, micro, usable_gb, reserve_fraction=0.10):
    """Run one probe for a pinned micro and enforce the memory reserve."""
    if not isinstance(micro, int) or micro <= 0:
        raise ValueError(f"micro must be a positive int, got {micro!r}.")
    limit = _limit_gb(usable_gb, reserve_fraction)
    try:
        result = probe(micro)
    except FitError as exc:
        raise FitError(
            f"Fit check FAILED at pinned micro={micro}: CUDA out of "
            f"memory (usable {usable_gb:.2f} GB). Lower "
            f"training.per_device_train_batch_size or shorten "
            f"model.max_seq_length."
        ) from exc
    peak = float(result["peak_gb"])
    if peak > limit:
        raise FitError(
            f"Fit check FAILED at pinned micro={micro}: peak "
            f"{peak:.2f} GB exceeds the {limit:.2f} GB limit "
            f"(usable {usable_gb:.2f} GB, reserve "
            f"{reserve_fraction:.0%}). Lower "
            f"training.per_device_train_batch_size, shorten "
            f"model.max_seq_length, or raise "
            f"training.vram_reserve_fraction deliberately.")
    return {
        "micro": micro,
        "peak_reserved_gb": round(peak, 3),
        "duration_s": result.get("duration_s"),
        "usable_gb": round(usable_gb, 3),
        "reserve_fraction": reserve_fraction,
        "limit_gb": round(limit, 3),
    }
