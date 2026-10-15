"""Pre-flight fit check at real dataset lengths.

No simulated padding, no prediction: tokenize the actual longest
training texts (plus the longest eval text), pad only to their true
max, and run real train steps plus an eval forward at the candidate
micro. Passes or raises FitError on OOM so the run fails fast with an
actionable message instead of OOMing mid-run.
"""

from contextlib import nullcontext


class FitError(RuntimeError):
    """Raised when the configured micro batch does not fit the GPU."""


# Headroom (GB) a passing probe's peak must leave below device total to
# be recommended. A probe peak is one shape, one compile: sustained
# training runs many shapes (length-grouped batches retrigger inductor
# autotune workspaces) plus optimizer/dataloader/fragmentation, so it
# sits GBs above the probe. Measured on RTX 3060 12GB, Qwen3.5-2B,
# max_len 550: probe(m) ~= 2.0 + 0.112*m GB (2.90 @ 8 ... 9.17 @ 64)
# while training peaked 10.28 GB at micro 32 (probe 5.54) and OOMed
# needing ~11.1 at 63-64 -- i.e. a ~4.7 GB sustained gap. Heuristic
# with the arithmetic on record; widen if training ever OOMs below a
# recommended micro.
_HEADROOM_GB = 5.0

# Max sustained-confirm attempts while stepping down visited levels.
_MAX_CONFIRMS = 3


def _device_total_gb():
    try:
        import torch
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        return None


def _reset_peak():
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_gb():
    try:
        import torch
        return torch.cuda.max_memory_allocated() / 1024**3
    except Exception:
        return 0.0


def resolve_batch_mode(training):
    """Config-key dispatch: 'search' or 'fixed'.

    global_batch_size present -> ("search", global): binary-search the
    micro. Absent -> ("fixed", micro): use the pinned
    per_device_train_batch_size. Both/neither (a fixed half-pair counts
    as neither) raises ValueError.
    """
    want_search = training.get("global_batch_size") is not None
    has_fixed = ("per_device_train_batch_size" in training
                 and "gradient_accumulation_steps" in training)
    if want_search == has_fixed:
        raise ValueError(
            "training needs exactly one batch mode: set global_batch_size "
            "(binary-search the micro) OR per_device_train_batch_size + "
            "gradient_accumulation_steps (pinned, no search).")
    if want_search:
        return "search", training["global_batch_size"]
    return "fixed", training["per_device_train_batch_size"]


def find_best_micro(model, tokenizer, train_texts, eval_text, max_length,
                    global_batch, dtype="bfloat16"):
    """Plain binary search for the largest fitting micro in [1, global].

    Textbook halving (mid = (lo + hi) // 2, no pre-fixed candidate list)
    with verify_fit as the oracle: FitError means "too big, go lower".
    Returns (micro, accum, longest, tried, probed_max).

    probed_max (m*) is the largest passing integer. micro is the
    recommended training micro: the largest visited-passing micro whose
    measured probe peak leaves _HEADROOM_GB below device total, then
    verified by one sustained mini-loop (several train steps plus one
    optimizer step, see verify_fit). Single-step passes overestimate
    what a sustained loop survives -- and even short sustained loops
    can squeak past a cliff that training then hits (confirm passed at
    63, trainer OOMed needing ~11.1/11.63 GB) -- so the GB headroom is
    the real margin and the sustained check the backstop. Confirm OOMs
    step down visited levels (max _MAX_CONFIRMS attempts). accum =
    global // micro (floor): the effective batch drifts at or below
    global and is reported, not hidden. micro=1 OOM propagates FitError.
    """
    if not isinstance(global_batch, int) or global_batch <= 0:
        raise ValueError(
            f"global_batch must be a positive int, got {global_batch!r}.")
    tried = []
    peaks = {}
    best = None
    longest = 0
    lo, hi = 1, global_batch
    while lo <= hi:
        micro = (lo + hi) // 2
        _reset_peak()
        try:
            longest = max(longest, verify_fit(
                model, tokenizer, list(train_texts[:max(1, micro)]),
                eval_text, micro, max_length, dtype=dtype))
        except FitError as exc:
            longest = max(longest, getattr(exc, "longest_tokens", 0))
            tried.append((micro, "OOM"))
            hi = micro - 1
        else:
            tried.append((micro, "OK"))
            peaks[micro] = _peak_gb()
            best = micro
            lo = micro + 1
    if best is None:
        # verify_fit already raised once for micro=1; raise a fresh one
        # with the search context (the original is chained implicitly).
        raise FitError(
            f"Micro search FAILED: even micro=1 OOMs at {longest} "
            f"tokens for global_batch={global_batch}. Shorten "
            f"model.max_seq_length or use a smaller model.")
    probed_max = best
    total = _device_total_gb()
    if total is None:
        selected = best  # no meter available: fall back to the edge
    else:
        selected = None
        for micro in sorted(peaks, reverse=True):
            if peaks[micro] + _HEADROOM_GB <= total:
                selected = micro
                break
    if selected is None:
        raise FitError(
            f"Micro search FAILED: no passing micro leaves "
            f"{_HEADROOM_GB} GB headroom on this GPU (single-step edge "
            f"{probed_max}). Pin per_device_train_batch_size manually "
            f"or shorten model.max_seq_length.")
    last_exc = None
    for cand in [m for m in sorted(peaks, reverse=True)
                 if m <= selected][:_MAX_CONFIRMS] or [selected]:
        try:
            longest = max(longest, verify_fit(
                model, tokenizer, list(train_texts[:max(1, cand)]),
                eval_text, cand, max_length, dtype=dtype,
                train_steps=3, materialize_optim=True))
        except FitError as exc:
            longest = max(longest, getattr(exc, "longest_tokens", 0))
            tried.append((cand, "OOM"))
            last_exc = exc
            continue
        tried.append((cand, "OK"))
        selected = cand
        break
    else:
        raise FitError(
            f"Micro search FAILED: sustained check keeps OOMing below "
            f"edge {probed_max}. Pin per_device_train_batch_size "
            f"manually or shorten model.max_seq_length.") from last_exc
    accum = max(1, global_batch // selected)
    return selected, accum, longest, tried, probed_max


def _is_oom(exc):
    return "out of memory" in str(exc).lower()


def verify_fit(model, tokenizer, train_texts, eval_text, micro,
               max_length, dtype="bfloat16", train_steps=1,
               materialize_optim=False):
    """Train steps + one eval forward at real lengths. Pure-ish (GPU).

    Returns the longest sequence length tested. Raises FitError on OOM;
    re-raises anything else. Clears grads and cache afterwards so
    training starts clean. train_steps > 1 with materialize_optim runs
    a sustained mini-loop (several forwards/backwards plus one optimizer
    step) approximating steady-state training memory: optimizer states,
    autotune workspaces and fragmentation do not show up in a single
    step (observed: single-step probe OK at 63-64, trainer OOM at
    ~10.8/11.6 GB). The optimizer uses lr=0.0, so its states allocate
    while weights stay untouched.
    """
    import torch

    if not torch.cuda.is_available():
        raise FitError("Fit check needs a CUDA GPU.")
    # Unwrap multimodal processors to the text tokenizer (their __call__
    # routes plain text through image fetching).
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    device = next(model.parameters()).device

    def _batch(texts, for_eval=False):
        # NOTE: text= as keyword (not positional): Unsloth multimodal
        # models return a Processor whose first positional arg is
        # `images`. Plain HF tokenizers treat text= identically.
        enc = tok(
            text=texts,
            truncation=True,
            max_length=max_length,
            padding="longest",
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        batch = {"input_ids": ids}
        mask = enc.get("attention_mask")
        if mask is not None:
            batch["attention_mask"] = mask.to(device)
        if not for_eval:
            labels = ids.clone()
            attn = batch.get("attention_mask")
            if attn is not None:
                labels[attn == 0] = -100
            batch["labels"] = labels
        return batch

    if dtype in ("float16", "bfloat16") and hasattr(torch, dtype):
        amp = torch.autocast(
            device_type="cuda", dtype=getattr(torch, dtype))
    else:
        amp = nullcontext()

    longest = 0
    model.train()
    try:
        batch = _batch(list(train_texts[:max(1, micro)]))
        longest = int(batch["input_ids"].shape[1])
        for _ in range(max(1, train_steps)):
            with amp:
                loss = model(**batch).loss
            loss.backward()
        if materialize_optim:
            # lr=0.0 keeps weights bit-identical; the optimizer states
            # still allocate, which is the memory we are probing.
            optim = torch.optim.AdamW(
                [p for p in model.parameters()], lr=0.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
        if eval_text:
            model.eval()
            with torch.no_grad(), amp:
                eval_batch = _batch([eval_text], for_eval=True)
                longest = max(
                    longest, int(eval_batch["input_ids"].shape[1]))
                model(**eval_batch)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        err = FitError(
            f"Fit check FAILED at micro={micro}, longest {longest} "
            f"tokens: CUDA out of memory. Lower "
            f"training.per_device_train_batch_size (or shorten "
            f"model.max_seq_length) and retry."
        )
        err.longest_tokens = longest
        raise err from exc
    finally:
        for param in model.parameters():
            if getattr(param, "grad", None) is not None:
                param.grad = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        model.train()
    return longest
