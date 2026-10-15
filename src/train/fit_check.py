"""Pre-flight fit check at real dataset lengths.

No simulated padding, no prediction: tokenize the actual longest
training texts (plus the longest eval text), pad only to their true
max, run one train step and one eval forward at the configured micro.
Passes (returns the longest length) or raises FitError on OOM so the
run fails fast with an actionable message instead of OOMing mid-run.
"""

from contextlib import nullcontext


class FitError(RuntimeError):
    """Raised when the configured micro batch does not fit the GPU."""


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


def micro_candidates(global_batch):
    """Sorted divisors of global_batch within [1, global_batch].

    Restricting the search to divisors keeps micro * accum == global
    exactly, so the effective batch (and hence the training math) never
    changes while searching.
    """
    if (not isinstance(global_batch, int) or global_batch <= 0):
        raise ValueError(
            f"global_batch must be a positive int, got {global_batch!r}.")
    return [m for m in range(1, global_batch + 1)
            if global_batch % m == 0]


def find_best_micro(model, tokenizer, train_texts, eval_text, max_length,
                    global_batch, dtype="bfloat16"):
    """Binary-search the largest fitting micro in [1, global_batch].

    Candidates are the divisors of global_batch (see micro_candidates),
    tested largest-fitting-first via binary search with verify_fit as the
    oracle: FitError means "too big, go lower". Returns
    (micro, accum, longest, tried, probed_max).

    micro is the recommended training micro: the largest passing
    candidate, backed off one step when the top candidate passed. A
    single probe step underestimates a sustained training loop (optimizer
    states, dataloader, inductor recompile/autotune on varying shapes);
    an all-pass search leaves the true edge unobserved, so taking hi
    verbatim OOMs mid-run (observed: probe-OK at 64, trainer OOM at
    ~10.8/11.6 GB). When an OOM *was* observed above, the edge is known
    and the largest passer already sits ~2x below it, so no back-off.
    probed_max is the largest passing candidate (== micro when no
    back-off); tried is the list of (micro, "OK"/"OOM") in test order.
    micro=1 OOM propagates FitError.
    """
    candidates = micro_candidates(global_batch)
    tried = []
    best = None
    longest = 0
    lo, hi = 0, len(candidates) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        micro = candidates[mid]
        try:
            longest = max(longest, verify_fit(
                model, tokenizer, list(train_texts[:max(1, micro)]),
                eval_text, micro, max_length, dtype=dtype))
        except FitError as exc:
            longest = max(longest, getattr(exc, "longest_tokens", 0))
            tried.append((micro, "OOM"))
            hi = mid - 1
        else:
            tried.append((micro, "OK"))
            best = micro
            lo = mid + 1
    if best is None:
        # verify_fit already raised once for micro=1; raise a fresh one
        # with the search context (the original is chained implicitly).
        raise FitError(
            f"Micro search FAILED: even micro=1 OOMs at {longest} "
            f"tokens for global_batch={global_batch}. Shorten "
            f"model.max_seq_length or use a smaller model.")
    probed_max = best
    idx = candidates.index(best)
    if idx == len(candidates) - 1 and len(candidates) > 1:
        best = candidates[idx - 1]  # top passed: edge unobserved, back off
    return best, global_batch // best, longest, tried, probed_max


def _is_oom(exc):
    return "out of memory" in str(exc).lower()


def verify_fit(model, tokenizer, train_texts, eval_text, micro,
               max_length, dtype="bfloat16"):
    """One train step + one eval forward at real lengths. Pure-ish (GPU).

    Returns the longest sequence length tested. Raises FitError on OOM;
    re-raises anything else. Clears grads and cache afterwards so
    training starts clean.
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
        with amp:
            loss = model(**batch).loss
        loss.backward()
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
