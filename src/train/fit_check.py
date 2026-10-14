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
        raise FitError(
            f"Fit check FAILED at micro={micro}, longest {longest} "
            f"tokens: CUDA out of memory. Lower "
            f"training.per_device_train_batch_size (or shorten "
            f"model.max_seq_length) and retry."
        ) from exc
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
