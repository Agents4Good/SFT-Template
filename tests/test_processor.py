"""GPU-free tests for token counting with multimodal processors."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset.processor import DatasetProcessor


class _TextTok:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts, **kwargs):
        self.calls += 1
        return {"input_ids": [[1, 2, 3] for _ in texts]}


class _VisionProc:
    """Mimics Gemma4Processor: __call__ routes through image fetching."""

    def __init__(self):
        self.tokenizer = _TextTok()

    def __call__(self, *args, **kwargs):
        raise AssertionError("processor __call__ must not count tokens")


def test_counts_use_inner_text_tokenizer():
    proc = DatasetProcessor({}, tokenizer=_VisionProc())
    assert proc.calculate_tokens_usage([{"text": "hi"}]) == [3]


def test_counts_fall_back_to_tokenizer_itself():
    tok = _TextTok()
    proc = DatasetProcessor({}, tokenizer=tok)
    assert proc.calculate_tokens_usage(
        [{"text": "hi"}, {"text": "yo"}]) == [3, 3]
    assert tok.calls == 1
