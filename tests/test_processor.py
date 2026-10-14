"""GPU-free tests for token counting with multimodal processors."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset.processor import DatasetProcessor, assistant_role_for_model


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


def test_assistant_role_defaults_to_assistant():
    assert assistant_role_for_model("unsloth/Qwen3.5-2B") == "assistant"
    assert assistant_role_for_model("meta-llama/Llama-3-8B") == "assistant"
    assert assistant_role_for_model("brand-new-arch") == "assistant"
    assert assistant_role_for_model("") == "assistant"
    assert assistant_role_for_model(None) == "assistant"


def test_assistant_role_keeps_gemma_model_role():
    assert assistant_role_for_model("unsloth/gemma-4-E2B-it") == "model"
    assert assistant_role_for_model("GOOGLE/GEMMA-2-9B-IT") == "model"


class _ChatTok(_TextTok):
    def __init__(self):
        super().__init__()
        self.roles_seen = []

    def apply_chat_template(self, messages, **kwargs):
        self.roles_seen = [m["role"] for m in messages]
        return " ".join(m["content"] for m in messages)


def test_format_uses_family_role():
    qwen = DatasetProcessor(
        {"model": {"name": "unsloth/Qwen3.5-2B"}}, tokenizer=_ChatTok())
    out = qwen.format_dataset(_tiny_gsm())
    assert qwen.tokenizer.roles_seen == ["user", "assistant"]
    assert out[0]["text"].startswith("Q:")

    gemma = DatasetProcessor(
        {"model": {"name": "unsloth/gemma-3-4b-it"}}, tokenizer=_ChatTok())
    gemma.format_dataset(_tiny_gsm())
    assert gemma.tokenizer.roles_seen == ["user", "model"]


def _tiny_gsm():
    from datasets import Dataset
    return Dataset.from_dict({
        "id": [0],
        "question": ["Q: 1+1?"],
        "answer": ["2"],
    })
