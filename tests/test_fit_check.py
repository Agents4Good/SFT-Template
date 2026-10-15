"""GPU-free tests for the pre-flight fit check (real lengths, no padding)."""

import os
import sys
import types
from contextlib import nullcontext

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.train.fit_check import (
    FitError, find_best_micro, micro_candidates, resolve_batch_mode,
    verify_fit)


class _FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.setitem_calls = []

    def to(self, device):
        return self

    def clone(self):
        return _FakeTensor(self.shape)

    def __eq__(self, other):
        return _FakeMask()

    def __setitem__(self, key, value):
        self.setitem_calls.append(value)


class _FakeMask:
    pass


class _FakeParam:
    def __init__(self):
        self.grad = None
        self.device = "cuda"


class _FakeLoss:
    def __init__(self, model):
        self.model = model

    def backward(self):
        self.model.backward_calls += 1
        for param in self.model.parameters():
            param.grad = object()


class _StubModel:
    def __init__(self, oom_at=None):
        self.params = [_FakeParam()]
        self.mode = None
        self.backward_calls = 0
        self.forward_calls = 0
        self.oom_at = oom_at

    def parameters(self):
        return iter(self.params)

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"

    def __call__(self, **kwargs):
        self.forward_calls += 1
        ids = kwargs.get("input_ids")
        n = ids.shape[0] if ids is not None else 0
        if self.oom_at is not None and n >= self.oom_at:
            raise RuntimeError("CUDA out of memory: stub OOM")
        from types import SimpleNamespace
        return SimpleNamespace(loss=_FakeLoss(self))


class _StubTok:
    def __call__(self, text, truncation=True, max_length=None,
                 padding="longest", return_tensors="pt"):
        longest = min(max_length or 0, max(len(t) for t in text))
        return {"input_ids": _FakeTensor((len(text), longest))}


def _stub_torch(monkeypatch, cuda=True):
    mod = types.ModuleType("torch")
    mod.float32 = "float32"
    mod.bfloat16 = "bfloat16"

    class _Cuda:
        @staticmethod
        def is_available():
            return cuda

        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def empty_cache():
            return None

    mod.cuda = _Cuda
    mod.autocast = lambda **kwargs: nullcontext()
    mod.no_grad = lambda: nullcontext()
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


def test_fit_passes_at_real_length(monkeypatch):
    _stub_torch(monkeypatch)
    model = _StubModel()
    # Caller pre-selects the longest texts (main.py sorts row lengths).
    longest = verify_fit(
        model, _StubTok(), ["aaaaaa", "aaaa"], "bb", micro=2,
        max_length=512)
    # micro=2 longest texts: 6 and 4 chars -> padded to true max 6.
    assert longest == 6
    assert model.backward_calls == 1
    assert model.forward_calls == 2  # one train step + one eval forward
    assert model.mode == "train"  # restored for training
    assert model.params[0].grad is None  # grads cleared afterwards


def test_fit_oom_raises_actionable_error(monkeypatch):
    _stub_torch(monkeypatch)
    model = _StubModel(oom_at=2)
    with pytest.raises(FitError, match="micro=2"):
        verify_fit(model, _StubTok(), ["aa", "aaaa"], "bb", micro=2,
                   max_length=512)
    try:
        verify_fit(model, _StubTok(), ["aa", "aaaa"], "bb", micro=2,
                   max_length=512)
    except FitError as exc:
        assert "per_device_train_batch_size" in str(exc)
    assert model.params[0].grad is None


def test_fit_without_cuda_fails(monkeypatch):
    _stub_torch(monkeypatch, cuda=False)
    with pytest.raises(FitError, match="CUDA"):
        verify_fit(_StubModel(), _StubTok(), ["aa"], None, micro=1,
                   max_length=512)


def test_fit_without_eval_text_skips_eval(monkeypatch):
    _stub_torch(monkeypatch)
    model = _StubModel()
    longest = verify_fit(model, _StubTok(), ["aaaa"], None, micro=1,
                         max_length=512)
    assert longest == 4
    assert model.forward_calls == 1


def test_micro_candidates_are_exact_divisors():
    assert micro_candidates(64) == [1, 2, 4, 8, 16, 32, 64]
    assert micro_candidates(48) == [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]
    assert micro_candidates(1) == [1]
    for bad in (0, -4, 4.5, "64", None):
        with pytest.raises(ValueError):
            micro_candidates(bad)


def test_search_brackets_largest_fitting(monkeypatch):
    _stub_torch(monkeypatch)
    model = _StubModel(oom_at=6)
    texts = ["a" * 8, "a" * 7, "a" * 6, "a" * 5,
             "a" * 4, "a" * 3, "a" * 2, "a"]
    micro, accum, longest, tried, probed_max = find_best_micro(
        model, _StubTok(), texts, "bb", 512, 8)
    # OOM observed at 8: edge known, no back-off; (4, 2) exact.
    assert (micro, accum, probed_max) == (4, 2, 4)
    assert micro * accum == 8
    assert longest == 8
    # Binary order over [1, 2, 4, 8]: 2 OK -> 4 OK -> 8 OOM.
    assert tried == [(2, "OK"), (4, "OK"), (8, "OOM")]


def test_search_backs_off_from_unobserved_edge(monkeypatch):
    _stub_torch(monkeypatch)
    micro, accum, longest, tried, probed_max = find_best_micro(
        _StubModel(), _StubTok(), ["a" * 8, "a" * 7], "bb", 512, 8)
    # All pass: top candidate's edge is unobserved -> one step back.
    assert (micro, accum, probed_max) == (4, 2, 8)
    assert tried == [(2, "OK"), (4, "OK"), (8, "OK")]


def test_search_single_candidate_never_backs_off(monkeypatch):
    _stub_torch(monkeypatch)
    micro, accum, longest, tried, probed_max = find_best_micro(
        _StubModel(), _StubTok(), ["aaaa"], "bb", 512, 1)
    assert (micro, accum, probed_max) == (1, 1, 1)


def test_search_fails_when_micro_1_ooms(monkeypatch):
    _stub_torch(monkeypatch)
    with pytest.raises(FitError, match="even micro=1 OOMs at 4 tokens"):
        find_best_micro(_StubModel(oom_at=1), _StubTok(),
                        ["aaaa", "bb"], "cc", 512, 8)


def test_resolve_batch_mode_dispatch():
    assert resolve_batch_mode({"global_batch_size": 64}) == ("search", 64)
    fixed = {"per_device_train_batch_size": 4,
             "gradient_accumulation_steps": 16}
    assert resolve_batch_mode(fixed) == ("fixed", 4)
    # global present wins even with a stray fixed half-pair.
    assert resolve_batch_mode(
        {"global_batch_size": 64,
         "per_device_train_batch_size": 4}) == ("search", 64)
    for bad in ({}, {"per_device_train_batch_size": 4},
                {"gradient_accumulation_steps": 16},
                {"global_batch_size": 64, **fixed}):
        with pytest.raises(ValueError, match="exactly one batch mode"):
            resolve_batch_mode(bad)
