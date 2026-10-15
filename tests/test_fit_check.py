"""GPU-free tests for the pre-flight fit check (real lengths, no padding)."""

import os
import sys
import types
from contextlib import nullcontext

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.train.fit_check import (
    FitError, find_best_micro, resolve_batch_mode, verify_fit)


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


class _FakeOptim:
    last = None

    def __init__(self, params, lr=0.0):
        self.params = list(params)
        self.lr = lr
        self.step_calls = 0
        _FakeOptim.last = self

    def step(self):
        self.step_calls += 1

    def zero_grad(self, set_to_none=False):
        pass


class _StubModel:
    def __init__(self, oom_at=None, oom_calls=()):
        self.params = [_FakeParam()]
        self.mode = None
        self.backward_calls = 0
        self.forward_calls = 0
        self.oom_at = oom_at
        self.oom_calls = set(oom_calls)

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
        if ((self.oom_at is not None and n >= self.oom_at)
                or self.forward_calls in self.oom_calls):
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
        total_memory = 12 * 1024**3
        peak_script = []

        @staticmethod
        def is_available():
            return cuda

        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def empty_cache():
            return None

        @classmethod
        def reset_peak_memory_stats(cls):
            return None

        @classmethod
        def max_memory_allocated(cls):
            if cls.peak_script:
                return cls.peak_script.pop(0)
            return 0

        @staticmethod
        def get_device_properties(dev):
            return types.SimpleNamespace(
                total_memory=_Cuda.total_memory)

    mod.cuda = _Cuda
    mod.autocast = lambda **kwargs: nullcontext()
    mod.no_grad = lambda: nullcontext()
    mod.optim = types.SimpleNamespace(AdamW=_FakeOptim)
    _FakeOptim.last = None
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


def _gb(*values):
    return [int(v * 1024**3) for v in values]


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


def test_verify_fit_sustained_loop_with_optim(monkeypatch):
    _stub_torch(monkeypatch)
    model = _StubModel()
    longest = verify_fit(model, _StubTok(), ["aaaa", "bb"], "cc", micro=2,
                         max_length=512, train_steps=3,
                         materialize_optim=True)
    assert longest == 4
    assert model.backward_calls == 3
    assert model.forward_calls == 4  # 3 train steps + 1 eval forward
    # lr=0.0: optimizer states allocate, weights stay untouched.
    assert _FakeOptim.last is not None
    assert _FakeOptim.last.lr == 0.0
    assert _FakeOptim.last.step_calls == 1
    assert model.params[0].grad is None


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


def test_search_rejects_bad_global_batch(monkeypatch):
    _stub_torch(monkeypatch)
    for bad in (0, -4, 4.5, "64", None):
        with pytest.raises(ValueError, match="positive int"):
            find_best_micro(_StubModel(), _StubTok(), ["aa"], "bb",
                            512, bad)


def test_search_halves_live_interval(monkeypatch):
    mod = _stub_torch(monkeypatch)
    model = _StubModel(oom_at=51)
    texts = ["x" * 600] * 64
    # Peaks (GB) for the passing walk probes 32/48/50 in order.
    mod.cuda.peak_script = _gb(5.0, 8.0, 10.6)
    micro, accum, longest, tried, probed_max = find_best_micro(
        model, _StubTok(), texts, "yy", 512, 64)
    # Plain halving over [1, 64], edge at 50: 32/48/56/52/50/51.
    # 51 is not a divisor of anything -- it was probed anyway.
    assert tried[:-1] == [(32, "OK"), (48, "OK"), (56, "OOM"),
                          (52, "OOM"), (50, "OK"), (51, "OOM")]
    assert probed_max == 50
    # Headroom gate (5 GB of 12): 50 needs 15.6, 48 needs 13 -> out;
    # 32 needs 10.0, confirmed sustained.
    assert tried[-1] == (32, "OK")
    assert (micro, accum) == (32, 2)
    assert micro * accum <= 64
    assert longest == 512


def test_search_first_probe_is_live_midpoint(monkeypatch):
    mod = _stub_torch(monkeypatch)
    model = _StubModel(oom_at=55)
    texts = ["x" * 600] * 96
    # Peaks for passing probes 48/54 in order.
    mod.cuda.peak_script = _gb(3.0, 10.9)
    micro, accum, longest, tried, probed_max = find_best_micro(
        model, _StubTok(), texts, "yy", 512, 96)
    # (1 + 96) // 2 = 48: the first probe is 48, organically.
    assert tried[0] == (48, "OK")
    assert tried[:-1] == [(48, "OK"), (72, "OOM"), (60, "OOM"),
                          (54, "OK"), (57, "OOM"), (55, "OOM")]
    assert probed_max == 54
    # 54 needs 15.9 GB -> out; 48 needs 8.0, confirmed sustained.
    assert tried[-1] == (48, "OK")
    assert (micro, accum) == (48, 2)


def test_search_confirms_margin_below_edge(monkeypatch):
    mod = _stub_torch(monkeypatch)
    model = _StubModel(oom_at=24)
    texts = ["x" * 600] * 100
    # Peaks for passing probes 12/18/21/23 in order.
    mod.cuda.peak_script = _gb(3.0, 5.0, 8.8, 10.8)
    micro, accum, longest, tried, probed_max = find_best_micro(
        model, _StubTok(), texts, "yy", 512, 100)
    assert tried[:-1] == [(50, "OOM"), (25, "OOM"), (12, "OK"),
                          (18, "OK"), (21, "OK"), (23, "OK"),
                          (24, "OOM")]
    assert probed_max == 23
    # 23 needs 15.8, 21 needs 13.8 -> out; 18 needs 10.0, sustained OK.
    assert tried[-1] == (18, "OK")
    assert len(tried) == 8
    assert (micro, accum) == (18, 5)
    assert micro * accum <= 100


def test_search_steps_one_below_top(monkeypatch):
    mod = _stub_torch(monkeypatch)
    # Peaks for passing probes 4/6/7/8 in order: 8 leaves no headroom.
    mod.cuda.peak_script = _gb(1.0, 2.0, 9.5, 11.0)
    micro, accum, longest, tried, probed_max = find_best_micro(
        _StubModel(), _StubTok(), ["a" * 8, "a" * 7], "bb", 512, 8)
    # Halving over [1, 8], all pass: edge unobserved.
    assert tried[:-1] == [(4, "OK"), (6, "OK"), (7, "OK"), (8, "OK")]
    assert probed_max == 8
    # 8 needs 16, 7 needs 14.5 -> out; 6 needs 7.0, sustained OK.
    assert tried[-1] == (6, "OK")
    assert (micro, accum) == (6, 1)


def test_search_walks_down_past_fragile_micros(monkeypatch):
    mod = _stub_torch(monkeypatch)
    # Rows fit singly, but the first sustained loop fails on call 5:
    # peaks reject 8 and 7 at the gate, the sustained confirm of 6
    # then fails, and 4 survives. eval=None keeps call counting to
    # train forwards only.
    mod.cuda.peak_script = _gb(1.0, 2.0, 9.5, 11.0)
    model = _StubModel(oom_calls={5})
    texts = ["x" * 600] * 8
    micro, accum, longest, tried, probed_max = find_best_micro(
        model, _StubTok(), texts, None, 512, 8)
    assert tried == [(4, "OK"), (6, "OK"), (7, "OK"), (8, "OK"),
                     (6, "OOM"), (4, "OK")]
    assert probed_max == 8
    assert (micro, accum) == (4, 2)
    assert micro * accum <= 8


def test_search_fails_when_sustained_keeps_ooming(monkeypatch):
    _stub_torch(monkeypatch)
    model = _StubModel(oom_calls={2, 3, 4, 5, 6, 7, 8, 9, 10})
    with pytest.raises(FitError, match="keeps OOMing"):
        find_best_micro(model, _StubTok(), ["aa", "bb"], None, 512, 2)


def test_search_floors_at_one(monkeypatch):
    _stub_torch(monkeypatch)
    micro, accum, longest, tried, probed_max = find_best_micro(
        _StubModel(), _StubTok(), ["aaaa"], "bb", 512, 1)
    assert (micro, accum, probed_max) == (1, 1, 1)
    # No room below: the sustained confirm re-probes 1 and passes.
    assert tried == [(1, "OK"), (1, "OK")]


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
