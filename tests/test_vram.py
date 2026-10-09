"""Tests for VRAM estimation + batch optimization (no GPU/model download).

Runs under the project venv (torch must be importable); the GPU is hidden
and all CUDA interaction is stubbed. Fake torch covers autocast/no_grad.
"""

import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.vram import (
    optimize_training_params,
    measure_static_gb,
    sizing_static_gb,
    training_baseline_gb,
    predicted_peak_gb,
    per_sample_gb_from_peaks,
    probe_per_sample_bytes,
)


def _metrics(static_gb, per_sample_gb):
    return {
        "total_static_gb": static_gb,
        "activations_per_sample_gb": per_sample_gb,
    }


def test_strict_divisor():
    # usable=(5-1.1)*0.9=3.51, available=2.51, max_micro=5 -> 4x16 strict.
    micro, accum, info = optimize_training_params(
        _metrics(1.0, 0.5),
        target_vram_gb=5.0,
        global_batch_size=64,
        safety=0.9,
        reserve_gb=0.8,
        eval_overhead_gb=0.3,
        allow_approximate=False,
    )
    assert (micro, accum) == (4, 16)
    assert info["approximate_used"] is False


def test_approximate_64_with_max6_gives_5x13():
    # usable=(5-1.1)*0.9=3.51, available=2.51, max_micro=6. Strict is 4x16;
    # approx candidates larger than 4 within 5%: 6x11=66 (3.1%), 5x13=65
    # (1.6%). Smallest deviation wins -> 5x13.
    micro, accum, info = optimize_training_params(
        _metrics(1.0, 0.4),
        target_vram_gb=5.0,
        global_batch_size=64,
        safety=0.9,
        reserve_gb=0.8,
        eval_overhead_gb=0.3,
        allow_approximate=True,
        max_deviation=0.05,
    )
    assert (micro, accum) == (5, 13)
    assert info["approximate_used"] is True
    assert info["effective_global_batch_size"] == 65


def test_prime_61_with_max8_gives_6x10():
    micro, accum, info = optimize_training_params(
        _metrics(1.0, 0.3),
        target_vram_gb=5.0,
        global_batch_size=61,
        safety=0.9,
        reserve_gb=0.8,
        eval_overhead_gb=0.3,
        allow_approximate=True,
        max_deviation=0.05,
    )
    assert (micro, accum) == (6, 10)
    assert info["effective_global_batch_size"] == 60


def test_static_exceeds_budget_falls_back_best_effort():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        micro, accum, info = optimize_training_params(
            _metrics(10.0, 0.5),
            target_vram_gb=5.0,
            global_batch_size=64,
        )
    assert (micro, accum) == (1, 64)
    assert any("exceeds usable" in w for w in info["warnings"])
    assert len(caught) == 1


def test_zero_per_sample_falls_back():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        micro, accum, _ = optimize_training_params(
            _metrics(1.0, 0.0), target_vram_gb=5.0, global_batch_size=32
        )
    assert (micro, accum) == (1, 32)


def test_invalid_inputs_raise():
    import pytest

    with pytest.raises(ValueError):
        optimize_training_params(_metrics(1.0, 0.5), target_vram_gb=5.0,
                                 global_batch_size=0)


# --- Measurement helpers (GPU-free via duck-typed fakes) ---

GB = 1024**3


class _FakeTensor:
    def __init__(self, numel, elsize):
        self._numel = numel
        self._elsize = elsize

    def numel(self):
        return self._numel

    def element_size(self):
        return self._elsize


class _FakeParam(_FakeTensor):
    def __init__(self, numel, elsize, requires_grad=True, device="cuda:0"):
        super().__init__(numel, elsize)
        self.requires_grad = requires_grad
        self.device = device


class _FakeModel:
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return iter(self._params)


def test_measure_static_sums_tensors():
    model = _FakeModel([
        _FakeParam(1_000_000, 1, requires_grad=True),   # 4-bit-ish weight
        _FakeParam(10_000, 2, requires_grad=True),      # LoRA param
        _FakeParam(500_000, 1, requires_grad=False),    # frozen
    ])
    opt_state = {0: {"exp_avg": _FakeTensor(10_000, 4),
                     "exp_avg_sq": _FakeTensor(10_000, 4)}}

    class _Opt:
        state = opt_state

    # tensor-only reporting: no context key, no CUDA sniffing
    tensor_total = 1_520_000 + 1_020_000 + 80_000
    m = measure_static_gb(model, _Opt())
    assert m["weights_gb"] == round(1_520_000 / GB, 3)
    assert m["gradients_gb"] == round(1_020_000 / GB, 3)
    assert m["optimizer_gb"] == round(80_000 / GB, 3)
    assert m["total_static_gb"] == round(tensor_total / GB, 3)
    assert "context_gb" not in m
    assert m["method"] == "measured"


def test_sizing_static_adds_second_opt_instance():
    measured = {"weights_gb": 7.0, "gradients_gb": 0.2, "optimizer_gb": 0.1,
                "total_static_gb": 7.3, "method": "measured"}
    got = sizing_static_gb(measured)
    assert got == round(7.3 + 0.1, 3)


def test_training_baseline_derives_floor_from_probe():
    # floor = smallest-micro peak minus its activation share, + trainer
    # optimizer instance, x fragmentation growth.
    static = {"weights_gb": 7.0, "gradients_gb": 0.2, "optimizer_gb": 0.1,
              "total_static_gb": 7.3, "method": "measured"}
    probe = {"reserved_peaks": {1: 9 * GB, 2: 9 * GB + GB // 10,
                                4: 9 * GB + 3 * GB // 10},
             "per_sample_reserved_gb": 0.1}
    got = training_baseline_gb(static, probe)
    assert got == round((9.0 - 0.1 + 0.1) * 1.1, 3)


def test_training_baseline_falls_back_without_peaks():
    static = {"weights_gb": 7.0, "gradients_gb": 0.2, "optimizer_gb": 0.1,
              "total_static_gb": 7.3, "method": "measured"}
    got = training_baseline_gb(static, {"reserved_peaks": {},
                                        "per_sample_reserved_gb": 0.0})
    assert got == round(7.3 + 0.1, 3)


def test_predicted_peak_adds_slopes_linearly():
    assert predicted_peak_gb(9.9, 0.1, 2, 0.3) == round(9.9 + 0.2 + 0.6, 3)


def test_slope_takes_max_consecutive():
    baseline = 2 * GB
    peaks = {1: baseline + 100, 2: baseline + 250, 4: baseline + 700}
    # slopes: 150/B, 225/B -> max wins
    assert per_sample_gb_from_peaks(peaks, baseline) == 225 / GB


def test_slope_single_point_and_empty():
    assert per_sample_gb_from_peaks({1: 1000}, 400) == 600 / GB
    assert per_sample_gb_from_peaks({}, 400) == 0.0
    assert per_sample_gb_from_peaks({1: 100}, 500) == 0.0  # negative clamped


# --- probe wiring with stubbed CUDA/model (verifies plumbing, not physics) ---

class _StubCuda:
    """Scripted peaks: returns queued max_memory_allocated values."""

    def __init__(self, baseline, peak_queue, reserved_queue=None,
                 reserved_baseline=None):
        self._baseline = baseline
        self._queue = list(peak_queue)
        self._rqueue = list(reserved_queue) if reserved_queue is not None else None
        self._reserved_baseline = (
            reserved_baseline if reserved_baseline is not None else baseline
        )
        self.is_available = lambda: True

    def memory_allocated(self):
        return self._baseline

    def memory_reserved(self):
        return self._reserved_baseline

    def max_memory_allocated(self):
        if not self._queue:
            raise AssertionError("more peak reads than scripted")
        v = self._queue.pop(0)
        if isinstance(v, Exception):
            raise v
        self._last = v
        return v

    def max_memory_reserved(self):
        if self._rqueue is None:
            # Default: reserved tracks allocated (no separate queue).
            return self._last
        if not self._rqueue:
            raise AssertionError("more reserved peak reads than scripted")
        v = self._rqueue.pop(0)
        if isinstance(v, Exception):
            raise v
        return v

    def reset_peak_memory_stats(self):
        pass

    def empty_cache(self):
        pass

    def synchronize(self):
        pass


class _StubLoss:
    def backward(self):
        pass


class _StubOut:
    loss = _StubLoss()


class _StubModel(_FakeModel):
    def __init__(self, params, oom_at=None):
        super().__init__(params)
        self.oom_at = oom_at
        self.train_calls = 0

    def train(self):
        self.train_calls += 1

    def eval(self):
        pass

    def __call__(self, **kwargs):
        if self.oom_at is not None and kwargs.get("n", 0) >= self.oom_at:
            raise RuntimeError("CUDA out of memory: stub OOM")
        return _StubOut()


class _StubOpt:
    state = {}

    def step(self):
        pass

    def zero_grad(self, set_to_none=True):
        pass


def _install_stub_torch():
    """Fake torch module: autocast/no_grad only (cuda passed explicitly).

    vram.py binds torch at import, so the fake is set as a module
    attribute (a sys.modules patch alone would no longer take effect).
    """
    import sys
    import types
    from contextlib import nullcontext

    import src.utils.vram as vram_module

    mod = types.ModuleType("torch")
    mod.autocast = lambda *a, **k: nullcontext()
    mod.no_grad = lambda *a, **k: nullcontext()
    sys.modules["torch"] = mod
    vram_module.torch = mod
    return mod


def test_probe_records_peaks_and_slope():
    _install_stub_torch()
    base = 3 * GB
    per = int(0.4 * GB)
    cuda = _StubCuda(base, [base + per, base + 2 * per])  # trials micro 1, 2
    model = _StubModel([_FakeParam(10, 4)])
    out = probe_per_sample_bytes(
        model, None, _StubOpt(), ["a", "b"], 128,
        micros=(1, 2), repeats=1, dtype="float32",
        _cuda=cuda, _batch_fn=lambda n: {"n": n},
    )
    assert out["peaks"] == {1: base + per, 2: base + 2 * per}
    assert out["per_sample_gb"] == per / GB
    assert out["oom_at"] is None
    assert out["warmed_up"] is True


def test_probe_oom_yields_partial_peaks():
    _install_stub_torch()
    base = 3 * GB
    per = int(0.4 * GB)
    cuda = _StubCuda(base, [base + per, base + 2 * per])
    model = _StubModel([_FakeParam(10, 4)], oom_at=4)
    out = probe_per_sample_bytes(
        model, None, _StubOpt(), ["a"] * 4, 128,
        micros=(1, 2, 4), repeats=1, dtype="float32",
        _cuda=cuda, _batch_fn=lambda n: {"n": n},
    )
    assert out["oom_at"] == 4
    assert out["per_sample_gb"] == per / GB


def test_probe_passes_text_as_keyword():
    """Multimodal processors take images positionally; text must be kwarg."""
    _install_stub_torch()
    seen = {}

    class _Tok:
        def __call__(self, *args, **kwargs):
            seen.update(kwargs)
            seen["positional"] = args

            class _T:
                def to(self, device):
                    return self

                def clone(self):
                    return self

                def __setitem__(self, k, v):
                    pass

            return {"input_ids": _T(), "attention_mask": None, "labels": _T()}

    base = 3 * GB
    per = int(0.4 * GB)
    cuda = _StubCuda(base, [base + per])
    model = _StubModel([_FakeParam(10, 4)])
    out = probe_per_sample_bytes(
        model, _Tok(), _StubOpt(), ["a"], 128,
        micros=(1,), repeats=1, dtype="float32", _cuda=cuda)
    assert seen["positional"] == ()
    assert seen["text"] == ["a"]
    assert seen["padding"] == "max_length"
    assert out["warmed_up"] is True


def test_resolve_budget_prefers_minimum():
    from src.utils.vram import resolve_budget
    import pytest

    assert resolve_budget(5.0, 11.6) == (5.0, [])
    b, notes = resolve_budget(12.0, 10.2)
    assert b == 10.2 and any("budgeting" in n for n in notes)
    assert resolve_budget(None, 8.0)[0] == 8.0
    assert resolve_budget(5.0, None)[0] == 5.0
    with pytest.raises(ValueError, match="No VRAM budget"):
        resolve_budget(None, None)


def test_probe_low_snr_detector():
    from src.utils.vram import probe_low_snr

    assert probe_low_snr({1: 100, 2: 90, 4: 110}, 50) is True
    assert probe_low_snr({1: 100, 2: 150, 4: 250}, 50) is False
    assert probe_low_snr({}, 50) is False
    assert probe_low_snr({1: 40}, 50) is True
    assert probe_low_snr({1: 60}, 50) is False


def test_probe_applies_margin_on_low_snr():
    _install_stub_torch()
    base = 3 * GB
    # non-monotonic: micro 2 peaks below micro 1 -> noise dominates
    cuda = _StubCuda(base, [base + 300, base + 100, base + 350])
    model = _StubModel([_FakeParam(10, 4)])
    out = probe_per_sample_bytes(
        model, None, _StubOpt(), ["a"] * 4, 128,
        micros=(1, 2, 4), repeats=1, dtype="float32",
        _cuda=cuda, _batch_fn=lambda n: {"n": n})
    assert out["low_snr"] is True
    # raw max slope would be (350-100)/2=125B; margined 2x
    assert out["per_sample_gb"] == 250 / GB


def test_fell_back_flag_set_only_on_fallback():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, _, info = optimize_training_params(
            {"total_static_gb": 10.0, "activations_per_sample_gb": 0.5},
            target_vram_gb=5.0, global_batch_size=64)
    assert info["fell_back"] is True
    assert any(issubclass(w.category, UserWarning) for w in caught)
    _, _, info = optimize_training_params(
        {"total_static_gb": 1.0, "activations_per_sample_gb": 0.5},
        target_vram_gb=5.0, global_batch_size=64, allow_approximate=False)
    assert info["fell_back"] is False


def test_explicit_target_not_clamped_to_current_free():
    # Regression: post-load free must not shrink an explicit total budget.
    _, _, info = optimize_training_params(
        {"total_static_gb": 1.0, "activations_per_sample_gb": 0.5},
        target_vram_gb=5.0, global_batch_size=64, allow_approximate=False)
    assert info["target_gb"] == 5.0


def test_probe_records_reserved_peaks_and_slope():
    _install_stub_torch()
    base = 3 * GB
    rbase = int(3.5 * GB)
    per = int(0.4 * GB)
    rper = int(0.5 * GB)
    cuda = _StubCuda(base, [base + per, base + 2 * per],
                     reserved_queue=[rbase + rper, rbase + 2 * rper],
                     reserved_baseline=rbase)
    model = _StubModel([_FakeParam(10, 4)])
    out = probe_per_sample_bytes(
        model, None, _StubOpt(), ["a", "b"], 128,
        micros=(1, 2), repeats=1, dtype="float32",
        _cuda=cuda, _batch_fn=lambda n: {"n": n},
    )
    assert out["reserved_peaks"] == {1: rbase + rper, 2: rbase + 2 * rper}
    assert out["per_sample_reserved_gb"] == rper / GB
    # allocated slope still reported independently
    assert out["per_sample_gb"] == per / GB


def test_probe_clamps_noisy_eval_to_train_slope():
    # Clamp applies only with checkpointing OFF (eval fwd-only < train).
    _install_stub_torch()
    base = 3 * GB
    per = int(0.1 * GB)

    def _run(ckpt):
        cuda = _StubCuda(base, [base + per, base + 2 * per,
                                base + per, base + 9 * per])
        model = _StubModel([_FakeParam(10, 4)])
        return probe_per_sample_bytes(
            model, None, _StubOpt(), ["a", "b"], 128,
            micros=(1, 2), repeats=1, dtype="float32",
            eval_texts=["e1", "e2"], _cuda=cuda,
            _batch_fn=lambda n: {"n": n},
            use_gradient_checkpointing=ckpt)

    out = _run(False)
    assert out["clamped_eval"] is True
    assert out["eval_per_sample_gb"] == out["per_sample_gb"]
    assert out["eval_per_sample_reserved_gb"] == out[
        "per_sample_reserved_gb"]

    # Checkpointing ON: eval stores full activations (no recompute), so a
    # larger eval slope stands -- clamping would under-charge eval ~20x.
    out = _run(True)
    assert out["clamped_eval"] is False
    assert out["eval_per_sample_gb"] == 8 * per / GB

    # Unknown: safe side, no clamp.
    out = _run(None)
    assert out["clamped_eval"] is False
