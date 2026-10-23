"""GPU-free tests for measured pre-flight micro-batch sizing."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.train.fit_check import (
    FitError, check_pinned, find_best_micro, resolve_batch_mode)


class Probe:
    """Fake probe: per-micro peak sequences, optional OOM at some micros."""

    def __init__(self, peaks=None, oom=()):
        self.peaks = {m: list(v) for m, v in (peaks or {}).items()}
        self.oom = set(oom)
        self.calls = []

    def __call__(self, micro):
        self.calls.append(micro)
        if micro in self.oom:
            exc = FitError(f"OOM at micro={micro}")
            exc.longest_tokens = 550
            raise exc
        return {
            "peak_gb": self.peaks[micro].pop(0),
            "duration_s": 1.25,
            "steps": 3,
        }


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


@pytest.mark.parametrize("global_batch", [0, -4, 4.5, "64", None])
def test_search_rejects_bad_global_batch(global_batch):
    with pytest.raises(ValueError, match="positive int"):
        find_best_micro(Probe(), 10.0, global_batch)


@pytest.mark.parametrize("max_micro", [0, -2])
def test_search_rejects_bad_max_micro(max_micro):
    with pytest.raises(ValueError, match="at least 1"):
        find_best_micro(Probe(), 10.0, 8, max_micro=max_micro)


@pytest.mark.parametrize("usable,reserve", [
    (0, 0.10), (-1, 0.10), (10, 1.0), (10, -0.10), (10, "0.1"),
])
def test_search_rejects_bad_memory_limit(usable, reserve):
    with pytest.raises((ValueError, TypeError)):
        find_best_micro(Probe(), usable, 8, reserve_fraction=reserve)


def test_search_all_fit_picks_global_without_confirm():
    probe = Probe({4: [4.0], 6: [6.0], 7: [7.0], 8: [8.0]})
    report = find_best_micro(probe, 10.0, 8, reserve_fraction=0.10)
    # 10 GB usable - 10% = 9 GB limit; the binary walk ends at 8.
    assert probe.calls == [4, 6, 7, 8]
    assert report["micro"] == 8
    assert report["accum"] == 1
    assert report["effective_batch_size"] == 8
    assert report["limit_gb"] == pytest.approx(9.0)
    assert report["confirm"] is None
    assert [c["result"] for c in report["candidates"]] == [
        "pass", "pass", "pass", "pass"]
    assert report["candidates"][-1]["peak_reserved_gb"] == 8.0


def test_search_walks_down_and_confirms_warm_cache():
    probe = Probe({4: [4.0], 6: [6.0], 7: [7.0, 7.0], 8: [10.0]})
    report = find_best_micro(probe, 10.0, 8, reserve_fraction=0.10)
    # 4/6/7 pass, 8 is over the 9 GB limit, so 7 is re-probed warm.
    assert probe.calls == [4, 6, 7, 8, 7]
    assert report["micro"] == 7
    assert report["accum"] == 1
    assert report["confirm"]["result"] == "confirm-pass"
    assert [(c["micro"], c["result"]) for c in report["candidates"]] == [
        (4, "pass"), (6, "pass"), (7, "pass"), (8, "over"),
        (7, "confirm-pass")]


def test_search_fails_when_warm_confirm_exceeds_limit():
    probe = Probe({4: [4.0], 6: [6.0], 7: [7.0, 9.5], 8: [10.0]})
    with pytest.raises(FitError, match="warm-cache confirm"):
        find_best_micro(probe, 10.0, 8, reserve_fraction=0.10)
    assert probe.calls == [4, 6, 7, 8, 7]


def test_search_skips_oom_candidates():
    probe = Probe({2: [2.0, 2.0], 1: [1.0]}, oom={4, 3})
    report = find_best_micro(probe, 10.0, 8, reserve_fraction=0.10)
    assert probe.calls == [4, 2, 3, 2]
    assert report["micro"] == 2
    assert report["accum"] == 4
    assert report["candidates"][0]["result"] == "oom"
    assert report["candidates"][0]["longest_tokens"] == 550
    assert [(c["micro"], c["result"]) for c in report["candidates"]] == [
        (4, "oom"), (2, "pass"), (3, "oom"), (2, "confirm-pass")]


def test_search_fails_when_micro_1_ooms():
    probe = Probe(oom={2, 1})
    with pytest.raises(FitError, match="no candidate") as info:
        find_best_micro(probe, 10.0, 4, reserve_fraction=0.10)
    assert probe.calls == [2, 1]
    assert "Tried" in str(info.value)


def test_search_caps_max_micro_to_global_and_reports_floor_accum():
    probe = Probe({4: [4.0], 2: [2.0], 3: [3.0]})
    report = find_best_micro(
        probe, 10.0, 10, reserve_fraction=0.10, max_micro=4)
    # max_micro is capped to global; the winner <= max_micro.
    assert probe.calls == [2, 3, 4]
    assert report["max_micro"] == 4
    assert report["micro"] == 4
    assert report["accum"] == 2
    assert report["effective_batch_size"] == 8


def test_search_budget_is_min_probes_for_global_64():
    probe = Probe({m: [m / 8.0] for m in range(1, 65)})
    report = find_best_micro(probe, 10.0, 64, reserve_fraction=0.10)
    # 64 max candidates, 9 GB limit -> 64 peaks at 8 GB and fits. The
    # binary walk is 6 probes (32, 48, 56, 60, 62, 63) plus 64 = 7,
    # no warm confirm because 64 is the final probe.
    assert report["micro"] == 64
    assert report["candidates"][-1]["result"] == "pass"
    assert len(probe.calls) == 7


def test_pinned_pass_reports_measurement():
    probe = Probe({4: [6.5]})
    report = check_pinned(probe, 4, 10.0, reserve_fraction=0.10)
    assert probe.calls == [4]
    assert report["micro"] == 4
    assert report["peak_reserved_gb"] == 6.5
    assert report["limit_gb"] == pytest.approx(9.0)
    assert report["usable_gb"] == 10.0
    assert report["reserve_fraction"] == 0.10


def test_pinned_over_limit_raises():
    with pytest.raises(FitError, match="exceeds") as info:
        check_pinned(Probe({4: [9.5]}), 4, 10.0, reserve_fraction=0.10)
    assert "per_device_train_batch_size" in str(info.value)


def test_pinned_oom_raises():
    with pytest.raises(FitError, match="CUDA out of memory"):
        check_pinned(Probe(oom={4}), 4, 10.0, reserve_fraction=0.10)


@pytest.mark.parametrize("micro", [0, -1, 2.5, "4", None])
def test_pinned_rejects_bad_micro(micro):
    with pytest.raises(ValueError, match="positive int"):
        check_pinned(Probe(), micro, 10.0)
