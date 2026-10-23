"""GPU-free tests for measured probes and session lifecycle."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.train.probe as probe_module
from src.train.fit_check import FitError
from src.train.probe import ProbeSession, usable_memory_gb


class FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def synchronize():
        return None

    @staticmethod
    def empty_cache():
        return None

    @staticmethod
    def reset_peak_memory_stats():
        return None

    @staticmethod
    def max_memory_reserved():
        return 5 * 1024**3

    @staticmethod
    def mem_get_info():
        return 6 * 1024**3, 12 * 1024**3

    @staticmethod
    def memory_reserved():
        return 2 * 1024**3


class FakeModel:
    def __init__(self):
        self.params = []

    def parameters(self):
        return self.params

    def train(self):
        return None

    def eval(self):
        return None


class FakeTrainer:
    def __init__(self, fail=None):
        self.args = types.SimpleNamespace(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
        )
        self._train_batch_size = 1
        self.control = types.SimpleNamespace(should_training_stop=False)
        self.eval_dataset = None
        self.model = FakeModel()
        self.trained = 0
        self.fail = fail

    def train(self):
        if self.fail is not None:
            raise self.fail
        self.trained += 1


def test_usable_memory_adds_own_reserved_to_free(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(cuda=FakeCuda))
    usable, total = usable_memory_gb()
    assert usable == pytest.approx(8.0)
    assert total == pytest.approx(12.0)


def test_usable_memory_requires_cuda(monkeypatch):
    class NoCuda:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(cuda=NoCuda))
    with pytest.raises(FitError, match="CUDA"):
        usable_memory_gb()


def test_open_removes_tmp_when_factory_fails(tmp_path, monkeypatch):
    tmp_dir = tmp_path / "probe"
    tmp_dir.mkdir()
    monkeypatch.setattr(
        probe_module.tempfile, "mkdtemp", lambda prefix: str(tmp_dir))

    def factory(tmp, steps):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        ProbeSession.open(factory)
    assert not tmp_dir.exists()


def test_close_is_idempotent_and_removes_tmp(monkeypatch):
    disposed = []
    monkeypatch.setattr(
        probe_module, "close_trainer", lambda trainer: disposed.append(trainer))

    session = ProbeSession.open(lambda tmp, steps: FakeTrainer(), steps=2)
    trainer = session.trainer
    tmp_dir = session.tmp_dir

    session.close()
    session.close()

    assert disposed == [trainer]
    assert session.trainer is None
    assert not os.path.exists(tmp_dir)


def test_release_transfers_ownership(monkeypatch):
    disposed = []
    monkeypatch.setattr(
        probe_module, "close_trainer", lambda trainer: disposed.append(trainer))

    session = ProbeSession.open(lambda tmp, steps: FakeTrainer(), steps=2)
    trainer = session.trainer
    tmp_dir = session.tmp_dir

    assert session.release_trainer() is trainer
    session.close()

    assert disposed == []
    assert not os.path.exists(tmp_dir)


def test_closed_or_released_session_rejects_probes():
    session = ProbeSession.open(lambda tmp, steps: FakeTrainer(), steps=2)
    session.release_trainer()
    with pytest.raises(RuntimeError, match="released"):
        session.run(2)
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.run(2)


def test_run_returns_measured_peak(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(cuda=FakeCuda))
    session = ProbeSession.open(lambda tmp, steps: FakeTrainer(), steps=3)

    result = session.run(4)

    assert result["peak_gb"] == pytest.approx(5.0)
    assert result["steps"] == 3
    assert session.probes == 1
    assert session.trainer.args.per_device_train_batch_size == 4
    session.close()


def test_run_maps_oom_to_fit_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(cuda=FakeCuda))
    oom = RuntimeError("CUDA out of memory. Tried to allocate")
    session = ProbeSession.open(
        lambda tmp, steps: FakeTrainer(fail=oom), steps=3)

    with pytest.raises(FitError, match="OOM at micro=4"):
        session.run(4)
    session.close()


def test_run_reraises_non_oom_errors(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(cuda=FakeCuda))
    session = ProbeSession.open(
        lambda tmp, steps: FakeTrainer(fail=RuntimeError("bad shape")),
        steps=3)

    with pytest.raises(RuntimeError, match="bad shape"):
        session.run(4)
    session.close()
