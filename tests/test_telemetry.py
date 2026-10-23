"""GPU-free tests for preflight and post-run telemetry."""

import json
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.train import telemetry


def _config():
    return {
        "training": {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
        },
        "peft": {"use_gradient_checkpointing": True},
    }


def _model():
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            model_type="stub",
            attn_implementation="eager",
        ))


def test_write_preflight_persists_report(tmp_path):
    report = {"mode": "search", "micro": 4}
    path = telemetry.write_preflight(str(tmp_path / "run"), report)

    assert path == os.path.join(str(tmp_path / "run"), "preflight.json")
    with open(path) as handle:
        assert json.load(handle) == report


def test_record_training_actuals_embeds_preflight(tmp_path, monkeypatch):
    class NoCuda:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(cuda=NoCuda))

    telemetry.record_training_actuals(
        str(tmp_path), _model(), _config(), preflight={"mode": "fixed"})

    with open(tmp_path / "peak_memory.json") as handle:
        actuals = json.load(handle)
    assert actuals["preflight"] == {"mode": "fixed"}
    assert actuals["per_device_train_batch_size"] == 4
    assert actuals["gradient_accumulation_steps"] == 2
    assert actuals["arch_fingerprint"]["model_type"] == "stub"
    assert actuals["arch_fingerprint"]["gradient_checkpointing"] is True


def test_record_training_actuals_never_raises(tmp_path):
    telemetry.record_training_actuals(str(tmp_path), _model(), {})

    assert not (tmp_path / "peak_memory.json").exists()


def test_arch_fingerprint_survives_missing_config():
    fingerprint = telemetry._arch_fingerprint(
        types.SimpleNamespace(), {"peft": {}})
    assert fingerprint["chunked_loss"] in (True, False)
    assert fingerprint["attn_implementation"] is None
