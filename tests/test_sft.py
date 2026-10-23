"""GPU-free tests for training orchestration."""

import os
import sys
import types

import pytest
import torch
from transformers import set_seed
from transformers.trainer_pt_utils import get_length_grouped_indices

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.train.sft as sft_module
from src.train.sft import SFTTrain


def _sft(tmp_path):
    sft = SFTTrain.__new__(SFTTrain)
    sft.config = {
        "training": {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 1,
        },
        "output": {
            "output_dir": str(tmp_path),
            "save_config": False,
            "save_dataset": False,
        },
        "dataset": {"path": "missing.jsonl"},
    }
    sft.final_path = str(tmp_path)
    sft.model = object()
    sft.trainer = None
    sft._env_applied = True
    return sft


class FakeTrainer:
    def __init__(self):
        self.args = types.SimpleNamespace(seed=123)
        self.calls = []

    def train(self):
        self.calls.append("train")


def test_train_reseeds_before_real_run(tmp_path, monkeypatch):
    sft = _sft(tmp_path)
    calls = []
    real_set_seed = sft_module.set_seed

    def record_set_seed(seed):
        calls.append(("seed", seed))
        real_set_seed(seed)

    monkeypatch.setattr(sft_module, "set_seed", record_set_seed)
    monkeypatch.setattr(
        sft_module.telemetry, "record_training_actuals",
        lambda *args, **kwargs: calls.append(("telemetry", None)))

    trainer = FakeTrainer()
    sft.train(trainer)

    assert calls == [("seed", 123), ("telemetry", None)]
    assert trainer.calls == ["train"]
    assert sft.trainer is trainer


def test_train_rejects_nonpositive_batch(tmp_path):
    sft = _sft(tmp_path)
    sft.config["training"]["per_device_train_batch_size"] = 0

    with pytest.raises(ValueError, match="positive"):
        sft.train(FakeTrainer())


def test_prepare_run_dir_backs_up_config_and_dataset(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dataset = source / "data.jsonl"
    dataset.write_text("{}\n")

    run_dir = tmp_path / "run"
    sft = _sft(run_dir)
    sft.config["output"]["save_config"] = True
    sft.config["output"]["save_dataset"] = True
    sft.config["dataset"]["path"] = str(dataset)

    sft._prepare_run_dir()

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "data.jsonl").exists()


def test_reseed_restores_grouped_sampler_order():
    lengths = list(range(256, 0, -1))
    torch.manual_seed(123)
    control = get_length_grouped_indices(lengths, 8)
    torch.manual_seed(123)
    for _ in range(7):
        get_length_grouped_indices(lengths, 8)
    set_seed(123)
    assert get_length_grouped_indices(lengths, 8) == control
