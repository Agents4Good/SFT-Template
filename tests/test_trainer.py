"""GPU-free tests for Trainer construction, promotion and disposal."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import EarlyStoppingCallback
from transformers.trainer_pt_utils import LengthGroupedSampler

import src.train.trainer as trainer_module
from src.train.trainer import (
    LengthGroupedSFTTrainer, _precision_flags, _sampling_strategy,
    build_args, build_callbacks, build_trainer, promote_trainer)


class _Dataset:
    """Minimal column-addressable dataset for sampler tests."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0])

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def __len__(self):
        return len(self.rows)


def _config():
    return {
        "model": {
            "name": "stub/model",
            # CPU-only tests: SFTConfig rejects bf16 without a GPU.
            "dtype": "float32",
            "max_seq_length": 550,
        },
        "training": {
            "optim": "adamw_8bit",
            "learning_rate": 3.0e-4,
            "lr_scheduler_type": "cosine",
            "warmup_steps": 100,
            "num_train_epochs": 5,
            "max_steps": -1,
            "eval_save_strategy": "steps",
            "eval_save_steps": 200,
            "save_total_limit": 2,
            "group_by_length": True,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "push_to_hub": False,
            "report_to": "tensorboard",
        },
        "peft": {"use_gradient_checkpointing": True},
        "output": {"output_dir": "/tmp"},
    }


class _FakeTrainer:
    def __init__(self):
        # A real SFTConfig: promote_trainer relies on properties such as
        # train_batch_size being available on trainer.args.
        self.args = build_args(_config(), 1, 1, "/tmp/probe", probe=True)
        self._train_batch_size = 1
        self.accelerator = types.SimpleNamespace(
            gradient_accumulation_steps=1)
        self.optimizer = object()
        self.lr_scheduler = object()
        self._created_lr_scheduler = True
        self.control = types.SimpleNamespace(
            should_training_stop=True,
            should_log=True,
            should_evaluate=True,
            should_save=True,
        )
        self.callback_handler = types.SimpleNamespace(callbacks=[])

    def add_callback(self, callback):
        self.callback_handler.callbacks.append(callback)


def test_probe_args_are_isolated_from_real_training():
    args = build_args(_config(), 8, 1, "/tmp/probe", probe=True,
                      probe_steps=3)
    assert args.per_device_train_batch_size == 8
    assert args.gradient_accumulation_steps == 1
    assert args.max_steps == 3
    assert args.learning_rate == 0.0
    assert args.weight_decay == 0.0
    assert args.num_train_epochs == 1.0
    assert args.warmup_steps == 0
    assert args.eval_strategy == "no"
    assert args.save_strategy == "no"
    assert args.logging_strategy == "no"
    assert args.load_best_model_at_end is False
    assert args.push_to_hub is False
    assert args.report_to == []
    assert args.train_sampling_strategy == "group_by_length"
    assert args.gradient_checkpointing is True
    assert args.max_length == 550


def test_real_args_keep_schedule_and_eval():
    args = build_args(_config(), 4, 16, "/tmp/real")
    assert args.per_device_train_batch_size == 4
    assert args.gradient_accumulation_steps == 16
    assert args.max_steps == -1
    assert args.num_train_epochs == 5
    assert args.learning_rate == 3.0e-4
    assert args.eval_strategy == "steps"
    assert args.eval_steps == 200
    assert args.save_steps == 200
    assert args.save_total_limit == 2
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "eval_loss"
    assert args.train_sampling_strategy == "group_by_length"


def test_precision_flags_map_config_dtype():
    config = _config()
    config["model"]["dtype"] = "float16"
    assert _precision_flags(config) == (True, False)
    config["model"]["dtype"] = "bfloat16"
    assert _precision_flags(config) == (False, True)
    config["model"]["dtype"] = "float32"
    assert _precision_flags(config) == (False, False)
    config["model"].pop("dtype")
    assert _precision_flags(config) == (False, False)


def test_explicit_lengths_from_column_or_input_ids():
    trainer = LengthGroupedSFTTrainer.__new__(LengthGroupedSFTTrainer)
    trainer.args = types.SimpleNamespace(length_column_name="length")
    dataset = _Dataset([
        {"length": 5, "input_ids": [1] * 5},
        {"length": 2, "input_ids": [1] * 2},
    ])
    assert trainer._explicit_lengths(dataset) == [5, 2]
    trainer.args.length_column_name = "missing"
    assert trainer._explicit_lengths(dataset) == [5, 2]


def test_grouped_sampler_uses_explicit_lengths_longest_first():
    trainer = LengthGroupedSFTTrainer.__new__(LengthGroupedSFTTrainer)
    trainer.args = types.SimpleNamespace(
        train_sampling_strategy="group_by_length",
        train_batch_size=2,
        gradient_accumulation_steps=1,
        length_column_name="length",
    )
    trainer.train_dataset = _Dataset([
        {"length": 1, "input_ids": [1]},
        {"length": 4, "input_ids": [1] * 4},
        {"length": 2, "input_ids": [1] * 2},
        {"length": 3, "input_ids": [1] * 3},
    ])
    sampler = trainer._get_train_sampler()
    assert isinstance(sampler, LengthGroupedSampler)
    indices = list(sampler)
    assert sorted(indices) == [0, 1, 2, 3]
    assert trainer.train_dataset["length"][indices[0]] == 4


def test_sampling_strategy_follows_config():
    training = _config()["training"]
    assert _sampling_strategy(training) == "group_by_length"
    training["group_by_length"] = False
    assert _sampling_strategy(training) == "random"
    training.pop("group_by_length")
    assert _sampling_strategy(training) == "random"


def test_build_callbacks_respects_early_stopping():
    config = _config()
    assert build_callbacks(config) == []
    config["training"]["early_stopping"] = {
        "enabled": True, "patience": 2, "threshold": 0.01}
    callbacks = build_callbacks(config)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], EarlyStoppingCallback)


def test_build_trainer_passes_component_through(monkeypatch):
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        trainer_module, "LengthGroupedSFTTrainer", FakeTrainer)
    args = build_args(_config(), 2, 3, "/tmp/x")
    build_trainer("model", "tokenizer", "train", "eval", args)
    assert captured["args"] is args
    assert captured["model"] == "model"
    assert captured["processing_class"] == "tokenizer"
    assert captured["train_dataset"] == "train"
    assert captured["eval_dataset"] == "eval"
    assert captured["callbacks"] == []


def test_promote_trainer_restores_real_args_and_accelerator():
    trainer = _FakeTrainer()
    args_object = trainer.args
    real = build_args(_config(), 4, 16, "/tmp/real")

    promote_trainer(trainer, real)

    assert trainer.args is args_object
    assert trainer.args.per_device_train_batch_size == 4
    assert trainer.args.gradient_accumulation_steps == 16
    assert trainer.args.learning_rate == 3.0e-4
    assert trainer.args.max_steps == -1
    assert trainer.args.output_dir == "/tmp/real"
    assert trainer.args.save_strategy == "steps"
    assert trainer._train_batch_size == trainer.args.train_batch_size
    # Accelerate divides the loss by this value at backward time.
    assert trainer.accelerator.gradient_accumulation_steps == 16
    assert trainer.optimizer is None
    assert trainer.lr_scheduler is None
    assert trainer._created_lr_scheduler is False
    assert trainer.control.should_training_stop is False
    assert trainer.control.should_log is False
    assert trainer.control.should_evaluate is False
    assert trainer.control.should_save is False


def test_promote_trainer_dedupes_callbacks():
    trainer = _FakeTrainer()
    real = build_args(_config(), 4, 16, "/tmp/real")
    config = _config()
    config["training"]["early_stopping"] = {
        "enabled": True, "patience": 2, "threshold": 0.01}

    promote_trainer(trainer, real, build_callbacks(config))
    promote_trainer(trainer, real, build_callbacks(config))

    callbacks = trainer.callback_handler.callbacks
    assert sum(isinstance(c, EarlyStoppingCallback) for c in callbacks) == 1
    assert len(callbacks) == len(set(callbacks))


def test_close_trainer_clears_grads(monkeypatch):
    cleared = []

    class Parameter:
        def __init__(self):
            self.grad = object()

    class Model:
        def __init__(self):
            self.params = [Parameter()]

        def parameters(self):
            return self.params

    model = Model()
    trainer = types.SimpleNamespace(model=model)
    monkeypatch.setattr(
        trainer_module.gc, "collect", lambda: cleared.append("gc"))

    trainer_module.close_trainer(trainer)

    assert model.params[0].grad is None
    assert "gc" in cleared
    trainer_module.close_trainer(None)
