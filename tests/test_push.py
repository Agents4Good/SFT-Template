"""GPU-free tests for the Hub push helper (HfApi stubbed)."""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import push as push_module
from src.utils.push import push_model, resolve_model_folder


def _make_run(tmp_path, name, files=("config.json",)):
    model_dir = tmp_path / name / "Model"
    model_dir.mkdir(parents=True)
    for f in files:
        (model_dir / f).write_text("{}")
    return str(model_dir)


def test_import_has_no_side_effects():
    assert callable(push_module.push_model)
    assert callable(push_module.main)


def test_explicit_folder_returned(tmp_path):
    model_dir = _make_run(tmp_path, "run-20261007_000000")
    assert resolve_model_folder(str(tmp_path), model_dir) == model_dir


def test_latest_run_resolves_newest(tmp_path):
    old = _make_run(tmp_path, "run-20261007_000000")
    time.sleep(0.01)
    new = _make_run(tmp_path, "run-20261007_000001")
    assert resolve_model_folder(str(tmp_path)) == new
    assert old != new


def test_no_runs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model_folder(str(tmp_path))


def test_missing_repo_id_raises(tmp_path):
    model_dir = _make_run(tmp_path, "run-20261007_000000")
    with pytest.raises(ValueError, match="repo_id"):
        push_model(model_dir, "")


def test_dry_run_uploads_nothing(tmp_path):
    model_dir = _make_run(tmp_path, "run-20261007_000000")
    out = push_model(model_dir, "user/model", dry_run=True)
    assert out == model_dir


def test_push_calls_api(tmp_path, monkeypatch):
    import types

    model_dir = _make_run(tmp_path, "run-20261007_000000")
    called = []

    class _Api:
        def upload_folder(self, **kwargs):
            called.append(kwargs)

    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = _Api
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    push_model(model_dir, "user/model", revision="dev", private=True)
    assert called == [{
        "folder_path": model_dir, "repo_id": "user/model",
        "repo_type": "model", "revision": "dev", "private": True,
    }]
