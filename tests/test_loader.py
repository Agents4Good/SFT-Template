"""GPU-free tests for the model loader abstraction (all backends stubbed)."""

import os
import sys
import types
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.factory import create_model_handler
from src.model import handler as handler_module
from src.model.hf_loader import (
    HFModel,
    LoaderError,
    _resolve_trust_remote_code,
)


def _base_config(**overrides):
    cfg = {
        "model": {
            "name": "test-model",
            "backend": "auto",
            "trust_remote_code": False,
            "dtype": "float32",
            "load_in_4bit": False,
            "max_seq_length": 128,
        },
        "peft": {"r": 8, "use_gradient_checkpointing": False},
    }
    cfg["model"].update(overrides.pop("model", {}))
    cfg.update(overrides)
    return cfg


def _stub_unsloth(monkeypatch, fail=False):
    torch_mod = types.ModuleType("torch")
    torch_mod.float32 = "torch.float32"
    torch_mod.bfloat16 = "torch.bfloat16"
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    mod = types.ModuleType("unsloth")

    class _Fast:
        @staticmethod
        def from_pretrained(**kwargs):
            if fail:
                raise RuntimeError("unsloth does not support test-model")
            return ("unsloth-model", "unsloth-tokenizer")

        @staticmethod
        def get_peft_model(model, **kwargs):
            return model

    mod.FastLanguageModel = _Fast
    monkeypatch.setitem(sys.modules, "unsloth", mod)
    return mod


class _StubHFConfig:
    def __init__(self, model_type="llama"):
        self.model_type = model_type
        self.use_cache = True


class _StubHFModel:
    def __init__(self):
        self.config = _StubHFConfig()
        self.calls = []
        self.moved_to = None

    def parameters(self):
        return iter([])

    def gradient_checkpointing_enable(self):
        self.calls.append("ckpt")

    def to(self, device):
        self.moved_to = device
        return self


class _StubTokenizer:
    def __init__(self, pad_token=None):
        self.pad_token = pad_token
        self.eos_token = "<eos>"
        self.model_max_length = 10**9


def _stub_transformers(monkeypatch, model_type="llama", pad_token=None,
                       fail_first_trc=False):
    """Stub transformers (+torch/peft) sufficient for HFModel._load."""
    torch_mod = types.ModuleType("torch")
    torch_mod.float32 = "torch.float32"
    torch_mod.bfloat16 = "torch.bfloat16"

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    torch_mod.cuda = _Cuda
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    tr = types.ModuleType("transformers")
    state = {"calls": 0, "model": _StubHFModel(), "tokenizer": _StubTokenizer(pad_token)}

    class _AutoConfig:
        @staticmethod
        def from_pretrained(name, trust_remote_code=False):
            state["calls"] += 1
            if fail_first_trc and not trust_remote_code:
                raise OSError(
                    f"{name} requires custom code: set trust_remote_code=True")
            return _StubHFConfig(model_type)

    class _AutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return state["model"]

    class _AutoTok:
        @staticmethod
        def from_pretrained(name, trust_remote_code=False):
            return state["tokenizer"]

    class _BNBConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    tr.AutoConfig = _AutoConfig
    tr.AutoModelForCausalLM = _AutoModel
    tr.AutoTokenizer = _AutoTok
    tr.BitsAndBytesConfig = _BNBConfig
    monkeypatch.setitem(sys.modules, "transformers", tr)

    peft = types.ModuleType("peft")

    class _TaskType:
        CAUSAL_LM = "CAUSAL_LM"

    captured = {}

    class _LoraConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    peft.TaskType = _TaskType
    peft.LoraConfig = _LoraConfig
    peft.get_peft_model = lambda model, cfg: model
    monkeypatch.setitem(sys.modules, "peft", peft)
    return state, captured


# --- factory routing ---

def test_explicit_unsloth_backend(monkeypatch):
    _stub_unsloth(monkeypatch)
    h = create_model_handler(_base_config(model={"backend": "unsloth"}))
    assert h.backend_name == "unsloth"
    assert h.get_model() == "unsloth-model"


def test_explicit_hf_backend(monkeypatch):
    _stub_transformers(monkeypatch)
    h = create_model_handler(_base_config(model={"backend": "hf"}))
    assert h.backend_name == "hf-peft"
    assert isinstance(h.get_model(), _StubHFModel)


def test_auto_falls_back_to_hf_with_warning(monkeypatch):
    _stub_unsloth(monkeypatch, fail=True)
    _stub_transformers(monkeypatch)
    with pytest.warns(UserWarning, match="falling back to HF"):
        h = create_model_handler(_base_config(model={"backend": "auto"}))
    assert h.backend_name == "hf-peft"


def test_auto_both_fail_reports_both(monkeypatch):
    _stub_unsloth(monkeypatch, fail=True)

    class _BadConfig:
        @staticmethod
        def from_pretrained(name, trust_remote_code=False):
            raise OSError("no such model")

    tr = types.ModuleType("transformers")
    tr.AutoConfig = _BadConfig
    monkeypatch.setitem(sys.modules, "transformers", tr)
    with pytest.warns(UserWarning, match="falling back to HF"):
        with pytest.raises(LoaderError, match="unsloth.*hf-peft"):
            create_model_handler(_base_config(model={"backend": "auto"}))


def test_invalid_backend_rejected(monkeypatch):
    with pytest.raises(LoaderError, match="one of"):
        create_model_handler(_base_config(model={"backend": "vllm"}))


def test_handler_exports_backends():
    assert handler_module.UnslothModel.backend_name == "unsloth"
    assert handler_module.HFModel.backend_name == "hf-peft"
    assert "Model" not in handler_module.__all__


# --- HF loader specifics ---

def test_arch_guard_refuses_seq2seq(monkeypatch):
    _stub_transformers(monkeypatch, model_type="t5")
    with pytest.raises(LoaderError, match="not a causal LM"):
        create_model_handler(_base_config(model={"backend": "hf"}))


def test_unknown_decoder_type_warns_and_proceeds(monkeypatch):
    _stub_transformers(monkeypatch, model_type="brand-new-arch")
    with pytest.warns(UserWarning, match="best-effort"):
        h = create_model_handler(_base_config(model={"backend": "hf"}))
    assert h.backend_name == "hf-peft"


def test_pad_token_falls_back_to_eos(monkeypatch):
    state, _ = _stub_transformers(monkeypatch, pad_token=None)
    h = create_model_handler(_base_config(model={"backend": "hf"}))
    assert h.get_tokenizer().pad_token == "<eos>"
    assert state["tokenizer"].model_max_length == 128


def test_lora_targets_reach_config(monkeypatch):
    _, captured = _stub_transformers(monkeypatch)
    cfg = _base_config(model={"backend": "hf"})
    cfg["peft"]["target_modules"] = ["q_proj", "v_proj"]
    create_model_handler(cfg)
    assert captured["target_modules"] == ["q_proj", "v_proj"]
    assert captured["r"] == 8


def test_trust_remote_code_auto_retries_once(monkeypatch):
    _stub_transformers(monkeypatch, fail_first_trc=True)
    cfg = _base_config(model={"backend": "hf"})
    cfg["model"]["trust_remote_code"] = "auto"
    with pytest.warns(UserWarning, match="executing remote code"):
        h = create_model_handler(cfg)
    assert h.backend_name == "hf-peft"


def test_resolve_trust_remote_code_policy():
    assert _resolve_trust_remote_code(True, "m") == (True, False)
    assert _resolve_trust_remote_code("false", "m") == (False, False)
    assert _resolve_trust_remote_code("auto", "m") == (False, True)
    with pytest.raises(LoaderError, match="auto\\|true\\|false"):
        _resolve_trust_remote_code("maybe", "m")


def test_full_precision_model_moved_to_cuda_when_available(monkeypatch):
    import src.model.hf_loader as hf_mod

    _stub_transformers(monkeypatch)
    torch_stub = sys.modules["torch"]
    torch_stub.cuda.is_available = staticmethod(lambda: True)
    h = create_model_handler(_base_config(model={"backend": "hf"}))
    assert h.get_model().moved_to == "cuda"
    # restore CPU-stub default for other tests (monkeypatch undoes anyway)
