"""HuggingFace + PEFT fallback loader for models Unsloth doesn't cover."""

import warnings

from src.core.base import AbstractModel

# Model types known to be decoder-only causal LMs trainable on this SFT path.
_CAUSAL_MODEL_TYPES = frozenset({
    "llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe",
    "gemma", "gemma2", "gemma3", "phi", "phi3", "falcon", "gpt2", "gpt_neox",
    "opt", "bloom", "deepseek_v2", "deepseek_v3", "command-r",
})
# Architectures this causal-LM SFT path structurally cannot train.
_NON_CAUSAL_MODEL_TYPES = frozenset({
    "t5", "mt5", "bart", "mbart", "bert", "roberta", "electra", "deberta",
    "deberta-v2", "vit", "clip",
})

_DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class LoaderError(RuntimeError):
    """Raised when no backend can load the requested model."""


def _resolve_trust_remote_code(policy, model_name):
    """Normalize the trust_remote_code policy.

    Returns (value, retried_flag). "auto" means: try False first; the caller
    retries once with True (loudly) if loading fails on custom code.
    """
    if isinstance(policy, bool):
        return policy, False
    if isinstance(policy, str) and policy.lower() in ("true", "false"):
        return policy.lower() == "true", False
    if policy is None or (isinstance(policy, str) and policy.lower() == "auto"):
        return False, True
    raise LoaderError(
        f"model.trust_remote_code must be auto|true|false, got {policy!r} "
        f"for model '{model_name}'."
    )


def _looks_like_custom_code_error(exc):
    msg = str(exc).lower()
    return any(s in msg for s in (
        "trust_remote_code", "custom code", "no model file", "does not exist",
        "configuration file", "not a valid model",
    ))


class HFModel(AbstractModel):
    """Model loader backed by transformers + PEFT (any causal LM).

    Covers models outside Unsloth's supported list: plain
    AutoModelForCausalLM load (optional 4-bit via BitsAndBytes), gradient
    checkpointing parity, and LoRA via peft. Refuses non-causal
    architectures (encoder-only / seq2seq) with a clear error.
    """

    backend_name = "hf-peft"

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        model_cfg = self.config["model"]
        peft_cfg = self.config.get("peft", {})
        model_name = model_cfg["name"]
        trc, allow_retry = _resolve_trust_remote_code(
            model_cfg.get("trust_remote_code", "auto"), model_name
        )
        try:
            self._load(trust_remote_code=trc)
        except Exception as exc:
            if allow_retry and _looks_like_custom_code_error(exc):
                warnings.warn(
                    f"'{model_name}' needs custom model code; retrying with "
                    "trust_remote_code=True. You are executing remote code.",
                    UserWarning,
                    stacklevel=2,
                )
                self._load(trust_remote_code=True)
            else:
                raise

    def _load(self, trust_remote_code):
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        model_cfg = self.config["model"]
        peft_cfg = self.config.get("peft", {})
        model_name = model_cfg["name"]

        hf_config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self._guard_architecture(hf_config, model_name)

        dtype_name = model_cfg.get("dtype", "bfloat16")
        load_in_4bit = model_cfg.get("load_in_4bit", False)
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as e:
                raise LoaderError(
                    f"load_in_4bit requested for '{model_name}' but "
                    f"BitsAndBytesConfig is unavailable: {e}"
                ) from e
            # bitsandbytes requires device placement via accelerate.
            model_kwargs["device_map"] = "auto"
            compute = getattr(torch, dtype_name, None)
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            torch_dtype = getattr(torch, dtype_name, None)
            if torch_dtype is not None:
                model_kwargs["dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not load_in_4bit and torch.cuda.is_available():
            # Single-GPU template scope: keep every param on cuda:0 so the
            # VRAM probe (one device) and training see the same placement.
            # Quantized loads are already placed via device_map="auto".
            model.to("cuda")

        if peft_cfg.get("use_gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = peft_cfg.get("target_modules") or _DEFAULT_TARGET_MODULES
        r = peft_cfg.get("r", 16)
        lora_config = LoraConfig(
            r=r,
            lora_alpha=peft_cfg.get("lora_alpha", r),
            lora_dropout=peft_cfg.get("lora_dropout", 0),
            bias=peft_cfg.get("bias", "none"),
            target_modules=list(target_modules),
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(model, lora_config)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        seq_len = model_cfg.get("max_seq_length")
        if seq_len is not None:
            try:
                tokenizer.model_max_length = int(seq_len)
            except (TypeError, ValueError):
                pass
        self.tokenizer = tokenizer

    @staticmethod
    def _guard_architecture(hf_config, model_name):
        model_type = getattr(hf_config, "model_type", "") or ""
        if model_type in _NON_CAUSAL_MODEL_TYPES:
            raise LoaderError(
                f"'{model_name}' is a '{model_type}' architecture, not a "
                "causal LM; this SFT path requires a decoder-only model."
            )
        if model_type and model_type not in _CAUSAL_MODEL_TYPES:
            warnings.warn(
                f"'{model_name}' has unrecognized model_type "
                f"'{model_type}'; attempting causal-LM load best-effort.",
                UserWarning,
                stacklevel=3,
            )

    def get_model(self):
        """Returns the loaded (PEFT-wrapped) model instance."""
        return self.model

    def get_tokenizer(self):
        """Returns the loaded tokenizer instance."""
        return self.tokenizer
