"""Backend selection for model loading.

auto (default): try Unsloth first (identical path/weights to the legacy
behavior when it works), fall back to HF+PEFT with a logged warning.
Explicit "unsloth"/"hf" skips the other backend entirely.
"""

import warnings

from src.core.base import AbstractModel
from src.model.hf_loader import HFModel, LoaderError
from src.model.unsloth_loader import UnslothModel

_BACKENDS = ("auto", "unsloth", "hf")


def create_model_handler(config) -> AbstractModel:
    """Instantiate and load the model handler for config['model']['backend']."""
    backend = (config.get("model", {}) or {}).get("backend", "auto")
    if backend not in _BACKENDS:
        raise LoaderError(
            f"model.backend must be one of {_BACKENDS}, got {backend!r}."
        )
    errors = {}
    if backend in ("auto", "unsloth"):
        handler = UnslothModel(config)
        try:
            handler.load_model()
            return handler
        except Exception as exc:  # noqa: BLE001 - aggregated below
            errors["unsloth"] = exc
            if backend == "unsloth":
                raise LoaderError(
                    f"Unsloth backend failed: {exc}"
                ) from exc
            warnings.warn(
                f"Unsloth load failed ({exc}); falling back to HF+PEFT.",
                UserWarning,
                stacklevel=2,
            )
    handler = HFModel(config)
    try:
        handler.load_model()
        return handler
    except Exception as exc:  # noqa: BLE001 - aggregated below
        errors["hf-peft"] = exc
        raise LoaderError(
            "No backend could load the model: "
            + "; ".join(f"{k}: {v}" for k, v in errors.items())
        ) from exc
