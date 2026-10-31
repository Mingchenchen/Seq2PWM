"""Safe, strict loading for Seq2PWM model weights."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch


WeightsMapping = Mapping[str, Any]


def _extract_state_dict(weights: WeightsMapping) -> Mapping[str, torch.Tensor]:
    """Return a raw state dict, accepting legacy training checkpoints too."""

    state = weights.get("model_state_dict", weights)
    if not isinstance(state, Mapping):
        raise TypeError("Weights file does not contain a model state dictionary.")
    if not state:
        raise ValueError("Model state dictionary is empty.")
    invalid = [
        key
        for key, value in state.items()
        if not isinstance(key, str) or not torch.is_tensor(value)
    ]
    if invalid:
        raise TypeError(
            "Model state dictionary must map parameter names to tensors; "
            f"invalid entries include {invalid[:3]!r}."
        )
    return state


def _strip_ddp_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = list(state)
    if keys and all(key.startswith("module.") for key in keys):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return dict(state)


def normalized_state_dict(weights: WeightsMapping) -> dict[str, torch.Tensor]:
    """Return parameters with only a universal DDP prefix removed."""

    return _strip_ddp_prefix(_extract_state_dict(weights))


def infer_model_family(
    weights: WeightsMapping,
) -> Literal["main", "ablation_none"]:
    """Identify the two full-model parameter namespaces used by public weights."""

    keys = set(normalized_state_dict(weights))
    main_markers = {"conv_norm.weight", "attn_norm.weight"}
    ablation_markers = {"feature_norm.weight", "context_norm.weight"}

    if main_markers <= keys and not (ablation_markers & keys):
        return "main"
    if ablation_markers <= keys and not (main_markers & keys):
        return "ablation_none"
    raise ValueError(
        "Weights architecture could not be identified. Expected either "
        "conv_norm/attn_norm (main) or feature_norm/context_norm "
        "(ablation-none) parameter keys."
    )


def read_weights(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> WeightsMapping:
    """Read a raw state dict without loading arbitrary Python classes."""

    weights = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(weights, Mapping):
        raise TypeError("Weights file must contain a mapping of parameter names to tensors.")
    _extract_state_dict(weights)
    return weights


def load_weights_strict(
    model: torch.nn.Module,
    weights_or_path: WeightsMapping | str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> WeightsMapping:
    """Load raw weights strictly, with legacy checkpoint compatibility.

    Public release files are raw ``model.state_dict()`` mappings. Legacy
    training checkpoints containing ``model_state_dict`` remain readable so
    existing experiments can be resumed or inspected. The only automatic
    migration is removal of a universal DistributedDataParallel ``module.``
    prefix; all other parameter names and tensor shapes must match exactly.
    """

    weights = (
        read_weights(weights_or_path, map_location=map_location)
        if isinstance(weights_or_path, (str, Path))
        else weights_or_path
    )
    state = normalized_state_dict(weights)
    model.load_state_dict(state, strict=True)
    return weights
