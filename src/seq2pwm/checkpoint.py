"""Backward-compatible aliases for the pre-v0.1 checkpoint API.

New code should import from :mod:`seq2pwm.weights` and use ``read_weights`` /
``load_weights_strict``. This shim can be removed in a future major release.
"""

from .weights import (  # noqa: F401
    infer_model_family,
    load_weights_strict,
    normalized_state_dict,
    read_weights,
)


read_checkpoint = read_weights
load_checkpoint_strict = load_weights_strict


__all__ = [
    "infer_model_family",
    "load_checkpoint_strict",
    "load_weights_strict",
    "normalized_state_dict",
    "read_checkpoint",
    "read_weights",
]
