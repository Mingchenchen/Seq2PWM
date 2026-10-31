"""Small configuration helpers shared by the training entry points."""

from __future__ import annotations

import string
from typing import Any


def resolve_placeholders(value: Any, **variables: object) -> Any:
    """Recursively substitute named placeholders in YAML string values."""
    if isinstance(value, dict):
        return {
            key: resolve_placeholders(item, **variables)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_placeholders(item, **variables) for item in value]
    if isinstance(value, str):
        fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(value)
            if field_name
        }
        unavailable = [name for name in fields if variables.get(name) is None]
        if unavailable:
            raise ValueError(
                f"Configuration value {value!r} requires --{unavailable[0]}."
            )
        try:
            return value.format(**variables)
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(
                f"Configuration value {value!r} requires --{missing}."
            ) from exc
    return value
