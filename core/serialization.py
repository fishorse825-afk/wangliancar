"""Recursive detached JSON views; preserve new contract fields automatically."""

import math


def to_dict(value):
    if hasattr(value, "__dict__"):
        return to_dict(vars(value))
    if isinstance(value, dict):
        return {str(k): to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def perception_to_dict(value):
    return to_dict(value)
