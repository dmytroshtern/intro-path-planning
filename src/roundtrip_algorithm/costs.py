"""
Cost functions for roundtrip path planning.

"""

from __future__ import annotations

from typing import Sequence
import math

import numpy as np


def config_distance(q_a: Sequence[float], q_b: Sequence[float], metric: str = "euclidean") -> float:
    a = np.asarray(q_a, dtype=float)
    b = np.asarray(q_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(f"Configurations must have same shape, got {a.shape} and {b.shape}.")

    if metric == "euclidean":
        return float(np.linalg.norm(b - a))

    if metric == "manhattan":
        return float(np.sum(np.abs(b - a)))

    raise ValueError(f"Unknown metric: {metric}")


def path_length(path_configs: Sequence[Sequence[float]], metric: str = "euclidean") -> float:
    if path_configs is None:
        return math.inf

    if len(path_configs) < 2:
        return 0.0

    total = 0.0

    for q_a, q_b in zip(path_configs, path_configs[1:]):
        total += config_distance(q_a, q_b, metric=metric)

    return float(total)


def count_path_points(path_configs: Sequence[Sequence[float]] | None) -> int:
    if path_configs is None:
        return 0

    return len(path_configs)