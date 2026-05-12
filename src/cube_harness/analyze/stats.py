"""Shared statistics helpers for experiment reporting.

Kept dependency-free (only the stdlib ``math``) so it can be imported from
``scripts/`` shells and notebooks without pulling in numpy/pandas.
"""

from __future__ import annotations

import math


def binomial_std_err(mean: float, n: int) -> float:
    """Standard error of a binary outcome ``Bernoulli(p)`` given the sample mean.

    Returns ``sqrt(p * (1 - p) / n)``. Returns ``0.0`` when ``n <= 1``
    (the formula degenerates) or ``n == 0``.

    For non-binary data, use the sample-std-based SE in
    ``cube_harness.analyze.inspect_results.get_sample_std_err`` instead.
    """
    if n <= 1:
        return 0.0
    return math.sqrt(mean * (1 - mean) / n)
