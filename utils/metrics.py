"""지표 계산 보조 함수."""

from __future__ import annotations

import numpy as np


def jains_fairness(values) -> float:
    """Jain 공정성 지수를 계산한다."""
    arr = np.asarray(values, dtype=np.float64)
    total = float(arr.sum())
    denom = float(np.square(arr).sum())
    if total <= 0.0 or denom <= 0.0:
        return 0.0
    return (total * total) / (float(arr.size) * denom)
