"""노드 배치 보조 함수."""

from __future__ import annotations

import math

import numpy as np


def place_nodes_on_ring(n_nodes: int, radius: float, seed: int | None = None):
    """게이트웨이 주변의 같은 원 위에 노드를 배치한다."""
    rng = np.random.RandomState(seed)
    # 각도만 무작위로 바꾸고 반경은 고정해, 모든 노드의 평균 경로손실이 같게 만든다.
    angles = rng.uniform(0.0, 2.0 * math.pi, n_nodes)
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    dists = np.full(n_nodes, float(radius), dtype=np.float64)
    return xs, ys, dists


def place_nodes_random(n_nodes: int, cell_radius: float, seed: int | None = None):
    """셀 원판 내부에 노드를 균일 무작위로 배치한다."""
    rng = np.random.RandomState(seed)
    # 원판 내부 균일 샘플링: r = R * sqrt(u), theta = 2pi * v
    u = rng.uniform(0.0, 1.0, n_nodes)
    v = rng.uniform(0.0, 1.0, n_nodes)
    r = float(cell_radius) * np.sqrt(u)
    theta = 2.0 * math.pi * v
    xs = r * np.cos(theta)
    ys = r * np.sin(theta)
    dists = r
    return xs, ys, dists
