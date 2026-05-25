"""LR-RL Baseline 팩토리."""
from __future__ import annotations

from agents.lr_rl_baseline import LRRLBaselineController


def make_lr_rl_controller(
    alpha: float = 0.4,
    gamma: float = 0.75,
    eps_start: float = 1.0,
    eps_min: float = 0.05,
    eps_decay: float = 0.999,
) -> LRRLBaselineController:
    return LRRLBaselineController(
        alpha=alpha, gamma=gamma,
        eps_start=eps_start, eps_min=eps_min, eps_decay=eps_decay,
        with_idle=False,
    )


def make_lr_rl_idle_controller(
    alpha: float = 0.4,
    gamma: float = 0.75,
    eps_start: float = 1.0,
    eps_min: float = 0.05,
    eps_decay: float = 0.999,
) -> LRRLBaselineController:
    return LRRLBaselineController(
        alpha=alpha, gamma=gamma,
        eps_start=eps_start, eps_min=eps_min, eps_decay=eps_decay,
        with_idle=True,
    )
