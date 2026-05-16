"""Thompson Sampling MAB 기준선 팩토리."""

from __future__ import annotations

from agents.thompson_mab import ThompsonMabController


def make_thompson_mab_controller(
    with_idle: bool = False,
    alpha_init: float = 1.0,
    beta_init: float = 1.0,
) -> ThompsonMabController:
    return ThompsonMabController(with_idle=with_idle, alpha_init=alpha_init, beta_init=beta_init)
