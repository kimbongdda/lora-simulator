"""Dual-MAB 기준선 팩토리."""

from __future__ import annotations

from agents.dual_mab import DualMabController


def make_dual_mab_controller(
    b: float = 0.3,
    eps_resource: float = 0.1,
    eps_backoff: float = 0.1,
    alpha: float = 0.1,
    lambda_tx: float = 1.0,
    lambda_col: float = 1.0,
    lambda_snr: float = 0.3,
) -> DualMabController:
    return DualMabController(
        b=b,
        eps_resource=eps_resource,
        eps_backoff=eps_backoff,
        alpha=alpha,
        lambda_tx=lambda_tx,
        lambda_col=lambda_col,
        lambda_snr=lambda_snr,
    )
