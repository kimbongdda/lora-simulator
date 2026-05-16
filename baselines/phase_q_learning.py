"""Phase Q-learning 기준선 사양."""

from __future__ import annotations

from agents.phase_q_learning import PhaseQLearningController
from agents.q_learning import DEFAULT_REWARD_VARIANT


def make_phase_q_learning_controller(
    alpha: float = 0.1,
    gamma_q: float = 0.9,
    epsilon: float = 1.0,
    eps_min: float = 0.05,
    eps_decay: float = 0.9995,
    frame_size: int = 18,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
) -> PhaseQLearningController:
    return PhaseQLearningController(
        alpha=alpha,
        gamma_q=gamma_q,
        epsilon=epsilon,
        eps_min=eps_min,
        eps_decay=eps_decay,
        frame_size=frame_size,
        reward_variant=reward_variant,
    )
