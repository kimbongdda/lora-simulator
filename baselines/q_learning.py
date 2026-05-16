"""Q-learning 기준선 사양."""

from __future__ import annotations

from agents.q_learning import DEFAULT_ACTION_VARIANT, DEFAULT_STATE_VARIANT, DecentralizedQLearningController


def make_decentralized_q_learning_controller(
    alpha: float = 0.1,
    gamma_q: float = 0.9,
    epsilon: float = 1.0,
    eps_min: float = 0.05,
    eps_decay: float = 0.9995,
    reward_variant: str = "v0_current",
    state_variant: str = DEFAULT_STATE_VARIANT,
    action_variant: str = DEFAULT_ACTION_VARIANT,
):
    return DecentralizedQLearningController(
        alpha=alpha,
        gamma_q=gamma_q,
        epsilon=epsilon,
        eps_min=eps_min,
        eps_decay=eps_decay,
        reward_variant=reward_variant,
        state_variant=state_variant,
        action_variant=action_variant,
    )
