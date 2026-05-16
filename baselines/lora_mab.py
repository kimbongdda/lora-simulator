"""LoRa-MAB (EXP3) 기준선 팩토리."""

from __future__ import annotations

from agents.exp3_mab import Exp3MabController


def make_lora_mab_controller(
    eta: float = 0.1,
    with_idle: bool = False,
    fail_reward: float = 0.0,
) -> Exp3MabController:
    return Exp3MabController(eta=eta, with_idle=with_idle, fail_reward=fail_reward)
