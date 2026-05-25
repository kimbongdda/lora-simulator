"""EXP3 Multi-Armed Bandit 컨트롤러 (LoRa-MAB 논문 기반).

각 노드가 독립적인 EXP3 에이전트로 동작한다.
arm = (SF_idx, channel_idx) 쌍 + IDLE arm, 총 N_SF × n_channels + 1 개.

Reference:
    LoRa-MAB: Toward an Intelligent Resource Allocation Approach for LoRaWAN
    (IEEE GLOBECOM Workshops, 2019)
"""

from __future__ import annotations

import math

from env.types import ControllerAction, OUTCOME_SUCCESS, OUTCOME_IDLE

N_SF = 6


class Exp3MabController:
    """노드별 독립 EXP3 에이전트.

    Parameters
    ----------
    eta : float
        EXP3 학습률. 클수록 최근 보상에 빠르게 반응하지만 탐색이 줄어든다.
    with_idle : bool
        True면 IDLE arm을 추가한다 (arm index = N_SF * n_channels).
    """

    key = "lora_mab"
    label = "LoRa-MAB (EXP3)"

    def __init__(self, eta: float = 0.1, with_idle: bool = False, fail_reward: float = 0.0) -> None:
        self.eta = eta
        self.with_idle = with_idle
        self.fail_reward = fail_reward

    def begin_run(self, nodes, config, rng) -> None:
        self.config = config
        self.rng = rng
        self._n_tx_arms = N_SF * config.n_channels
        self._idle_arm = self._n_tx_arms if self.with_idle else None
        self._n_arms = self._n_tx_arms + (1 if self.with_idle else 0)

        # 노드 ID → weight 배열 (균등 초기화)
        self._weights: dict[int, list[float]] = {
            node.node_id: [1.0] * self._n_arms for node in nodes
        }
        # 직전 선택을 observe()에서 참조하기 위해 저장
        self._last_arm: dict[int, int] = {}
        self._last_prob: dict[int, float] = {}

    def _arm_to_sf_ch(self, arm: int) -> tuple[int, int]:
        return divmod(arm, self.config.n_channels)

    def _sample_arm(self, node_id: int) -> tuple[int, float]:
        """EXP3 확률 분포로 arm을 샘플하고 (arm_idx, prob)를 반환한다."""
        w = self._weights[node_id]
        total = sum(w)
        probs = [wi / total for wi in w]

        r = self.rng.random()
        cumulative = 0.0
        for arm, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return arm, p
        return len(probs) - 1, probs[-1]

    def choose_action(self, node, slot: int, gateway_info: dict | None = None) -> ControllerAction:
        arm, prob = self._sample_arm(node.node_id)
        self._last_arm[node.node_id] = arm
        self._last_prob[node.node_id] = prob

        if arm == self._idle_arm:
            return ControllerAction(
                transmit=False,
                sf_idx=node.sf_idx,
                channel_idx=node.channel_idx,
                prev_channel_idx=node.channel_idx,
                internal_action=arm,
            )

        sf_idx, ch_idx = self._arm_to_sf_ch(arm)
        return ControllerAction(
            transmit=True,
            sf_idx=sf_idx,
            channel_idx=ch_idx,
            prev_channel_idx=node.channel_idx,
            internal_action=arm,
        )

    def observe(self, node, action: ControllerAction, outcome: int, slot: int, **_) -> None:
        if action.internal_action is None:
            return

        reward = 1.0 if outcome == OUTCOME_SUCCESS else (0.0 if outcome == OUTCOME_IDLE else self.fail_reward)
        arm = self._last_arm.get(node.node_id)
        prob = self._last_prob.get(node.node_id, 1.0)
        if arm is None:
            return

        # EXP3 importance-weighted update
        r_hat = reward / max(prob, 1e-8)
        w = self._weights[node.node_id]
        exponent = min(self.eta * r_hat / self._n_arms, 700.0)  # exp(709) ≈ max float
        w[arm] *= math.exp(exponent)

        # 수치 안정: 최대값으로 정규화
        max_w = max(w)
        if max_w > 1e6:
            self._weights[node.node_id] = [wi / max_w for wi in w]

    def end_slot(self, slot: int) -> None:
        return None

    def get_mean_q_array(self):
        return None
