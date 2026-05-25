"""Thompson Sampling Multi-Armed Bandit 컨트롤러.

Bernoulli 보상 구조에서 EXP3보다 sparse reward에 강하다.
각 arm마다 Beta(alpha, beta) 사후 분포를 유지하고,
샘플링으로 탐색/이용 균형을 자연스럽게 달성한다.

arm = (SF_idx, channel_idx) 쌍 + 선택적 IDLE arm.
"""

from __future__ import annotations

from env.types import ControllerAction, OUTCOME_SUCCESS, OUTCOME_IDLE

N_SF = 6


class ThompsonMabController:
    """노드별 독립 Thompson Sampling 에이전트.

    Parameters
    ----------
    with_idle : bool
        True면 IDLE arm을 추가한다 (arm index = N_SF * n_channels).
    alpha_init, beta_init : float
        Beta 사전 분포 초기값. 기본 (1, 1) = 균등 사전분포.
    """

    key = "thompson_mab"
    label = "LoRa-MAB (Thompson Sampling)"

    def __init__(
        self,
        with_idle: bool = False,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
    ) -> None:
        self.with_idle = with_idle
        self.alpha_init = alpha_init
        self.beta_init = beta_init

    def begin_run(self, nodes, config, rng) -> None:
        self.config = config
        self.rng = rng
        self._n_tx_arms = N_SF * config.n_channels
        self._idle_arm = self._n_tx_arms if self.with_idle else None
        self._n_arms = self._n_tx_arms + (1 if self.with_idle else 0)

        # 노드 ID → (alpha 배열, beta 배열)
        self._alpha: dict[int, list[float]] = {
            node.node_id: [self.alpha_init] * self._n_arms for node in nodes
        }
        self._beta: dict[int, list[float]] = {
            node.node_id: [self.beta_init] * self._n_arms for node in nodes
        }
        self._last_arm: dict[int, int] = {}

    def _arm_to_sf_ch(self, arm: int) -> tuple[int, int]:
        return divmod(arm, self.config.n_channels)

    def _sample_arm(self, node_id: int) -> int:
        """각 arm의 Beta 분포에서 샘플하고 최대값 arm을 반환한다."""
        alpha = self._alpha[node_id]
        beta = self._beta[node_id]
        best_arm, best_sample = 0, -1.0
        for arm in range(self._n_arms):
            # rng.betavariate는 Python 표준 random 모듈에 있음
            sample = self.rng.betavariate(alpha[arm], beta[arm])
            if sample > best_sample:
                best_sample = sample
                best_arm = arm
        return best_arm

    def choose_action(self, node, slot: int, gateway_info: dict | None = None) -> ControllerAction:
        arm = self._sample_arm(node.node_id)
        self._last_arm[node.node_id] = arm

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
        if action.internal_action is None or outcome == OUTCOME_IDLE:
            return
        arm = self._last_arm.get(node.node_id)
        if arm is None:
            return

        if outcome == OUTCOME_SUCCESS:
            self._alpha[node.node_id][arm] += 1.0
        else:
            self._beta[node.node_id][arm] += 1.0

    def end_slot(self, slot: int) -> None:
        return None

    def get_mean_q_array(self):
        return None
