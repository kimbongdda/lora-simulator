"""Dual-MAB Distributed LoRa Random Access Controller.

두 개의 독립 MAB 에이전트를 조합한다.
- Resource-MAB  : 어떤 (SF, channel)로 전송할지 학습 (ε-greedy + EMA)
- Backoff-MAB   : ACB 차단 시 얼마나 기다릴지 학습 (ε-greedy + EMA, differential reward)

노드별 완전 분산 동작, 게이트웨이 협조 없음.
"""

from __future__ import annotations

from env.types import (
    ControllerAction,
    OUTCOME_IDLE,
    OUTCOME_SUCCESS,
    OUTCOME_FAIL_COLLISION,
)

N_SF = 6
_W_MAX_ARMS = (1, 2, 4, 8, 16)


class DualMabController:
    """Resource-MAB + Backoff-MAB 이중 에이전트.

    Parameters
    ----------
    b : float
        ACB 차단 확률. 0 이면 항상 전송 허가 (Backoff-MAB 비활성).
    eps_resource : float
        Resource-MAB ε-greedy 탐색률.
    eps_backoff : float
        Backoff-MAB ε-greedy 탐색률.
    alpha : float
        EMA 학습률 (양쪽 MAB 공용).
    lambda_tx : float
        전송 성공 시 Backoff-MAB 보상.
    lambda_col : float
        충돌 실패 시 Backoff-MAB 페널티.
    lambda_snr : float
        링크 실패 시 Backoff-MAB 페널티 (충돌보다 약하게).
    """

    key = "dual_mab"
    label = "Dual-MAB (Resource + Backoff)"

    def __init__(
        self,
        b: float = 0.3,
        eps_resource: float = 0.1,
        eps_backoff: float = 0.1,
        alpha: float = 0.1,
        lambda_tx: float = 1.0,
        lambda_col: float = 1.0,
        lambda_snr: float = 0.3,
    ) -> None:
        self.b = b
        self.eps_resource = eps_resource
        self.eps_backoff = eps_backoff
        self.alpha = alpha
        self.lambda_tx = lambda_tx
        self.lambda_col = lambda_col
        self.lambda_snr = lambda_snr

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_run(self, nodes, config, rng) -> None:
        self.config = config
        self.rng = rng
        self._n_res_arms = N_SF * config.n_channels
        self._n_bk_arms = len(_W_MAX_ARMS)

        # Resource-MAB Q값: 0.5로 낙관적 초기화
        self._res_q: dict[int, list[float]] = {
            n.node_id: [0.5] * self._n_res_arms for n in nodes
        }
        # Backoff-MAB Q값: 0으로 초기화 (차동 보상 범위가 음수 포함)
        self._bk_q: dict[int, list[float]] = {
            n.node_id: [0.0] * self._n_bk_arms for n in nodes
        }
        self._last_res_arm: dict[int, int] = {}
        self._last_bk_arm: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def _eps_greedy(self, q: list[float], eps: float) -> int:
        if self.rng.random() < eps:
            return self.rng.randrange(len(q))
        return max(range(len(q)), key=lambda i: q[i])

    def choose_action(self, node, slot: int, gateway_info: dict | None = None) -> ControllerAction:
        u = self.rng.random()

        if u < self.b:
            # ACB 차단 → Backoff-MAB 로 W_max 결정 후 대기
            arm = self._eps_greedy(self._bk_q[node.node_id], self.eps_backoff)
            self._last_bk_arm[node.node_id] = arm
            W_max = _W_MAX_ARMS[arm]
            node.backoff_slots = self.rng.randint(0, W_max)
            return ControllerAction(
                transmit=False,
                sf_idx=node.sf_idx,
                channel_idx=node.channel_idx,
                prev_channel_idx=node.channel_idx,
            )

        # 전송 허가 → Resource-MAB 로 (SF, channel) 결정
        arm = self._eps_greedy(self._res_q[node.node_id], self.eps_resource)
        self._last_res_arm[node.node_id] = arm
        sf_idx, ch_idx = divmod(arm, self.config.n_channels)
        return ControllerAction(
            transmit=True,
            sf_idx=sf_idx,
            channel_idx=ch_idx,
            prev_channel_idx=node.channel_idx,
            internal_action=arm,
        )

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def observe(self, node, action: ControllerAction, outcome: int, slot: int, **_) -> None:
        if outcome == OUTCOME_IDLE:
            return  # ACB 차단 슬롯 — 학습 대상 없음

        node_id = node.node_id
        alpha = self.alpha

        # Resource-MAB: EMA 업데이트
        res_arm = self._last_res_arm.get(node_id)
        if res_arm is not None:
            reward = 1.0 if outcome == OUTCOME_SUCCESS else 0.0
            q = self._res_q[node_id]
            q[res_arm] = (1.0 - alpha) * q[res_arm] + alpha * reward

        # Backoff-MAB: 차동 보상으로 EMA 업데이트
        bk_arm = self._last_bk_arm.get(node_id)
        if bk_arm is not None:
            if outcome == OUTCOME_SUCCESS:
                bk_reward = self.lambda_tx
            elif outcome == OUTCOME_FAIL_COLLISION:
                bk_reward = -self.lambda_col   # 혼잡 → 백오프가 부족했음
            else:                               # link failure
                bk_reward = -self.lambda_snr   # SNR 문제 → 백오프와 무관
            q = self._bk_q[node_id]
            q[bk_arm] = (1.0 - alpha) * q[bk_arm] + alpha * bk_reward

    def end_slot(self, slot: int) -> None:
        return None

    def get_mean_q_array(self):
        return None
