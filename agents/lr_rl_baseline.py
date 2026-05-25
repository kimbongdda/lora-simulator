"""LR-RL Baseline Controller — Hong et al. (2023) (SF, 채널) joint 확장.

원본 논문: Hong et al., "Reinforcement Learning Approach for SF Allocation
in LoRa Network", IEEE IoT Journal, 2023.

핵심 특징
----------
- Q-table 대신 **V-table**: (SF, 채널) 조합별 가치값. 상태 개념 없음 (stateless).
- 크기: N_SF × N_ch = 6 × 3 = 18 엔트리 (노드별 독립).
- 패킷 도착 시 **반드시 전송**. IDLE 액션 없음 (원본).
- 보상: PCR(충돌률)과 패킷 비율(P_pro_i)로 정규화 (원본 수식 17 확장).
- V-table 업데이트: TD(0) — (1-α)*V + α*(R + γ*max(V)).

with_idle=True 변형
-------------------
- V-table에 IDLE arm 1개 추가 → 크기 N_SF*N_ch + 1 = 19 엔트리.
- IDLE 선택 시 transmit=False; V_idle 업데이트: R=0, γ*max(전송 arms 만).
- 전송 arm이 모두 음수로 수렴할 때 IDLE이 자연스럽게 선택됨.

하이퍼파라미터 기본값
---------------------
α=0.4, γ=0.75, ε₀=1.0, ε_decay=0.999, ε_min=0.05 (원본 논문 기재값)
"""
from __future__ import annotations

import numpy as np

from env.types import (
    ControllerAction,
    OUTCOME_SUCCESS,
    OUTCOME_FAIL_COLLISION,
    OUTCOME_FAIL_LINK,
    OUTCOME_IDLE,
)

_EPS_CLIP = 1e-6  # 0 나눗셈 방지 최소값


class LRRLBaselineController:
    """Hong et al. (2023) LR-RL — (SF, 채널) joint 선택 확장."""

    SUPPORTS_BATCH = False

    def __init__(
        self,
        alpha: float = 0.4,
        gamma: float = 0.75,
        eps_start: float = 1.0,
        eps_min: float = 0.05,
        eps_decay: float = 0.999,
        with_idle: bool = False,
    ) -> None:
        self.alpha     = alpha
        self.gamma     = gamma
        self.eps_start = eps_start
        self.eps_min   = eps_min
        self.eps_decay = eps_decay
        self.with_idle = with_idle

        if with_idle:
            self.key   = "lr_rl_idle"
            self.label = "LR-RL+IDLE (V-table)"
        else:
            self.key   = "lr_rl"
            self.label = "LR-RL (V-table)"

    # ── 시뮬레이터 인터페이스 ──────────────────────────────────────────────

    def begin_run(self, nodes, config, rng) -> None:
        N   = len(nodes)
        nch = config.n_channels
        nsf = 6  # SF7~SF12

        self._N    = N
        self._nch  = nch
        self._nsf  = nsf
        self._rng  = rng
        self._n_tx = nsf * nch  # 전송 arm 수

        # IDLE arm은 전송 arm 뒤에 붙음 (인덱스 n_tx)
        self._idle_arm = self._n_tx if self.with_idle else None
        n_arms = self._n_tx + (1 if self.with_idle else 0)

        # 노드별 V-table: shape (N, n_arms), 초기값 0
        self._V: np.ndarray = np.zeros((N, n_arms), dtype=np.float64)

        # 노드별 탐색률 (독립 감쇠)
        self._eps: list[float] = [self.eps_start] * N

        # 노드별 로컬 통계 (원본 수식 17)
        self._N_cur: list[int]       = [0] * N   # 총 전송 횟수
        self._n_col: list[int]       = [0] * N   # 총 충돌 횟수
        self._n_i: list[np.ndarray]  = [
            np.zeros(self._n_tx, dtype=np.int64) for _ in range(N)
        ]

    def choose_action(self, node, slot: int, gateway_info: dict) -> ControllerAction:
        nid = node.node_id

        if self._rng.random() < self._eps[nid]:
            arm = self._rng.randint(0, len(self._V[nid]) - 1)
        else:
            arm = int(np.argmax(self._V[nid]))

        # IDLE arm 선택
        if self.with_idle and arm == self._idle_arm:
            return ControllerAction(
                transmit=False,
                sf_idx=node.sf_idx,
                channel_idx=node.channel_idx,
                prev_channel_idx=node.channel_idx,
                internal_action=arm,
                state=None,
            )

        # 전송 arm: arm 인덱스 = sf * nch + ch
        sf, ch = divmod(arm, self._nch)
        return ControllerAction(
            transmit=True,
            sf_idx=sf,
            channel_idx=ch,
            prev_channel_idx=node.channel_idx,
            internal_action=arm,
            state=None,
        )

    def observe(self, node, action: ControllerAction, outcome: int, slot: int, gateway_info=None) -> None:
        nid = node.node_id
        arm = action.internal_action

        # IDLE arm 업데이트 (V_idle = (1-α)*V_idle + α*(0 + γ*max(전송 arms)))
        if self.with_idle and outcome == OUTCOME_IDLE:
            if arm == self._idle_arm:
                v_tx_max = float(np.max(self._V[nid, : self._n_tx]))
                old = self._V[nid, self._idle_arm]
                self._V[nid, self._idle_arm] = (
                    (1.0 - self.alpha) * old + self.alpha * self.gamma * v_tx_max
                )
            return

        if outcome == OUTCOME_IDLE:
            return

        sf  = action.sf_idx
        ch  = action.channel_idx
        tx_arm = sf * self._nch + ch  # 전송 arm 인덱스 (n_i 인덱싱용)

        # 로컬 통계 업데이트
        self._N_cur[nid] += 1
        self._n_i[nid][tx_arm] += 1

        is_fail = outcome in (OUTCOME_FAIL_COLLISION, OUTCOME_FAIL_LINK)
        if is_fail:
            self._n_col[nid] += 1

        N_cur  = self._N_cur[nid]
        P_pcr  = self._n_col[nid] / N_cur
        P_pro  = self._n_i[nid][tx_arm] / N_cur

        # 원본 수식 17 — 0 나눗셈 방지 클리핑
        if not is_fail:
            R = 1.0 / (max(1.0 - P_pcr, _EPS_CLIP) * max(P_pro, _EPS_CLIP))
        else:
            R = -1.0 / (max(P_pcr, _EPS_CLIP) * max(P_pro, _EPS_CLIP))

        # TD(0) V-table 업데이트 (전송 arm만 사용해 max 계산)
        v_max = float(np.max(self._V[nid, : self._n_tx]))
        old   = self._V[nid, tx_arm]
        self._V[nid, tx_arm] = (1.0 - self.alpha) * old + self.alpha * (R + self.gamma * v_max)

    def end_slot(self, slot: int) -> None:
        for i in range(self._N):
            self._eps[i] = max(self.eps_min, self._eps[i] * self.eps_decay)


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
