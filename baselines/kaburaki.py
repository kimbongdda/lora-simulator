"""Kaburaki et al. (2021) 타이밍 오프셋 Q-learning 베이스라인.

원 논문: Kaburaki et al., "Q-Learning-based Timing Offset Control for LoRa Networks", 2021.

알고리즘 개요
-------------
각 노드 n은 J+1개의 오프셋 후보 집합 D_n = {0, D_n1, ..., D_nJ} 를 보유한다.
  - D_n[0] = 0 (즉시 전송)
  - D_n[1..J] ~ U[1, D_max] 균등 분포로 초기화

SF 와 채널은 begin_run() 시 한 번 랜덤으로 배정되고 이후 변경하지 않는다.

상태: s_n = j ∈ {0, 1, ..., J}  (현재 오프셋 후보 인덱스)
액션: a ∈ {0, +1, -1}           (오프셋 인덱스 변화)

패킷이 있는 노드는 오프셋 j* 를 ε-greedy 로 선택하고
node.backoff_slots = D_n[j*] 를 설정한다.
backoff 가 만료되면 고정 SF/채널로 전송한다.

보상: r = +1 (SUCCESS), -1 (FAIL)

ε 감쇠: ε_t = max(ε_min, 1 − t / Z)   (Z = 총 슬롯 수)

기존 시스템 적용 시 조정 사항
------------------------------
- 실패 시 재전송: 원 논문은 1회 전송 모델이지만, 기존 시뮬레이터는
  accumulate 큐를 유지한다. 여기서는 전송 후(성공/실패 무관)
  pending 플래그를 리셋해 다음 슬롯에 새 오프셋을 선택하게 한다.
  이는 "실패 시 새 오프셋으로 재시도" 로 해석되며 accumulate 모드에 자연스럽다.
- fresh 모드: backoff 만료 슬롯에 패킷이 없으면 전송이 발생하지 않는다.
  accumulate 모드를 권장한다.
"""
from __future__ import annotations

import random as _random

import numpy as np

from env.types import (
    ControllerAction,
    OUTCOME_SUCCESS,
    OUTCOME_IDLE,
)
from env.channel import clamp_sf_index

# 액션 인덱스 → 오프셋 인덱스 변화량
_ACTION_DELTA = [0, +1, -1]   # a=0: 유지, a=1: +1, a=2: -1


class KaburakiController:
    """타이밍 오프셋 Q-learning (Kaburaki et al., 2021).

    Parameters
    ----------
    J : int
        비-0 오프셋 후보 수. 후보 집합 크기 = J+1 (0 포함).
    D_max : int
        오프셋 최대값 (슬롯 수).
    alpha : float
        Q-learning 학습률.
    gamma : float
        할인율.
    eps_min : float
        최소 탐색률 ε_min.
    """

    key   = "kaburaki"
    label = "Kaburaki (offset Q-lrn)"
    SUPPORTS_BATCH = False

    def __init__(
        self,
        J: int = 3,
        D_max: int = 64,
        alpha: float = 0.3,
        gamma: float = 0.95,
        eps_min: float = 0.05,
    ) -> None:
        self.J = J
        self.D_max = D_max
        self.alpha = alpha
        self.gamma = gamma
        self.eps_min = eps_min

    # ── 시뮬레이터 인터페이스 ───────────────────────────────────────────────

    def begin_run(self, nodes, config, rng) -> None:
        N = len(nodes)
        self._N = N
        self._n_slots = config.n_slots

        # 오프셋 후보 집합: D[n][0]=0, D[n][1..J] ~ U[1, D_max]
        self._D: list[list[int]] = []
        for _ in range(N):
            cands = [0] + [rng.randint(1, max(1, self.D_max)) for _ in range(self.J)]
            self._D.append(cands)

        # Q 테이블: Q[n][j][a]  j ∈ {0..J}, a ∈ {0,1,2}
        J1 = self.J + 1
        self._Q: list[list[list[float]]] = [
            [[0.0] * 3 for _ in range(J1)] for _ in range(N)
        ]

        # 현재 오프셋 인덱스 (상태)
        self._j: list[int] = [0] * N

        # 이번 전송 결정에 쓴 (이전 j, action_idx) 기록
        self._prev_j:   list[int] = [0] * N
        self._prev_a:   list[int] = [0] * N

        # 오프셋을 이미 설정했는지 (True = backoff 대기 중 또는 전송 직전)
        self._pending: list[bool] = [False] * N

        # SF / 채널 고정
        self._fixed_sf: list[int] = [node.sf_idx for node in nodes]
        self._fixed_ch: list[int] = [node.channel_idx for node in nodes]

        # SF를 node에도 반영 (혹시 begin_run 이후 변경 방지)
        for node in nodes:
            node.sf_idx = self._fixed_sf[node.node_id]
            node.channel_idx = self._fixed_ch[node.node_id]

    def choose_action(self, node, slot: int, gateway_info: dict) -> ControllerAction:
        nid = node.node_id
        sf  = self._fixed_sf[nid]
        ch  = self._fixed_ch[nid]

        if self._pending[nid]:
            # backoff 만료 → 전송
            return ControllerAction(
                transmit=True,
                sf_idx=sf,
                channel_idx=ch,
                internal_action=self._prev_a[nid],
                state=(self._prev_j[nid],),
            )

        # 새 오프셋 선택 (ε-greedy)
        j = self._j[nid]
        eps = max(self.eps_min, 1.0 - slot / max(self._n_slots, 1))

        if _random.random() < eps:
            a_idx = _random.randint(0, 2)
        else:
            a_idx = int(np.argmax(self._Q[nid][j]))

        # 오프셋 인덱스 업데이트 (클램프)
        new_j = max(0, min(self.J, j + _ACTION_DELTA[a_idx]))
        D = self._D[nid][new_j]

        # 이전 상태·액션 기록 (observe 에서 Q 업데이트 시 사용)
        self._prev_j[nid] = j
        self._prev_a[nid] = a_idx
        self._j[nid]      = new_j
        self._pending[nid] = True

        if D > 0:
            node.backoff_slots = D
            return ControllerAction(
                transmit=False,
                sf_idx=sf,
                channel_idx=ch,
                internal_action=a_idx,
                state=(j,),
            )
        else:
            # D=0: 즉시 전송 (backoff 불필요)
            return ControllerAction(
                transmit=True,
                sf_idx=sf,
                channel_idx=ch,
                internal_action=a_idx,
                state=(j,),
            )

    def observe(self, node, action, outcome: int, slot: int, gateway_info=None) -> None:
        if outcome == OUTCOME_IDLE:
            return

        nid = node.node_id
        self._pending[nid] = False   # 다음 패킷을 위해 리셋

        reward = 1.0 if outcome == OUTCOME_SUCCESS else -1.0

        j       = self._prev_j[nid]
        a_idx   = self._prev_a[nid]
        next_j  = self._j[nid]

        q_old  = self._Q[nid][j][a_idx]
        q_next = max(self._Q[nid][next_j])
        self._Q[nid][j][a_idx] = q_old + self.alpha * (
            reward + self.gamma * q_next - q_old
        )

    def end_slot(self, slot: int) -> None:
        pass


def make_kaburaki_controller(
    J: int = 3,
    D_max: int = 64,
    alpha: float = 0.3,
    gamma: float = 0.95,
    eps_min: float = 0.05,
) -> KaburakiController:
    return KaburakiController(J=J, D_max=D_max, alpha=alpha, gamma=gamma, eps_min=eps_min)
