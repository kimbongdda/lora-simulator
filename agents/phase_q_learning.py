"""슬롯 페이즈(phase) 학습 Q-learning 컨트롤러.

상태: (phase_bin, sf_idx)  — phase * N_SF + sf_idx (flat index)
액션: 기존 relative SF±1×CH  +  wait_k (N슬롯 대기)

    0            : IDLE (1슬롯 패스, 보상=0)
    1 ~ 3*n_ch   : 전송 (SF±1 × channel, 기존과 동일)
    3*n_ch+1     : wait_short  (frame_size // 4 슬롯 대기, 보상=0)
    3*n_ch+2     : wait_medium (frame_size // 2 슬롯 대기, 보상=0)
    3*n_ch+3     : wait_long   (frame_size - 1 슬롯 대기, 보상=0)

wait_k 액션을 선택하면 backoff_arr 에 대기 슬롯 수를 설정하여
다음 preferred phase가 올 때까지 시뮬레이터 레벨에서 잠든다.
대기 보상은 0(중립) — 패킷이 쌓여 있어도 전략적 대기에 패널티 없음.

frame_size <= 3 이면 wait 액션 불필요 → 추가하지 않음 (IDLE만 사용).

시뮬레이터와의 인터페이스:
    choose_actions_batch → 6-tuple (... , backoff_req) 반환
    simulator.py 배치 경로가 len==6 일 때 backoff_arr 에 반영.
    기존 5-tuple 컨트롤러는 영향 없음.
"""

from __future__ import annotations

import numpy as np

from agents.q_learning import (
    ACTION_IDLE,
    DEFAULT_REWARD_VARIANT,
    N_SF,
    N_SF_CHANGES,
    REWARD_VARIANTS,
    decode_action,
    clamp_sf_index,
    compute_reward,
)
from env.types import (
    ControllerAction,
    OUTCOME_FAIL_COLLISION,
    OUTCOME_FAIL_LINK,
    OUTCOME_IDLE,
    OUTCOME_SUCCESS,
)

_NO_WAIT = object()   # wait 액션 없음을 나타내는 sentinel


class PhaseQLearningController:
    """슬롯 페이즈를 상태에 포함하고 멀티슬롯 대기 액션을 갖는 Q-learning 컨트롤러.

    state  = phase * N_SF + sf_idx
    action = IDLE | transmit(SF±1×CH) | wait_short | wait_medium | wait_long
    """

    key = "phase_q_learning"
    label = "Q-learning (phase)"

    def __init__(
        self,
        alpha: float = 0.1,
        gamma_q: float = 0.9,
        epsilon: float = 1.0,
        eps_min: float = 0.05,
        eps_decay: float = 0.9995,
        frame_size: int = 18,
        reward_variant: str = DEFAULT_REWARD_VARIANT,
    ):
        self.alpha = alpha
        self.gamma_q = gamma_q
        self.initial_epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.frame_size = max(1, int(frame_size))
        self.reward_variant = reward_variant

    @property
    def SUPPORTS_BATCH(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # 내부: 액션 공간 & wait 설계
    # ------------------------------------------------------------------

    def _build_action_space(self, n_channels: int) -> None:
        """액션 수와 wait 지속 시간을 결정한다."""
        self._n_tx_actions = 1 + N_SF_CHANGES * n_channels   # IDLE + transmit
        fs = self.frame_size

        if fs > 3:
            # frame_size 기반으로 wait 지속 시간 계산 (최소 1)
            self._wait_durations = [
                max(1, fs // 4),           # wait_short
                max(2, fs // 2),           # wait_medium
                max(3, fs - 1),            # wait_long (거의 한 프레임)
            ]
            self._wait_action_start = self._n_tx_actions   # wait 액션 첫 인덱스
            self._n_actions = self._n_tx_actions + len(self._wait_durations)
        else:
            # 짧은 프레임은 wait 의미 없음 — tx 액션만
            self._wait_durations = []
            self._wait_action_start = self._n_tx_actions   # sentinel (사용 안 됨)
            self._n_actions = self._n_tx_actions

    def _is_wait(self, action_idx: int) -> bool:
        return bool(self._wait_durations) and action_idx >= self._wait_action_start

    def _wait_duration(self, action_idx: int) -> int:
        offset = action_idx - self._wait_action_start
        return self._wait_durations[offset]

    # ------------------------------------------------------------------
    # 상태 인코딩
    # ------------------------------------------------------------------

    def _state(self, sf_arr: np.ndarray, slot: int) -> np.ndarray:
        phase = int(slot) % self.frame_size
        return (phase * N_SF + sf_arr).astype(np.int32)

    # ------------------------------------------------------------------
    # 생명주기
    # ------------------------------------------------------------------

    def begin_run(self, nodes, config, rng) -> None:
        N = len(nodes)
        self._n_channels = config.n_channels
        self._build_action_space(config.n_channels)
        self._n_states = self.frame_size * N_SF
        self._epoch_slots = config.epoch_slots
        self._rng = rng

        # dense Q-table: (N_nodes, n_states, n_actions)
        self._q = np.zeros((N, self._n_states, self._n_actions), dtype=np.float64)

        # 배치 경로용
        self._epsilon_arr = np.full(N, self.initial_epsilon, dtype=np.float64)
        self._success_count_arr = np.zeros(N, dtype=np.int64)
        self._last_slot = 0
        np_seed = rng.randint(0, 2 ** 31)
        self._np_rng = np.random.RandomState(np_seed)

        # 표준 경로용
        self._node_ids = [n.node_id for n in nodes]
        self._node_epsilons: dict[int, float] = {n.node_id: self.initial_epsilon for n in nodes}
        self._std_success_count: dict[int, int] = {n.node_id: 0 for n in nodes}
        self._gateway_g_bins: tuple = tuple(1 for _ in range(self._n_channels))

    def end_slot(self, slot: int) -> None:
        np.maximum(self.eps_min, self._epsilon_arr * self.eps_decay, out=self._epsilon_arr)
        r_cfg = REWARD_VARIANTS.get(self.reward_variant, {})
        rtype = r_cfg.get("type", "standard")
        if rtype in ("fairness", "log_thr") and (slot + 1) % self._epoch_slots == 0:
            self._success_count_arr[:] = 0

    def end_run(self) -> None:
        pass

    # ------------------------------------------------------------------
    # 보상 계산 헬퍼
    # ------------------------------------------------------------------

    def _reward_for_outcome(
        self,
        outcome: int,
        is_wait_or_idle: bool,
        node_cnt: int,
        mean_cnt: float,
        has_packet: bool,
    ) -> float:
        """대기/IDLE은 항상 0, 전송 결과는 reward variant에 따라 계산."""
        if is_wait_or_idle:
            return 0.0
        r_cfg = REWARD_VARIANTS.get(self.reward_variant, REWARD_VARIANTS[DEFAULT_REWARD_VARIANT])
        rtype = r_cfg.get("type", "standard")
        if outcome == OUTCOME_SUCCESS:
            if rtype in ("fairness", "log_thr"):
                mean_n = max(1.0, mean_cnt)
                norm = node_cnt / mean_n
                if rtype == "fairness":
                    return 1.0 / (1.0 + norm)
                else:
                    import math
                    return math.log(norm + 2.0) - math.log(norm + 1.0)
            if rtype == "composite":
                return float(r_cfg.get("success_base", 1.0))
            return float(r_cfg["success"])
        if outcome in (OUTCOME_FAIL_COLLISION, OUTCOME_FAIL_LINK):
            return float(r_cfg["fail"])
        # OUTCOME_IDLE + has_packet (전송 의도 없이 패스한 경우)
        return float(r_cfg["idle_pkt"]) if has_packet else float(r_cfg["idle_no_pkt"])

    # ------------------------------------------------------------------
    # 표준 경로
    # ------------------------------------------------------------------

    def choose_action(self, node, slot: int, gateway_info: dict | None = None) -> ControllerAction:
        if gateway_info is not None:
            self._gateway_g_bins = tuple(gateway_info.get("per_channel_g_bin", self._gateway_g_bins))

        sf = np.array([node.sf_idx], dtype=np.int32)
        state_idx = int(self._state(sf, slot)[0])
        eps = self._node_epsilons.get(node.node_id, self.initial_epsilon)

        if self._rng.random() < eps:
            action = self._rng.randint(0, self._n_actions - 1)
        else:
            action = int(np.argmax(self._q[node.node_id, state_idx]))

        if self._is_wait(action):
            backoff = self._wait_duration(action)
            transmit = False
            next_sf = node.sf_idx
            channel_idx = node.channel_idx
        else:
            backoff = 0
            transmit, sf_delta, channel_idx = decode_action(action, self._n_channels)
            next_sf = clamp_sf_index(node.sf_idx + sf_delta)

        return ControllerAction(
            transmit=transmit,
            sf_idx=next_sf,
            channel_idx=channel_idx,
            prev_channel_idx=node.channel_idx,
            internal_action=action,
            state=state_idx,
            retry_count_before=node.retry_count,
            backoff_slots=backoff,
        )

    def observe(self, node, action: ControllerAction, outcome: int, slot: int,
                gateway_info: dict | None = None) -> None:
        if gateway_info is not None:
            self._gateway_g_bins = tuple(gateway_info.get("per_channel_g_bin", self._gateway_g_bins))
        if action.state is None or action.internal_action is None:
            return

        nid = node.node_id
        is_wait_or_idle = (action.internal_action == ACTION_IDLE or self._is_wait(action.internal_action))
        node_cnt = self._std_success_count.get(nid, 0)
        mean_cnt = sum(self._std_success_count.values()) / max(1, len(self._std_success_count))

        reward = self._reward_for_outcome(outcome, is_wait_or_idle, node_cnt, mean_cnt, node.has_packet)
        if outcome == OUTCOME_SUCCESS:
            self._std_success_count[nid] = node_cnt + 1

        sf = np.array([node.sf_idx], dtype=np.int32)
        next_state = int(self._state(sf, slot + 1)[0])

        current_q = self._q[nid, action.state, action.internal_action]
        best_next = float(self._q[nid, next_state].max())
        td = reward + self.gamma_q * best_next - current_q
        self._q[nid, action.state, action.internal_action] += self.alpha * td

        eps = self._node_epsilons.get(nid, self.initial_epsilon)
        self._node_epsilons[nid] = max(self.eps_min, eps * self.eps_decay)

    # ------------------------------------------------------------------
    # 배치 경로
    # ------------------------------------------------------------------

    def choose_actions_batch(
        self,
        sf_arr: np.ndarray,
        ch_arr: np.ndarray,
        last_outcome_arr: np.ndarray,
        retry_arr: np.ndarray,
        active_ids: np.ndarray,
        g_max: int,
        slot: int,
        gateway_g_bins: tuple = (),
    ) -> tuple:
        """6-tuple 반환: (state_idx, int_actions, transmit, new_sf, new_ch, backoff_req)

        backoff_req: (n_active,) int32 — wait 액션 선택 시 해당 대기 슬롯 수, 나머지 0.
        시뮬레이터가 이 값으로 backoff_arr 를 갱신한다.
        """
        self._last_slot = slot
        n_active = len(active_ids)
        if n_active == 0:
            empty_i = np.array([], dtype=np.int32)
            return empty_i, empty_i, np.array([], dtype=bool), empty_i, empty_i, empty_i

        state_idx = self._state(sf_arr[active_ids], slot)

        q_vals = self._q[active_ids, state_idx]
        greedy = q_vals.argmax(axis=1).astype(np.int32)

        eps = self._epsilon_arr[active_ids]
        explore = self._np_rng.random(n_active) < eps
        rand_acts = self._np_rng.randint(0, self._n_actions, n_active).astype(np.int32)
        int_actions = np.where(explore, rand_acts, greedy).astype(np.int32)

        # wait 액션 마스크
        if self._wait_durations:
            wait_mask = int_actions >= self._wait_action_start
            wait_offsets = np.clip(int_actions - self._wait_action_start, 0, len(self._wait_durations) - 1)
            durations = np.array(self._wait_durations, dtype=np.int32)
            backoff_req = np.where(wait_mask, durations[wait_offsets], 0).astype(np.int32)
        else:
            wait_mask = np.zeros(n_active, dtype=bool)
            backoff_req = np.zeros(n_active, dtype=np.int32)

        # wait 또는 IDLE → 전송 안 함
        transmit = (int_actions != ACTION_IDLE) & (~wait_mask)

        new_ch = np.where(
            transmit,
            (int_actions - 1) % self._n_channels,
            ch_arr[active_ids],
        ).astype(np.int32)

        # SF: 전송 액션만 delta 적용, 나머지는 현재 SF 유지
        sf_change_idx = np.where(transmit, (int_actions - 1) // self._n_channels, 0)
        sf_deltas = np.array([0, 1, -1], dtype=np.int32)[sf_change_idx]
        new_sf = np.clip(sf_arr[active_ids] + np.where(transmit, sf_deltas, 0), 0, 5).astype(np.int32)

        return state_idx, int_actions, transmit, new_sf, new_ch, backoff_req

    def observe_batch(
        self,
        active_ids: np.ndarray,
        state_idx: np.ndarray,
        int_actions: np.ndarray,
        outcomes_arr: np.ndarray,
        sf_arr: np.ndarray,
        last_outcome_arr: np.ndarray,
        retry_arr: np.ndarray,
        queue_arr: np.ndarray,
        retry_before_active: np.ndarray,
        gateway_g_bins: tuple,
        ch_arr: np.ndarray | None = None,
    ) -> None:
        n_active = len(active_ids)
        if n_active == 0:
            return

        active_outcomes = outcomes_arr[active_ids]
        r_cfg = REWARD_VARIANTS.get(self.reward_variant, REWARD_VARIANTS[DEFAULT_REWARD_VARIANT])
        rtype = r_cfg.get("type", "standard")

        # wait 액션: 항상 0 (긴 대기에 슬롯별 보상 누적 방지)
        # IDLE(1슬롯): reward variant의 idle_pkt 사용 (v10 등에서 양수 가능)
        if self._wait_durations:
            wait_mask = int_actions >= self._wait_action_start
        else:
            wait_mask = np.zeros(n_active, dtype=bool)
        idle_mask = int_actions == ACTION_IDLE

        rewards = np.zeros(n_active, dtype=np.float64)

        # IDLE: idle_pkt (패킷 있는 노드) / idle_no_pkt (패킷 없는 노드)
        has_pkt_mask = idle_mask & (queue_arr[active_ids] > 0)
        no_pkt_mask  = idle_mask & (queue_arr[active_ids] == 0)
        rewards[has_pkt_mask] = float(r_cfg["idle_pkt"])
        rewards[no_pkt_mask]  = float(r_cfg["idle_no_pkt"])
        # wait: 0 유지 (이미 np.zeros)

        # 전송 결과만 reward variant 계산
        tx_mask = ~(wait_mask | idle_mask)
        if tx_mask.any():
            s_mask = tx_mask & (active_outcomes == OUTCOME_SUCCESS)
            f_mask = tx_mask & (
                (active_outcomes == OUTCOME_FAIL_COLLISION) |
                (active_outcomes == OUTCOME_FAIL_LINK)
            )

            if rtype in ("fairness", "log_thr"):
                mean_n = max(1.0, float(self._success_count_arr.mean()))
                counts = self._success_count_arr[active_ids[s_mask]].astype(np.float64)
                norm = counts / mean_n
                if rtype == "fairness":
                    rewards[s_mask] = 1.0 / (1.0 + norm)
                else:
                    rewards[s_mask] = np.log(norm + 2.0) - np.log(norm + 1.0)
            elif rtype == "composite":
                rewards[s_mask] = float(r_cfg.get("success_base", 1.0))
            else:
                rewards[s_mask] = float(r_cfg["success"])

            rewards[f_mask] = float(r_cfg["fail"])

        self._success_count_arr[active_ids[tx_mask & (active_outcomes == OUTCOME_SUCCESS)]] += 1

        # next state: slot+1 의 phase
        next_state_idx = self._state(sf_arr[active_ids], self._last_slot + 1)

        current_q = self._q[active_ids, state_idx, int_actions]
        best_next = self._q[active_ids, next_state_idx].max(axis=1)
        td_errors = rewards + self.gamma_q * best_next - current_q
        self._q[active_ids, state_idx, int_actions] += self.alpha * td_errors
