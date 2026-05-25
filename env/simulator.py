"""충돌 중심 LoRaWAN 동작을 재현하는 공유 시뮬레이터."""

from __future__ import annotations

import random
from collections import deque

import numpy as np

from .channel import GATEWAY_WINDOW, N_SF, clamp_sf_index, quantize_g, total_resources
from .collision import evaluate_transmissions
from .layout import place_nodes_on_ring, place_nodes_random
from .link import ALL_PROFILES, SNR_THRESH, compute_cell_radius, compute_mean_snr
from .types import (
    ControllerAction,
    NodeState,
    OUTCOME_FAIL_COLLISION,
    OUTCOME_FAIL_LINK,
    OUTCOME_IDLE,
    OUTCOME_SUCCESS,
    ScenarioConfig,
    ScheduledTransmission,
)
from utils.metrics import jains_fairness


def _build_nodes(config: ScenarioConfig):
    """설정에 맞춰 노드 배치를 만들고 초기 SF와 SNR을 계산한다."""
    profile = ALL_PROFILES[config.profile_name]
    cell_radius_m = compute_cell_radius(profile)
    fixed_radius_m = float(config.fixed_distance_ratio) * float(cell_radius_m)
    # 배치 시드: layout_seed가 지정된 경우 그것을 사용하고, 없으면 seed를 그대로 쓴다.
    # 여러 베이스라인을 비교할 때 layout_seed를 고정하면 모든 베이스라인이
    # 동일한 노드 위치를 공유하고, seed만 달리해 시뮬레이션 RNG를 독립적으로 유지한다.
    _layout_seed = config.seed if config.layout_seed is None else config.layout_seed
    if config.layout == "random":
        xs, ys, dists = place_nodes_random(config.n_nodes, cell_radius_m, seed=_layout_seed)
    else:
        xs, ys, dists = place_nodes_on_ring(config.n_nodes, fixed_radius_m, seed=_layout_seed)

    sf_rng = np.random.RandomState(_layout_seed)
    nodes: list[NodeState] = []
    for idx in range(config.n_nodes):
        mean_snr_db = compute_mean_snr(dists[idx], profile)
        init_sf_idx = int(sf_rng.randint(0, N_SF))
        nodes.append(
            NodeState(
                node_id=idx,
                x=float(xs[idx]),
                y=float(ys[idx]),
                dist_m=float(dists[idx]),
                mean_snr_db=float(mean_snr_db),
                sf_idx=init_sf_idx,
            )
        )

    return profile, cell_radius_m, fixed_radius_m, np.asarray(xs), np.asarray(ys), np.asarray(dists), nodes


def _update_gateway(ch_history: deque, ch_counts: list[int], n_channels: int,
                    gw_obs_mode: str = "attempt",
                    gw_thresholds: tuple[float, ...] = (0.7, 1.3)) -> tuple[tuple, tuple]:
    """ch_history를 갱신하고 (채널별 G bin 튜플, 채널별 G float 튜플)을 반환한다."""
    ch_history.append(ch_counts)
    window = len(ch_history)
    history_arr = np.array(ch_history, dtype=np.float64)   # (W, n_ch)
    ch_g = history_arr.sum(axis=0) / (window * N_SF)
    bins = tuple(int(quantize_g(float(g), gw_thresholds)) for g in ch_g)
    raw = tuple(float(g) for g in ch_g)
    return bins, raw


def _epoch_row(epoch_rows, slot, epoch_generated, epoch_attempts, epoch_successes,
               epoch_collisions, epoch_link_failures, epoch_slots,
               epoch_backlog_sum, epoch_backlog_max,
               epoch_node_successes, epoch_node_attempts, n_nodes,
               epoch_g_sum=None):
    per_ch_g = (epoch_g_sum / epoch_slots).tolist() if epoch_slots and epoch_g_sum is not None else []
    epoch_rows.append({
        "epoch": len(epoch_rows),
        "slot_end": slot,
        "generated": epoch_generated,
        "attempts": epoch_attempts,
        "successes": epoch_successes,
        "collisions": epoch_collisions,
        "link_failures": epoch_link_failures,
        "success_rate": epoch_successes / epoch_attempts if epoch_attempts else 0.0,
        "collision_rate": epoch_collisions / epoch_attempts if epoch_attempts else 0.0,
        "throughput": epoch_successes / epoch_slots if epoch_slots else 0.0,
        "mean_backlog_total": epoch_backlog_sum / epoch_slots if epoch_slots else 0.0,
        "mean_backlog_per_node": (
            epoch_backlog_sum / (epoch_slots * n_nodes) if epoch_slots and n_nodes > 0 else 0.0
        ),
        "max_backlog_total": epoch_backlog_max,
        "per_node_successes": epoch_node_successes.tolist(),
        "per_node_attempts": epoch_node_attempts.tolist(),
        "per_channel_g": per_ch_g,
    })


def run_simulation(config: ScenarioConfig, controller) -> dict:
    """지정한 컨트롤러로 시뮬레이션 1회를 실행하고 요약 결과를 반환한다.

    컨트롤러가 SUPPORTS_BATCH=True를 노출하면 numpy 벡터화 배치 경로를 사용한다.
    그렇지 않으면 기존 per-node Python 루프를 사용한다.
    """
    rng = random.Random(config.seed)
    profile, cell_radius_m, fixed_radius_m, xs, ys, dists, nodes = _build_nodes(config)
    controller.begin_run(nodes, config, rng)

    use_batch = getattr(controller, 'SUPPORTS_BATCH', False)

    # 에너지 비용: 컨트롤러가 _energy_cost를 가지면 사용, 없으면 기본값 (SF7~SF12)
    _default_energy_cost: dict[int, float] = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0}
    _energy_cost: dict[int, float] = getattr(controller, '_energy_cost', None) or _default_energy_cost

    # --- 공통 집계 변수 ---
    N = config.n_nodes
    measured_attempts = 0
    measured_successes = 0
    measured_collisions = 0
    measured_link_failures = 0
    measured_generated = 0
    measured_backlog_sum = 0.0
    measured_backlog_max = 0

    sf_usage = np.zeros((N, N_SF), dtype=np.int64)
    node_idle_arr = np.zeros(N, dtype=np.int64)
    node_collisions_arr = np.zeros(N, dtype=np.int64)
    node_link_failures_arr = np.zeros(N, dtype=np.int64)
    node_successes_arr = np.zeros(N, dtype=np.int64)
    node_attempts_arr = np.zeros(N, dtype=np.int64)
    node_energy_total = np.zeros(N, dtype=np.float64)
    node_energy_fail = np.zeros(N, dtype=np.float64)

    epoch_rows: list[dict] = []
    epoch_generated = 0
    epoch_attempts = 0
    epoch_successes = 0
    epoch_collisions = 0
    epoch_link_failures = 0
    epoch_slots = 0
    epoch_backlog_sum = 0.0
    epoch_backlog_max = 0
    epoch_node_successes = np.zeros(N, dtype=np.int64)
    epoch_node_attempts = np.zeros(N, dtype=np.int64)

    _ch_history: deque = deque(maxlen=GATEWAY_WINDOW)
    _gateway_g_bins: tuple = tuple(1 for _ in range(config.n_channels))
    _gateway_g_raw: tuple = tuple(0.0 for _ in range(config.n_channels))
    epoch_g_sum = np.zeros(config.n_channels, dtype=np.float64)

    # ==================== 배치 경로 (numpy 벡터화) ====================
    if use_batch:
        np_rng = np.random.RandomState(config.seed)
        snr_arr = np.array([n.mean_snr_db for n in nodes], dtype=np.float64)
        sf_arr = np.array([n.sf_idx for n in nodes], dtype=np.int32)
        ch_arr = np.zeros(N, dtype=np.int32)
        queue_arr = np.zeros(N, dtype=np.int32)
        backoff_arr = np.zeros(N, dtype=np.int32)
        retry_arr = np.zeros(N, dtype=np.int32)
        last_outcome_arr = np.zeros(N, dtype=np.int32)

        for slot in range(config.n_slots):
            is_meas = slot >= config.warmup_slots

            # 패킷 도착 (벡터화)
            arrived = np_rng.random(N) < config.p_arrival
            if config.queue_mode == "fresh":
                queue_arr[:] = arrived
                # retry_arr는 리셋하지 않음 — 연속 실패 횟수는 패킷 경계와 무관하게 유지,
                # 성공 시에만 리셋 (accumulate 모드와 동일한 의미론)
            else:
                queue_arr += arrived
            if is_meas:
                n_arr = int(arrived.sum())
                measured_generated += n_arr
                epoch_generated += n_arr

            # 백오프 감소 (벡터화)
            np.maximum(0, backoff_arr - 1, out=backoff_arr)

            # 액션 결정 (배치)
            active_mask = (queue_arr > 0) & (backoff_arr == 0)
            active_ids = np.where(active_mask)[0]
            retry_before_active = retry_arr[active_ids].copy()

            g_max = int(max(_gateway_g_bins)) if _gateway_g_bins else 1
            _batch_result = controller.choose_actions_batch(
                sf_arr, ch_arr, last_outcome_arr, retry_arr, active_ids, g_max, slot, _gateway_g_bins
            )
            if len(_batch_result) == 6:
                state_idx, int_actions, transmit_local, new_sf, new_ch, _backoff_req = _batch_result
                # wait 액션을 지원하는 컨트롤러: 요청된 backoff를 배열에 반영
                if len(active_ids) > 0:
                    backoff_arr[active_ids] = np.maximum(backoff_arr[active_ids], _backoff_req)
            else:
                state_idx, int_actions, transmit_local, new_sf, new_ch = _batch_result

            # SF 업데이트 (전송 여부 무관)
            sf_arr[active_ids] = new_sf

            # 전송 노드
            tx_local_mask = transmit_local
            tx_ids = active_ids[tx_local_mask]
            if len(tx_ids) > 0:
                ch_arr[tx_ids] = new_ch[tx_local_mask]

            # 충돌 + 링크 판정 (벡터화)
            outcomes_arr = np.zeros(N, dtype=np.int32)
            if len(tx_ids) > 0:
                tx_sf = sf_arr[tx_ids]
                tx_ch = ch_arr[tx_ids]
                tx_resource = tx_sf * config.n_channels + tx_ch
                _, inv, counts_res = np.unique(tx_resource, return_inverse=True, return_counts=True)
                col_local = counts_res[inv] >= 2

                outcomes_arr[tx_ids[col_local]] = OUTCOME_FAIL_COLLISION

                solo_ids = tx_ids[~col_local]
                if len(solo_ids) > 0:
                    if config.enable_rician_fading:
                        k = config.rician_k_factor
                        mean_lin = 10.0 ** ((snr_arr[solo_ids] + config.rician_fade_margin_db) / 10.0)
                        a = np.sqrt(k / (k + 1.0))
                        sigma = np.sqrt(1.0 / (2.0 * (k + 1.0)))
                        h_re = a + np_rng.normal(0.0, sigma, size=len(solo_ids))
                        h_im = np_rng.normal(0.0, sigma, size=len(solo_ids))
                        g_arr = np.maximum(h_re ** 2 + h_im ** 2, 1e-10)
                        inst_snr = 10.0 * np.log10(mean_lin * g_arr)
                        margin = inst_snr - SNR_THRESH[sf_arr[solo_ids]]
                    elif config.enable_rayleigh_fading:
                        mean_lin = 10.0 ** ((snr_arr[solo_ids] + config.rayleigh_fade_margin_db) / 10.0)
                        g_arr = np_rng.exponential(1.0, size=len(solo_ids))
                        inst_snr = 10.0 * np.log10(mean_lin * g_arr)
                        margin = inst_snr - SNR_THRESH[sf_arr[solo_ids]]
                    else:
                        margin = snr_arr[solo_ids] - SNR_THRESH[sf_arr[solo_ids]]
                    link_ok = margin >= 0.0
                    outcomes_arr[solo_ids[link_ok]] = OUTCOME_SUCCESS
                    outcomes_arr[solo_ids[~link_ok]] = OUTCOME_FAIL_LINK

            # 게이트웨이 G 추정 (벡터화)
            ch_counts = [0] * config.n_channels
            if len(tx_ids) > 0:
                if config.gw_obs_mode == "success":
                    suc_ids_gw = tx_ids[outcomes_arr[tx_ids] == OUTCOME_SUCCESS]
                    if len(suc_ids_gw) > 0:
                        counts_np = np.bincount(ch_arr[suc_ids_gw], minlength=config.n_channels)
                        ch_counts = counts_np.tolist()
                else:  # "attempt"
                    counts_np = np.bincount(ch_arr[tx_ids], minlength=config.n_channels)
                    ch_counts = counts_np.tolist()
            _gateway_g_bins, _gateway_g_raw = _update_gateway(_ch_history, ch_counts, config.n_channels, config.gw_obs_mode, config.gw_thresholds)

            # 통계 집계 (벡터화)
            if is_meas and len(tx_ids) > 0:
                tx_outs = outcomes_arr[tx_ids]
                n_att = len(tx_ids)
                suc_ids = tx_ids[tx_outs == OUTCOME_SUCCESS]
                col_ids = tx_ids[tx_outs == OUTCOME_FAIL_COLLISION]
                lf_ids = tx_ids[tx_outs == OUTCOME_FAIL_LINK]

                measured_attempts += n_att
                measured_successes += len(suc_ids)
                measured_collisions += len(col_ids)
                measured_link_failures += len(lf_ids)
                epoch_attempts += n_att
                epoch_successes += len(suc_ids)
                epoch_collisions += len(col_ids)
                epoch_link_failures += len(lf_ids)

                node_attempts_arr[tx_ids] += 1
                node_successes_arr[suc_ids] += 1
                node_collisions_arr[col_ids] += 1
                node_link_failures_arr[lf_ids] += 1
                sf_usage[tx_ids, sf_arr[tx_ids]] += 1

                epoch_node_attempts[tx_ids] += 1
                epoch_node_successes[suc_ids] += 1

                # 에너지 추적
                _e_per_tx = np.array([_energy_cost.get(int(sf_arr[tid]), 0.0) for tid in tx_ids], dtype=np.float64)
                node_energy_total[tx_ids] += _e_per_tx
                _fail_mask_e = tx_outs != OUTCOME_SUCCESS
                node_energy_fail[tx_ids[_fail_mask_e]] += _e_per_tx[_fail_mask_e]

            if is_meas and len(active_ids) > 0:
                idle_ids = active_ids[~tx_local_mask]
                node_idle_arr[idle_ids] += 1

            # 상태 업데이트 (벡터화)
            suc_mask = outcomes_arr == OUTCOME_SUCCESS
            fail_mask = (outcomes_arr == OUTCOME_FAIL_COLLISION) | (outcomes_arr == OUTCOME_FAIL_LINK)
            queue_arr[suc_mask] = np.maximum(0, queue_arr[suc_mask] - 1)
            retry_arr[suc_mask] = 0
            retry_arr[fail_mask] += 1
            last_outcome_arr[:] = outcomes_arr

            # 백로그 통계
            if is_meas:
                total_bl = int(queue_arr.sum())
                measured_backlog_sum += total_bl
                measured_backlog_max = max(measured_backlog_max, total_bl)
                epoch_backlog_sum += total_bl
                epoch_backlog_max = max(epoch_backlog_max, total_bl)
                epoch_g_sum += np.array(_gateway_g_raw, dtype=np.float64)
                epoch_slots += 1

                if epoch_slots == config.epoch_slots or slot == config.n_slots - 1:
                    _epoch_row(
                        epoch_rows, slot,
                        epoch_generated, epoch_attempts, epoch_successes,
                        epoch_collisions, epoch_link_failures, epoch_slots,
                        epoch_backlog_sum, epoch_backlog_max,
                        epoch_node_successes, epoch_node_attempts, N,
                        epoch_g_sum,
                    )
                    epoch_generated = epoch_attempts = epoch_successes = 0
                    epoch_collisions = epoch_link_failures = 0
                    epoch_slots = 0
                    epoch_backlog_sum = 0.0
                    epoch_backlog_max = 0
                    epoch_node_successes = np.zeros(N, dtype=np.int64)
                    epoch_node_attempts = np.zeros(N, dtype=np.int64)
                    epoch_g_sum = np.zeros(config.n_channels, dtype=np.float64)

            # observe 배치
            controller.observe_batch(
                active_ids, state_idx, int_actions, outcomes_arr,
                sf_arr, last_outcome_arr, retry_arr, queue_arr,
                retry_before_active, _gateway_g_bins,
                ch_arr=ch_arr,
            )

            controller.end_slot(slot)

        success_vector = node_successes_arr.astype(np.float64)
        attempt_vector = node_attempts_arr.astype(np.float64)
        final_backlog_total = int(queue_arr.sum())

    # ==================== 표준 경로 (per-node Python 루프) ====================
    else:
        _gateway_info: dict = {
            "per_channel_g": [0.0] * config.n_channels,
            "per_channel_g_bin": _gateway_g_bins,
        }
        np_rng = np.random.RandomState(config.seed)

        for slot in range(config.n_slots):
            is_measured_slot = slot >= config.warmup_slots

            # 패킷 도착 — numpy로 N개 난수를 한 번에 생성
            rand_vals = np_rng.random(N)
            for i, node in enumerate(nodes):
                arrived = rand_vals[i] < config.p_arrival
                if config.queue_mode == "fresh":
                    node.queue_len = 1 if arrived else 0
                    # retry_count는 리셋하지 않음 — 연속 실패 횟수는 패킷 경계와 무관하게 유지,
                    # 성공 시에만 리셋 (accumulate 모드와 동일한 의미론)
                    if arrived:
                        node.generated_packets += 1
                        if is_measured_slot:
                            measured_generated += 1
                            epoch_generated += 1
                else:
                    if arrived:
                        node.queue_len += 1
                        node.generated_packets += 1
                        if is_measured_slot:
                            measured_generated += 1
                            epoch_generated += 1
                if node.backoff_slots > 0:
                    node.backoff_slots -= 1

            actions: list[ControllerAction] = []
            transmissions: list[ScheduledTransmission] = []

            for node in nodes:
                if not node.has_packet or node.backoff_slots > 0:
                    action = ControllerAction(
                        transmit=False,
                        sf_idx=node.sf_idx,
                        channel_idx=node.channel_idx,
                        prev_channel_idx=node.channel_idx,
                    )
                else:
                    action = controller.choose_action(node, slot, _gateway_info)
                    action.sf_idx = clamp_sf_index(action.sf_idx)
                    action.channel_idx = max(0, min(config.n_channels - 1, action.channel_idx))
                    action.prev_channel_idx = node.channel_idx
                    node.sf_idx = action.sf_idx

                    if action.transmit:
                        node.channel_idx = action.channel_idx
                        transmissions.append(
                            ScheduledTransmission(
                                node_id=node.node_id,
                                sf_idx=node.sf_idx,
                                channel_idx=node.channel_idx,
                                mean_snr_db=node.mean_snr_db,
                            )
                        )
                        if is_measured_slot:
                            sf_usage[node.node_id, node.sf_idx] += 1
                    elif is_measured_slot:
                        node_idle_arr[node.node_id] += 1
                actions.append(action)

            outcomes, slot_collisions, slot_link_failures = evaluate_transmissions(
                transmissions, rng,
                config.enable_rayleigh_fading, config.rayleigh_fade_margin_db,
                config.enable_rician_fading, config.rician_k_factor, config.rician_fade_margin_db,
            )

            # 게이트웨이 G 추정
            ch_counts = [0] * config.n_channels
            if config.gw_obs_mode == "success":
                for tx in transmissions:
                    if outcomes.get(tx.node_id) == OUTCOME_SUCCESS:
                        ch_counts[tx.channel_idx] += 1
            else:  # "attempt"
                for tx in transmissions:
                    ch_counts[tx.channel_idx] += 1
            _gateway_g_bins, _gateway_g_raw = _update_gateway(_ch_history, ch_counts, config.n_channels, config.gw_obs_mode, config.gw_thresholds)
            _gateway_info = {
                "per_channel_g": list(_gateway_g_raw),
                "per_channel_g_bin": _gateway_g_bins,
            }

            for node, action in zip(nodes, actions):
                if not action.transmit:
                    node.last_outcome = OUTCOME_IDLE
                    controller.observe(node, action, OUTCOME_IDLE, slot, gateway_info=_gateway_info)
                    continue

                outcome = outcomes.get(node.node_id, OUTCOME_IDLE)
                node.last_outcome = outcome

                if is_measured_slot:
                    measured_attempts += 1
                    epoch_attempts += 1
                    epoch_node_attempts[node.node_id] += 1
                    node.attempts_measured += 1
                    _e = _energy_cost.get(node.sf_idx, 0.0)
                    node_energy_total[node.node_id] += _e
                    if outcome != OUTCOME_SUCCESS:
                        node_energy_fail[node.node_id] += _e

                if outcome == OUTCOME_SUCCESS:
                    if node.queue_len > 0:
                        node.queue_len -= 1
                    node.retry_count = 0
                    node.backoff_slots = 0
                    if is_measured_slot:
                        measured_successes += 1
                        epoch_successes += 1
                        epoch_node_successes[node.node_id] += 1
                        node.successes_measured += 1
                        node_successes_arr[node.node_id] += 1
                elif outcome in (OUTCOME_FAIL_COLLISION, OUTCOME_FAIL_LINK):
                    node.retry_count += 1
                    if is_measured_slot and outcome == OUTCOME_FAIL_COLLISION:
                        measured_collisions += 1
                        node_collisions_arr[node.node_id] += 1
                    if is_measured_slot and outcome == OUTCOME_FAIL_LINK:
                        measured_link_failures += 1
                        node_link_failures_arr[node.node_id] += 1

                controller.observe(node, action, outcome, slot, gateway_info=_gateway_info)

            if is_measured_slot:
                epoch_collisions += slot_collisions
                epoch_link_failures += slot_link_failures
                epoch_slots += 1
                slot_backlog_total = sum(node.queue_len for node in nodes)
                measured_backlog_sum += slot_backlog_total
                measured_backlog_max = max(measured_backlog_max, slot_backlog_total)
                epoch_backlog_sum += slot_backlog_total
                epoch_backlog_max = max(epoch_backlog_max, slot_backlog_total)
                epoch_g_sum += np.array(_gateway_g_raw, dtype=np.float64)

                if epoch_slots == config.epoch_slots or slot == config.n_slots - 1:
                    _epoch_row(
                        epoch_rows, slot,
                        epoch_generated, epoch_attempts, epoch_successes,
                        epoch_collisions, epoch_link_failures, epoch_slots,
                        epoch_backlog_sum, epoch_backlog_max,
                        epoch_node_successes, epoch_node_attempts, N,
                        epoch_g_sum,
                    )
                    epoch_generated = epoch_attempts = epoch_successes = 0
                    epoch_collisions = epoch_link_failures = 0
                    epoch_slots = 0
                    epoch_backlog_sum = 0.0
                    epoch_backlog_max = 0
                    epoch_node_successes = np.zeros(N, dtype=np.int64)
                    epoch_node_attempts = np.zeros(N, dtype=np.int64)
                    epoch_g_sum = np.zeros(config.n_channels, dtype=np.float64)

            controller.end_slot(slot)

        success_vector = np.array([node.successes_measured for node in nodes], dtype=np.float64)
        attempt_vector = np.array([node.attempts_measured for node in nodes], dtype=np.float64)
        final_backlog_total = int(sum(node.queue_len for node in nodes))

    # ==================== 공통 반환 dict ====================
    # attempt_vector는 배치/표준 경로 공통으로 사용하는 변수
    _total_energy = float(node_energy_total.sum())
    _fail_energy = float(node_energy_fail.sum())
    _active_nodes = int((attempt_vector > 0).sum())

    sf_totals = sf_usage.sum(axis=0)
    sf_total_count = int(sf_totals.sum())
    n_resources = config.n_resources
    mean_backlog_total = (
        measured_backlog_sum / config.measured_slots if config.measured_slots else 0.0
    )
    mean_backlog_per_node = (
        measured_backlog_sum / (config.measured_slots * N)
        if config.measured_slots and N > 0
        else 0.0
    )

    return {
        "baseline_key": getattr(controller, "key", controller.__class__.__name__.lower()),
        "baseline_label": getattr(controller, "label", controller.__class__.__name__),
        "profile_name": profile.name,
        "cell_radius_m": float(cell_radius_m),
        "fixed_radius_m": float(fixed_radius_m),
        "mean_snr_db": float(nodes[0].mean_snr_db if nodes else 0.0),
        "n_nodes": N,
        "target_g": float(config.target_g),
        "raw_p_arrival": float(config.raw_p_arrival),
        "p_arrival": float(config.p_arrival),
        "p_arrival_is_clipped": bool(config.p_arrival_is_clipped),
        "n_channels": config.n_channels,
        "n_resources": n_resources,
        "n_slots": config.n_slots,
        "warmup_slots": config.warmup_slots,
        "measured_slots": config.measured_slots,
        "generated": measured_generated,
        "attempts": measured_attempts,
        "successes": measured_successes,
        "collisions": measured_collisions,
        "link_failures": measured_link_failures,
        "success_rate": measured_successes / measured_attempts if measured_attempts else 0.0,
        "collision_rate": measured_collisions / measured_attempts if measured_attempts else 0.0,
        "throughput": measured_successes / config.measured_slots if config.measured_slots else 0.0,
        "pdr": measured_successes / measured_generated if measured_generated else 0.0,
        "arrival_load_realized": (
            measured_generated / (n_resources * config.measured_slots)
            if config.measured_slots else 0.0
        ),
        "attempt_load_realized": (
            measured_attempts / (n_resources * config.measured_slots)
            if config.measured_slots else 0.0
        ),
        "mean_backlog_total": mean_backlog_total,
        "mean_backlog_per_node": mean_backlog_per_node,
        "max_backlog_total": measured_backlog_max,
        "final_backlog_total": final_backlog_total,
        "final_backlog_per_node": final_backlog_total / N if N > 0 else 0.0,
        "fairness": jains_fairness(success_vector[attempt_vector > 0]) if (attempt_vector > 0).any() else 0.0,
        "fairness_asr": (
            jains_fairness(
                success_vector[attempt_vector > 0] / attempt_vector[attempt_vector > 0]
            ) if (attempt_vector > 0).any() else 0.0
        ),
        # Jain FI on per-node throughput (S_i / T_meas) — 전체 N 노드 포함, 0-attempt 노드도 0으로 반영
        "fairness_thr": (
            jains_fairness(success_vector / config.measured_slots)
            if config.measured_slots else 0.0
        ),
        "sf_usage": sf_usage,
        "idle_usage": node_idle_arr,
        "sf_fractions": (
            (sf_totals / sf_total_count).tolist() if sf_total_count else [0.0] * N_SF
        ),
        "per_node_success": success_vector.tolist(),
        "per_node_attempts": attempt_vector.tolist(),
        "per_node_collisions": node_collisions_arr.tolist(),
        "per_node_link_failures": node_link_failures_arr.tolist(),
        "per_node_energy_total": node_energy_total.tolist(),
        "per_node_energy_fail": node_energy_fail.tolist(),
        "total_energy": _total_energy,
        "fail_energy": _fail_energy,
        "mae": _total_energy / _active_nodes if _active_nodes > 0 else 0.0,
        "elr": _fail_energy / _total_energy if _total_energy > 0 else 0.0,
        "energy_efficiency": measured_successes / _total_energy if _total_energy > 0 else 0.0,
        "epoch_log": epoch_rows,
        "xs": xs.tolist(),
        "ys": ys.tolist(),
        "dists_m": dists.tolist(),
        "q_table_data": getattr(controller, "get_mean_q_array", lambda: None)(),
        "q_per_node": getattr(controller, "get_per_node_q_snapshot", lambda: None)(),
    }
