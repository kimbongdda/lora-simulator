"""충돌이 잦은 LoRaWAN 환경을 위한 표 형태 Q-learning 컨트롤러.
이 모듈은 액션, 상태, 보상 설계를 모두 실험 가능하게 분리해 둔다.
그래서 같은 시뮬레이션 조건에서 학습 정책만 바꿔 비교할 수 있다.

액션 공간
----------
액션 0은 해당 슬롯에서 쉬는 것을 뜻하고, 그 외의 액션은
특정 SF 변화와 채널 선택을 동시에 지정한다.

    action = 1 + sf_change_idx * n_channels + ch_idx

    sf_change_idx  0 = 현재 SF 유지
                   1 = SF + 1  (더 안정적)
                   2 = SF - 1  (더 빠름)

상태 변형
---------
STATE_VARIANTS로 상태 벡터 구성을 바꿀 수 있다.
각 변형은 어떤 특징을 상태에 넣을지 여부를 불리언 플래그로 가진다.
encode_state()는 선택된 특징만 순서대로 묶어 Q-table 키를 만든다.

    s0_full      : snr_margin_bin까지 포함한 기본 상태
    s1_realistic : snr_margin_bin 제거, 실제 노드가 알 수 있는 정보만 사용

보상 변형
---------
같은 충돌 중심 환경에서도 보상 설계에 따라 학습이 어떻게 달라지는지
보기 위해 여러 reward variant를 제공한다.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np

from env.channel import clamp_sf_index
from env.link import SNR_THRESH
from env.types import ControllerAction, OUTCOME_FAIL_COLLISION, OUTCOME_FAIL_LINK, OUTCOME_IDLE, OUTCOME_SUCCESS

ACTION_IDLE = 0
N_SF_CHANGES = 3
N_SF = 6
CHANNEL_SWITCH_PENALTY = 0.05

ACTION_VARIANTS: dict[str, dict] = {
    "relative": {
        "label": "Relative (SF±1 × CH)",
        "description": (
            "SF 유지/+1/-1 세 가지 변화 × 채널 선택. 기본 액션공간. "
            "n_channels=3 기준 10 액션 (0=IDLE, 1~9=SF변화×CH)."
        ),
    },
    "absolute": {
        "label": "Absolute (SF6 × CH)",
        "description": (
            "SF0~5 직접 선택 × 채널 선택. 상대 변화 대신 절대 SF를 지정. "
            "n_channels=3 기준 19 액션 (0=IDLE, 1~18=SF×CH)."
        ),
    },
}
DEFAULT_ACTION_VARIANT = "relative"


def n_actions(n_channels: int, action_variant: str = DEFAULT_ACTION_VARIANT) -> int:
    """이산 액션 공간의 크기를 반환한다."""
    if action_variant == "absolute":
        return 1 + N_SF * int(n_channels)
    return 1 + N_SF_CHANGES * int(n_channels)


def decode_action(action: int, n_channels: int) -> tuple[bool, int, int]:
    """relative 액션 id를 (전송 여부, SF 변화량, 채널 인덱스)로 푼다."""
    if action == ACTION_IDLE:
        return False, 0, 0

    offset = action - 1
    sf_change_idx = offset // int(n_channels)
    channel_idx = offset % int(n_channels)
    sf_delta = (0, +1, -1)[sf_change_idx]
    return True, sf_delta, channel_idx


def decode_action_absolute(action: int, n_channels: int) -> tuple[bool, int, int]:
    """absolute 액션 id를 (전송 여부, 절대 SF 인덱스, 채널 인덱스)로 푼다."""
    if action == ACTION_IDLE:
        return False, 0, 0
    offset = action - 1
    sf_idx = offset // int(n_channels)
    channel_idx = offset % int(n_channels)
    return True, sf_idx, channel_idx


def _snr_margin_bin(margin_db: float) -> int:
    if margin_db < 0.0:
        return 0
    if margin_db < 3.0:
        return 1
    return 2


def _retry_bin(retry_count: int) -> int:
    return min(int(retry_count), 3)


def _last_result_bin(last_outcome: int) -> int:
    if last_outcome == OUTCOME_SUCCESS:
        return 1
    if last_outcome in (OUTCOME_FAIL_COLLISION, OUTCOME_FAIL_LINK):
        return 2
    return 0


def _failure_level(last_outcome: int, retry_count: int) -> int:
    """직전 결과와 retry 횟수를 4단계 실패 레벨로 합친다.

    0 = 직전 성공 또는 retry=0 (fresh)
    1 = 첫 실패 (retry=1)
    2 = 2~3회 연속 실패
    3 = 4회 이상 연속 실패

    last_result와 retry_bin 조합을 압축해 상태 차원을 줄인다.
    """
    if last_outcome == OUTCOME_SUCCESS or retry_count == 0:
        return 0
    if retry_count == 1:
        return 1
    if retry_count <= 3:
        return 2
    return 3


STATE_VARIANTS: dict[str, dict] = {
    "s0_full": {
        "label": "s0 전체 (기본)",
        "description": (
            "현재 기본 상태 공간. snr_margin_bin 포함(simulator privilege). "
            "n_channels=3 기준 34,992 상태."
        ),
        "has_packet":      True,
        "last_result_bin": True,
        "retry_bin":       True,
        "sf_idx":          True,
        "snr_margin_bin":  True,   # ⚠️ simulator privilege: 현실 노드는 모름
        "channel_idx":     True,
        "gateway_g_bins":  True,   # 채널별 3개 bin
        "failure_level":   False,
        "gateway_g_max":   False,
    },
    "s1_realistic": {
        "label": "s1 현실적",
        "description": (
            "snr_margin_bin 제거 — 노드가 직접 알 수 있는 정보만 사용. "
            "n_channels=3 기준 11,664 상태. 학습 속도 향상 기대."
        ),
        "has_packet":      True,
        "last_result_bin": True,
        "retry_bin":       True,
        "sf_idx":          True,
        "snr_margin_bin":  False,
        "channel_idx":     True,
        "gateway_g_bins":  True,
        "failure_level":   False,
        "gateway_g_max":   False,
    },
    "s2_compact": {
        "label": "s2 압축 (SF+FL+Gmax)",
        "description": (
            "4가지 최적화 적용. "
            "(1) has_packet 제거: choose_action은 패킷 있을 때만 호출됨. "
            "(2) channel_idx 제거: 액션 공간에 이미 인코딩, reward switch_pen과 중복. "
            "(3) last_result+retry → failure_level 4단계 통합: 0=성공/fresh, 1=첫실패, 2=2~3회, 3=4회+. "
            "(4) per-channel g_bins → 최대 혼잡도 1개 bin으로 단일화. "
            "SF(6)×FL(4)×Gmax(3) = 72 상태. 배치 경로 지원."
        ),
        "has_packet":      False,
        "last_result_bin": False,
        "retry_bin":       False,
        "sf_idx":          True,
        "snr_margin_bin":  False,
        "channel_idx":     False,
        "gateway_g_bins":  False,
        "failure_level":   True,
        "gateway_g_max":   True,
    },
    "s3_no_sf": {
        "label": "s3 SF 제거 (FL+Gmax)",
        "description": (
            "s2_compact에서 SF 인덱스 제거. "
            "SF 선택을 액션 공간(SF 변경 delta)에만 위임하고, "
            "현재 SF 자체는 상태에 포함하지 않음. "
            "FL(4)×Gmax(3) = 12 상태. 배치 경로 지원."
        ),
        "has_packet":      False,
        "last_result_bin": False,
        "retry_bin":       False,
        "sf_idx":          False,
        "snr_margin_bin":  False,
        "channel_idx":     False,
        "gateway_g_bins":  False,
        "failure_level":   True,
        "gateway_g_max":   True,
    },
    "s4_no_fl": {
        "label": "s4 실패레벨 제거 (SF+Gmax)",
        "description": (
            "s2_compact에서 실패 레벨 제거. "
            "retry 이력 없이 현재 SF와 채널 혼잡도만으로 판단. "
            "SF(6)×Gmax(3) = 18 상태. 배치 경로 지원."
        ),
        "has_packet":      False,
        "last_result_bin": False,
        "retry_bin":       False,
        "sf_idx":          True,
        "snr_margin_bin":  False,
        "channel_idx":     False,
        "gateway_g_bins":  False,
        "failure_level":   False,
        "gateway_g_max":   True,
    },
    "s5_no_gmax": {
        "label": "s5 혼잡도 제거 (SF+FL)",
        "description": (
            "s2_compact에서 최대 혼잡도 제거. "
            "GW 피드백 없이 노드 자체 상태(SF, 실패 이력)만 사용. "
            "SF(6)×FL(4) = 24 상태. 배치 경로 지원."
        ),
        "has_packet":      False,
        "last_result_bin": False,
        "retry_bin":       False,
        "sf_idx":          True,
        "snr_margin_bin":  False,
        "channel_idx":     False,
        "gateway_g_bins":  False,
        "failure_level":   True,
        "gateway_g_max":   False,
    },
    "s6_sf_only": {
        "label": "s6 SF만 (SF)",
        "description": (
            "SF 인덱스만 상태로 사용. 실패 이력·혼잡도 피드백 없음. "
            "SF(6) = 6 상태. 배치 경로 지원."
        ),
        "has_packet":      False,
        "last_result_bin": False,
        "retry_bin":       False,
        "sf_idx":          True,
        "snr_margin_bin":  False,
        "channel_idx":     False,
        "gateway_g_bins":  False,
        "failure_level":   False,
        "gateway_g_max":   False,
    },
    "s7_sf_gch": {
        "label": "s7 SF+채널별혼잡도 (SF+Gch)",
        "description": (
            "SF 인덱스 + 채널별 혼잡도 bin 3개. 실패 이력 없음. "
            "어떤 채널이 혼잡한지 직접 식별 가능. "
            "SF(6)×Gch0(3)×Gch1(3)×Gch2(3) = 162 상태 (n_channels=3 기준). 배치 경로 지원."
        ),
        "has_packet":      False,
        "last_result_bin": False,
        "retry_bin":       False,
        "sf_idx":          True,
        "snr_margin_bin":  False,
        "channel_idx":     False,
        "gateway_g_bins":  True,   # 채널별 3개 bin 개별 사용
        "failure_level":   False,
        "gateway_g_max":   False,
    },
    "s8_sf_delta": {
        "label": "s8 SF방향+FL+Gmax (SF_delta)",
        "description": (
            "s2_compact에서 현재 SF 절대값 대신 직전 액션의 SF 변화 방향을 상태로 사용. "
            "down(-1)→0, stay(0)→1, up(+1)→2. action space의 relative SF와 대칭. "
            "SF_delta(3)×FL(4)×Gmax(3) = 36 상태. 배치 경로 지원."
        ),
        "has_packet":      False,
        "last_result_bin": False,
        "retry_bin":       False,
        "sf_idx":          False,
        "sf_delta":        True,
        "snr_margin_bin":  False,
        "channel_idx":     False,
        "gateway_g_bins":  False,
        "failure_level":   True,
        "gateway_g_max":   True,
    },
    "s9_sf_delta_gmax": {
        "label": "s9 SF방향+Gmax (SF_delta+Gmax)",
        "description": (
            "s8_sf_delta에서 실패 이력(FL) 제거. "
            "SF 변화 방향과 채널 최대 혼잡도만 사용. "
            "SF_delta(3)×Gmax(3) = 9 상태. 배치 경로 지원."
        ),
        "has_packet":      False,
        "last_result_bin": False,
        "retry_bin":       False,
        "sf_idx":          False,
        "sf_delta":        True,
        "snr_margin_bin":  False,
        "channel_idx":     False,
        "gateway_g_bins":  False,
        "failure_level":   False,
        "gateway_g_max":   True,
    },
    "s10_sf_delta_fl": {
        "label": "s10 SF방향+FL (SF_delta+FL)",
        "description": (
            "s8_sf_delta에서 혼잡도(Gmax) 제거. "
            "SF 변화 방향과 실패 이력만 사용. GW 피드백 없음. "
            "SF_delta(3)×FL(4) = 12 상태. 배치 경로 지원."
        ),
        "has_packet":      False,
        "last_result_bin": False,
        "retry_bin":       False,
        "sf_idx":          False,
        "sf_delta":        True,
        "snr_margin_bin":  False,
        "channel_idx":     False,
        "gateway_g_bins":  False,
        "failure_level":   True,
        "gateway_g_max":   False,
    },
    "s12_sf_ch_gmax": {
        "label": "s12 SF+채널+Gmax",
        "description": (
            "현재 SF, 선택한 채널 인덱스, 채널 최대 혼잡도 bin. "
            "채널 선택 이력이 상태에 포함되어 채널 전환 학습 가능. "
            "SF(6)×CH(n_ch)×Gmax(n_bins) = 6×3×3=54 상태 (n_ch=3, bins=3). 배치 경로 지원."
        ),
        "has_packet":      False,
        "last_result_bin": False,
        "retry_bin":       False,
        "sf_idx":          True,
        "snr_margin_bin":  False,
        "channel_idx":     True,
        "gateway_g_bins":  False,
        "failure_level":   False,
        "gateway_g_max":   True,
    },
    "s13_sf_ch": {
        "label": "s13 SF+채널 (SF+CH)",
        "description": (
            "현재 SF 인덱스 + 선택한 채널 인덱스. Gmax·실패이력 없음. "
            "SF(6)×CH(n_ch) = 6×3=18 상태 (n_ch=3). 배치 경로 지원."
        ),
        "has_packet":      False,
        "last_result_bin": False,
        "retry_bin":       False,
        "sf_idx":          True,
        "snr_margin_bin":  False,
        "channel_idx":     True,
        "gateway_g_bins":  False,
        "failure_level":   False,
        "gateway_g_max":   False,
    },
    "s14_sf_delta_ch": {
        "label": "s14 SF방향+채널 (SF_delta+CH)",
        "description": (
            "직전 SF 변화 방향 + 선택한 채널 인덱스. Gmax·실패이력 없음. "
            "SF_delta(3)×CH(n_ch) = 3×3=9 상태 (n_ch=3). 배치 경로 지원."
        ),
        "has_packet":      False,
        "last_result_bin": False,
        "retry_bin":       False,
        "sf_idx":          False,
        "sf_delta":        True,
        "snr_margin_bin":  False,
        "channel_idx":     True,
        "gateway_g_bins":  False,
        "failure_level":   False,
        "gateway_g_max":   False,
    },
    "s11_sf_delta_only": {
        "label": "s11 SF방향만 (SF_delta only)",
        "description": (
            "SF 변화 방향(down/stay/up)만 상태로 사용. "
            "실패 이력·혼잡도 피드백 없음. 최소 상태 공간. "
            "SF_delta(3) = 3 상태. 배치 경로 지원."
        ),
        "has_packet":      False,
        "last_result_bin": False,
        "retry_bin":       False,
        "sf_idx":          False,
        "sf_delta":        True,
        "snr_margin_bin":  False,
        "channel_idx":     False,
        "gateway_g_bins":  False,
        "failure_level":   False,
        "gateway_g_max":   False,
    },
}

DEFAULT_STATE_VARIANT = "s4_no_fl"


def encode_state(node, gateway_g_bins: tuple[int, ...], state_variant: str = DEFAULT_STATE_VARIANT, sf_delta: int = 0) -> tuple[int, ...]:
    """큐 테이블 조회용 상태 키를 만든다.

    노드의 현재 패킷 여부, 직전 결과, retry 횟수, SF, 채널,
    그리고 게이트웨이 혼잡 정보 중 활성화된 항목만 순서대로 넣는다.
    """
    cfg = STATE_VARIANTS.get(state_variant, STATE_VARIANTS[DEFAULT_STATE_VARIANT])
    parts: list[int] = []

    if cfg["has_packet"]:
        parts.append(int(node.has_packet))
    if cfg["last_result_bin"]:
        parts.append(_last_result_bin(node.last_outcome))
    if cfg["retry_bin"]:
        parts.append(_retry_bin(node.retry_count))
    if cfg["sf_idx"]:
        parts.append(int(node.sf_idx))
    if cfg.get("sf_delta"):
        parts.append(int(sf_delta) + 1)  # -1→0, 0→1, +1→2
    if cfg["snr_margin_bin"]:
        margin_db = float(node.mean_snr_db) - float(SNR_THRESH[node.sf_idx])
        parts.append(_snr_margin_bin(margin_db))
    if cfg["channel_idx"]:
        parts.append(int(node.channel_idx))
    if cfg["gateway_g_bins"]:
        parts.extend(gateway_g_bins)
    if cfg["failure_level"]:
        parts.append(_failure_level(node.last_outcome, node.retry_count))
    if cfg["gateway_g_max"]:
        parts.append(max(gateway_g_bins) if gateway_g_bins else 1)

    return tuple(parts)


REWARD_VARIANTS: dict[str, dict] = {
    "v6_signal": {
        "label": "v6 기준선",
        "plot_label": "v6 Signal",
        "description": (
            "s2_compact 상태와 함께 사용 권장. "
            "retry_coef=0: failure_level이 상태에 있으므로 중복 페널티 불필요. "
            "switch_pen=0: channel_idx가 상태에 없으므로 불필요."
        ),
        "type": "standard",
        "success": 1.0,
        "fail": -1.0,
        "idle_pkt": -0.02,
        "idle_no_pkt": 0.0,
        "retry_coef": 0.0,
        "switch_pen": 0.0,
    },
    "v7_asymmetric": {
        "label": "v7 균형 최적화",
        "plot_label": "v7 Balanced",
        "description": (
            "ASR·처리량·공정성 균형 최적화. "
            "성공 보상 = 0.5 + 0.5/(1+norm) → [0.5, 1.0]. "
            "기저값(0.5)으로 처리량 인센티브 유지, 공정성 항(0.5/(1+norm))으로 균등화. "
            "fail=-1.0(ASR 보호), idle=-0.02(v6 동일)."
        ),
        "type": "composite",
        "success_base": 0.5,
        "success_fair": 0.5,
        "fail": -1.0,
        "idle_pkt": -0.02,
        "idle_no_pkt": 0.0,
        "retry_coef": 0.0,
        "switch_pen": 0.0,
    },
    "v8_fairness": {
        "label": "v8 공정성 강화",
        "plot_label": "v8 Fairness+",
        "description": (
            "공정성 우선 다목적 최적화. "
            "성공 보상 = 0.3 + 0.7/(1+norm) → [0.3, 1.0]. "
            "공정성 항 비중(0.7)이 가장 높아 성공 많은 노드 보상을 강하게 억제. "
            "fail=-1.0, idle=-0.02."
        ),
        "type": "composite",
        "success_base": 0.3,
        "success_fair": 0.7,
        "fail": -1.0,
        "idle_pkt": -0.02,
        "idle_no_pkt": 0.0,
        "retry_coef": 0.0,
        "switch_pen": 0.0,
    },
    "v9_log_thr": {
        "label": "v9 탐색 허용",
        "plot_label": "v9 Explore",
        "description": (
            "탐색 허용 다목적 최적화. "
            "성공 보상 = 0.6 + 0.4/(1+norm) → [0.6, 1.0]. "
            "fail=-0.7로 충돌 페널티를 완화해 더 넓은 SF/채널 탐색 허용. "
            "idle=-0.02, 공정성 항 포함."
        ),
        "type": "composite",
        "success_base": 0.6,
        "success_fair": 0.4,
        "fail": -0.7,
        "idle_pkt": -0.02,
        "idle_no_pkt": 0.0,
        "retry_coef": 0.0,
        "switch_pen": 0.0,
    },
    "v10_phase_idle": {
        "label": "v10 페이즈/IDLE 특화",
        "plot_label": "v10 Phase+Idle",
        "description": (
            "Phase Learning 및 전략적 IDLE 활용에 최적화. "
            "idle_pkt=+0.10: 패킷이 있어도 IDLE을 양수 보상으로 장려 → "
            "노드가 혼잡 슬롯 회피를 학습. "
            "fail=-0.7: 페널티 완화로 탐색 유지 (충돌을 너무 두려워하지 않음). "
            "success=1.0 고정. "
            "실험 결과: phase F=40 기준 v6 대비 fairness +0.05~0.10, "
            "throughput +1.0~1.5 향상. ASR은 소폭 하락(선택적 전송 증가 때문)."
        ),
        "type": "standard",
        "success": 1.0,
        "fail": -0.7,
        "idle_pkt": 0.10,
        "idle_no_pkt": 0.0,
        "retry_coef": 0.0,
        "switch_pen": 0.0,
    },
}

DEFAULT_REWARD_VARIANT = "v6_signal"


def compute_reward(
    outcome: int,
    retry_count_before: int,
    channel_changed: bool = False,
    has_packet: bool = True,
    variant: str = DEFAULT_REWARD_VARIANT,
    node_success_count: int = 0,
    mean_success_count: float = 1.0,
) -> float:
    """슬롯 결과에 대한 스칼라 보상을 계산한다.

    노드는 충돌과 링크 실패를 구분하지 못하므로 둘 다 실패로 취급한다.

    node_success_count: 이 노드의 에폭 내 성공 횟수 (v8, v9 variant에서 사용).
    mean_success_count: 전체 노드 평균 에폭 성공 횟수. v8/v9 정규화에 사용.
    """
    reward_cfg = REWARD_VARIANTS.get(variant, REWARD_VARIANTS[DEFAULT_REWARD_VARIANT])
    retry_penalty = float(reward_cfg["retry_coef"]) * float(retry_count_before)
    switch_penalty = float(reward_cfg["switch_pen"]) if channel_changed else 0.0
    rtype = reward_cfg.get("type", "standard")

    if outcome == OUTCOME_SUCCESS:
        if rtype in ("fairness", "log_thr", "composite"):
            # 전체 노드 평균 대비 정규화: 보상이 시뮬레이션 길이와 무관하게 안정 유지
            mean_n = max(1.0, float(mean_success_count))
            norm = float(node_success_count) / mean_n
            if rtype == "fairness":
                r = 1.0 / (1.0 + norm)
            elif rtype == "log_thr":
                r = math.log(norm + 2.0) - math.log(norm + 1.0)
            else:  # composite: base + fair/(1+norm)
                base = float(reward_cfg.get("success_base", 0.5))
                fair = float(reward_cfg.get("success_fair", 0.5))
                r = base + fair / (1.0 + norm)
        else:
            r = float(reward_cfg["success"])
        return r - switch_penalty
    if outcome in (OUTCOME_FAIL_COLLISION, OUTCOME_FAIL_LINK):
        return float(reward_cfg["fail"]) - retry_penalty - switch_penalty
    if has_packet:
        return float(reward_cfg["idle_pkt"])
    return float(reward_cfg["idle_no_pkt"])


def _new_q_table(n_act: int):
    return defaultdict(lambda: np.zeros(n_act, dtype=np.float64))


class _QLearningControllerBase:
    """공유 테이블형과 분산형 큐 학습이 함께 쓰는 공통 로직."""

    key = "q_learning"
    label = "Q-learning"
    shared_table = True

    def __init__(
        self,
        alpha: float = 0.1,
        gamma_q: float = 0.9,
        epsilon: float = 1.0,
        eps_min: float = 0.05,
        eps_decay: float = 0.9995,
        reward_variant: str = DEFAULT_REWARD_VARIANT,
        state_variant: str = DEFAULT_STATE_VARIANT,
        action_variant: str = DEFAULT_ACTION_VARIANT,
    ) -> None:
        self.alpha = alpha
        self.gamma_q = gamma_q
        self.initial_epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.reward_variant = reward_variant
        self.state_variant = state_variant
        self.action_variant = action_variant
        self._rng = random.Random()

        self._n_channels = 3
        self._n_actions = n_actions(3, action_variant)

        self._shared_q_table = _new_q_table(self._n_actions)
        self._shared_epsilon = epsilon
        self._node_q_tables: dict[int, defaultdict] = {}
        self._node_epsilons: dict[int, float] = {}
        self._node_ids: list[int] = []
        self._gateway_g_bins: tuple[int, ...] = (1, 1, 1)

    def begin_run(self, nodes, config, rng) -> None:
        self._rng = rng
        self._n_channels = config.n_channels
        self._n_actions = n_actions(config.n_channels, self.action_variant)
        self._gateway_g_bins = tuple(1 for _ in range(config.n_channels))
        self._node_ids = [node.node_id for node in nodes]
        self._epoch_slots = config.epoch_slots
        # v8/v9 표준 경로 전용 에폭 내 성공 카운터
        self._std_success_count: dict[int, int] = {nid: 0 for nid in self._node_ids}

        # s8_sf_delta 표준 경로: 노드별 직전 SF delta 추적 (-1/0/+1)
        self._sf_delta_map: dict[int, int] = {nid: 0 for nid in self._node_ids}

        if self.shared_table:
            self._shared_q_table = _new_q_table(self._n_actions)
            self._shared_epsilon = self.initial_epsilon
            self._node_q_tables = {}
            self._node_epsilons = {}
        else:
            self._shared_q_table = _new_q_table(self._n_actions)
            self._shared_epsilon = self.initial_epsilon
            self._node_q_tables = {
                node_id: _new_q_table(self._n_actions) for node_id in self._node_ids
            }
            self._node_epsilons = {
                node_id: self.initial_epsilon for node_id in self._node_ids
            }

    def _get_q_table(self, node_id: int):
        if self.shared_table:
            return self._shared_q_table
        return self._node_q_tables[node_id]

    def _get_epsilon(self, node_id: int) -> float:
        if self.shared_table:
            return self._shared_epsilon
        return self._node_epsilons[node_id]

    def choose_action(self, node, slot: int, gateway_info: dict | None = None) -> ControllerAction:
        if gateway_info is not None:
            self._gateway_g_bins = tuple(gateway_info.get("per_channel_g_bin", self._gateway_g_bins))

        q_table = self._get_q_table(node.node_id)
        epsilon = self._get_epsilon(node.node_id)
        sf_delta_prev = self._sf_delta_map.get(node.node_id, 0)
        state = encode_state(node, self._gateway_g_bins, self.state_variant, sf_delta=sf_delta_prev)

        if self._rng.random() < epsilon:
            internal_action = self._rng.randint(0, self._n_actions - 1)
        else:
            internal_action = int(np.argmax(q_table[state]))

        if self.action_variant == "absolute":
            transmit, abs_sf, channel_idx = decode_action_absolute(internal_action, self._n_channels)
            next_sf_idx = abs_sf if transmit else node.sf_idx
        else:
            transmit, sf_delta, channel_idx = decode_action(internal_action, self._n_channels)
            next_sf_idx = clamp_sf_index(node.sf_idx + sf_delta)

        # 다음 슬롯 상태 인코딩에 쓸 SF 변화 방향 저장 (IDLE이면 0, sign으로 {-1,0,+1} 보장)
        raw = next_sf_idx - node.sf_idx
        self._sf_delta_map[node.node_id] = (1 if raw > 0 else -1 if raw < 0 else 0) if transmit else 0

        return ControllerAction(
            transmit=transmit,
            sf_idx=next_sf_idx,
            channel_idx=channel_idx,
            prev_channel_idx=node.channel_idx,
            internal_action=internal_action,
            state=state,
            retry_count_before=node.retry_count,
        )

    def observe(self, node, action: ControllerAction, outcome: int, slot: int, gateway_info: dict | None = None) -> None:
        state = action.state
        internal_action = action.internal_action
        if state is None or internal_action is None:
            return

        # 다음 상태 계산이 맞도록 슬롯 종료 후의 게이트웨이 bin을 반영한다.
        if gateway_info is not None:
            self._gateway_g_bins = tuple(gateway_info.get("per_channel_g_bin", self._gateway_g_bins))

        channel_changed = action.transmit and action.channel_idx != action.prev_channel_idx
        nid = node.node_id
        node_cnt = self._std_success_count.get(nid, 0)
        mean_cnt = (sum(self._std_success_count.values()) / max(1, len(self._std_success_count)))
        reward = compute_reward(
            outcome=outcome,
            retry_count_before=action.retry_count_before,
            channel_changed=channel_changed,
            has_packet=node.has_packet,
            variant=self.reward_variant,
            node_success_count=node_cnt,
            mean_success_count=mean_cnt,
        )
        if outcome == OUTCOME_SUCCESS:
            self._std_success_count[nid] = node_cnt + 1
        next_state = encode_state(node, self._gateway_g_bins, self.state_variant,
                                   sf_delta=self._sf_delta_map.get(node.node_id, 0))

        q_table = self._get_q_table(node.node_id)
        current_q = q_table[state][internal_action]
        best_next = float(np.max(q_table[next_state]))
        td_error = reward + self.gamma_q * best_next - current_q
        q_table[state][internal_action] += self.alpha * td_error

    def end_slot(self, slot: int) -> None:
        # v8/v9: 에폭 경계마다 성공 카운터 리셋
        rtype = REWARD_VARIANTS.get(self.reward_variant, {}).get("type", "standard")
        if rtype in ("fairness", "log_thr", "composite") and (slot + 1) % self._epoch_slots == 0:
            for nid in self._node_ids:
                self._std_success_count[nid] = 0

        if self.shared_table:
            self._shared_epsilon = max(self.eps_min, self._shared_epsilon * self.eps_decay)
            return

        for node_id in self._node_ids:
            self._node_epsilons[node_id] = max(
                self.eps_min,
                self._node_epsilons[node_id] * self.eps_decay,
            )

    def get_q_arrays(self) -> dict[int, np.ndarray]:
        """각 노드의 큐 테이블을 조밀한 numpy 배열로 반환한다.

        방문한 상태만 포함하고, 아직 방문하지 않은 상태는 출력에서 제외한다.
        그래서 결과가 훨씬 작고 읽기 쉬워진다. 공유 테이블형은 {-1: array}
        형태로 반환한다.
        """
        result = {}
        if self.shared_table:
            table = self._shared_q_table
            if table:
                result[-1] = np.array(list(table.values()), dtype=np.float64)
        else:
            for node_id, table in self._node_q_tables.items():
                if table:
                    result[node_id] = np.array(list(table.values()), dtype=np.float64)
        return result

    def get_per_node_q_snapshot(self) -> dict[int, np.ndarray] | None:
        """노드별 Q-values(방문 상태 평균)를 반환한다.

        Returns: {node_id: np.ndarray (n_actions,)} or None
        공유 테이블형(centralized)은 per-node 구분이 없으므로 None을 반환한다.
        """
        if self.shared_table:
            return None
        out = {}
        for nid, table in self._node_q_tables.items():
            if table:
                out[nid] = np.mean(list(table.values()), axis=0)
        return out or None

    def get_mean_q_array(self) -> tuple[list[tuple], np.ndarray] | None:
        """(states, 노드 평균 Q-array)를 반환한다.

        데이터가 없으면 None을 반환한다. 분산형 컨트롤러는 모든 노드가
        방문한 상태의 합집합을 모아 상태별 Q값 평균을 계산한다.
        """
        if self.shared_table:
            table = self._shared_q_table
            if not table:
                return None
            states = list(table.keys())
            arr = np.array([table[s] for s in states], dtype=np.float64)
            return states, arr

        all_states: dict[tuple, list[np.ndarray]] = {}
        for table in self._node_q_tables.values():
            for state, qvals in table.items():
                all_states.setdefault(state, []).append(qvals)

        if not all_states:
            return None

        states = sorted(all_states.keys())
        arr = np.array(
            [np.mean(all_states[s], axis=0) for s in states],
            dtype=np.float64,
        )
        return states, arr


class DecentralizedQLearningController(_QLearningControllerBase):
    """노드별로 독립적인 큐 테이블을 쓰는 분산형 컨트롤러.

    s2_compact 상태 변형에서는 dense numpy Q-table을 사용한
    완전 벡터화 배치 경로(SUPPORTS_BATCH)를 제공한다.
    시뮬레이터가 SUPPORTS_BATCH를 감지하면 슬롯당 N번 개별 호출 대신
    choose_actions_batch / observe_batch를 호출해 속도를 크게 높인다.
    """

    key = "decentralized_q_learning"
    label = "Q-learning (decentralized)"
    shared_table = False

    # s2* 계열 공통 차원 상수
    _S2_N_SF = 6
    _S2_N_FL = 4   # failure_level 단계 수

    # 배치 경로를 지원하는 상태 변형 이름 집합 (상태 수는 n_bins에 따라 런타임에 결정)
    _S2_DENSE_VARIANTS: frozenset[str] = frozenset({
        "s2_compact", "s3_no_sf", "s4_no_fl", "s5_no_gmax", "s6_sf_only", "s7_sf_gch",
        "s8_sf_delta", "s9_sf_delta_gmax", "s10_sf_delta_fl", "s11_sf_delta_only",
        "s12_sf_ch_gmax", "s13_sf_ch", "s14_sf_delta_ch",
    })

    @classmethod
    def _compute_n_states(cls, variant: str, n_bins: int, n_channels: int) -> int:
        """상태 변형과 bin 수로부터 총 상태 수를 계산한다."""
        if variant == "s2_compact":  return cls._S2_N_SF * cls._S2_N_FL * n_bins
        if variant == "s3_no_sf":    return cls._S2_N_FL * n_bins
        if variant == "s4_no_fl":    return cls._S2_N_SF * n_bins
        if variant == "s5_no_gmax":  return cls._S2_N_SF * cls._S2_N_FL
        if variant == "s6_sf_only":  return cls._S2_N_SF
        if variant == "s7_sf_gch":         return cls._S2_N_SF * (n_bins ** n_channels)
        if variant == "s8_sf_delta":        return 3 * cls._S2_N_FL * n_bins   # SF_delta(3)×FL(4)×Gmax
        if variant == "s9_sf_delta_gmax":   return 3 * n_bins                  # SF_delta(3)×Gmax
        if variant == "s10_sf_delta_fl":    return 3 * cls._S2_N_FL            # SF_delta(3)×FL(4)
        if variant == "s11_sf_delta_only":  return 3                            # SF_delta(3)
        if variant == "s12_sf_ch_gmax":     return cls._S2_N_SF * n_channels * n_bins  # SF(6)×CH×Gmax
        if variant == "s13_sf_ch":          return cls._S2_N_SF * n_channels            # SF(6)×CH
        if variant == "s14_sf_delta_ch":    return 3 * n_channels                       # SF_delta(3)×CH
        return 0

    @property
    def SUPPORTS_BATCH(self) -> bool:
        return getattr(self, '_use_dense', False)

    def begin_run(self, nodes, config, rng) -> None:
        super().begin_run(nodes, config, rng)
        self._epoch_slots = config.epoch_slots
        self._n_bins = len(getattr(config, "gw_thresholds", (0.7, 1.3))) + 1
        self._n_channels = config.n_channels
        self._use_dense = self.state_variant in self._S2_DENSE_VARIANTS
        if self._use_dense:
            N = len(nodes)
            n_states = self._compute_n_states(self.state_variant, self._n_bins, self._n_channels)
            self._q_dense = np.zeros((N, n_states, self._n_actions), dtype=np.float64)
            self._epsilon_arr = np.full(N, self.initial_epsilon, dtype=np.float64)
            self._success_count_arr = np.zeros(N, dtype=np.int64)  # v8/v9 보상용 에폭 내 성공 수
            self._sf_delta_arr = np.zeros(N, dtype=np.int32)       # s8_sf_delta 배치 경로용
            # 배치 경로 전용 numpy RNG (컨트롤러 rng와 독립)
            np_seed = rng.randint(0, 2 ** 31)
            self._np_rng = np.random.RandomState(np_seed)

    # ------------------------------------------------------------------
    # 배치 경로 전용 메서드 (s2_compact + dense Q-table)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_failure_level(last_outcome_arr: np.ndarray,
                               retry_arr: np.ndarray) -> np.ndarray:
        """failure_level 배열을 계산한다 (0=성공/fresh, 1=첫실패, 2=2~3회, 3=4회+)."""
        fl = np.zeros(len(last_outcome_arr), dtype=np.int32)
        fresh_or_suc = (last_outcome_arr == OUTCOME_SUCCESS) | (retry_arr == 0)
        fl[~fresh_or_suc & (retry_arr == 1)] = 1
        fl[~fresh_or_suc & (retry_arr >= 2) & (retry_arr <= 3)] = 2
        fl[~fresh_or_suc & (retry_arr >= 4)] = 3
        return fl

    def _encode_batch(self, sf_arr: np.ndarray, last_outcome_arr: np.ndarray,
                      retry_arr: np.ndarray, g_max: int,
                      gateway_g_bins: tuple = (),
                      sf_delta_arr: np.ndarray | None = None,
                      ch_arr: np.ndarray | None = None) -> np.ndarray:
        """s2* 계열 상태를 variant에 따라 플랫 인덱스로 일괄 인코딩한다.

        s2_compact    : sf * 12 + fl * 3 + gm            (72 states)
        s3_no_sf      : fl * 3  + gm                     (12 states)
        s4_no_fl      : sf * 3  + gm                     (18 states)
        s5_no_gmax    : sf * 4  + fl                     (24 states)
        s6_sf_only    : sf                                ( 6 states)
        s7_sf_gch     : sf * 27 + g0 * 9 + g1 * 3 + g2  (162 states, n_channels=3)
        s8_sf_delta   : sd * 12 + fl * 3 + gm            (36 states, sd=delta+1)
        s12_sf_ch_gmax: sf * (n_ch * nb) + ch * nb + gm  (54 states, n_ch=3, nb=3)
        """
        v = self.state_variant
        nb = self._n_bins
        fl = self._compute_failure_level(last_outcome_arr, retry_arr)
        if v == "s2_compact":
            idx = sf_arr * (self._S2_N_FL * nb) + fl * nb + g_max
        elif v == "s3_no_sf":
            idx = fl * nb + g_max
        elif v == "s4_no_fl":
            idx = sf_arr * nb + g_max
        elif v == "s5_no_gmax":
            idx = sf_arr * self._S2_N_FL + fl
        elif v == "s6_sf_only":
            idx = sf_arr
        elif v == "s12_sf_ch_gmax":
            ch = ch_arr if ch_arr is not None else np.zeros_like(sf_arr)
            idx = sf_arr * (self._n_channels * nb) + ch * nb + g_max
        elif v == "s13_sf_ch":
            ch = ch_arr if ch_arr is not None else np.zeros_like(sf_arr)
            idx = sf_arr * self._n_channels + ch
        elif v == "s14_sf_delta_ch":
            ch = ch_arr if ch_arr is not None else np.zeros_like(sf_arr)
            sd = (sf_delta_arr if sf_delta_arr is not None else np.zeros_like(sf_arr)) + 1
            idx = sd * self._n_channels + ch
        elif v in ("s8_sf_delta", "s9_sf_delta_gmax", "s10_sf_delta_fl", "s11_sf_delta_only"):
            sd = (sf_delta_arr if sf_delta_arr is not None else np.zeros_like(sf_arr)) + 1  # -1→0, 0→1, +1→2
            if v == "s8_sf_delta":
                idx = sd * (self._S2_N_FL * nb) + fl * nb + g_max
            elif v == "s9_sf_delta_gmax":
                idx = sd * nb + g_max
            elif v == "s10_sf_delta_fl":
                idx = sd * self._S2_N_FL + fl
            else:  # s11_sf_delta_only
                idx = sd
        else:  # s7_sf_gch — 채널별 bin 개별 인코딩
            # sf * nb^n_ch + g0 * nb^(n_ch-1) + g1 * nb^(n_ch-2) + ... + g_{n_ch-1}
            idx = sf_arr * (nb ** self._n_channels)
            for c in range(self._n_channels):
                g_c = int(gateway_g_bins[c]) if c < len(gateway_g_bins) else 0
                idx = idx + g_c * (nb ** (self._n_channels - 1 - c))
        return idx.astype(np.int32)

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """active_ids 노드에 대해 액션을 벡터화하여 선택한다.

        Returns
        -------
        state_idx    : (n_active,) int32  — 현재 상태 플랫 인덱스
        int_actions  : (n_active,) int32  — 선택된 액션 ID
        transmit     : (n_active,) bool   — 전송 여부
        new_sf       : (n_active,) int32  — 적용할 SF 인덱스
        new_ch       : (n_active,) int32  — 적용할 채널 인덱스
        """
        n_active = len(active_ids)
        if n_active == 0:
            empty_i = np.array([], dtype=np.int32)
            return empty_i, empty_i, np.array([], dtype=bool), empty_i, empty_i

        state_idx = self._encode_batch(
            sf_arr[active_ids], last_outcome_arr[active_ids], retry_arr[active_ids], g_max, gateway_g_bins,
            sf_delta_arr=self._sf_delta_arr[active_ids] if hasattr(self, '_sf_delta_arr') else None,
            ch_arr=ch_arr[active_ids],
        )

        q_vals = self._q_dense[active_ids, state_idx]           # (n_active, A)
        greedy = q_vals.argmax(axis=1).astype(np.int32)

        eps = self._epsilon_arr[active_ids]
        explore = self._np_rng.random(n_active) < eps
        rand_acts = self._np_rng.randint(0, self._n_actions, n_active).astype(np.int32)
        int_actions = np.where(explore, rand_acts, greedy).astype(np.int32)

        transmit = (int_actions != ACTION_IDLE)
        new_ch = np.where(
            int_actions > 0,
            (int_actions - 1) % self._n_channels,
            ch_arr[active_ids],
        ).astype(np.int32)
        if self.action_variant == "absolute":
            # 절대 SF: (action-1) // n_ch 가 SF 인덱스 0~5
            new_sf = np.where(
                int_actions > 0,
                (int_actions - 1) // self._n_channels,
                sf_arr[active_ids],
            ).astype(np.int32)
        else:
            # 상대 SF: (action-1) // n_ch → sf_change_idx → delta (0, +1, -1)
            sf_change_idx = np.where(int_actions > 0, (int_actions - 1) // self._n_channels, 0)
            sf_deltas = np.array([0, 1, -1], dtype=np.int32)[sf_change_idx]
            new_sf = np.clip(sf_arr[active_ids] + sf_deltas, 0, 5).astype(np.int32)

        # s8_sf_delta 배치 경로: 다음 슬롯 상태 인코딩용 delta 저장 (IDLE→0)
        # np.sign으로 방향만 저장 → absolute 액션 변형에서도 {-1, 0, +1} 보장
        if hasattr(self, '_sf_delta_arr'):
            applied_delta = np.sign(new_sf - sf_arr[active_ids]).astype(np.int32)
            self._sf_delta_arr[active_ids] = np.where(transmit, applied_delta, 0)

        return state_idx, int_actions, transmit, new_sf, new_ch

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
        """active_ids 노드에 대해 TD 업데이트를 벡터화하여 수행한다."""
        n_active = len(active_ids)
        if n_active == 0:
            return

        active_outcomes = outcomes_arr[active_ids]
        r_cfg = REWARD_VARIANTS.get(self.reward_variant, REWARD_VARIANTS[DEFAULT_REWARD_VARIANT])
        rtype = r_cfg.get("type", "standard")

        rewards = np.full(n_active, float(r_cfg['idle_pkt']), dtype=np.float64)
        s_mask = active_outcomes == OUTCOME_SUCCESS
        f_mask = (active_outcomes == OUTCOME_FAIL_COLLISION) | (active_outcomes == OUTCOME_FAIL_LINK)
        no_pkt_mask = (active_outcomes == OUTCOME_IDLE) & (queue_arr[active_ids] == 0)

        # 성공 보상 계산 (type에 따라 분기)
        if rtype in ("fairness", "log_thr", "composite"):
            # 전체 노드 평균 대비 정규화: 보상이 시뮬레이션 길이와 무관하게 안정 유지
            mean_n = max(1.0, float(self._success_count_arr.mean()))
            suc_ids = active_ids[s_mask]
            counts = self._success_count_arr[suc_ids].astype(np.float64)
            norm = counts / mean_n  # 평균 대비 상대 성공 비율
            if rtype == "fairness":
                rewards[s_mask] = 1.0 / (1.0 + norm)
            elif rtype == "log_thr":
                rewards[s_mask] = np.log(norm + 2.0) - np.log(norm + 1.0)
            else:  # composite: base + fair/(1+norm)
                base = float(r_cfg.get("success_base", 0.5))
                fair = float(r_cfg.get("success_fair", 0.5))
                rewards[s_mask] = base + fair / (1.0 + norm)
        else:
            rewards[s_mask] = float(r_cfg['success'])

        retry_pen = float(r_cfg['retry_coef']) * retry_before_active.astype(np.float64)
        rewards[f_mask] = float(r_cfg['fail']) - retry_pen[f_mask]
        rewards[no_pkt_mask] = float(r_cfg['idle_no_pkt'])

        # 누적 성공 카운터 업데이트 (보상 계산 이후)
        self._success_count_arr[active_ids[s_mask]] += 1

        g_max_next = int(max(gateway_g_bins)) if gateway_g_bins else 1
        next_state_idx = self._encode_batch(
            sf_arr[active_ids], last_outcome_arr[active_ids], retry_arr[active_ids], g_max_next, gateway_g_bins,
            sf_delta_arr=self._sf_delta_arr[active_ids] if hasattr(self, '_sf_delta_arr') else None,
            ch_arr=ch_arr[active_ids] if ch_arr is not None else None,
        )

        current_q = self._q_dense[active_ids, state_idx, int_actions]
        best_next = self._q_dense[active_ids, next_state_idx].max(axis=1)
        td_errors = rewards + self.gamma_q * best_next - current_q
        self._q_dense[active_ids, state_idx, int_actions] += self.alpha * td_errors

    def end_slot(self, slot: int) -> None:
        if self._use_dense:
            np.maximum(self.eps_min, self._epsilon_arr * self.eps_decay, out=self._epsilon_arr)
            # v8/v9: 에폭 경계마다 성공 카운터 리셋 (보상 크기가 시뮬레이션 길이에 비례해 0으로 붕괴하는 것을 방지)
            rtype = REWARD_VARIANTS.get(self.reward_variant, {}).get("type", "standard")
            if rtype in ("fairness", "log_thr", "composite") and (slot + 1) % self._epoch_slots == 0:
                self._success_count_arr[:] = 0
            return
        super().end_slot(slot)

    def get_per_node_q_snapshot(self) -> dict[int, np.ndarray] | None:
        if self._use_dense and hasattr(self, "_q_dense"):
            N = self._q_dense.shape[0]
            return {i: self._q_dense[i].mean(axis=0) for i in range(N)}
        return super().get_per_node_q_snapshot()

    def get_mean_q_array(self) -> tuple[list[tuple], np.ndarray] | None:
        if self._use_dense:
            v = self.state_variant
            nb = self._n_bins
            if v == "s2_compact":
                states = [(sf, fl, gm)
                          for sf in range(self._S2_N_SF)
                          for fl in range(self._S2_N_FL)
                          for gm in range(nb)]
            elif v == "s3_no_sf":
                states = [(fl, gm)
                          for fl in range(self._S2_N_FL)
                          for gm in range(nb)]
            elif v == "s4_no_fl":
                states = [(sf, gm)
                          for sf in range(self._S2_N_SF)
                          for gm in range(nb)]
            elif v == "s5_no_gmax":
                states = [(sf, fl)
                          for sf in range(self._S2_N_SF)
                          for fl in range(self._S2_N_FL)]
            elif v == "s6_sf_only":
                states = [(sf,) for sf in range(self._S2_N_SF)]
            elif v == "s8_sf_delta":
                states = [(sd - 1, fl, gm)
                          for sd in range(3)
                          for fl in range(self._S2_N_FL)
                          for gm in range(nb)]
            elif v == "s9_sf_delta_gmax":
                states = [(sd - 1, gm)
                          for sd in range(3)
                          for gm in range(nb)]
            elif v == "s10_sf_delta_fl":
                states = [(sd - 1, fl)
                          for sd in range(3)
                          for fl in range(self._S2_N_FL)]
            elif v == "s11_sf_delta_only":
                states = [(sd - 1,) for sd in range(3)]
            elif v == "s12_sf_ch_gmax":
                states = [(sf, ch, gm)
                          for sf in range(self._S2_N_SF)
                          for ch in range(self._n_channels)
                          for gm in range(nb)]
            elif v == "s13_sf_ch":
                states = [(sf, ch)
                          for sf in range(self._S2_N_SF)
                          for ch in range(self._n_channels)]
            elif v == "s14_sf_delta_ch":
                states = [(sd - 1, ch)
                          for sd in range(3)
                          for ch in range(self._n_channels)]
            else:  # s7_sf_gch
                from itertools import product
                states = [(sf, *gs)
                          for sf in range(self._S2_N_SF)
                          for gs in product(range(nb), repeat=self._n_channels)]
            mean_q = self._q_dense.mean(axis=0)
            return states, mean_q
        return super().get_mean_q_array()
