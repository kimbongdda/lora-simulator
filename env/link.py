"""충돌 중심 LoRaWAN 실험을 위한 링크 및 ADR 보조 함수."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


SNR_THRESH = np.array([-7.5, -10.0, -12.5, -15.0, -17.5, -20.0], dtype=np.float64)
# 단순화한 경로손실 모델에 쓰는 기본 잡음/송신 전력 가정.
NOISE_FLOOR_DBM: float = -119.0
CELL_EDGE_MARGIN_DB: float = 2.0
P_TX_DBM: float = 14.0
GL_DB: float = 0.0


@dataclass(frozen=True)
class RadioProfile:
    name: str
    d0: float
    lpl_d0: float
    gamma: float
    description: str


PROFILE_SHORT = RadioProfile("short", 1.0, 40.0, 3.5, "Urban short-range")
PROFILE_MEDIUM = RadioProfile("medium", 1.0, 37.0, 3.0, "Suburban medium-range")
PROFILE_LONG = RadioProfile("long", 1.0, 34.0, 2.5, "Rural long-range")
PROFILE_LORASIM = RadioProfile("lorasim", 40.0, 127.41, 2.08, "LoRaSim paper-based (d0=40m, lpl=127.41dB, γ=2.08)")

ALL_PROFILES = {
    "short": PROFILE_SHORT,
    "medium": PROFILE_MEDIUM,
    "long": PROFILE_LONG,
    "lorasim": PROFILE_LORASIM,
}


def apply_rayleigh_fading(mean_snr_db: float, rng, fade_margin_db: float = 0.0) -> float:
    """Rayleigh fading 적용 후 순시 SNR(dB)를 반환한다.

    g = |h|² ~ Exp(mean=1) 를 linear SNR에 곱한 뒤 dB로 변환한다.
    fade_margin_db만큼 평균 SNR을 올린 뒤 페이딩을 적용하므로
    아웃에이지 확률을 실용적인 수준으로 낮출 수 있다.
    rng는 Python standard random.Random 인스턴스.
    """
    mean_linear = 10.0 ** ((mean_snr_db + fade_margin_db) / 10.0)
    g = rng.expovariate(1.0)
    return 10.0 * math.log10(mean_linear * g)


def link_success(margin_db: float) -> bool:
    """SNR 여유분이 0 이상이면 패킷 성공, 미만이면 실패 (하드 임계값 모델).

    P_rx >= SNR_THRESH[sf_idx]  →  True  (성공)
    P_rx <  SNR_THRESH[sf_idx]  →  False (링크 실패)
    """
    return float(margin_db) >= 0.0


def compute_cell_radius(
    profile: RadioProfile,
    sf_idx: int = 5,
    edge_margin_db: float = CELL_EDGE_MARGIN_DB,
    p_tx_dbm: float = P_TX_DBM,
    gl_db: float = GL_DB,
) -> float:
    """선택한 SF가 간신히 동작 가능한 반경을 계산한다."""
    # 선택한 SF가 셀 가장자리에서도 목표 SNR 기준을 조금 넘도록 반경을 계산한다.
    target_snr = float(SNR_THRESH[sf_idx]) + float(edge_margin_db)
    target_rssi = NOISE_FLOOR_DBM + target_snr
    lpl_target = p_tx_dbm - gl_db - target_rssi
    exponent = (lpl_target - profile.lpl_d0) / (10.0 * profile.gamma)
    return profile.d0 * (10.0 ** exponent)


def compute_mean_rssi(dist_m: float, profile: RadioProfile) -> float:
    """주어진 거리에서의 평균 RSSI를 계산한다."""
    dist_m = max(float(dist_m), profile.d0)
    lpl = profile.lpl_d0 + 10.0 * profile.gamma * math.log10(dist_m / profile.d0)
    return P_TX_DBM - GL_DB - lpl


def compute_mean_snr(dist_m: float, profile: RadioProfile) -> float:
    """주어진 거리에서의 평균 SNR을 계산한다."""
    return compute_mean_rssi(dist_m, profile) - NOISE_FLOOR_DBM


def simple_distance_sf_index(dist_m: float, cell_radius_m: float) -> int:
    """
    Map distance to a fixed SF by linearly dividing the cell radius into 6 bins.

    이 함수는 이전 코드베이스의 단순 초기화 규칙을 그대로 따른다.
    즉, ADR을 쓰지 않는 기하학적 기준선으로 활용한다.
    """
    if cell_radius_m <= 0:
        return 0
    # 의도적으로 거친 기하학적 휴리스틱이다. 더 먼 노드일수록 더 큰 SF를 받는다.
    return int(np.clip(math.floor(float(dist_m) / float(cell_radius_m) * 6.0), 0, 5))


def adr_static_sf_index(mean_snr_db: float, margin_db: float = 1.0) -> int:
    """
    목표 SNR 여유를 만족하는 가장 빠른 SF를 고른다.

    이 함수는 표준 ADR 구현이 아니라 ADR의 아이디어를 흉내 낸 휴리스틱이다.
    네트워크가 SNR/RSSI 여유를 보고 데이터율을 조정해 용량을 높인다는
    LoRaWAN ADR의 동작을 단순화해 반영한다.
    """
    # 노드의 SNR로 만족 가능한 가장 빠른 SF를 고른다.
    required = SNR_THRESH + float(margin_db)
    feasible = np.where(float(mean_snr_db) >= required)[0]
    if feasible.size == 0:
        return 5
    return int(feasible[0])
