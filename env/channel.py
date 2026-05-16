"""충돌 중심 LoRaWAN 설정에서 쓰는 채널/자원 보조 함수."""

from __future__ import annotations


N_SF = 6
MIN_SF = 7
MAX_SF = 12

# attempt 모드: 전송 시도 기반 G, 상한 없음. G=1 피크를 중심 구간에 배치.
G_LOW_THRESHOLD_ATTEMPT: float = 0.7
G_HIGH_THRESHOLD_ATTEMPT: float = 1.3

# success 모드: 성공 수신 기반 G, 상한 = 1.0 (슬롯당 최대 N_SF개 성공).
# [0, 1] 구간을 균등 3분할. 슬롯 ALOHA 최적 처리량(≈1/e≈0.37)이 Mid 구간에 오도록 설계.
G_LOW_THRESHOLD_SUCCESS: float = 0.33
G_HIGH_THRESHOLD_SUCCESS: float = 0.67

# 하위 호환: attempt 모드 임계값을 기본으로 유지.
G_LOW_THRESHOLD: float = G_LOW_THRESHOLD_ATTEMPT
G_HIGH_THRESHOLD: float = G_HIGH_THRESHOLD_ATTEMPT

# 시뮬레이터가 채널별 G를 추정할 때 쓰는 롤링 윈도우 길이(슬롯).
GATEWAY_WINDOW: int = 50


def total_resources(n_channels: int, n_sf: int = N_SF) -> int:
    """직교 자원(SF × 채널)의 총 개수를 반환한다."""
    return int(n_channels) * int(n_sf)


def arrival_probability_from_g(target_g: float, n_nodes: int, n_channels: int) -> float:
    """
    명목 제공 부하 G를 베르누이 도착 확률로 바꾼다.

    이 환경에서는 제공 부하 G를 다음처럼 정의한다.

        G = N * p_arrival / (N_SF * n_channels)

    시뮬레이터가 SF별 충돌 분리를 명시적으로 모델링하므로,
    SF × 채널 수로 정규화하는 해석이 가장 일관적이다.
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    # 제공 부하는 직교 자원(SF × 채널) 수로 정규화한다.
    # 패킷은 같은 자원을 경쟁할 때만 충돌하므로 이 기준이 가장 일관적이다.
    n_resources = total_resources(n_channels)
    return min(max(float(target_g), 0.0) * float(n_resources) / float(n_nodes), 1.0)


def clamp_sf_index(sf_idx: int) -> int:
    """SF 인덱스를 유효한 LoRa 범위 SF7..SF12 -> index 0..5로 제한한다."""
    return max(0, min(N_SF - 1, int(sf_idx)))


def quantize_g(g_val: float, thresholds: tuple[float, ...] = (G_LOW_THRESHOLD_ATTEMPT, G_HIGH_THRESHOLD_ATTEMPT)) -> int:
    """채널별 부하 추정값을 순서형 bin으로 양자화한다.

    thresholds는 bin 경계값 튜플 (오름차순). bin 수 = len(thresholds) + 1.

    기본값 (0.7, 1.3) — attempt 모드 3-bin:
        0  low  (G < 0.7)
        1  mid  (0.7 <= G < 1.3)
        2  high (G >= 1.3)

    예: thresholds=(0.5, 1.0, 1.5) → 4-bin
    """
    g = float(g_val)
    for i, t in enumerate(thresholds):
        if g < t:
            return i
    return len(thresholds)
