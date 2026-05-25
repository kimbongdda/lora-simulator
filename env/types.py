"""시뮬레이터와 컨트롤러가 함께 쓰는 공통 데이터 타입."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .channel import arrival_probability_from_g, total_resources


# 시뮬레이터와 모든 컨트롤러가 공통으로 쓰는 슬롯 단위 결과 코드.
# 0은 전송 없음, 1은 성공, 나머지 두 코드는 충돌 실패와 링크 실패를 분리한다.
OUTCOME_IDLE = 0
OUTCOME_SUCCESS = 1
OUTCOME_FAIL_COLLISION = 2
OUTCOME_FAIL_LINK = 3


@dataclass(frozen=True)
class ScenarioConfig:
    n_nodes: int
    target_g: float
    n_slots: int
    warmup_slots: int
    epoch_slots: int
    n_channels: int = 3
    fixed_distance_ratio: float = 0.5
    profile_name: str = "short"
    seed: int = 42
    layout_seed: int | None = None  # 노드 배치 전용 시드. None이면 seed와 동일하게 사용.
    queue_mode: str = "accumulate"  # "accumulate"는 적재를 유지하고, "fresh"는 슬롯마다 이전 패킷을 버린다.
    layout: str = "ring"            # "ring"은 같은 반경 배치, "random"은 셀 내부 무작위 배치.
    gw_obs_mode: str = "attempt"    # "attempt": 전송 시도 수 기반 (현재), "success": 성공 수만 기반 (현실적)
    gw_thresholds: tuple[float, ...] = (0.7, 1.3)  # GW 혼잡도 양자화 bin 경계값. len+1 = bin 수.
    enable_rayleigh_fading: bool = False            # True면 순시 Rayleigh fading SNR 적용.
    rayleigh_fade_margin_db: float = 10.0          # 페이딩 마진(dB). 평균 SNR에 더한 뒤 페이딩 적용.
    enable_rician_fading: bool = False             # True면 Rician fading 적용. Rayleigh보다 우선.
    rician_k_factor: float = 4.0                  # Rician K-factor (LOS/산란 전력 비). K≈4: 교외, K≈10: 개활지.
    rician_fade_margin_db: float = 5.0            # Rician용 페이딩 마진(dB). Rayleigh보다 작아도 됨.

    @property
    def n_resources(self) -> int:
        return total_resources(self.n_channels)

    @property
    def raw_p_arrival(self) -> float:
        if self.n_nodes <= 0:
            raise ValueError("n_nodes must be positive")
        return float(self.target_g) * float(self.n_resources) / float(self.n_nodes)

    @property
    def p_arrival(self) -> float:
        return arrival_probability_from_g(self.target_g, self.n_nodes, self.n_channels)

    @property
    def p_arrival_is_clipped(self) -> bool:
        return self.raw_p_arrival > 1.0

    @property
    def measured_slots(self) -> int:
        return max(0, self.n_slots - self.warmup_slots)


@dataclass
class NodeState:
    node_id: int
    x: float
    y: float
    dist_m: float
    mean_snr_db: float
    sf_idx: int
    queue_len: int = 0
    retry_count: int = 0  # 큐 맨 앞 패킷 기준 연속 실패 횟수.
    backoff_slots: int = 0  # 다음 전송 시도까지 남은 백오프 슬롯 수.
    last_outcome: int = OUTCOME_IDLE
    generated_packets: int = 0
    attempts_measured: int = 0
    successes_measured: int = 0
    channel_idx: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sf(self) -> int:
        return self.sf_idx + 7

    @property
    def has_packet(self) -> bool:
        return self.queue_len > 0


@dataclass
class ControllerAction:
    transmit: bool
    sf_idx: int
    channel_idx: int = 0
    prev_channel_idx: int = 0  # 이번 결정 직전에 사용하던 채널. 채널 변경 패널티 계산에 필요하다.
    internal_action: int | None = None
    state: tuple | None = None
    retry_count_before: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScheduledTransmission:
    node_id: int
    sf_idx: int
    channel_idx: int
    mean_snr_db: float
