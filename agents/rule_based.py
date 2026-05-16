"""규칙 기반 기준선 컨트롤러."""

from __future__ import annotations

from env.link import adr_static_sf_index
from env.types import ControllerAction, OUTCOME_FAIL_COLLISION, OUTCOME_FAIL_LINK, OUTCOME_SUCCESS


class _BaseController:
    key = "base"
    label = "Base"

    def begin_run(self, nodes, config, rng) -> None:
        self.nodes = nodes
        self.config = config
        self.rng = rng
        # 도착 확률이 이미 G를 조절하므로, 전송 여부는 패킷 보유 상태만 본다.
        # 즉, 패킷이 있으면 매 슬롯 전송 시도 확률을 1.0으로 둔다.
        self.tx_probability = 1.0

    def observe(self, node, action: ControllerAction, outcome: int, slot: int, **_) -> None:
        return None

    def end_slot(self, slot: int) -> None:
        return None

    def _random_channel(self) -> int:
        return self.rng.randrange(self.config.n_channels)


class PureAlohaController(_BaseController):
    """
    순수 ALOHA 형태의 기준선.

    각 슬롯마다 SF와 채널을 완전히 무작위로 고른다.
    즉, 6개 SF와 n_channels개의 채널이 만드는 모든 자원에 대해
    랜덤 접근을 하는 가장 단순한 기준선이다.
    """

    key = "pure_aloha"
    label = "Pure ALOHA"

    def choose_action(self, node, slot: int, gateway_info: dict | None = None) -> ControllerAction:
        return ControllerAction(
            transmit=self.rng.random() < self.tx_probability,
            sf_idx=self.rng.randrange(6),
            channel_idx=self._random_channel(),
            prev_channel_idx=node.channel_idx,
        )


class AdrLikeController(_BaseController):
    """
    ADR 유사 기준선.

    각 노드의 평균 SNR을 기준으로 고정 SF를 하나 배정하고,
    실험이 끝날 때까지 그 SF를 유지한다.
    채널은 매 슬롯 무작위로 선택한다.
    """

    key = "adr_like"
    label = "ADR-like"

    def __init__(self, margin_db: float = 1.0) -> None:
        self.margin_db = margin_db

    def begin_run(self, nodes, config, rng) -> None:
        super().begin_run(nodes, config, rng)
        # 노드별 평균 SNR을 보고 SF를 배정해 여러 SF로 분산시킨다.
        # 이렇게 하면 6개의 SF와 채널 조합을 더 폭넓게 사용하게 된다.
        for node in nodes:
            node.sf_idx = adr_static_sf_index(node.mean_snr_db, margin_db=self.margin_db)

    def choose_action(self, node, slot: int, gateway_info: dict | None = None) -> ControllerAction:
        return ControllerAction(
            transmit=self.rng.random() < self.tx_probability,
            sf_idx=node.sf_idx,
            channel_idx=self._random_channel(),
            prev_channel_idx=node.channel_idx,
        )


class RetryAwareController(_BaseController):
    """
    슬롯 기반 ALOHA의 재시도 인지 휴리스틱.

    노드는 ADR 유사 SF를 고정으로 사용하고,
    실패할 때마다 retry 횟수에 따라 랜덤 백오프 창을 늘린다.
    채널은 매 슬롯 무작위로 고른다.
    """

    key = "retry_aware"
    label = "Retry-aware"

    def __init__(self, margin_db: float = 1.0, base_window: int = 2, max_window: int = 64) -> None:
        self.margin_db = margin_db
        self.base_window = base_window
        self.max_window = max_window

    def begin_run(self, nodes, config, rng) -> None:
        super().begin_run(nodes, config, rng)
        for node in nodes:
            node.sf_idx = adr_static_sf_index(node.mean_snr_db, margin_db=self.margin_db)

    def choose_action(self, node, slot: int, gateway_info: dict | None = None) -> ControllerAction:
        return ControllerAction(
            transmit=self.rng.random() < self.tx_probability,
            sf_idx=node.sf_idx,
            channel_idx=self._random_channel(),
            prev_channel_idx=node.channel_idx,
        )

    def observe(self, node, action: ControllerAction, outcome: int, slot: int, **_) -> None:
        if outcome == OUTCOME_SUCCESS:
            node.backoff_slots = 0
            return

        if outcome not in (OUTCOME_FAIL_COLLISION, OUTCOME_FAIL_LINK):
            return

        # 실패가 반복될수록 백오프 창을 지수적으로 넓혀 충돌 재발을 줄인다.
        exponent = max(0, min(node.retry_count - 1, 5))
        window = min(self.max_window, self.base_window * (2 ** exponent))
        node.backoff_slots = self.rng.randint(0, window)
