"""재시도 인지 슬롯 ALOHA 기준선 사양."""

from __future__ import annotations

from agents.rule_based import RetryAwareController


def make_retry_aware_controller(margin_db: float = 1.0, base_window: int = 2, max_window: int = 64):
    return RetryAwareController(
        margin_db=margin_db,
        base_window=base_window,
        max_window=max_window,
    )
