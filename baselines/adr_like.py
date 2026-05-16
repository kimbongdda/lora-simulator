"""ADR 유사 기준선 사양."""

from __future__ import annotations

from agents.rule_based import AdrLikeController


def make_adr_like_controller(margin_db: float = 1.0):
    return AdrLikeController(margin_db=margin_db)
