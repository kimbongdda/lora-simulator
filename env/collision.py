"""슬롯 기반 LoRaWAN 자원의 충돌 판정 로직."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .link import SNR_THRESH, apply_rayleigh_fading, link_success
from .types import OUTCOME_FAIL_COLLISION, OUTCOME_FAIL_LINK, OUTCOME_SUCCESS, ScheduledTransmission


def evaluate_transmissions(
    transmissions: Iterable[ScheduledTransmission],
    rng,
    enable_rayleigh_fading: bool = False,
    fade_margin_db: float = 0.0,
):
    """한 슬롯에 발생한 전송들을 평가한다.

    같은 슬롯에서 두 개 이상의 노드가 같은 (SF, 채널) 자원을 쓰면
    먼저 충돌로 판정한다. 충돌이 없을 때만 SNR 여유분 기반의
    부드러운 성공 확률 모델을 적용한다.
    """
    transmissions = list(transmissions)
    if not transmissions:
        return {}, 0, 0

    # 이 슬롯에서 각 직교 자원(SF, 채널)을 몇 개 노드가 선택했는지 센다.
    # 충돌 여부는 SF와 채널이 둘 다 같은지로만 판단한다.
    counts = Counter((tx.sf_idx, tx.channel_idx) for tx in transmissions)
    outcomes = {}
    collisions = 0
    link_failures = 0

    for tx in transmissions:
        if counts[(tx.sf_idx, tx.channel_idx)] > 1:
            # 같은 (SF, 채널)을 여러 노드가 쓰면 링크 모델을 보기 전에 충돌로 처리한다.
            outcomes[tx.node_id] = OUTCOME_FAIL_COLLISION
            collisions += 1
            continue

        # 충돌 없는 전송: P_rx >= SNR_THRESH[sf] 이면 성공, 미만이면 링크 실패.
        snr_db = apply_rayleigh_fading(tx.mean_snr_db, rng, fade_margin_db) if enable_rayleigh_fading else tx.mean_snr_db
        margin_db = float(snr_db) - float(SNR_THRESH[tx.sf_idx])
        if link_success(margin_db):
            outcomes[tx.node_id] = OUTCOME_SUCCESS
        else:
            outcomes[tx.node_id] = OUTCOME_FAIL_LINK
            link_failures += 1

    return outcomes, collisions, link_failures
