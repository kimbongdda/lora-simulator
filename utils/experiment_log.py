"""실험 로그 기록/조회 유틸리티.

모든 실험 실행 결과를 outputs/experiment_log.csv 에 누적 저장한다.
각 행은 실험 하나(또는 비교 모드에서는 변형 하나)를 나타낸다.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any

LOG_PATH = os.path.join("outputs", "experiment_log.csv")

COLUMNS = [
    "timestamp",
    "mode",
    "n_nodes",
    "target_g",
    "n_slots",
    "warmup_slots",
    "n_channels",
    "layout",
    "profile",
    "reward_variant",
    "state_variant",
    "action_variant",
    "frame_size",
    "series_label",       # 비교 모드에서 각 계열 이름
    "asr",
    "throughput",
    "fairness",
    "collision_rate",
    "mean_backlog",
    "final_backlog",
    "attempts",
    "notes",
    "output_dir",
]


def _ensure_log() -> None:
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()


def _row(**kwargs: Any) -> dict:
    """COLUMNS 에 맞춰 빈 값을 채운 행 딕셔너리를 반환한다."""
    base = {c: "" for c in COLUMNS}
    base["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base.update({k: v for k, v in kwargs.items() if k in COLUMNS})
    return base


def append_rows(rows: list[dict]) -> None:
    """행 목록을 로그 파일에 추가한다."""
    if not rows:
        return
    _ensure_log()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        for row in rows:
            writer.writerow(row)


def read_log() -> list[dict]:
    """로그 전체를 읽어 행 목록으로 반환한다. 파일 없으면 빈 리스트."""
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------
# 모드별 로그 빌더
# ------------------------------------------------------------------

def _base(mode: str, cfg_dict: dict) -> dict:
    return dict(
        mode=mode,
        n_nodes=cfg_dict.get("n_nodes", ""),
        target_g=cfg_dict.get("target_g", ""),
        n_slots=cfg_dict.get("n_slots", ""),
        warmup_slots=cfg_dict.get("warmup_slots", ""),
        n_channels=cfg_dict.get("n_channels", ""),
        layout=cfg_dict.get("layout", ""),
        profile=cfg_dict.get("profile_name", ""),
        output_dir=cfg_dict.get("output_dir", ""),
    )


def log_single(mode: str, cfg, result: dict, notes: str = "") -> None:
    """단일 결과 모드 (Node Analysis, Epoch Timeseries)."""
    import dataclasses
    cd = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    row = _row(
        **_base(mode, cd),
        reward_variant=cd.get("reward_variant", ""),
        state_variant=cd.get("state_variant", ""),
        series_label=mode,
        asr=f"{result.get('success_rate', ''):.4f}" if result.get("success_rate") is not None else "",
        throughput=f"{result.get('throughput', ''):.4f}" if result.get("throughput") is not None else "",
        fairness=f"{result.get('fairness', ''):.4f}" if result.get("fairness") is not None else "",
        collision_rate=f"{result.get('collision_rate', ''):.4f}" if result.get("collision_rate") is not None else "",
        mean_backlog=f"{result.get('mean_backlog_per_node', ''):.2f}" if result.get("mean_backlog_per_node") is not None else "",
        final_backlog=f"{result.get('final_backlog_per_node', ''):.2f}" if result.get("final_backlog_per_node") is not None else "",
        attempts=result.get("attempts", ""),
        notes=notes,
    )
    append_rows([row])


def log_compare(mode: str, cfg, final_rows: list[dict],
                label_col: str, notes: str = "") -> None:
    """비교 모드 (Reward/State/Action/Phase Variant Compare).
    final_rows 의 각 항목을 개별 행으로 기록한다.
    """
    import dataclasses
    cd = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    base = _base(mode, cd)
    rows = []
    for r in final_rows:
        row = _row(
            **base,
            reward_variant=r.get("reward_variant") or r.get("av_key") or cd.get("reward_variant", ""),
            state_variant=r.get("sv_key") or r.get("state_variant") or cd.get("state_variant", ""),
            action_variant=r.get("av_key", ""),
            frame_size=r.get("frame_size", ""),
            series_label=r.get(label_col, ""),
            asr=f"{r['success_rate']:.4f}" if "success_rate" in r else "",
            throughput=f"{r['throughput']:.4f}" if "throughput" in r else "",
            fairness=f"{r['fairness']:.4f}" if "fairness" in r else "",
            collision_rate=f"{r.get('collision_rate', ''):.4f}" if r.get("collision_rate") is not None else "",
            mean_backlog=f"{r.get('mean_backlog_per_node', ''):.2f}" if r.get("mean_backlog_per_node") is not None else "",
            final_backlog=f"{r.get('final_backlog_per_node', ''):.2f}" if r.get("final_backlog_per_node") is not None else "",
            attempts=r.get("total_attempts", ""),
            notes=notes,
        )
        rows.append(row)
    append_rows(rows)
