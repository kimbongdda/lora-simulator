"""작은 파일 출력 보조 함수."""

from __future__ import annotations

import csv
import json
import os


def ensure_dir(path: str) -> None:
    """출력 경로가 없으면 디렉터리를 만든다."""
    os.makedirs(path, exist_ok=True)


def write_csv_rows(path: str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """딕셔너리 행 목록을 CSV 파일로 저장한다."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload) -> None:
    """JSON 직렬화 결과를 보기 좋은 형식으로 저장한다."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
