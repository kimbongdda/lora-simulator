"""실험용 공통 시각화 보조 함수."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np


N_SF = 6
SF_LABELS = [f"SF{i + 7}" for i in range(N_SF)]
N_DISTANCE_BINS = 12


def plot_sf_distance_heatmap(
    sf_data: dict[str, tuple],
    labels: dict[str, str],
    keys: list[str] | tuple[str, ...],
    suptitle: str,
    out_path: str,
) -> None:
    """
    거리별 SF 선택 히트맵을 시리즈별로 그린다.
    IDLE 데이터가 있으면 각 히트맵 아래에 거리별 IDLE 비율 바 차트를 추가한다.

    Parameters
    ----------
    sf_data : {key: (sf_usage, idle_usage, dists_m, cell_radius_m)
                 or (sf_usage, dists_m, cell_radius_m)}  -- idle_usage: (n_nodes,) array
    labels  : {key: 표시 이름}
    keys    : 그릴 시리즈의 순서 있는 목록
    suptitle: 전체 그림 제목
    out_path: 저장할 PNG 경로
    """
    n_series = len(keys)
    ncols = min(n_series, 3)
    nrows = (n_series + ncols - 1) // ncols

    # 4-tuple이면 idle 데이터 있음
    has_idle = any(len(sf_data[k]) == 4 for k in keys)
    subplot_rows_per_series = 2 if has_idle else 1
    nrows_total = nrows * subplot_rows_per_series

    fig, axes = plt.subplots(
        nrows_total, ncols,
        figsize=(6 * ncols, (3 if has_idle else 4) * nrows_total),
        squeeze=False,
    )
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    for idx, key in enumerate(keys):
        col = idx % ncols
        sf_row = (idx // ncols) * subplot_rows_per_series
        ax = axes[sf_row][col]

        entry = sf_data[key]
        if len(entry) == 4:
            sf_usage, idle_usage, dists_m, cell_radius_m = entry
            idle_usage = np.asarray(idle_usage, dtype=np.float64)
        else:
            sf_usage, dists_m, cell_radius_m = entry
            idle_usage = None

        sf_usage = np.asarray(sf_usage, dtype=np.float64)   # (n_nodes, N_SF)
        dists_m = np.asarray(dists_m, dtype=np.float64)

        bin_edges = np.linspace(0.0, float(cell_radius_m), N_DISTANCE_BINS + 1)
        heatmap = np.zeros((N_DISTANCE_BINS, N_SF))
        idle_bin = np.zeros(N_DISTANCE_BINS)
        total_bin = np.zeros(N_DISTANCE_BINS)

        for node_id in range(len(dists_m)):
            bin_idx = int(np.searchsorted(bin_edges[1:], dists_m[node_id]))
            bin_idx = min(bin_idx, N_DISTANCE_BINS - 1)
            heatmap[bin_idx] += sf_usage[node_id]
            if idle_usage is not None:
                idle_bin[bin_idx] += idle_usage[node_id]
                total_bin[bin_idx] += sf_usage[node_id].sum() + idle_usage[node_id]

        # 각 거리 bin을 정규화해 색이 선택 비율을 나타내도록 한다.
        row_sums = heatmap.sum(axis=1, keepdims=True)
        heatmap_norm = np.where(row_sums > 0, heatmap / row_sums, 0.0)

        im = ax.pcolormesh(
            bin_edges,
            np.arange(-0.5, N_SF),
            heatmap_norm.T,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
        )
        fig.colorbar(im, ax=ax, label="Fraction of selections")

        ax.set_yticks(range(N_SF))
        ax.set_yticklabels(SF_LABELS, fontsize=8)
        ax.set_xlabel("Distance from gateway (m)", fontsize=9)
        ax.set_ylabel("SF", fontsize=9)
        ax.set_title(labels.get(key, key), fontsize=10)

        # IDLE 비율 바 차트
        if has_idle:
            ax_idle = axes[sf_row + 1][col]
            if idle_usage is not None:
                idle_ratio = np.where(total_bin > 0, idle_bin / total_bin, 0.0)
                bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                bin_width = bin_edges[1] - bin_edges[0]
                ax_idle.bar(bin_centres, idle_ratio, width=bin_width * 0.85,
                            color="#5588cc", alpha=0.8, edgecolor="white", linewidth=0.5)
                ax_idle.set_ylim(0.0, 1.0)
            ax_idle.set_xlabel("Distance from gateway (m)", fontsize=9)
            ax_idle.set_ylabel("IDLE ratio", fontsize=9)
            ax_idle.set_title(f"{labels.get(key, key)} — IDLE", fontsize=9)
            ax_idle.grid(True, axis="y", alpha=0.3)

    for idx in range(n_series, nrows * ncols):
        col = idx % ncols
        sf_row = (idx // ncols) * subplot_rows_per_series
        axes[sf_row][col].set_visible(False)
        if has_idle:
            axes[sf_row + 1][col].set_visible(False)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def plot_channel_g_timeseries(
    epoch_data: dict[str, list[dict]],
    labels: dict[str, str],
    colors: dict[str, str],
    n_channels: int,
    out_path: str,
    suptitle: str = "",
    target_g: float | None = None,
) -> None:
    """채널별 평균 G 추이를 시리즈(베이스라인/변형)별로 그린다.

    Parameters
    ----------
    epoch_data : {series_key: epoch_log 리스트}
    labels     : {series_key: 표시 이름}
    colors     : {series_key: 색상 문자열}
    n_channels : 채널 수 (서브플롯 수)
    out_path   : 저장 PNG 경로
    suptitle   : 전체 그림 제목
    target_g   : 목표 G (수평 점선으로 표시)
    """
    fig, axes = plt.subplots(1, n_channels, figsize=(6 * n_channels, 4), squeeze=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    for ch in range(n_channels):
        ax = axes[0, ch]
        for series_key, epoch_log in epoch_data.items():
            if not epoch_log:
                continue
            xs = [ep["slot_end"] for ep in epoch_log]
            ys = [ep["per_channel_g"][ch] if len(ep.get("per_channel_g", [])) > ch else 0.0
                  for ep in epoch_log]
            color = colors.get(series_key, "#888888")
            label = labels.get(series_key, series_key)
            ax.plot(xs, ys, label=label, color=color, linewidth=1.7, alpha=0.9)

        if target_g is not None:
            ax.axhline(target_g, color="#444444", linewidth=1.0, linestyle="--",
                       alpha=0.6, label=f"target G={target_g}")

        ax.set_title(f"Channel {ch} G", fontsize=10)
        ax.set_xlabel("Slot", fontsize=9)
        ax.set_ylabel("G (offered load)", fontsize=9)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.25)
        if ch == 0:
            ax.legend(loc="best", fontsize=8, framealpha=0.9)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93) if suptitle else (0.0, 0.0, 1.0, 1.0))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


ACTION_LABELS = [
    "IDLE",
    "SF= ch0", "SF= ch1", "SF= ch2",
    "SF+1 ch0", "SF+1 ch1", "SF+1 ch2",
    "SF-1 ch0", "SF-1 ch1", "SF-1 ch2",
]
SF_LABELS = [f"SF{i + 7}" for i in range(6)]
FAILURE_LEVEL_LABELS = ["OK/fresh", "fail(r=1)", "fail(r=2~3)", "fail(r=4+)"]
G_BIN_LABELS = ["G Low", "G Mid", "G High"]


def plot_q_table_heatmap(
    q_table_data: tuple[list[tuple], np.ndarray],
    state_variant: str,
    suptitle: str,
    out_path: str,
) -> None:
    """큐 테이블을 히트맵으로 그린다. (failure_level, g_bin) 조합마다 패널을 둔다.

    s2_compact 상태 레이아웃만 구조화된 패널 뷰를 지원한다.
    상태 형식은 (sf_idx, failure_level, max_g_bin)이다.
    다른 상태 변형은 상태×액션 원시 히트맵으로 보여준다.
    """
    states, q_arr = q_table_data  # q_arr: (상태 수, 액션 수)
    n_actions = q_arr.shape[1]
    act_labels = ACTION_LABELS[:n_actions]

    if state_variant == "s2_compact":
        # 구조화 뷰: failure_level × g_bin 조합마다 하나의 패널을 만든다.
        n_fail = 4
        n_gbin = 3
        n_sf = 6
        # 조밀한 그리드를 만든다: 행=failure_level, 열=g_bin, 각 칸은 SF×액션 히트맵.
        grid = np.full((n_fail, n_gbin, n_sf, n_actions), np.nan)
        for state, qvals in zip(states, q_arr):
            if len(state) == 3:
                sf, fl, gb = state
                if 0 <= sf < n_sf and 0 <= fl < n_fail and 0 <= gb < n_gbin:
                    grid[fl, gb, sf, :] = qvals

        fig, axes = plt.subplots(n_fail, n_gbin, figsize=(5 * n_gbin, 4 * n_fail), squeeze=False)
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")

        vmax = np.nanmax(np.abs(grid)) or 1.0

        for fl in range(n_fail):
            for gb in range(n_gbin):
                ax = axes[fl][gb]
                data = grid[fl, gb]  # (n_sf, n_actions)
                if np.all(np.isnan(data)):
                    ax.set_visible(False)
                    continue
                im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                               vmin=-vmax, vmax=vmax, interpolation="nearest")
                ax.set_xticks(range(n_actions))
                ax.set_xticklabels(act_labels, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(n_sf))
                ax.set_yticklabels(SF_LABELS, fontsize=8)
                ax.set_title(
                    f"{FAILURE_LEVEL_LABELS[fl]} | {G_BIN_LABELS[gb]}",
                    fontsize=9,
                )
                # 각 칸의 값을 숫자로 표기한다.
                for r in range(n_sf):
                    for c in range(n_actions):
                        val = data[r, c]
                        if not np.isnan(val):
                            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                                    fontsize=6, color="black")
                fig.colorbar(im, ax=ax, shrink=0.8)

    else:
        # 원시 뷰: 상태를 그대로 쌓아 단일 히트맵으로 그린다.
        n_states = len(states)
        fig, ax = plt.subplots(figsize=(max(10, n_actions * 0.8), max(6, n_states * 0.25)))
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
        vmax = float(np.max(np.abs(q_arr))) or 1.0
        im = ax.imshow(q_arr, aspect="auto", cmap="RdYlGn",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(n_actions))
        ax.set_xticklabels(act_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("State index", fontsize=9)
        fig.colorbar(im, ax=ax, label="Q-value")

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")
