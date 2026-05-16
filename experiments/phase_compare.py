"""슬롯 페이즈 학습 비교 실험.

frame_size별 PhaseQLearningController를 돌려
페이즈 자기 조직화가 성능·공정성에 미치는 영향을 비교한다.

frame_size=1  → 페이즈 비활성 (SF-only Q-learning, baseline)
frame_size=F  → F주기 페이즈 학습

상태: (phase_bin, sf_idx)  — SF만 사용, FL/Gmax/delta 없음
액션: relative SF±1 × channel
"""

from __future__ import annotations

import dataclasses
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

from agents.q_learning import DEFAULT_REWARD_VARIANT
from agents.phase_q_learning import PhaseQLearningController
from env import ScenarioConfig, run_simulation
from experiments.plots import plot_channel_g_timeseries
from utils.io import ensure_dir, write_csv_rows, write_json


FRAME_COLORS = [
    "#d55e00",   # frame=1  (no phase)
    "#0072b2",   # frame=8
    "#009e73",   # frame=18
    "#cc79a7",   # frame=30
    "#56b4e9",   # frame=60
    "#f0e442",
    "#999999",
]


@dataclasses.dataclass(frozen=True)
class PhaseCompareConfig:
    n_nodes: int = 60
    target_g: float = 1.0
    n_slots: int = 20_000
    warmup_slots: int = 5_000
    epoch_slots: int = 500
    n_channels: int = 3
    fixed_distance_ratio: float = 0.5
    profile_name: str = "short"
    seed: int = 42
    queue_mode: str = "accumulate"
    gw_obs_mode: str = "attempt"
    enable_rayleigh_fading: bool = False
    rayleigh_fade_margin_db: float = 10.0
    gw_thresholds: tuple[float, ...] = (0.7, 1.3)
    layout: str = "ring"
    reward_variant: str = "v10_phase_idle"   # IDLE 보상 양수, fail 완화 — 페이즈 학습 최적화
    # 비교할 frame_size 목록.
    # 1 = 페이즈 비활성 baseline, N = n_nodes 와 동일하게 두면 이상적 분리.
    frame_sizes: tuple[int, ...] = (1, 8, 18, 30, 60)
    output_dir: str = os.path.join("outputs", "phase_compare")


def _frame_label(frame_size: int, n_nodes: int) -> str:
    if frame_size <= 1:
        return "No phase (F=1)"
    suffix = " =N" if frame_size == n_nodes else ""
    return f"Phase F={frame_size}{suffix}"


def run_phase_compare(config: PhaseCompareConfig | None = None) -> dict:
    config = config or PhaseCompareConfig()
    out_dir = os.path.abspath(config.output_dir)
    ensure_dir(out_dir)

    epoch_data: dict[str, list[dict]] = {}
    epoch_rows: list[dict] = []
    final_rows: list[dict] = []
    per_node_rows: list[dict] = []

    for idx, frame_size in enumerate(config.frame_sizes):
        key = f"phase_f{frame_size}"
        label = _frame_label(frame_size, config.n_nodes)

        scenario = ScenarioConfig(
            n_nodes=config.n_nodes,
            target_g=config.target_g,
            n_slots=config.n_slots,
            warmup_slots=config.warmup_slots,
            epoch_slots=config.epoch_slots,
            n_channels=config.n_channels,
            fixed_distance_ratio=config.fixed_distance_ratio,
            profile_name=config.profile_name,
            seed=config.seed + 10_000 * (idx + 1),
            layout_seed=config.seed,
            queue_mode=config.queue_mode,
            gw_obs_mode=config.gw_obs_mode,
            enable_rayleigh_fading=config.enable_rayleigh_fading,
            rayleigh_fade_margin_db=config.rayleigh_fade_margin_db,
            gw_thresholds=config.gw_thresholds,
            layout=config.layout,
        )
        controller = PhaseQLearningController(
            alpha=0.1,
            gamma_q=0.9,
            epsilon=1.0,
            eps_min=0.05,
            eps_decay=0.9995,
            frame_size=frame_size,
            reward_variant=config.reward_variant,
        )
        result = run_simulation(scenario, controller)

        annotated = []
        for ep in result["epoch_log"]:
            row = dict(ep)
            row["series_key"] = key
            row["frame_size"] = frame_size
            annotated.append(row)
            epoch_rows.append(dict(row))
        epoch_data[key] = annotated

        n_resources = config.n_channels * 6
        measured = result["measured_slots"]
        per_success = np.asarray(result["per_node_success"], dtype=np.float64)
        per_attempts = np.asarray(result["per_node_attempts"], dtype=np.float64)
        per_collisions = np.asarray(result["per_node_collisions"], dtype=np.float64)
        per_link_failures = np.asarray(result["per_node_link_failures"], dtype=np.float64)

        for node_id in range(config.n_nodes):
            att = per_attempts[node_id]
            suc = per_success[node_id]
            per_node_rows.append({
                "series_key": key,
                "frame_size": frame_size,
                "node_id": node_id,
                "attempts": int(att),
                "successes": int(suc),
                "collisions": int(per_collisions[node_id]),
                "link_failures": int(per_link_failures[node_id]),
                "asr": suc / att if att > 0 else 0.0,
                "throughput": suc / measured if measured > 0 else 0.0,
                "resource_load": att / (n_resources * measured) if measured > 0 else 0.0,
            })

        pns = per_success
        min_max_ratio = float(pns.min() / pns.max()) if pns.max() > 0 else 0.0
        n_states = frame_size * 6
        final_rows.append({
            "series_key": key,
            "frame_size": frame_size,
            "label": label,
            "n_states": n_states,
            "n_nodes": result["n_nodes"],
            "target_g": result["target_g"],
            "success_rate": result["success_rate"],
            "collision_rate": result["collision_rate"],
            "throughput": result["throughput"],
            "total_attempts": result["attempts"],
            "arrival_load_realized": result["arrival_load_realized"],
            "attempt_load_realized": result["attempt_load_realized"],
            "mean_backlog_per_node": result["mean_backlog_per_node"],
            "final_backlog_per_node": result["final_backlog_per_node"],
            "fairness": result["fairness"],
            "min_max_ratio": min_max_ratio,
        })

        print(
            f"  {label:<22} F={frame_size:>3}  states={n_states:>4}  "
            f"ASR={result['success_rate']:.3f}  "
            f"thr={result['throughput']:.4f}  "
            f"jain={result['fairness']:.4f}  "
            f"backlog/node={result['final_backlog_per_node']:.2f}"
        )

    write_csv_rows(os.path.join(out_dir, "phase_compare_epochs.csv"), epoch_rows)
    write_csv_rows(os.path.join(out_dir, "phase_compare_summary.csv"), final_rows)
    write_csv_rows(os.path.join(out_dir, "phase_compare_per_node.csv"), per_node_rows)
    write_json(
        os.path.join(out_dir, "metadata.json"),
        {
            "config": dataclasses.asdict(config),
            "notes": {
                "state": "(phase_bin, sf_idx) — SF만 사용",
                "action": "relative SF±1 × channel",
                "frame_size_1": "phase 비활성 baseline (SF-only)",
            },
        },
    )

    labels = {f"phase_f{fs}": _frame_label(fs, config.n_nodes) for fs in config.frame_sizes}
    colors = {f"phase_f{fs}": FRAME_COLORS[i % len(FRAME_COLORS)] for i, fs in enumerate(config.frame_sizes)}

    p_gen = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    suptitle_base = (
        f"Phase Learning | N={config.n_nodes} | G={config.target_g:.2f} | "
        f"p_gen={p_gen:.3f} | reward={config.reward_variant}"
    )

    _plot_timeseries(epoch_data, labels, colors, out_dir, config, suptitle_base)
    _plot_summary_bar(final_rows, labels, colors, out_dir, config, suptitle_base)
    plot_channel_g_timeseries(
        epoch_data=epoch_data,
        labels=labels,
        colors=colors,
        n_channels=config.n_channels,
        out_path=os.path.join(out_dir, "channel_g.png"),
        suptitle=f"Per-Channel G | {suptitle_base}",
        target_g=config.target_g,
    )

    return {
        "output_dir": out_dir,
        "epoch_data": epoch_data,
        "final_rows": final_rows,
        "per_node_rows": per_node_rows,
        "channel_g_image": os.path.join(out_dir, "channel_g.png"),
    }


def _plot_timeseries(epoch_data, labels, colors, out_dir, config, suptitle):
    panels = [
        ("success_rate",          "ASR per Epoch",            "Success rate"),
        ("throughput",            "Throughput per Epoch",     "Successes / slot"),
        ("collision_rate",        "Collision Rate per Epoch", "Collision rate"),
        ("attempts",              "Attempts per Epoch",       "Transmissions"),
        ("mean_backlog_per_node", "Mean Backlog per Epoch",   "Packets / node"),
        ("fairness",              "Jain Fairness per Epoch",  "Fairness index"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    fig.suptitle(f"Phase Learning — Epoch Timeseries\n{suptitle}", fontsize=11, fontweight="bold")

    for ax, (metric, title, ylabel) in zip(axes.flat, panels):
        for key, log in epoch_data.items():
            if not log:
                continue
            xs = [e["slot_end"] for e in log]
            ys = [e.get(metric, 0.0) for e in log]
            ax.plot(xs, ys, label=labels.get(key, key), color=colors.get(key), linewidth=1.7, alpha=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Slot", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        if metric in ("success_rate", "collision_rate", "fairness"):
            ax.set_ylim(-0.02, 1.05)

    axes[0, 0].legend(loc="best", fontsize=8, framealpha=0.9)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = os.path.join(out_dir, "phase_compare.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {path}")


def _plot_summary_bar(final_rows, labels, colors, out_dir, config, suptitle):
    metrics = [
        ("success_rate",          "ASR",                   (0.0, 1.05)),
        ("throughput",            "Throughput (pkts/slot)", None),
        ("collision_rate",        "Collision Rate",         (0.0, 1.05)),
        ("fairness",              "Jain Fairness",          (0.0, 1.05)),
        ("mean_backlog_per_node", "Mean Backlog / Node",    None),
    ]
    keys = [f"phase_f{fs}" for fs in config.frame_sizes]
    row_by_key = {r["series_key"]: r for r in final_rows}
    x_labels = [labels.get(k, k) for k in keys]
    x_pos = list(range(len(keys)))

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    fig.suptitle(f"Phase Learning — Summary\n{suptitle}", fontsize=10, fontweight="bold")

    for ax, (metric, title, ylim) in zip(axes, metrics):
        vals = [row_by_key[k].get(metric, 0.0) if k in row_by_key else 0.0 for k in keys]
        bar_colors = [colors.get(k, "#aaaaaa") for k in keys]
        ax.bar(x_pos, vals, color=bar_colors, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    path = os.path.join(out_dir, "summary_bar.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {path}")


def main() -> None:
    result = run_phase_compare()
    print(f"Outputs saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
