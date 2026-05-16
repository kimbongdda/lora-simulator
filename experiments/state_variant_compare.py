"""Q-learning 상태공간 변형을 비교한다."""

from __future__ import annotations

import dataclasses
import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

from agents.q_learning import DEFAULT_REWARD_VARIANT, STATE_VARIANTS
from baselines import make_controller
from env import ScenarioConfig, run_simulation
from experiments.plots import plot_channel_g_timeseries, plot_sf_distance_heatmap
from utils.io import ensure_dir, write_csv_rows, write_json


VARIANT_COLORS = [
    "#d55e00",
    "#0072b2",
    "#009e73",
    "#cc79a7",
    "#56b4e9",
    "#f0e442",
    "#999999",
]

Q_CONTROLLER_KEYS = ("decentralized_q_learning",)

# 배치 경로를 지원하는 s2* 계열 상태 변형
S2_DENSE_VARIANTS = (
    "s2_compact", "s3_no_sf", "s4_no_fl", "s5_no_gmax",
    "s8_sf_delta", "s9_sf_delta_gmax", "s10_sf_delta_fl", "s11_sf_delta_only",
    "s12_sf_ch_gmax", "s13_sf_ch", "s14_sf_delta_ch",
)


@dataclasses.dataclass(frozen=True)
class StateVariantConfig:
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
    gw_thresholds: tuple[float, ...] = (0.7, 1.3)  # bin 경계값. bin 수 = len+1.
    layout: str = "ring"
    controller_key: str = "decentralized_q_learning"
    reward_variant: str = DEFAULT_REWARD_VARIANT
    state_variant_keys: tuple[str, ...] = S2_DENSE_VARIANTS
    output_dir: str = os.path.join("outputs", "state_variant_compare")


def _validate_config(config: StateVariantConfig) -> tuple[str, ...]:
    if config.controller_key not in Q_CONTROLLER_KEYS:
        raise ValueError(f"controller_key must be one of {Q_CONTROLLER_KEYS}, got {config.controller_key!r}")
    valid_keys = tuple(k for k in config.state_variant_keys if k in STATE_VARIANTS)
    if not valid_keys:
        raise ValueError("state_variant_keys must contain at least one valid state variant key.")
    return valid_keys


def run_state_variant_compare(config: StateVariantConfig | None = None) -> dict:
    config = config or StateVariantConfig()
    sv_keys = _validate_config(config)
    out_dir = os.path.abspath(config.output_dir)
    ensure_dir(out_dir)


    epoch_data: dict[str, list[dict]] = {}
    epoch_rows: list[dict] = []
    final_rows: list[dict] = []
    per_node_rows: list[dict] = []
    node_positions: dict[str, tuple] = {}
    sf_data: dict[str, tuple] = {}
    q_table_data_all: dict[str, tuple] = {}

    for idx, sv_key in enumerate(sv_keys):
        sv_spec = STATE_VARIANTS[sv_key]
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
        controller = make_controller(
            config.controller_key,
            reward_variant=config.reward_variant,
            state_variant=sv_key,
        )
        result = run_simulation(scenario, controller)
        if result.get("q_table_data") is not None:
            q_table_data_all[sv_key] = result["q_table_data"]

        annotated_log = []
        for ep in result["epoch_log"]:
            row = dict(ep)
            row["baseline_label"] = sv_spec["label"]
            row["sv_key"] = sv_key
            annotated_log.append(row)
        epoch_data[sv_key] = annotated_log

        node_positions[sv_key] = (result["xs"], result["ys"], result["dists_m"])
        sf_data[sv_key] = (
            np.asarray(result["sf_usage"]),
            np.asarray(result["idle_usage"]),
            result["dists_m"],
            result["cell_radius_m"],
        )

        measured = result["measured_slots"]
        n_resources = config.n_channels * 6
        per_success = np.asarray(result["per_node_success"], dtype=np.float64)
        per_attempts = np.asarray(result["per_node_attempts"], dtype=np.float64)
        per_collisions = np.asarray(result["per_node_collisions"], dtype=np.float64)
        per_link_failures = np.asarray(result["per_node_link_failures"], dtype=np.float64)

        for node_id in range(config.n_nodes):
            attempts = per_attempts[node_id]
            successes = per_success[node_id]
            collisions = per_collisions[node_id]
            link_failures = per_link_failures[node_id]
            per_node_rows.append({
                "sv_key": sv_key,
                "sv_label": sv_spec["label"],
                "node_id": node_id,
                "attempts": int(attempts),
                "successes": int(successes),
                "collisions": int(collisions),
                "link_failures": int(link_failures),
                "asr": successes / attempts if attempts > 0 else 0.0,
                "throughput": successes / measured if measured > 0 else 0.0,
                "resource_load": attempts / (n_resources * measured) if measured > 0 else 0.0,
            })

        pns = per_success
        min_max_ratio = float(pns.min() / pns.max()) if pns.max() > 0 else 0.0
        final_rows.append({
            "sv_key": sv_key,
            "sv_label": sv_spec["label"],
            "n_states": _n_states_for(sv_key, len(config.gw_thresholds) + 1, config.n_channels),
            "sv_description": sv_spec["description"],
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

        for epoch in result["epoch_log"]:
            row = dict(epoch)
            row["sv_key"] = sv_key
            row["sv_label"] = sv_spec["label"]
            epoch_rows.append(row)

        print(
            f"  {sv_spec['label']:<28} "
            f"states={_n_states_for(sv_key, len(config.gw_thresholds) + 1, config.n_channels):3d}  "
            f"ASR={result['success_rate']:.3f}  "
            f"thr={result['throughput']:.4f}  "
            f"jain={result['fairness']:.4f}  "
            f"attempts={result['attempts']:,}  "
            f"backlog/node={result['final_backlog_per_node']:.2f}"
        )

    write_csv_rows(os.path.join(out_dir, "state_variant_epochs.csv"), epoch_rows)
    write_csv_rows(os.path.join(out_dir, "state_variant_summary.csv"), final_rows)
    write_csv_rows(os.path.join(out_dir, "state_variant_per_node.csv"), per_node_rows)
    write_json(
        os.path.join(out_dir, "metadata.json"),
        {
            "config": dataclasses.asdict(config),
            "state_variants": {k: STATE_VARIANTS[k] for k in sv_keys},
            "notes": {
                "controller_key": "State-variant comparison uses decentralized Q-learning only.",
                "reward_variant": config.reward_variant,
            },
        },
    )

    p_gen_equiv = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    _plot(epoch_data, out_dir, config, sv_keys)
    _plot_per_node(per_node_rows, out_dir, config, sv_keys)
    _plot_layout(per_node_rows, node_positions, out_dir, config, sv_keys)
    _plot_summary_bar(final_rows, out_dir, config, sv_keys)
    plot_sf_distance_heatmap(
        sf_data=sf_data,
        labels={k: STATE_VARIANTS[k]["label"] for k in sv_keys},
        keys=list(sv_keys),
        suptitle=(
            f"SF Selection vs Distance | Q-learning state variants | "
            f"N={config.n_nodes} | G={config.target_g:.2f} | p_gen={p_gen_equiv:.3f}"
        ),
        out_path=os.path.join(out_dir, "sf_heatmap.png"),
    )
    _svc_labels = {k: STATE_VARIANTS[k]["label"] for k in sv_keys}
    _svc_colors = {k: VARIANT_COLORS[i % len(VARIANT_COLORS)] for i, k in enumerate(sv_keys)}
    plot_channel_g_timeseries(
        epoch_data=epoch_data,
        labels=_svc_labels,
        colors=_svc_colors,
        n_channels=config.n_channels,
        out_path=os.path.join(out_dir, "channel_g.png"),
        suptitle=(
            f"Per-Channel G | Q-learning state variants | "
            f"N={config.n_nodes} | G={config.target_g:.2f}"
        ),
        target_g=config.target_g,
    )

    return {
        "output_dir": out_dir,
        "epoch_data": epoch_data,
        "epoch_rows": epoch_rows,
        "final_rows": final_rows,
        "per_node_rows": per_node_rows,
        "node_positions": node_positions,
        "sf_heatmap_image": os.path.join(out_dir, "sf_heatmap.png"),
        "channel_g_image": os.path.join(out_dir, "channel_g.png"),
        "q_table_data_all": q_table_data_all,
    }


def _n_states_for(sv_key: str, n_bins: int = 3, n_channels: int = 3) -> int:
    """상태 변형의 총 상태 수를 반환한다."""
    from agents.q_learning import DecentralizedQLearningController
    return DecentralizedQLearningController._compute_n_states(sv_key, n_bins, n_channels)


def _sv_color(sv_key: str, sv_keys: tuple[str, ...]) -> str:
    idx = list(sv_keys).index(sv_key) if sv_key in sv_keys else 0
    return VARIANT_COLORS[idx % len(VARIANT_COLORS)]


def _plot(
    epoch_data: dict[str, list[dict]],
    out_dir: str,
    config: StateVariantConfig,
    sv_keys: tuple[str, ...],
) -> None:
    panels = [
        ("success_rate",          "ASR per Epoch",            "Success rate",            False),
        ("throughput",            "Throughput per Epoch",     "Successes / slot",        False),
        ("collision_rate",        "Collision Rate per Epoch", "Collision rate",           False),
        ("attempts",              "Attempts per Epoch",       "Transmissions",            False),
        ("mean_backlog_per_node", "Mean Backlog per Epoch",   "Packets queued / node",   False),
    ]

    p_gen = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    fig.suptitle(
        "State Variant Comparison\n"
        f"Q-learning (decentralized) | N={config.n_nodes} | G={config.target_g:.2f} | "
        f"p_gen={p_gen:.3f} | reward={config.reward_variant} | slots={config.n_slots}",
        fontsize=12,
        fontweight="bold",
    )

    for ax, (metric, title, ylabel, log_y) in zip(axes.flat, panels):
        for idx, sv_key in enumerate(sv_keys):
            log = epoch_data.get(sv_key, [])
            if not log:
                continue
            xs = [e["slot_end"] for e in log]
            ys = [e.get(metric, 0.0) for e in log]
            color = VARIANT_COLORS[idx % len(VARIANT_COLORS)]
            n_st = _n_states_for(sv_key, len(config.gw_thresholds) + 1, config.n_channels)
            label = f"{STATE_VARIANTS[sv_key]['label']} ({n_st}st)"
            if log_y:
                ys = [max(v, 1e-3) for v in ys]
                ax.semilogy(xs, ys, label=label, color=color, linewidth=1.7, alpha=0.9)
            else:
                ax.plot(xs, ys, label=label, color=color, linewidth=1.7, alpha=0.9)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Slot", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        if metric in ("success_rate", "collision_rate"):
            ax.set_ylim(-0.02, 1.05)

    axes[0, 0].legend(loc="best", fontsize=8, framealpha=0.9)
    # panels가 5개이므로 마지막(2행 3열) subplot은 숨긴다
    axes[1, 2].set_visible(False)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path = os.path.join(out_dir, "state_variant_compare.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def _plot_per_node(
    rows: list[dict],
    out_dir: str,
    config: StateVariantConfig,
    sv_keys: tuple[str, ...],
) -> None:
    grouped: dict[str, dict] = {}
    for row in rows:
        key = row["sv_key"]
        grouped.setdefault(key, {
            "label": row["sv_label"],
            "node_ids": [],
            "asr": [],
            "throughput": [],
            "resource_load": [],
        })
        grouped[key]["node_ids"].append(row["node_id"])
        grouped[key]["asr"].append(row["asr"])
        grouped[key]["throughput"].append(row["throughput"])
        grouped[key]["resource_load"].append(row["resource_load"])

    p_gen = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"Per-node Breakdown | N={config.n_nodes} | G={config.target_g:.2f} | "
        f"p_gen={p_gen:.3f} | layout={config.layout}",
        fontsize=12,
        fontweight="bold",
    )

    scatter_metrics = [
        ("asr",           "Per-node ASR",            "ASR"),
        ("throughput",    "Per-node Throughput",      "Successes / measured slot"),
        ("resource_load", "Per-node Resource Load",   "Attempts / (resource × slot)"),
    ]

    for col, (metric, title, ylabel) in enumerate(scatter_metrics):
        ax = axes[col]
        for sv_key in sv_keys:
            data = grouped.get(sv_key)
            if not data:
                continue
            color = _sv_color(sv_key, sv_keys)
            n_st = _n_states_for(sv_key, len(config.gw_thresholds) + 1, config.n_channels)
            label = f"{STATE_VARIANTS[sv_key]['label']} ({n_st}st)"
            ax.scatter(data["node_ids"], data[metric], color=color, s=18, alpha=0.55, label=label, zorder=3)
            ax.axhline(float(np.mean(data[metric])), color=color, linewidth=1.5, alpha=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Node ID", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        if metric == "asr":
            ax.set_ylim(-0.02, 1.05)

    axes[0].legend(loc="best", fontsize=8, framealpha=0.9)

    # 실패 원인 분해 막대
    ax = axes[3]
    labels, suc_frac, coll_frac, link_frac = [], [], [], []
    for sv_key in sv_keys:
        key_rows = [r for r in rows if r["sv_key"] == sv_key]
        successes  = sum(r["successes"]     for r in key_rows)
        collisions = sum(r["collisions"]    for r in key_rows)
        link_fail  = sum(r["link_failures"] for r in key_rows)
        total = successes + collisions + link_fail
        n_st = _n_states_for(sv_key, len(config.gw_thresholds) + 1, config.n_channels)
        labels.append(f"{STATE_VARIANTS[sv_key]['label']}\n({n_st}st)")
        suc_frac.append(successes  / total if total else 0.0)
        coll_frac.append(collisions / total if total else 0.0)
        link_frac.append(link_fail  / total if total else 0.0)

    x_idx = list(range(len(labels)))
    ax.bar(x_idx, suc_frac,  label="Success",   color="#2ca02c", alpha=0.85)
    ax.bar(x_idx, coll_frac, bottom=suc_frac,
           label="Collision", color="#d62728", alpha=0.85)
    ax.bar(x_idx, link_frac,
           bottom=[s + c for s, c in zip(suc_frac, coll_frac)],
           label="Link Fail", color="#ff7f0e", alpha=0.85)
    ax.set_title("실패 원인 분해", fontsize=10)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of attempts", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out_path = os.path.join(out_dir, "per_node_breakdown.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def _plot_layout(
    rows: list[dict],
    node_positions: dict[str, tuple],
    out_dir: str,
    config: StateVariantConfig,
    sv_keys: tuple[str, ...],
) -> None:
    n_variants = len(sv_keys)
    ncols = min(n_variants, 3)
    nrows = (n_variants + ncols - 1) // ncols
    p_gen = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(
        f"Node Layout (ASR color) | N={config.n_nodes} | G={config.target_g:.2f} | "
        f"p_gen={p_gen:.3f} | layout={config.layout}",
        fontsize=12,
        fontweight="bold",
    )

    asr_lookup = {(r["sv_key"], r["node_id"]): r["asr"] for r in rows}

    for idx, sv_key in enumerate(sv_keys):
        ax = axes[idx // ncols][idx % ncols]
        xs, ys, _ = node_positions[sv_key]
        asr_vals = [asr_lookup.get((sv_key, nid), 0.0) for nid in range(len(xs))]
        sc = ax.scatter(xs, ys, c=asr_vals, cmap="RdYlGn", vmin=0.0, vmax=1.0,
                        s=60, alpha=0.85, edgecolors="none", zorder=3)
        ax.scatter([0], [0], c="black", marker="*", s=200, zorder=5, label="Gateway")
        fig.colorbar(sc, ax=ax, label="ASR")
        n_st = _n_states_for(sv_key, len(config.gw_thresholds) + 1, config.n_channels)
        ax.set_title(f"{STATE_VARIANTS[sv_key]['label']} ({n_st}st)", fontsize=10)
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("y (m)", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")

    for idx in range(n_variants, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path = os.path.join(out_dir, "node_layout.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def _plot_summary_bar(
    final_rows: list[dict],
    out_dir: str,
    config: StateVariantConfig,
    sv_keys: tuple[str, ...],
) -> None:
    """ASR / 처리량 / Jain 공정성 / min-max 비율을 나란히 보여주는 요약 막대 그래프."""
    metrics = [
        ("success_rate",    "ASR"),
        ("throughput",      "Throughput (suc/slot)"),
        ("fairness",        "Jain Fairness"),
        ("min_max_ratio",   "Min/Max Ratio"),
        ("total_attempts",  "Total Attempts"),
    ]

    row_by_key = {r["sv_key"]: r for r in final_rows}
    x = list(range(len(sv_keys)))
    x_labels = [f"{STATE_VARIANTS[k]['label']}\n({_n_states_for(k)}st)" for k in sv_keys]
    colors = [VARIANT_COLORS[i % len(VARIANT_COLORS)] for i in range(len(sv_keys))]

    p_gen = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5))
    fig.suptitle(
        f"State Variant Summary | N={config.n_nodes} | G={config.target_g:.2f} | "
        f"p_gen={p_gen:.3f} | reward={config.reward_variant}",
        fontsize=12,
        fontweight="bold",
    )

    for ax, (metric, ylabel) in zip(axes, metrics):
        vals = [row_by_key.get(k, {}).get(metric, 0.0) for k in sv_keys]
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            label_txt = f"{int(val):,}" if metric == "total_attempts" else f"{val:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(val * 0.01, 0.005),
                    label_txt, ha="center", va="bottom", fontsize=8)
        ax.set_title(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=10, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        if metric in ("success_rate", "fairness", "min_max_ratio"):
            ax.set_ylim(0, 1.05)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out_path = os.path.join(out_dir, "summary_bar.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")
