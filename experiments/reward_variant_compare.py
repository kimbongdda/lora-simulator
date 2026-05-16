"""Q-learning 기준선의 보상 변형을 비교한다."""

from __future__ import annotations

import dataclasses
import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

from agents.q_learning import REWARD_VARIANTS
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
]

Q_CONTROLLER_KEYS = ("decentralized_q_learning",)


@dataclasses.dataclass(frozen=True)
class RewardVariantConfig:
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
    layout: str = "ring"
    controller_key: str = "decentralized_q_learning"
    variant_keys: tuple[str, ...] = tuple(REWARD_VARIANTS)
    state_variant: str = "s0_full"
    output_dir: str = os.path.join("outputs", "reward_variant_compare")


def _validate_config(config: RewardVariantConfig) -> tuple[str, ...]:
    if config.controller_key not in Q_CONTROLLER_KEYS:
        raise ValueError(f"controller_key must be one of {Q_CONTROLLER_KEYS}, got {config.controller_key!r}")

    variant_keys = tuple(key for key in config.variant_keys if key in REWARD_VARIANTS)
    if not variant_keys:
        raise ValueError("variant_keys must contain at least one valid reward variant key.")
    return variant_keys


def run_reward_variant_compare(config: RewardVariantConfig | None = None) -> dict:
    config = config or RewardVariantConfig()
    variant_keys = _validate_config(config)
    out_dir = os.path.abspath(config.output_dir)
    ensure_dir(out_dir)

    epoch_data: dict[str, list[dict]] = {}
    epoch_rows: list[dict] = []
    final_rows: list[dict] = []
    per_node_rows: list[dict] = []
    node_positions: dict[str, tuple] = {}
    sf_data: dict[str, tuple] = {}
    last_q_table_data = None
    last_q_per_node = None
    last_q_dists_m = None

    for idx, variant_key in enumerate(variant_keys):
        variant_spec = REWARD_VARIANTS[variant_key]
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
            layout=config.layout,
        )
        controller = make_controller(config.controller_key, reward_variant=variant_key, state_variant=config.state_variant)
        result = run_simulation(scenario, controller)
        if result.get("q_table_data") is not None:
            last_q_table_data = result["q_table_data"]
        if result.get("q_per_node") is not None:
            last_q_per_node = result["q_per_node"]
            last_q_dists_m = result["dists_m"]

        annotated_log = []
        for ep in result["epoch_log"]:
            row = dict(ep)
            row["baseline_label"] = variant_spec["label"]
            row["variant_key"] = variant_key
            annotated_log.append(row)
        epoch_data[variant_key] = annotated_log

        node_positions[variant_key] = (result["xs"], result["ys"], result["dists_m"])
        sf_data[variant_key] = (
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
                "variant_key": variant_key,
                "variant_label": variant_spec["label"],
                "node_id": node_id,
                "attempts": int(attempts),
                "successes": int(successes),
                "collisions": int(collisions),
                "link_failures": int(link_failures),
                "asr": successes / attempts if attempts > 0 else 0.0,
                "throughput": successes / measured if measured > 0 else 0.0,
                "resource_load": attempts / (n_resources * measured) if measured > 0 else 0.0,
            })

        final_rows.append(
            {
                "variant_key": variant_key,
                "variant_label": variant_spec["label"],
                "variant_description": variant_spec["description"],
                "baseline_key": result["baseline_key"],
                "baseline_label": result["baseline_label"],
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
            }
        )

        for epoch in result["epoch_log"]:
            row = dict(epoch)
            row["variant_key"] = variant_key
            row["variant_label"] = variant_spec["label"]
            row["baseline_key"] = result["baseline_key"]
            row["baseline_label"] = result["baseline_label"]
            epoch_rows.append(row)

        print(
            f"  {variant_spec['label']:<14} "
            f"ASR={result['success_rate']:.3f} "
            f"thr={result['throughput']:.4f} "
            f"attempts={result['attempts']:,}  "
            f"backlog/node={result['final_backlog_per_node']:.2f}"
        )

    write_csv_rows(os.path.join(out_dir, "reward_variant_epochs.csv"), epoch_rows)
    write_csv_rows(os.path.join(out_dir, "reward_variant_summary.csv"), final_rows)
    write_csv_rows(os.path.join(out_dir, "reward_variant_per_node.csv"), per_node_rows)
    write_json(
        os.path.join(out_dir, "metadata.json"),
        {
            "config": dataclasses.asdict(config),
            "variants": {key: REWARD_VARIANTS[key] for key in variant_keys},
            "notes": {
                "controller_key": "Reward-variant comparison uses decentralized Q-learning only.",
                "queue_model": "Accumulating queue per node.",
            },
        },
    )

    p_gen_equiv = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    _plot(epoch_data, out_dir, config, variant_keys)
    _plot_per_node(per_node_rows, out_dir, config, variant_keys)
    _plot_layout(per_node_rows, node_positions, out_dir, config, variant_keys)
    plot_sf_distance_heatmap(
        sf_data=sf_data,
        labels={k: REWARD_VARIANTS[k].get("plot_label", REWARD_VARIANTS[k]["label"]) for k in variant_keys},
        keys=list(variant_keys),
        suptitle=(
            f"SF Selection vs Distance | Q-learning reward variants | "
            f"N={config.n_nodes} | G={config.target_g:.2f} | p_gen≈{p_gen_equiv:.3f}"
        ),
        out_path=os.path.join(out_dir, "sf_heatmap.png"),
    )
    _rvc_labels = {k: REWARD_VARIANTS[k].get("plot_label", REWARD_VARIANTS[k]["label"]) for k in variant_keys}
    _rvc_colors = {k: VARIANT_COLORS[i % len(VARIANT_COLORS)] for i, k in enumerate(variant_keys)}
    plot_channel_g_timeseries(
        epoch_data=epoch_data,
        labels=_rvc_labels,
        colors=_rvc_colors,
        n_channels=config.n_channels,
        out_path=os.path.join(out_dir, "channel_g.png"),
        suptitle=(
            f"Per-Channel G | Q-learning reward variants | "
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
        "q_table_data": last_q_table_data,
        "q_per_node": last_q_per_node,
        "q_dists_m": last_q_dists_m,
    }


def _variant_color(variant_key: str, variant_keys: tuple[str, ...]) -> str:
    idx = list(variant_keys).index(variant_key) if variant_key in variant_keys else 0
    return VARIANT_COLORS[idx % len(VARIANT_COLORS)]


def _plot(
    epoch_data: dict[str, list[dict]],
    out_dir: str,
    config: RewardVariantConfig,
    variant_keys: tuple[str, ...],
) -> None:
    panels = [
        ("success_rate",          "ASR per Epoch",            "Success rate",          False),
        ("throughput",            "Throughput per Epoch",     "Successes / slot",      False),
        ("collision_rate",        "Collision Rate per Epoch", "Collision rate",         False),
        ("attempts",              "Attempts per Epoch",       "Transmissions",          False),
        ("mean_backlog_per_node", "Mean Backlog per Epoch",   "Packets queued / node", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    p_gen_equiv = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    fig.suptitle(
        "Reward Variant Comparison\n"
        f"Q-learning (decentralized) | N={config.n_nodes} | G={config.target_g:.2f} | "
        f"p_gen≈{p_gen_equiv:.3f} | queue={config.queue_mode} | slots={config.n_slots}",
        fontsize=12,
        fontweight="bold",
    )

    for ax, (metric, title, ylabel, log_y) in zip(axes.flat, panels):
        for idx, variant_key in enumerate(variant_keys):
            epoch_log = epoch_data.get(variant_key, [])
            if not epoch_log:
                continue
            xs = [epoch["slot_end"] for epoch in epoch_log]
            ys = [epoch.get(metric, 0.0) for epoch in epoch_log]
            color = VARIANT_COLORS[idx % len(VARIANT_COLORS)]
            label = REWARD_VARIANTS[variant_key].get("plot_label", REWARD_VARIANTS[variant_key]["label"])
            if log_y:
                ys = [max(value, 1e-3) for value in ys]
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
    out_path = os.path.join(out_dir, "reward_variant_compare.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def _plot_per_node(
    rows: list[dict],
    out_dir: str,
    config: RewardVariantConfig,
    variant_keys: tuple[str, ...],
) -> None:
    grouped: dict[str, dict] = {}
    for row in rows:
        key = row["variant_key"]
        grouped.setdefault(key, {
            "label": row["variant_label"],
            "node_ids": [],
            "asr": [],
            "throughput": [],
            "resource_load": [],
        })
        grouped[key]["node_ids"].append(row["node_id"])
        grouped[key]["asr"].append(row["asr"])
        grouped[key]["throughput"].append(row["throughput"])
        grouped[key]["resource_load"].append(row["resource_load"])

    p_gen_equiv = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"Per-node Breakdown | N={config.n_nodes} | G={config.target_g:.2f} | "
        f"p_gen≈{p_gen_equiv:.3f} | layout={config.layout}",
        fontsize=12,
        fontweight="bold",
    )

    scatter_metrics = [
        ("asr", "Per-node ASR", "ASR"),
        ("throughput", "Per-node Throughput", "Successes / measured slot"),
        ("resource_load", "Per-node Resource Load", "Attempts / (resource × slot)"),
    ]

    for col, (metric, title, ylabel) in enumerate(scatter_metrics):
        ax = axes[col]
        for variant_key in variant_keys:
            data = grouped.get(variant_key)
            if not data:
                continue
            color = _variant_color(variant_key, variant_keys)
            label = REWARD_VARIANTS[variant_key].get("plot_label", REWARD_VARIANTS[variant_key]["label"])
            ax.scatter(data["node_ids"], data[metric], color=color, s=18, alpha=0.55, label=label, zorder=3)
            ax.axhline(float(np.mean(data[metric])), color=color, linewidth=1.5, alpha=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Node ID", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        if metric == "asr":
            ax.set_ylim(-0.02, 1.05)

    axes[0].legend(loc="best", fontsize=8, framealpha=0.9)

    # 실패 원인을 누적 막대로 분해해 보여준다.
    ax = axes[3]
    labels, suc_frac, coll_frac, link_frac = [], [], [], []
    for variant_key in variant_keys:
        key_rows = [r for r in rows if r["variant_key"] == variant_key]
        successes = sum(r["successes"] for r in key_rows)
        collisions = sum(r["collisions"] for r in key_rows)
        link_failures = sum(r["link_failures"] for r in key_rows)
        total = successes + collisions + link_failures
        labels.append(REWARD_VARIANTS[variant_key].get("plot_label", REWARD_VARIANTS[variant_key]["label"]))
        suc_frac.append(successes / total if total else 0.0)
        coll_frac.append(collisions / total if total else 0.0)
        link_frac.append(link_failures / total if total else 0.0)

    x_idx = list(range(len(labels)))
    ax.bar(x_idx, suc_frac, label="Success", color="#2ca02c", alpha=0.85)
    ax.bar(x_idx, coll_frac, bottom=suc_frac, label="Collision", color="#d62728", alpha=0.85)
    ax.bar(x_idx, link_frac, bottom=[s + c for s, c in zip(suc_frac, coll_frac)], label="Link Fail", color="#ff7f0e", alpha=0.85)
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
    config: RewardVariantConfig,
    variant_keys: tuple[str, ...],
) -> None:
    n_variants = len(variant_keys)
    ncols = min(n_variants, 3)
    nrows = (n_variants + ncols - 1) // ncols
    p_gen_equiv = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(
        f"Node Layout (ASR color) | N={config.n_nodes} | G={config.target_g:.2f} | "
        f"p_gen≈{p_gen_equiv:.3f} | layout={config.layout}",
        fontsize=12,
        fontweight="bold",
    )

    asr_lookup = {(r["variant_key"], r["node_id"]): r["asr"] for r in rows}

    for idx, variant_key in enumerate(variant_keys):
        ax = axes[idx // ncols][idx % ncols]
        xs, ys, _ = node_positions[variant_key]
        asr_vals = [asr_lookup.get((variant_key, node_id), 0.0) for node_id in range(len(xs))]
        sc = ax.scatter(xs, ys, c=asr_vals, cmap="RdYlGn", vmin=0.0, vmax=1.0, s=60, alpha=0.85, edgecolors="none", zorder=3)
        ax.scatter([0], [0], c="black", marker="*", s=200, zorder=5, label="Gateway")
        fig.colorbar(sc, ax=ax, label="ASR")
        label = REWARD_VARIANTS[variant_key].get("plot_label", REWARD_VARIANTS[variant_key]["label"])
        ax.set_title(label, fontsize=10)
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
