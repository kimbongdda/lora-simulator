"""충돌 중심 LoRaWAN 설정에서 노드별 분해 분석을 수행한다."""

from __future__ import annotations

import dataclasses
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import BASELINE_ORDER, expand_run_specs, get_baseline_specs, make_controller
from env import ScenarioConfig, run_simulation
from experiments.plots import plot_channel_g_timeseries, plot_sf_distance_heatmap
from experiments.style import series_color, series_linestyle, series_marker
from utils.io import ensure_dir, write_csv_rows, write_json


@dataclasses.dataclass(frozen=True)
class PerNodeConfig:
    n_nodes: int = 60
    target_g: float = 1.0
    n_slots: int = 20_000
    warmup_slots: int = 5_000
    epoch_slots: int = 1_000
    n_channels: int = 3
    fixed_distance_ratio: float = 0.5
    profile_name: str = "short"
    seed: int = 42
    queue_mode: str = "accumulate"
    gw_obs_mode: str = "attempt"
    enable_rayleigh_fading: bool = False
    rayleigh_fade_margin_db: float = 10.0
    dual_mab_b: float = 0.3
    layout: str = "random"
    reward_variant: str = "v0_current"
    reward_variants: tuple[str, ...] = ()
    state_variant: str = "s0_full"
    output_dir: str = os.path.join("outputs", "per_node_analysis")
    baseline_keys: tuple[str, ...] = tuple(BASELINE_ORDER)


def _resolved_reward_variants(config: PerNodeConfig) -> tuple[str, ...]:
    return tuple(config.reward_variants) if config.reward_variants else (config.reward_variant,)


def _run_all(config: PerNodeConfig):
    rows = []
    meta_rows = []
    epoch_data: dict[str, list[dict]] = {}
    node_positions: dict[str, tuple] = {}
    sf_data: dict[str, tuple] = {}
    last_q_table_data = None
    last_q_per_node = None
    last_q_dists_m = None
    run_specs = expand_run_specs(config.baseline_keys, _resolved_reward_variants(config))

    for idx, run_spec in enumerate(run_specs):
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
        controller = make_controller(run_spec.base_key, run_spec.reward_variant or config.reward_variant, config.state_variant, dual_mab_b=config.dual_mab_b)
        result = run_simulation(scenario, controller)
        if result.get("q_table_data") is not None:
            last_q_table_data = result["q_table_data"]
        if result.get("q_per_node") is not None:
            last_q_per_node = result["q_per_node"]
            last_q_dists_m = result["dists_m"]

        annotated_epoch_log = []
        for ep in result["epoch_log"]:
            row = dict(ep)
            row["baseline_key"] = run_spec.series_key
            row["base_baseline_key"] = run_spec.base_key
            row["baseline_label"] = run_spec.label
            row["reward_variant"] = run_spec.reward_variant or ""
            annotated_epoch_log.append(row)
        epoch_data[run_spec.series_key] = annotated_epoch_log
        node_positions[run_spec.series_key] = (result["xs"], result["ys"], result["dists_m"])
        sf_data[run_spec.series_key] = (
            np.asarray(result["sf_usage"]),
            np.asarray(result["idle_usage"]),
            result["dists_m"],
            result["cell_radius_m"],
        )

        measured = result["measured_slots"]
        per_success = np.asarray(result["per_node_success"], dtype=np.float64)
        per_attempts = np.asarray(result["per_node_attempts"], dtype=np.float64)
        per_collisions = np.asarray(result["per_node_collisions"], dtype=np.float64)
        per_link_failures = np.asarray(result["per_node_link_failures"], dtype=np.float64)
        n_resources = config.n_channels * 6

        for node_id in range(config.n_nodes):
            attempts = per_attempts[node_id]
            successes = per_success[node_id]
            collisions = per_collisions[node_id]
            link_failures = per_link_failures[node_id]
            rows.append(
                {
                    "baseline_key": run_spec.series_key,
                    "base_baseline_key": run_spec.base_key,
                    "baseline_label": run_spec.label,
                    "reward_variant": run_spec.reward_variant or "",
                    "node_id": node_id,
                    "attempts": int(attempts),
                    "successes": int(successes),
                    "collisions": int(collisions),
                    "link_failures": int(link_failures),
                    "asr": successes / attempts if attempts > 0 else 0.0,
                    "throughput": successes / measured if measured > 0 else 0.0,
                    "resource_load": attempts / (n_resources * measured) if measured > 0 else 0.0,
                }
            )

        meta_rows.append(
            {
                "baseline_key": run_spec.series_key,
                "base_baseline_key": run_spec.base_key,
                "baseline_label": run_spec.label,
                "reward_variant": run_spec.reward_variant or "",
                "system_success_rate": result["success_rate"],
                "system_collision_rate": result["collision_rate"],
                "system_throughput": result["throughput"],
                "fairness": result["fairness"],
                "mean_backlog_per_node": result["mean_backlog_per_node"],
            }
        )

    return rows, meta_rows, epoch_data, node_positions, sf_data, last_q_table_data, last_q_per_node, last_q_dists_m, run_specs


def _plot(rows: list[dict], epoch_data: dict[str, list[dict]], out_path: str, config: PerNodeConfig) -> None:
    grouped: dict[str, dict] = {}
    for row in rows:
        key = row["baseline_key"]
        grouped.setdefault(
            key,
            {
                "label": row["baseline_label"],
                "base_key": row["base_baseline_key"],
                "reward_variant": row["reward_variant"],
                "node_ids": [],
                "asr": [],
                "throughput": [],
                "resource_load": [],
            },
        )
        grouped[key]["node_ids"].append(row["node_id"])
        grouped[key]["asr"].append(row["asr"])
        grouped[key]["throughput"].append(row["throughput"])
        grouped[key]["resource_load"].append(row["resource_load"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    p_gen_equiv = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    fig.suptitle(
        f"Node Analysis | N={config.n_nodes} | G={config.target_g} | p_gen≈{p_gen_equiv:.3f} | "
        f"profile={config.profile_name} | channels={config.n_channels}",
        fontsize=12,
        fontweight="bold",
    )

    scatter_metrics = [
        ("asr", "Per-node ASR", "Success rate (per node)"),
        ("throughput", "Per-node Throughput", "Successes / measured slot"),
        ("resource_load", "Per-node Resource Load", "Attempts / (resource * slot)"),
    ]

    for col, (metric, title, ylabel) in enumerate(scatter_metrics):
        ax = axes[0, col]
        for key, data in grouped.items():
            color = series_color(data["base_key"], data["reward_variant"])
            marker = series_marker(data["base_key"], data["reward_variant"])
            ax.scatter(
                data["node_ids"],
                data[metric],
                color=color,
                marker=marker,
                s=22,
                alpha=0.65,
                label=data["label"],
                zorder=3,
            )
            ax.axhline(
                float(np.mean(data[metric])),
                color=color,
                linewidth=1.5,
                linestyle=series_linestyle(data["base_key"], data["reward_variant"]),
                alpha=0.9,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Node ID", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        if metric == "asr":
            ax.set_ylim(-0.02, 1.05)

    axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.9)

    ts_panels = [
        ("success_rate", "Success Rate vs Slot", "ASR per epoch"),
        ("throughput", "Throughput vs Slot", "Successes / slot per epoch"),
    ]
    for col, (metric, title, ylabel) in enumerate(ts_panels):
        ax = axes[1, col]
        for key, epoch_log in epoch_data.items():
            if not epoch_log:
                continue
            base_key = epoch_log[0]["base_baseline_key"]
            reward_variant = epoch_log[0]["reward_variant"]
            color = series_color(base_key, reward_variant)
            linestyle = series_linestyle(base_key, reward_variant)
            xs = [ep["slot_end"] for ep in epoch_log]
            ys = [ep.get(metric, 0.0) for ep in epoch_log]
            ax.plot(xs, ys, label=epoch_log[0]["baseline_label"], color=color, linestyle=linestyle, linewidth=1.8, alpha=0.9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Slot", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        if metric == "success_rate":
            ax.set_ylim(-0.02, 1.05)

    ax = axes[1, 2]
    labels = []
    suc_frac = []
    coll_frac = []
    link_frac = []
    for key, data in grouped.items():
        key_rows = [row for row in rows if row["baseline_key"] == key]
        successes = sum(row["successes"] for row in key_rows)
        collisions = sum(row["collisions"] for row in key_rows)
        link_failures = sum(row["link_failures"] for row in key_rows)
        total = successes + collisions + link_failures
        labels.append(data["label"])
        suc_frac.append(successes / total if total else 0.0)
        coll_frac.append(collisions / total if total else 0.0)
        link_frac.append(link_failures / total if total else 0.0)
    x_idx = list(range(len(labels)))
    ax.bar(x_idx, suc_frac, label="Success", color="#2ca02c", alpha=0.85)
    ax.bar(x_idx, coll_frac, bottom=suc_frac, label="Collision", color="#d62728", alpha=0.85)
    ax.bar(x_idx, link_frac, bottom=[s + c for s, c in zip(suc_frac, coll_frac)], label="Link Fail", color="#ff7f0e", alpha=0.85)
    ax.set_title("Failure Cause Breakdown", fontsize=11)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of attempts", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def _plot_layout(rows: list[dict], node_positions: dict[str, tuple], out_path: str, config: PerNodeConfig) -> None:
    keys = list(node_positions.keys())
    n_baselines = len(keys)
    ncols = min(n_baselines, 3)
    nrows = (n_baselines + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    p_gen_equiv = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    fig.suptitle(
        f"Node Layout (ASR color) | N={config.n_nodes} | G={config.target_g} | p_gen≈{p_gen_equiv:.3f} | layout={config.layout}",
        fontsize=12,
        fontweight="bold",
    )

    asr_lookup = {(row["baseline_key"], row["node_id"]): row["asr"] for row in rows}

    for idx, key in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        xs, ys, _ = node_positions[key]
        asr_vals = [asr_lookup.get((key, node_id), 0.0) for node_id in range(len(xs))]
        sc = ax.scatter(xs, ys, c=asr_vals, cmap="RdYlGn", vmin=0.0, vmax=1.0, s=60, alpha=0.85, edgecolors="none", zorder=3)
        ax.scatter([0], [0], c="black", marker="*", s=200, zorder=5, label="Gateway")
        fig.colorbar(sc, ax=ax, label="ASR")
        label = next(row["baseline_label"] for row in rows if row["baseline_key"] == key)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("y (m)", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")

    for idx in range(n_baselines, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def run_per_node_analysis(config: PerNodeConfig | None = None) -> dict:
    config = config or PerNodeConfig()
    out_dir = os.path.abspath(config.output_dir)
    ensure_dir(out_dir)

    rows, meta_rows, epoch_data, node_positions, sf_data, last_q_table_data, last_q_per_node, last_q_dists_m, run_specs = _run_all(config)
    write_csv_rows(os.path.join(out_dir, "per_node.csv"), rows)
    write_csv_rows(os.path.join(out_dir, "system_summary.csv"), meta_rows)
    write_json(
        os.path.join(out_dir, "metadata.json"),
        {
            "config": dataclasses.asdict(config),
            "run_specs": [dataclasses.asdict(spec) for spec in run_specs],
            "baselines": [dataclasses.asdict(spec) for spec in get_baseline_specs() if spec.key in config.baseline_keys],
            "metrics": {
                "asr": "successes / attempts (per node, 0 if no attempts)",
                "throughput": "successes / measured_slots (per node)",
                "resource_load": "attempts / (n_resources * measured_slots) (per node, G units)",
            },
        },
    )
    p_gen_equiv = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    _plot(rows, epoch_data, os.path.join(out_dir, "per_node.png"), config)
    _plot_layout(rows, node_positions, os.path.join(out_dir, "node_layout.png"), config)
    plot_sf_distance_heatmap(
        sf_data=sf_data,
        labels={key: next(r["baseline_label"] for r in rows if r["baseline_key"] == key) for key in sf_data},
        keys=list(sf_data.keys()),
        suptitle=(
            f"SF Selection vs Distance | N={config.n_nodes} | G={config.target_g:.2f} | "
            f"p_gen≈{p_gen_equiv:.3f} | layout={config.layout}"
        ),
        out_path=os.path.join(out_dir, "sf_heatmap.png"),
    )
    _ch_labels = {k: next(r["baseline_label"] for r in rows if r["baseline_key"] == k) for k in epoch_data}
    _ch_colors = {
        k: series_color(epoch_data[k][0]["base_baseline_key"], epoch_data[k][0].get("reward_variant", ""))
        if epoch_data[k] else "#888888"
        for k in epoch_data
    }
    plot_channel_g_timeseries(
        epoch_data=epoch_data,
        labels=_ch_labels,
        colors=_ch_colors,
        n_channels=config.n_channels,
        out_path=os.path.join(out_dir, "channel_g.png"),
        suptitle=(
            f"Per-Channel G | N={config.n_nodes} | G={config.target_g:.2f} | "
            f"p_gen≈{p_gen_equiv:.3f}"
        ),
        target_g=config.target_g,
    )
    return {
        "output_dir": out_dir,
        "rows": rows,
        "meta_rows": meta_rows,
        "epoch_data": epoch_data,
        "node_positions": node_positions,
        "sf_heatmap_image": os.path.join(out_dir, "sf_heatmap.png"),
        "channel_g_image": os.path.join(out_dir, "channel_g.png"),
        "q_table_data": last_q_table_data,
        "q_per_node": last_q_per_node,
        "q_dists_m": last_q_dists_m,
    }


def main() -> None:
    result = run_per_node_analysis()
    print(f"Outputs saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
