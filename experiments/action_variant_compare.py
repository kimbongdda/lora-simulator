"""Q-learning 액션공간 변형(relative vs absolute, 채널혼잡도 유무)을 비교한다."""

from __future__ import annotations

import dataclasses
import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

from agents.q_learning import ACTION_VARIANTS, DEFAULT_REWARD_VARIANT, DEFAULT_STATE_VARIANT, n_actions
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

# 액션 변형 × 상태 변형 조합으로 정의하는 시리즈.
# state_variant=None이면 ActionVariantConfig.state_variant를 사용한다.
SERIES_DEFS: dict[str, dict] = {
    "relative": {
        "label": "Relative (SF±1)",
        "av_key": "relative",
    },
    "absolute": {
        "label": "Absolute (SF직접)",
        "av_key": "absolute",
    },
}


@dataclasses.dataclass(frozen=True)
class ActionVariantConfig:
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
    enable_rician_fading: bool = False
    rician_k_factor: float = 4.0
    rician_fade_margin_db: float = 5.0
    gw_thresholds: tuple[float, ...] = (0.7, 1.3)
    layout: str = "ring"
    controller_key: str = "decentralized_q_learning"
    reward_variant: str = DEFAULT_REWARD_VARIANT
    state_variant: str = DEFAULT_STATE_VARIANT
    action_series_keys: tuple[str, ...] = tuple(SERIES_DEFS)
    psi: float = 0.0
    E0: float = 10.0
    W: int = 20
    er_mode: str = "ETD"
    mu: float = 0.5
    output_dir: str = os.path.join("outputs", "action_variant_compare")


def _validate_config(config: ActionVariantConfig) -> tuple[str, ...]:
    if config.controller_key not in Q_CONTROLLER_KEYS:
        raise ValueError(f"controller_key must be one of {Q_CONTROLLER_KEYS}, got {config.controller_key!r}")
    valid_keys = tuple(k for k in config.action_series_keys if k in SERIES_DEFS)
    if not valid_keys:
        raise ValueError("action_series_keys must contain at least one valid series key.")
    return valid_keys


def run_action_variant_compare(config: ActionVariantConfig | None = None) -> dict:
    config = config or ActionVariantConfig()
    series_keys = _validate_config(config)
    out_dir = os.path.abspath(config.output_dir)
    ensure_dir(out_dir)

    epoch_data: dict[str, list[dict]] = {}
    epoch_rows: list[dict] = []
    final_rows: list[dict] = []
    per_node_rows: list[dict] = []
    node_positions: dict[str, tuple] = {}
    sf_data: dict[str, tuple] = {}

    for idx, series_key in enumerate(series_keys):
        series = SERIES_DEFS[series_key]
        av_key = series["av_key"]
        sv = config.state_variant

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
            enable_rician_fading=config.enable_rician_fading,
            rician_k_factor=config.rician_k_factor,
            rician_fade_margin_db=config.rician_fade_margin_db,
            gw_thresholds=config.gw_thresholds,
            layout=config.layout,
        )
        controller = make_controller(
            config.controller_key,
            reward_variant=config.reward_variant,
            state_variant=sv,
            action_variant=av_key,
            psi=config.psi,
            E0=config.E0,
            W=config.W,
            er_mode=config.er_mode,
            mu=config.mu,
        )
        result = run_simulation(scenario, controller)

        annotated_log = []
        for ep in result["epoch_log"]:
            row = dict(ep)
            row["baseline_label"] = series["label"]
            row["series_key"] = series_key
            annotated_log.append(row)
        epoch_data[series_key] = annotated_log

        node_positions[series_key] = (result["xs"], result["ys"], result["dists_m"])
        sf_data[series_key] = (
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
                "series_key": series_key,
                "av_label": series["label"],
                "node_id": node_id,
                "attempts": int(attempts),
                "successes": int(successes),
                "collisions": int(collisions),
                "link_failures": int(link_failures),
                "asr": successes / attempts if attempts > 0 else 0.0,
                "throughput": successes / measured if measured > 0 else 0.0,
                "resource_load": attempts / (n_resources * measured) if measured > 0 else 0.0,
            })

        n_act = n_actions(config.n_channels, av_key)
        pns = per_success
        min_max_ratio = float(pns.min() / pns.max()) if pns.max() > 0 else 0.0
        final_rows.append({
            "av_key": series_key,
            "av_label": series["label"],
            "n_actions": n_act,
            "state_variant": sv,
            "av_description": ACTION_VARIANTS[av_key]["description"],
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
            row["series_key"] = series_key
            row["av_label"] = series["label"]
            epoch_rows.append(row)

        print(
            f"  {series['label']:<36} "
            f"av={av_key:<8} sv={sv:<14} "
            f"n_actions={n_act:2d}  "
            f"ASR={result['success_rate']:.3f}  "
            f"thr={result['throughput']:.4f}  "
            f"jain={result['fairness']:.4f}  "
            f"backlog/node={result['final_backlog_per_node']:.2f}"
        )

    write_csv_rows(os.path.join(out_dir, "action_variant_epochs.csv"), epoch_rows)
    write_csv_rows(os.path.join(out_dir, "action_variant_summary.csv"), final_rows)
    write_csv_rows(os.path.join(out_dir, "action_variant_per_node.csv"), per_node_rows)
    write_json(
        os.path.join(out_dir, "metadata.json"),
        {
            "config": dataclasses.asdict(config),
            "series": {k: SERIES_DEFS[k] for k in series_keys},
            "notes": {
                "controller_key": "Action-variant comparison uses decentralized Q-learning only.",
                "default_state_variant": config.state_variant,
                "reward_variant": config.reward_variant,
            },
        },
    )

    p_gen_equiv = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    _plot(epoch_data, out_dir, config, series_keys)
    _plot_summary_bar(final_rows, out_dir, config, series_keys)
    plot_sf_distance_heatmap(
        sf_data=sf_data,
        labels={k: SERIES_DEFS[k]["label"] for k in series_keys},
        keys=list(series_keys),
        suptitle=(
            f"SF Selection vs Distance | Q-learning action variants | "
            f"N={config.n_nodes} | G={config.target_g:.2f} | p_gen={p_gen_equiv:.3f}"
        ),
        out_path=os.path.join(out_dir, "sf_heatmap.png"),
    )
    _av_labels = {k: SERIES_DEFS[k]["label"] for k in series_keys}
    _av_colors = {k: VARIANT_COLORS[i % len(VARIANT_COLORS)] for i, k in enumerate(series_keys)}
    plot_channel_g_timeseries(
        epoch_data=epoch_data,
        labels=_av_labels,
        colors=_av_colors,
        n_channels=config.n_channels,
        out_path=os.path.join(out_dir, "channel_g.png"),
        suptitle=(
            f"Per-Channel G | Q-learning action variants | "
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
    }


def _series_color(series_key: str, series_keys: tuple[str, ...]) -> str:
    idx = list(series_keys).index(series_key) if series_key in series_keys else 0
    return VARIANT_COLORS[idx % len(VARIANT_COLORS)]


def _plot(
    epoch_data: dict[str, list[dict]],
    out_dir: str,
    config: ActionVariantConfig,
    series_keys: tuple[str, ...],
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
        "Action Variant Comparison\n"
        f"Q-learning (decentralized) | N={config.n_nodes} | G={config.target_g:.2f} | "
        f"p_gen={p_gen:.3f} | default_state={config.state_variant} | reward={config.reward_variant} | slots={config.n_slots}",
        fontsize=12,
        fontweight="bold",
    )

    for ax, (metric, title, ylabel, log_y) in zip(axes.flat, panels):
        for idx, series_key in enumerate(series_keys):
            log = epoch_data.get(series_key, [])
            if not log:
                continue
            series = SERIES_DEFS[series_key]
            av_key = series["av_key"]
            sv = config.state_variant
            xs = [e["slot_end"] for e in log]
            ys = [e.get(metric, 0.0) for e in log]
            color = VARIANT_COLORS[idx % len(VARIANT_COLORS)]
            n_act = n_actions(config.n_channels, av_key)
            label = f"{series['label']} ({n_act}act, {sv})"
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

    axes[0, 0].legend(loc="best", fontsize=7.5, framealpha=0.9)
    axes[1, 2].set_visible(False)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path = os.path.join(out_dir, "action_variant_compare.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def _plot_summary_bar(
    final_rows: list[dict],
    out_dir: str,
    config: ActionVariantConfig,
    series_keys: tuple[str, ...],
) -> None:
    metrics = [
        ("success_rate",          "ASR",                  (0.0, 1.05)),
        ("throughput",            "Throughput (pkts/slot)", None),
        ("collision_rate",        "Collision Rate",        (0.0, 1.05)),
        ("mean_backlog_per_node", "Mean Backlog / Node",   None),
        ("fairness",              "Jain's Fairness",       (0.0, 1.05)),
    ]

    p_gen = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    fig.suptitle(
        f"Action Variant Summary | N={config.n_nodes} | G={config.target_g:.2f} | "
        f"p_gen={p_gen:.3f} | default_state={config.state_variant}",
        fontsize=11,
        fontweight="bold",
    )

    row_by_key = {r["av_key"]: r for r in final_rows}
    x_labels = []
    for k in series_keys:
        series = SERIES_DEFS[k]
        av_key = series["av_key"]
        sv = config.state_variant
        n_act = n_actions(config.n_channels, av_key)
        x_labels.append(f"{series['label']}\n({n_act}act)")
    x_pos = list(range(len(series_keys)))

    for ax, (metric, title, ylim) in zip(axes, metrics):
        vals = [row_by_key[k].get(metric, 0.0) if k in row_by_key else 0.0 for k in series_keys]
        colors = [VARIANT_COLORS[i % len(VARIANT_COLORS)] for i in range(len(series_keys))]
        ax.bar(x_pos, vals, color=colors, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out_path = os.path.join(out_dir, "summary_bar.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def main() -> None:
    result = run_action_variant_compare()
    print(f"Outputs saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
