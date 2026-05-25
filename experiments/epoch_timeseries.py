"""에폭 단위 시계열 진단 도구."""

from __future__ import annotations

import dataclasses
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import BASELINE_ORDER, expand_run_specs, get_baseline_specs, make_controller
from env import ScenarioConfig, run_simulation
from experiments.style import series_color, series_linestyle
from utils.io import ensure_dir, write_csv_rows, write_json


@dataclasses.dataclass(frozen=True)
class TimeseriesConfig:
    n_nodes: int = 60
    target_g: float = 1.0
    n_slots: int = 30_000
    warmup_slots: int = 0
    epoch_slots: int = 500
    n_channels: int = 3
    fixed_distance_ratio: float = 0.5
    profile_name: str = "short"
    seed: int = 42
    queue_mode: str = "accumulate"
    gw_obs_mode: str = "attempt"
    enable_rayleigh_fading: bool = False
    enable_rician_fading: bool = False
    rician_k_factor: float = 4.0
    rician_fade_margin_db: float = 5.0
    dual_mab_b: float = 0.3
    psi: float = 0.0
    E0: float = 10.0
    W: int = 20
    er_mode: str = "ETD"
    mu: float = 0.5
    rayleigh_fade_margin_db: float = 10.0
    reward_variant: str = "v0_current"
    reward_variants: tuple[str, ...] = ()
    state_variant: str = "s0_full"
    output_dir: str = os.path.join("outputs", "epoch_timeseries")
    baseline_keys: tuple[str, ...] = tuple(BASELINE_ORDER)


def _resolved_reward_variants(config: TimeseriesConfig) -> tuple[str, ...]:
    return tuple(config.reward_variants) if config.reward_variants else (config.reward_variant,)


def _run_all(config: TimeseriesConfig):
    epoch_data: dict[str, list[dict]] = {}
    summary_rows = []
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
            enable_rician_fading=config.enable_rician_fading,
            rician_k_factor=config.rician_k_factor,
            rician_fade_margin_db=config.rician_fade_margin_db,
        )
        controller = make_controller(run_spec.base_key, run_spec.reward_variant or config.reward_variant, config.state_variant, dual_mab_b=config.dual_mab_b, psi=config.psi, E0=config.E0, W=config.W, er_mode=config.er_mode, mu=config.mu)
        result = run_simulation(scenario, controller)

        annotated_log = []
        for ep in result["epoch_log"]:
            row = dict(ep)
            row["baseline_key"] = run_spec.series_key
            row["base_baseline_key"] = run_spec.base_key
            row["baseline_label"] = run_spec.label
            row["reward_variant"] = run_spec.reward_variant or ""
            summary_rows.append(row)
            annotated_log.append(row)
        epoch_data[run_spec.series_key] = annotated_log

        print(
            f"  {run_spec.label:<42} "
            f"final backlog/node={result['final_backlog_per_node']:>8.1f} "
            f"system ASR={result['success_rate']:.3f}"
        )

    return epoch_data, summary_rows, run_specs


def _epoch_series(epoch_log: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs = [ep["slot_end"] for ep in epoch_log]
    ys = [ep.get(key, 0.0) for ep in epoch_log]
    return xs, ys


def _plot(epoch_data: dict[str, list[dict]], out_path: str, config: TimeseriesConfig) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Epoch Time-Series | N={config.n_nodes} | G={config.target_g} | "
        f"profile={config.profile_name} | channels={config.n_channels} | "
        f"slots={config.n_slots} | epoch={config.epoch_slots} | warmup={config.warmup_slots}",
        fontsize=12,
        fontweight="bold",
    )

    panels_top = [
        ("attempt_load_realized", "Realized Attempt Load (G_att)", "attempts / (resources * epoch_slots)"),
        ("success_rate", "ASR per Epoch", "Success rate"),
        ("throughput", "Throughput per Epoch", "Successes / slot"),
    ]
    panels_bottom = [
        ("mean_backlog_per_node", "Mean Backlog / Node", "Packets queued / node", False),
        ("collision_rate", "Collision Rate per Epoch", "Collision rate", False),
        ("mean_backlog_per_node", "Mean Backlog / Node (log scale)", "Packets queued / node", True),
    ]

    for col, (metric, title, ylabel) in enumerate(panels_top):
        ax = axes[0, col]
        for series_key, epoch_log in epoch_data.items():
            if not epoch_log:
                continue
            base_key = epoch_log[0].get("base_baseline_key", series_key)
            reward_variant = epoch_log[0].get("reward_variant", "")
            color = series_color(base_key, reward_variant)
            linestyle = series_linestyle(base_key, reward_variant)
            if metric == "attempt_load_realized":
                n_resources = config.n_channels * 6
                xs = [ep["slot_end"] for ep in epoch_log]
                ys = [ep["attempts"] / (n_resources * config.epoch_slots) for ep in epoch_log]
            else:
                xs, ys = _epoch_series(epoch_log, metric)
            ax.plot(xs, ys, label=epoch_log[0]["baseline_label"], color=color, linestyle=linestyle, linewidth=1.8, alpha=0.9)

        if metric == "attempt_load_realized":
            ax.axhline(config.target_g, color="#444444", linewidth=1.2, linestyle="--", alpha=0.7, label=f"target G={config.target_g}")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Slot", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        if metric in ("success_rate", "collision_rate"):
            ax.set_ylim(-0.02, 1.05)

    for col, (metric, title, ylabel, log_y) in enumerate(panels_bottom):
        ax = axes[1, col]
        for series_key, epoch_log in epoch_data.items():
            if not epoch_log:
                continue
            base_key = epoch_log[0].get("base_baseline_key", series_key)
            reward_variant = epoch_log[0].get("reward_variant", "")
            color = series_color(base_key, reward_variant)
            linestyle = series_linestyle(base_key, reward_variant)
            xs, ys = _epoch_series(epoch_log, metric)
            if log_y:
                ys = [max(value, 1e-3) for value in ys]
                ax.semilogy(xs, ys, label=epoch_log[0]["baseline_label"], color=color, linestyle=linestyle, linewidth=1.8, alpha=0.9)
            else:
                ax.plot(xs, ys, label=epoch_log[0]["baseline_label"], color=color, linestyle=linestyle, linewidth=1.8, alpha=0.9)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Slot", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        if metric == "collision_rate":
            ax.set_ylim(-0.02, 1.05)

    axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def run_epoch_timeseries(config: TimeseriesConfig | None = None) -> dict:
    config = config or TimeseriesConfig()
    out_dir = os.path.abspath(config.output_dir)
    ensure_dir(out_dir)

    epoch_data, all_epoch_rows, run_specs = _run_all(config)
    write_csv_rows(os.path.join(out_dir, "epoch_timeseries.csv"), all_epoch_rows)
    write_json(
        os.path.join(out_dir, "metadata.json"),
        {
            "config": dataclasses.asdict(config),
            "run_specs": [dataclasses.asdict(spec) for spec in run_specs],
            "baselines": [dataclasses.asdict(spec) for spec in get_baseline_specs() if spec.key in config.baseline_keys],
            "note": (
                "warmup_slots=0 so the full transient from slot 0 is visible. "
                "Unbounded backlog growth indicates the system cannot drain the queue at this G."
            ),
        },
    )
    _plot(epoch_data, os.path.join(out_dir, "epoch_timeseries.png"), config)
    return {"output_dir": out_dir, "epoch_data": epoch_data, "epoch_rows": all_epoch_rows}


def main() -> None:
    result = run_epoch_timeseries()
    print(f"Outputs saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
