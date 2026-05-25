"""ER-Q 변형 비교: baseline vs ER-ETD vs ER-EM."""

from __future__ import annotations

import dataclasses
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines.catalog import make_controller
from env import ScenarioConfig, run_simulation
from experiments.style import configure_fonts
from utils.io import ensure_dir, write_csv_rows, write_json


ER_SERIES: list[dict] = [
    {"key": "baseline",       "label": "Q-learning (no ER)",          "color": "#4b0082", "linestyle": (0, (3, 1, 1, 1))},
    {"key": "er_etd",         "label": "ER-ETD",                      "color": "#d62728", "linestyle": "-"},
    {"key": "er_em",          "label": "ER-EM",                       "color": "#ff7f0e", "linestyle": "--"},
    {"key": "er_etd_sfch",    "label": "ER-ETD (SF+CH)",              "color": "#1f77b4", "linestyle": "-"},
    {"key": "er_etd_rel_sfch","label": "ER-ETD (SF-delta+CH)",        "color": "#2ca02c", "linestyle": "--"},
]


@dataclasses.dataclass(frozen=True)
class ERCompareConfig:
    n_nodes: int = 60
    target_g: float = 1.0
    n_slots: int = 30_000
    warmup_slots: int = 0
    epoch_slots: int = 500
    n_channels: int = 3
    profile_name: str = "short"
    seed: int = 42
    queue_mode: str = "accumulate"
    gw_obs_mode: str = "attempt"
    reward_variant: str = "v0_current"
    state_variant: str = "s0_full"
    psi: float = 0.6
    E0: float = 10.0
    W: int = 20
    mu: float = 0.5
    series_keys: tuple[str, ...] = ("baseline", "er_etd", "er_em", "er_etd_sfch", "er_etd_rel_sfch")
    output_dir: str = os.path.join("outputs", "er_compare")


def _make_scenario(config: ERCompareConfig, seed_offset: int) -> ScenarioConfig:
    return ScenarioConfig(
        n_nodes=config.n_nodes,
        target_g=config.target_g,
        n_slots=config.n_slots,
        warmup_slots=config.warmup_slots,
        epoch_slots=config.epoch_slots,
        n_channels=config.n_channels,
        profile_name=config.profile_name,
        seed=config.seed + seed_offset,
        layout_seed=config.seed,
        queue_mode=config.queue_mode,
        gw_obs_mode=config.gw_obs_mode,
    )


def _run_all(config: ERCompareConfig):
    epoch_data: dict[str, list[dict]] = {}
    summary_rows: list[dict] = []

    series_map = {s["key"]: s for s in ER_SERIES}

    for idx, series_key in enumerate(config.series_keys):
        meta = series_map.get(series_key, {"key": series_key, "label": series_key, "color": "#888888", "linestyle": "-"})

        if series_key == "baseline":
            psi, er_mode, mu = 0.0, "ETD", 0.5
            state_variant  = config.state_variant
            action_variant = "relative"
        elif series_key == "er_etd":
            psi, er_mode, mu = config.psi, "ETD", 0.5
            state_variant  = config.state_variant
            action_variant = "relative"
        elif series_key == "er_em":
            psi, er_mode, mu = config.psi, "EM", config.mu
            state_variant  = config.state_variant
            action_variant = "relative"
        elif series_key == "er_etd_sfch":
            psi, er_mode, mu = config.psi, "ETD", 0.5
            state_variant  = "s13_sf_ch"
            action_variant = "absolute"
        else:  # er_etd_rel_sfch
            psi, er_mode, mu = config.psi, "ETD", 0.5
            state_variant  = "s14_sf_delta_ch"
            action_variant = "relative"

        scenario = _make_scenario(config, seed_offset=10_000 * (idx + 1))
        controller = make_controller(
            "decentralized_q_learning",
            config.reward_variant,
            state_variant,
            action_variant=action_variant,
            psi=psi,
            E0=config.E0,
            W=config.W,
            er_mode=er_mode,
            mu=mu,
        )
        result = run_simulation(scenario, controller)

        annotated: list[dict] = []
        for ep in result["epoch_log"]:
            row = dict(ep)
            row["series_key"] = series_key
            row["series_label"] = meta["label"]
            summary_rows.append(row)
            annotated.append(row)
        epoch_data[series_key] = annotated

        print(
            f"  {meta['label']:<30} "
            f"final backlog/node={result['final_backlog_per_node']:>8.1f} "
            f"system ASR={result['success_rate']:.3f}"
        )

    return epoch_data, summary_rows


def _epoch_series(epoch_log: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs = [ep["slot_end"] for ep in epoch_log]
    ys = [ep.get(key, 0.0) for ep in epoch_log]
    return xs, ys


def _plot(epoch_data: dict[str, list[dict]], out_path: str, config: ERCompareConfig) -> None:
    configure_fonts()
    series_map = {s["key"]: s for s in ER_SERIES}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"ER Compare: Q-learning vs ETD vs EM | N={config.n_nodes} G={config.target_g} "
        f"ψ={config.psi} E₀={config.E0} W={config.W} μ={config.mu} | "
        f"slots={config.n_slots} epoch={config.epoch_slots}",
        fontsize=11,
        fontweight="bold",
    )

    panels_top = [
        ("attempt_load_realized", "Realized Attempt Load (G_att)", "attempts / (resources × epoch_slots)"),
        ("success_rate", "ASR per Epoch", "Success rate"),
        ("throughput", "Throughput per Epoch", "Successes / slot"),
    ]
    panels_bottom = [
        ("mean_backlog_per_node", "Mean Backlog / Node", "Packets queued / node", False),
        ("collision_rate", "Collision Rate per Epoch", "Collision rate", False),
        ("mean_backlog_per_node", "Mean Backlog / Node (log scale)", "Packets queued / node", True),
    ]

    n_resources = config.n_channels * 6

    for col, (metric, title, ylabel) in enumerate(panels_top):
        ax = axes[0, col]
        for series_key, epoch_log in epoch_data.items():
            if not epoch_log:
                continue
            meta = series_map.get(series_key, {"label": series_key, "color": "#888888", "linestyle": "-"})
            if metric == "attempt_load_realized":
                xs = [ep["slot_end"] for ep in epoch_log]
                ys = [ep["attempts"] / (n_resources * config.epoch_slots) for ep in epoch_log]
            else:
                xs, ys = _epoch_series(epoch_log, metric)
            ax.plot(xs, ys, label=meta["label"], color=meta["color"],
                    linestyle=meta["linestyle"], linewidth=1.8, alpha=0.9)

        if metric == "attempt_load_realized":
            ax.axhline(config.target_g, color="#444444", linewidth=1.2, linestyle="--",
                       alpha=0.7, label=f"target G={config.target_g}")

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
            meta = series_map.get(series_key, {"label": series_key, "color": "#888888", "linestyle": "-"})
            xs, ys = _epoch_series(epoch_log, metric)
            if log_y:
                ys = [max(v, 1e-3) for v in ys]
                ax.semilogy(xs, ys, label=meta["label"], color=meta["color"],
                            linestyle=meta["linestyle"], linewidth=1.8, alpha=0.9)
            else:
                ax.plot(xs, ys, label=meta["label"], color=meta["color"],
                        linestyle=meta["linestyle"], linewidth=1.8, alpha=0.9)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Slot", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        if metric == "collision_rate":
            ax.set_ylim(-0.02, 1.05)

    axes[0, 0].legend(loc="upper right", fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def run_er_compare(config: ERCompareConfig | None = None) -> dict:
    config = config or ERCompareConfig()
    out_dir = os.path.abspath(config.output_dir)
    ensure_dir(out_dir)

    epoch_data, all_rows = _run_all(config)
    write_csv_rows(os.path.join(out_dir, "er_compare.csv"), all_rows)
    write_json(
        os.path.join(out_dir, "metadata.json"),
        {"config": dataclasses.asdict(config)},
    )
    out_img = os.path.join(out_dir, "er_compare.png")
    _plot(epoch_data, out_img, config)
    return {"output_dir": out_dir, "epoch_data": epoch_data, "epoch_rows": all_rows}


def main() -> None:
    result = run_er_compare()
    print(f"Outputs saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
