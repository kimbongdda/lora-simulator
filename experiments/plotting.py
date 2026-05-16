"""기준선 비교 스윕용 시각화 보조 함수."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.style import DEFAULT_MARKERSIZE, configure_fonts, series_color, series_linestyle, series_marker


def _group_by_baseline(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["baseline_key"], []).append(row)
    for items in grouped.values():
        items.sort(key=lambda rec: rec["x_value"])
    return grouped


def plot_summary_figure(g_rows: list[dict], n_rows: list[dict], out_path: str, meta: dict) -> None:
    configure_fonts()

    g_grouped = _group_by_baseline(g_rows)
    n_grouped = _group_by_baseline(n_rows)

    n_label = meta.get("n_sweep_label", f"G = {meta.get('n_ref_g', '?')}")

    fig, axes = plt.subplots(4, 4, figsize=(22, 17))
    # G-sweep 행은 3개 패널만 사용 → 4번째 열 숨김
    axes[0, 3].set_visible(False)
    axes[1, 3].set_visible(False)
    axes[3, 3].set_visible(False)
    fig.suptitle(
        "Collision-Focused LoRaWAN Baseline Comparison\n"
        f"Profile={meta['profile_name']} | R_fixed={meta['fixed_radius_m']:.1f} m | "
        f"channels={meta['n_channels']} | resources={meta['n_resources']} | "
        f"slots={meta['n_slots']} | warmup={meta['warmup_slots']} | queue=accumulating",
        fontsize=12,
        fontweight="bold",
    )

    g_sub = f"N = {meta['g_ref_n']}"
    panels = [
        ("collision_rate",        "Collision Rate vs G",        "Collision rate",           axes[0, 0], g_grouped, g_sub),
        ("success_rate",          "PDR vs G",                   "PDR (success rate)",       axes[0, 1], g_grouped, g_sub),
        ("throughput",            "Throughput vs G",            "Throughput (succ/slot)",   axes[0, 2], g_grouped, g_sub),
        ("arrival_load_realized", "Realized Arrival Load vs G", "Arrival load / resource",  axes[1, 0], g_grouped, g_sub),
        ("attempt_load_realized", "Realized Attempt Load vs G", "Attempt load / resource",  axes[1, 1], g_grouped, g_sub),
        ("mean_backlog_per_node", "Mean Backlog vs G",          "Mean backlog / node",      axes[1, 2], g_grouped, g_sub),
        # N-sweep: 원시 지표
        ("collision_rate",        "Collision Rate vs N",        "Collision rate",            axes[2, 0], n_grouped, n_label),
        ("success_rate",          "PDR vs N",                   "PDR (success rate)",       axes[2, 1], n_grouped, n_label),
        ("throughput",            "Throughput vs N",            "Throughput (succ/slot)",   axes[2, 2], n_grouped, n_label),
        ("mean_backlog_per_node", "Mean Backlog vs N",          "Mean backlog / node",      axes[2, 3], n_grouped, n_label),
        # N-sweep: 노드 스케일 지표
        ("throughput_per_node",   "Throughput/Node vs N",       "Succ / slot / node",       axes[3, 0], n_grouped, n_label),
        ("success_rate",          "PDR vs N (node scale)",      "PDR (success rate)",       axes[3, 1], n_grouped, n_label),
        ("attempt_load_realized", "Attempt Load vs N",          "Attempt load / resource",  axes[3, 2], n_grouped, n_label),
    ]

    for metric, title, y_label, ax, grouped, subtitle in panels:
        for baseline_key, rows in grouped.items():
            xs = [row["x_value"] for row in rows]
            if metric == "throughput_per_node":
                ys = [row["throughput"] / max(row.get("n_nodes", 1), 1) for row in rows]
            else:
                ys = [row[metric] for row in rows]
            label = rows[0]["baseline_label"]
            base_key = rows[0].get("base_baseline_key", baseline_key)
            reward_variant = rows[0].get("reward_variant", "")
            ax.plot(
                xs,
                ys,
                label=label,
                color=series_color(base_key, reward_variant),
                linestyle=series_linestyle(base_key, reward_variant),
                linewidth=1.8,
                marker=series_marker(base_key, reward_variant),
                markersize=DEFAULT_MARKERSIZE,
                markeredgecolor="white",
                markeredgewidth=0.6,
            )

        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xlabel("G" if "vs G" in title else "N", fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(True, alpha=0.25)

        if metric in ("collision_rate", "success_rate"):
            ax.set_ylim(-0.02, 1.02)

        if metric == "arrival_load_realized":
            x_vals = [row["x_value"] for row in g_rows]
            if x_vals:
                ax.plot(
                    [min(x_vals), max(x_vals)],
                    [min(x_vals), max(x_vals)],
                    color="#444444",
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.8,
                    label="Target G",
                )
                ax.text(
                    0.98, 0.04, "dashed: target G",
                    transform=ax.transAxes,
                    ha="right", va="bottom",
                    fontsize=8, color="#444444",
                )

    axes[0, 0].legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
