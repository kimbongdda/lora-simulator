"""Streamlit GUI for the collision-focused LoRaWAN simulator.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from agents.q_learning import DEFAULT_REWARD_VARIANT, DEFAULT_STATE_VARIANT, REWARD_VARIANTS, STATE_VARIANTS
from baselines.catalog import BASELINE_ORDER, BASELINE_SPECS
from experiments.action_variant_compare import ActionVariantConfig, SERIES_DEFS, run_action_variant_compare
from experiments.collision_sweeps import ComparisonConfig, run_comparison
from experiments.epoch_timeseries import TimeseriesConfig, run_epoch_timeseries
from experiments.per_node_analysis import PerNodeConfig, run_per_node_analysis
from experiments.phase_compare import PhaseCompareConfig, run_phase_compare
from experiments.reward_variant_compare import RewardVariantConfig, run_reward_variant_compare
from experiments.state_variant_compare import StateVariantConfig, run_state_variant_compare
from experiments.style import configure_fonts
from utils.experiment_log import log_compare, read_log


MODE_NODE = "Node Analysis"
MODE_EPOCH = "Epoch Timeseries"
MODE_SWEEP = "G / N Sweep"
MODE_VARIANT = "Reward Variant Compare"
MODE_STATE = "State Variant Compare"
MODE_ACTION = "Action Variant Compare"
MODE_PHASE = "Phase Learning"
MODE_LOG = "Experiment Log"

FIXED_MODES = (MODE_NODE, MODE_EPOCH, MODE_VARIANT, MODE_STATE, MODE_ACTION, MODE_PHASE)


def _baseline_label(key: str) -> str:
    if key in BASELINE_SPECS:
        return BASELINE_SPECS[key].label
    base_key = key.split("__", 1)[0]
    if base_key in BASELINE_SPECS:
        return BASELINE_SPECS[base_key].label
    return key


def _epoch_series_label(epoch_data: dict[str, list[dict]], key: str) -> str:
    log = epoch_data.get(key, [])
    if log and "baseline_label" in log[0]:
        return log[0]["baseline_label"]
    return _baseline_label(key)


def _reward_label(key: str) -> str:
    return REWARD_VARIANTS[key]["label"]


def _variant_table_rows() -> list[dict]:
    rows = []
    for key, spec in REWARD_VARIANTS.items():
        rows.append(
            {
                "variant_key": key,
                "variant_label": spec["label"],
                "description": spec["description"],
            }
        )
    return rows


def _store_result(**kwargs) -> None:
    st.session_state["last_result"] = kwargs


def _render_q_dist_plot(q_per_node, dists_m, n_channels: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    configure_fonts()

    if q_per_node is None:
        st.info("No Q-table data available (not a Q-learning controller).")
        return

    node_ids = sorted(q_per_node.keys())
    dists = np.array([dists_m[i] for i in node_ids])
    q_matrix = np.array([q_per_node[i] for i in node_ids])   # (N, n_actions)
    n_actions = q_matrix.shape[1]

    has_idle = (n_actions % n_channels == 1)
    if has_idle:
        group_labels  = ["IDLE", "SF= (keep)", "SF+1 (raise)", "SF-1 (lower)"]
        group_indices = [
            [0],
            list(range(1,               n_channels + 1)),
            list(range(n_channels + 1,  2 * n_channels + 1)),
            list(range(2 * n_channels + 1, 3 * n_channels + 1)),
        ]
    else:
        group_labels  = ["SF= (keep)", "SF+1 (raise)", "SF-1 (lower)"]
        group_indices = [
            list(range(0,             n_channels)),
            list(range(n_channels,    2 * n_channels)),
            list(range(2 * n_channels, 3 * n_channels)),
        ]

    group_q = np.stack(
        [q_matrix[:, idx].max(axis=1) for idx in group_indices], axis=1
    )

    n_bins = min(20, max(5, len(node_ids) // 3))
    bin_edges = np.linspace(dists.min(), dists.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.digitize(dists, bin_edges[1:-1])

    heatmap = np.full((len(group_labels), n_bins), np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            heatmap[:, b] = group_q[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.suptitle("Q-table Policy: Preferred action by distance (mean Q-value)", fontsize=12, fontweight="bold")

    vabs = float(np.nanmax(np.abs(heatmap))) or 1.0
    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn",
                   vmin=-vabs, vmax=vabs, interpolation="nearest")

    # annotate each cell with its value
    for r in range(len(group_labels)):
        for c in range(n_bins):
            v = heatmap[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    ax.set_yticks(range(len(group_labels)))
    ax.set_yticklabels(group_labels, fontsize=10)
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([f"{c:.0f}" for c in bin_centers], rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Distance from GW (m)", fontsize=10)
    ax.set_ylabel("Action Group", fontsize=10)
    fig.colorbar(im, ax=ax, label="Mean Q-value", shrink=0.85)

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    st.pyplot(fig)
    plt.close(fig)


def _render_node_detail(epoch_data: dict[str, list[dict]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_fonts()
    import numpy as np

    if not epoch_data:
        return

    st.divider()
    st.subheader("Per-node Cumulative Time-Series")

    baseline_keys = list(epoch_data)
    selected_key = st.selectbox(
        "Baseline",
        options=baseline_keys,
        format_func=lambda key: _epoch_series_label(epoch_data, key),
        key="node_analysis_baseline_select",
    )

    log = epoch_data.get(selected_key, [])
    if not log or "per_node_successes" not in log[0]:
        st.info("This result does not include per-node epoch traces.")
        return

    xs = [epoch["slot_end"] for epoch in log]
    if len(xs) >= 2:
        default_duration = xs[1] - xs[0]
    else:
        default_duration = max(1, int(log[0].get("attempts", 1)))

    durations = []
    for idx in range(len(log)):
        if idx == 0:
            durations.append(default_duration)
        else:
            durations.append(max(1, xs[idx] - xs[idx - 1]))

    n_nodes = len(log[0]["per_node_successes"])
    n_epochs = len(log)
    suc_mat = np.zeros((n_nodes, n_epochs))
    att_mat = np.zeros((n_nodes, n_epochs))
    dur_arr = np.asarray(durations, dtype=float)

    for epoch_idx, epoch in enumerate(log):
        suc_mat[:, epoch_idx] = epoch["per_node_successes"]
        att_mat[:, epoch_idx] = epoch["per_node_attempts"]

    cum_suc = np.cumsum(suc_mat, axis=1)
    cum_att = np.cumsum(att_mat, axis=1)
    cum_dur = np.cumsum(dur_arr)

    asr_plot = np.where(cum_att > 0, cum_suc / cum_att, np.nan)
    thr_plot = cum_suc / cum_dur[np.newaxis, :]

    panels = [
        (asr_plot, "Per-node Cumulative ASR", "Cumulative ASR"),
        (thr_plot, "Per-node Cumulative Throughput", "Cumulative throughput"),
    ]

    for values, title, ylabel in panels:
        fig, ax = plt.subplots(figsize=(13, 4))
        for node_id in range(n_nodes):
            ax.plot(xs, values[node_id], linewidth=0.7, alpha=0.35)
        ax.plot(xs, np.nanmean(values, axis=0), color="black", linewidth=2.0, label="Mean")
        ax.set_title(f"{title} | {log[0].get('baseline_label', selected_key)}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Slot", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.2)
        if "ASR" in ylabel:
            ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8, framealpha=0.9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


st.set_page_config(page_title="LoRaWAN Simulator", page_icon="📡", layout="wide")
st.title("LoRaWAN Collision-Focused Simulator")

with st.sidebar:
    st.header("Experiment")
    mode = st.radio(
        "Mode",
        options=[MODE_NODE, MODE_EPOCH, MODE_SWEEP, MODE_VARIANT, MODE_STATE, MODE_ACTION, MODE_PHASE, MODE_LOG],
        help=(
            "Node Analysis: fixed (N, G) per-node plots\n"
            "Epoch Timeseries: fixed (N, G) time evolution\n"
            "G / N Sweep: compare baselines over G and N sweeps\n"
            "Reward Variant Compare: compare decentralized Q-learning reward variants at once\n"
            "State Variant Compare: compare Q-learning state space variants at once\n"
            "Action Variant Compare: relative (SF±1) vs absolute (SF direct) action space comparison\n"
            "Phase Learning: slot-phase self-organization comparison by frame_size\n"
            "Experiment Log: view cumulative experiment records"
        ),
    )

    st.divider()
    st.header("Radio Setup")
    profile = st.selectbox("Profile", options=["short", "medium", "long", "lorasim"], index=3)
    channels = st.select_slider("Channels", options=[1, 2, 3, 6], value=3)

    st.divider()
    st.header("Traffic")
    queue_mode = st.radio(
        "Queue mode",
        options=["accumulate", "fresh"],
        index=1,
        help=(
            "accumulate: packets queue up at each node\n"
            "fresh: only one fresh packet per slot, no queue build-up"
        ),
    )
    gw_obs_mode = st.radio(
        "GW observation mode",
        options=["attempt", "success"],
        index=1,
        help=(
            "attempt: GW estimates load from all transmission attempts (current default)\n"
            "success: GW estimates load from successfully decoded packets only (more realistic)"
        ),
    )

    enable_rayleigh_fading = st.toggle("Rayleigh Fading", value=False,
                                       help="Apply Rayleigh fading (|h|²~Exp(1)) to instantaneous SNR")
    if enable_rayleigh_fading:
        rayleigh_fade_margin_db = st.slider(
            "Fade Margin (dB)", min_value=0.0, max_value=30.0, value=10.0, step=1.0,
            help="Margin added to mean SNR. Higher = less outage. 10 dB → ~9% outage at 2 dB SNR margin",
        )
    else:
        rayleigh_fade_margin_db = 10.0

    st.divider()
    st.header("Simulation Length")
    if mode == MODE_EPOCH:
        default_slots = 30_000
        default_warmup = 0
    elif mode == MODE_SWEEP:
        default_slots = 20_000
        default_warmup = 5_000
    else:
        default_slots = 20_000
        default_warmup = 5_000

    n_slots = st.number_input("Slots", min_value=1_000, max_value=500_000, value=default_slots, step=1_000)
    warmup_slots = st.number_input("Warmup slots", min_value=0, max_value=100_000, value=default_warmup, step=500)
    epoch_slots = st.number_input("Epoch slots", min_value=10, max_value=20_000, value=100, step=10)

    st.divider()
    st.header("Scenario")
    if mode in FIXED_MODES:
        n_nodes = st.slider("N nodes", min_value=10, max_value=200, value=60, step=5)
        target_g = st.slider("Target G", min_value=0.1, max_value=3.0, value=3.0, step=0.1)
        if mode in (MODE_NODE, MODE_VARIANT, MODE_STATE, MODE_ACTION, MODE_PHASE):
            layout = st.radio("Node layout", options=["ring", "random"], index=1)
        else:
            layout = "ring"
            st.caption("Epoch Timeseries uses the ring layout by default for collision-only comparison.")
        p_gen_equiv = target_g * channels * 6 / max(n_nodes, 1)
        st.caption(f"Equivalent packet generation probability ≈ {p_gen_equiv:.3f}")
        n_runs_sweep = 1
    else:
        n_nodes = st.slider("G-sweep fixed N", min_value=10, max_value=200, value=60, step=5)
        target_g = st.slider("N-sweep fixed G", min_value=0.1, max_value=3.0, value=3.0, step=0.1)
        layout = st.radio("Node layout", options=["random", "ring"], index=0,
                          help="random: uniform placement inside cell (good for ADR distance diversity). ring: equal distance (controlled experiment)")
        n_runs_sweep = st.number_input(
            "Runs to average", min_value=1, max_value=30, value=1, step=1,
            help="Number of independent seeds per data point. Results are averaged. More runs = smoother curves but proportionally longer runtime.",
        )
        if n_runs_sweep > 1:
            st.caption(f"Each data point averaged over {n_runs_sweep} seeds — runtime ×{n_runs_sweep}.")

    st.divider()
    st.header("Q-learning State Space")
    if mode == MODE_STATE:
        st.caption("State Variant Compare mode uses decentralized Q-learning only.")
        from experiments.state_variant_compare import S2_DENSE_VARIANTS
        selected_state_variant_keys = st.multiselect(
            "State variants",
            options=list(STATE_VARIANTS),
            default=list(S2_DENSE_VARIANTS),
            format_func=lambda k: STATE_VARIANTS[k]["label"],
            help="Select state space variants to compare.",
        )
        state_variant = DEFAULT_STATE_VARIANT

        st.caption("GW Congestion Bins")
        _n_bins_ui = st.number_input(
            "Bin count", min_value=2, max_value=8, value=3, step=1,
            help="Number of GW congestion quantization bins. bins = thresholds + 1.",
        )
        _default_thresholds = {2: "1.0", 3: "0.7, 1.3", 4: "0.5, 1.0, 1.5", 5: "0.4, 0.8, 1.2, 1.6"}
        _thresh_str = st.text_input(
            "Bin thresholds (comma-separated)",
            value=_default_thresholds.get(int(_n_bins_ui), "0.7, 1.3"),
            help="Ascending comma-separated thresholds. Count must equal Bin count - 1.",
        )
        try:
            _parsed = tuple(float(x.strip()) for x in _thresh_str.split(",") if x.strip())
            if len(_parsed) != int(_n_bins_ui) - 1:
                st.warning(f"Threshold count ({len(_parsed)}) does not match bin count-1 ({int(_n_bins_ui)-1}).")
                _parsed = (0.7, 1.3)
        except ValueError:
            st.warning("Threshold parse error — using default (0.7, 1.3)")
            _parsed = (0.7, 1.3)
        gw_thresholds_ui = _parsed
    elif mode == MODE_ACTION:
        st.caption("Action Variant Compare mode: select a fixed state variant.")
        selected_action_variant_keys = st.multiselect(
            "Action series",
            options=list(SERIES_DEFS),
            default=list(SERIES_DEFS),
            format_func=lambda k: SERIES_DEFS[k]["label"],
            help="Select action space variants to compare.",
        )
        state_variant = st.selectbox(
            "Fixed state variant",
            options=list(STATE_VARIANTS),
            index=list(STATE_VARIANTS).index(DEFAULT_STATE_VARIANT),
            format_func=lambda k: STATE_VARIANTS[k]["label"],
        )
        selected_state_variant_keys = []
        gw_thresholds_ui = (0.7, 1.3)
    elif mode == MODE_PHASE:
        st.caption("Phase Learning mode: state = (phase_bin, sf_idx). No state variant selection.")
        selected_state_variant_keys = []
        selected_action_variant_keys = []
        state_variant = DEFAULT_STATE_VARIANT
        gw_thresholds_ui = (0.7, 1.3)
        _default_frames = "1, 8, 18, 30, 60"
        _frames_str = st.text_input("Frame sizes (comma-separated)", value=_default_frames,
                                    help="Frame sizes to compare. 1 = phase disabled (baseline).")
        try:
            _frame_sizes_ui = tuple(int(x.strip()) for x in _frames_str.split(",") if x.strip())
            if not _frame_sizes_ui:
                _frame_sizes_ui = (1, 8, 18, 30, 60)
        except ValueError:
            st.warning("Frame size parse error — using default values")
            _frame_sizes_ui = (1, 8, 18, 30, 60)
    else:
        selected_state_variant_keys = []
        selected_action_variant_keys = []
        gw_thresholds_ui = (0.7, 1.3)
        state_variant = st.selectbox(
            "State variant",
            options=list(STATE_VARIANTS),
            index=list(STATE_VARIANTS).index(DEFAULT_STATE_VARIANT),
            format_func=lambda k: STATE_VARIANTS[k]["label"],
            help="\n".join(f"{k}: {v['description']}" for k, v in STATE_VARIANTS.items()),
        )

    st.divider()
    st.header("Q-learning Reward")
    if mode == MODE_VARIANT:
        controller_key = "decentralized_q_learning"
        st.caption("Reward Variant Compare mode uses decentralized Q-learning only.")
        selected_variant_keys = st.multiselect(
            "Reward variants",
            options=list(REWARD_VARIANTS),
            default=list(REWARD_VARIANTS),
            format_func=_reward_label,
            help="Choose one or more reward variants to compare side by side.",
        )
        st.dataframe(pd.DataFrame(_variant_table_rows()), width="stretch", hide_index=True)
        reward_variant = DEFAULT_REWARD_VARIANT
    elif mode == MODE_STATE:
        controller_key = "decentralized_q_learning"
        st.caption("State Variant Compare mode: select a fixed reward variant.")
        reward_variant = st.selectbox(
            "Fixed reward variant",
            options=list(REWARD_VARIANTS),
            index=list(REWARD_VARIANTS).index(DEFAULT_REWARD_VARIANT),
            format_func=_reward_label,
        )
        selected_variant_keys = [reward_variant]
    elif mode == MODE_ACTION:
        controller_key = "decentralized_q_learning"
        st.caption("Action Variant Compare mode: select a fixed reward variant.")
        reward_variant = st.selectbox(
            "Fixed reward variant",
            options=list(REWARD_VARIANTS),
            index=list(REWARD_VARIANTS).index(DEFAULT_REWARD_VARIANT),
            format_func=_reward_label,
            key="action_mode_reward",
        )
        selected_variant_keys = [reward_variant]
    elif mode == MODE_PHASE:
        controller_key = "phase_q_learning"
        st.caption("Phase Learning mode: select a reward variant.")
        reward_variant = st.selectbox(
            "Reward variant",
            options=list(REWARD_VARIANTS),
            index=list(REWARD_VARIANTS).index(DEFAULT_REWARD_VARIANT),
            format_func=_reward_label,
            key="phase_mode_reward",
        )
        selected_variant_keys = [reward_variant]
    else:
        controller_key = "decentralized_q_learning"
        selected_variant_keys = st.multiselect(
            "Reward variants",
            options=list(REWARD_VARIANTS),
            default=list(REWARD_VARIANTS),
            format_func=lambda key: f"{REWARD_VARIANTS[key]['label']} | {REWARD_VARIANTS[key]['description']}",
            help="If a Q-learning baseline is selected, it will be expanded once per selected reward variant.",
        )
        reward_variant = selected_variant_keys[0] if selected_variant_keys else DEFAULT_REWARD_VARIANT
        st.caption("Selected Q-learning baselines will be duplicated once per chosen reward variant and plotted separately.")

    st.divider()
    st.header("Baselines")
    selected_baselines: list[str] = []
    if mode in (MODE_VARIANT, MODE_STATE, MODE_ACTION):
        st.caption("This mode compares Q-learning variants only.")
        spec = BASELINE_SPECS["decentralized_q_learning"]
        st.checkbox(spec.label, value=True, disabled=True, key="variant_mode_decentralized_q_learning")
    elif mode == MODE_PHASE:
        st.caption("Phase Learning mode uses the phase Q-learning controller only.")
        st.checkbox("Q-learning (phase)", value=True, disabled=True, key="phase_mode_controller")
    elif mode == MODE_LOG:
        st.caption("Experiment Log mode: browse recorded experiment results.")
    else:
        for key in BASELINE_ORDER:
            spec = BASELINE_SPECS[key]
            checked = st.checkbox(spec.label, value=True, help=spec.description, key=f"baseline_{key}")
            if checked:
                selected_baselines.append(key)

    if "dual_mab" in selected_baselines:
        dual_mab_b = st.slider(
            "Dual-MAB ACB barring prob (b)", min_value=0.0, max_value=1.0,
            value=0.3, step=0.05,
            help="b=0: ACB disabled (always transmit), b=1: always barred. Default 0.3.",
        )
    else:
        dual_mab_b = 0.3

    run_clicked = st.button("Run Simulation", type="primary", width="stretch")


if run_clicked:
    output_tag = mode.lower().replace(" ", "_").replace("/", "_")
    output_dir = os.path.join("outputs", f"gui_{output_tag}")
    status = st.status("Running simulation...", expanded=True)

    try:
        if mode == MODE_LOG:
            st.stop()
        if mode not in (MODE_VARIANT, MODE_STATE, MODE_ACTION, MODE_PHASE) and not selected_baselines:
            st.error("Select at least one baseline.")
            st.stop()
        if mode not in (MODE_VARIANT, MODE_STATE, MODE_ACTION, MODE_PHASE) and "decentralized_q_learning" in selected_baselines and not selected_variant_keys:
            st.error("Select at least one reward variant for the Q-learning baselines.")
            st.stop()
        if mode == MODE_VARIANT and not selected_variant_keys:
            st.error("Select at least one reward variant.")
            st.stop()
        if mode == MODE_STATE and not selected_state_variant_keys:
            st.error("Select at least one state variant.")
            st.stop()
        if mode == MODE_ACTION and not selected_action_variant_keys:
            st.error("Select at least one action series.")
            st.stop()

        if mode == MODE_NODE:
            cfg = PerNodeConfig(
                n_nodes=int(n_nodes),
                target_g=float(target_g),
                n_slots=int(n_slots),
                warmup_slots=int(warmup_slots),
                epoch_slots=int(epoch_slots),
                n_channels=int(channels),
                profile_name=profile,
                queue_mode=queue_mode,
                gw_obs_mode=gw_obs_mode,
                enable_rayleigh_fading=enable_rayleigh_fading,
                rayleigh_fade_margin_db=rayleigh_fade_margin_db,
                dual_mab_b=dual_mab_b,
                layout=layout,
                reward_variant=reward_variant,
                reward_variants=tuple(selected_variant_keys),
                state_variant=state_variant,
                output_dir=output_dir,
                baseline_keys=tuple(selected_baselines),
            )
            with status:
                st.write("Running per-node baseline comparison...")
            result = run_per_node_analysis(cfg)
            _store_result(
                mode=mode,
                image_path=os.path.join(result["output_dir"], "per_node.png"),
                table_rows=result["meta_rows"],
                table_columns=[
                    "baseline_label",
                    "reward_variant",
                    "system_success_rate",
                    "system_collision_rate",
                    "system_throughput",
                    "fairness",
                    "mean_backlog_per_node",
                ],
                queue_mode=queue_mode,
                layout_image=os.path.join(result["output_dir"], "node_layout.png"),
                sf_heatmap_image=result.get("sf_heatmap_image"),
                epoch_data=result["epoch_data"],
                q_per_node=result.get("q_per_node"),
                q_dists_m=result.get("q_dists_m"),
                n_channels=int(channels),
                state_variant=state_variant,
                extra_images=[result["channel_g_image"]] if result.get("channel_g_image") else [],
            )
            try:
                _node_log_rows = [
                    {
                        "reward_variant": mr.get("reward_variant", ""),
                        "state_variant": mr.get("state_variant", state_variant),
                        "series_label": mr.get("baseline_label", ""),
                        "success_rate": mr.get("system_success_rate"),
                        "collision_rate": mr.get("system_collision_rate"),
                        "throughput": mr.get("system_throughput"),
                        "fairness": mr.get("fairness"),
                        "mean_backlog_per_node": mr.get("mean_backlog_per_node"),
                        "final_backlog_per_node": mr.get("final_backlog_per_node"),
                        "total_attempts": mr.get("attempts", ""),
                    }
                    for mr in result.get("meta_rows", [])
                ]
                log_compare(mode, cfg, _node_log_rows, "series_label", notes=queue_mode)
            except Exception:
                pass

        elif mode == MODE_EPOCH:
            cfg = TimeseriesConfig(
                n_nodes=int(n_nodes),
                target_g=float(target_g),
                n_slots=int(n_slots),
                warmup_slots=int(warmup_slots),
                epoch_slots=int(epoch_slots),
                n_channels=int(channels),
                profile_name=profile,
                queue_mode=queue_mode,
                gw_obs_mode=gw_obs_mode,
                enable_rayleigh_fading=enable_rayleigh_fading,
                rayleigh_fade_margin_db=rayleigh_fade_margin_db,
                dual_mab_b=dual_mab_b,
                reward_variant=reward_variant,
                reward_variants=tuple(selected_variant_keys),
                state_variant=state_variant,
                output_dir=output_dir,
                baseline_keys=tuple(selected_baselines),
            )
            with status:
                st.write("Running epoch-level timeseries...")
            result = run_epoch_timeseries(cfg)
            _store_result(
                mode=mode,
                image_path=os.path.join(result["output_dir"], "epoch_timeseries.png"),
                table_rows=[],
                table_columns=[],
                queue_mode=queue_mode,
            )

        elif mode == MODE_SWEEP:
            cfg = ComparisonConfig(
                profile_name=profile,
                n_channels=int(channels),
                n_slots=int(n_slots),
                warmup_slots=int(warmup_slots),
                epoch_slots=int(epoch_slots),
                queue_mode=queue_mode,
                gw_obs_mode=gw_obs_mode,
                enable_rayleigh_fading=enable_rayleigh_fading,
                rayleigh_fade_margin_db=rayleigh_fade_margin_db,
                dual_mab_b=dual_mab_b,
                reward_variant=reward_variant,
                reward_variants=tuple(selected_variant_keys),
                state_variant=state_variant,
                g_ref_n=int(n_nodes),
                n_ref_g=float(target_g),
                output_dir=output_dir,
                workers=None,
                layout=layout,
                baseline_keys=tuple(selected_baselines),
                n_runs=int(n_runs_sweep),
            )
            with status:
                st.write("Running G / N sweep comparison...")
            result = run_comparison(cfg)
            _store_result(
                mode=mode,
                image_path=os.path.join(result["output_dir"], "baseline_comparison.png"),
                table_rows=result["g_rows"],
                table_columns=[
                    "baseline_label",
                    "reward_variant",
                    "target_g",
                    "success_rate",
                    "collision_rate",
                    "throughput",
                    "attempt_load_realized",
                    "mean_backlog_per_node",
                ],
                n_rows=result["n_rows"],
                n_runs=int(n_runs_sweep),
                queue_mode=queue_mode,
            )

        elif mode == MODE_STATE:
            cfg = StateVariantConfig(
                n_nodes=int(n_nodes),
                target_g=float(target_g),
                n_slots=int(n_slots),
                warmup_slots=int(warmup_slots),
                epoch_slots=int(epoch_slots),
                n_channels=int(channels),
                profile_name=profile,
                queue_mode=queue_mode,
                gw_obs_mode=gw_obs_mode,
                enable_rayleigh_fading=enable_rayleigh_fading,
                rayleigh_fade_margin_db=rayleigh_fade_margin_db,
                gw_thresholds=gw_thresholds_ui,
                layout=layout,
                controller_key=controller_key,
                reward_variant=reward_variant,
                state_variant_keys=tuple(selected_state_variant_keys),
                output_dir=output_dir,
            )
            with status:
                st.write("Running state variant comparison...")
            result = run_state_variant_compare(cfg)
            _store_result(
                mode=mode,
                image_path=os.path.join(result["output_dir"], "state_variant_compare.png"),
                table_rows=result["final_rows"],
                table_columns=[
                    "sv_label",
                    "n_states",
                    "success_rate",
                    "collision_rate",
                    "throughput",
                    "total_attempts",
                    "attempt_load_realized",
                    "mean_backlog_per_node",
                    "fairness",
                    "min_max_ratio",
                ],
                queue_mode=queue_mode,
                epoch_data=result["epoch_data"],
                per_node_image=os.path.join(result["output_dir"], "per_node_breakdown.png"),
                layout_image=os.path.join(result["output_dir"], "node_layout.png"),
                sf_heatmap_image=result.get("sf_heatmap_image"),
                q_table_data=None,
                state_variant=state_variant,
                extra_images=[
                    os.path.join(result["output_dir"], "summary_bar.png"),
                    *(([result["channel_g_image"]] if result.get("channel_g_image") else [])),
                ],
            )
            try:
                log_compare(mode, cfg, result["final_rows"], "sv_label", notes=queue_mode)
            except Exception:
                pass

        elif mode == MODE_ACTION:
            cfg = ActionVariantConfig(
                n_nodes=int(n_nodes),
                target_g=float(target_g),
                n_slots=int(n_slots),
                warmup_slots=int(warmup_slots),
                epoch_slots=int(epoch_slots),
                n_channels=int(channels),
                profile_name=profile,
                queue_mode=queue_mode,
                gw_obs_mode=gw_obs_mode,
                enable_rayleigh_fading=enable_rayleigh_fading,
                rayleigh_fade_margin_db=rayleigh_fade_margin_db,
                gw_thresholds=gw_thresholds_ui,
                layout=layout,
                controller_key=controller_key,
                reward_variant=reward_variant,
                state_variant=state_variant,
                action_series_keys=tuple(selected_action_variant_keys),
                output_dir=output_dir,
            )
            with status:
                st.write("Running action variant comparison...")
            result = run_action_variant_compare(cfg)
            _store_result(
                mode=mode,
                image_path=os.path.join(result["output_dir"], "action_variant_compare.png"),
                table_rows=result["final_rows"],
                table_columns=[
                    "av_label",
                    "n_actions",
                    "success_rate",
                    "collision_rate",
                    "throughput",
                    "total_attempts",
                    "attempt_load_realized",
                    "mean_backlog_per_node",
                    "fairness",
                ],
                queue_mode=queue_mode,
                epoch_data=result["epoch_data"],
                sf_heatmap_image=result.get("sf_heatmap_image"),
                extra_images=[
                    os.path.join(result["output_dir"], "summary_bar.png"),
                    *(([result["channel_g_image"]] if result.get("channel_g_image") else [])),
                ],
                state_variant=state_variant,
            )
            try:
                log_compare(mode, cfg, result["final_rows"], "av_label", notes=queue_mode)
            except Exception:
                pass

        elif mode == MODE_PHASE:
            cfg = PhaseCompareConfig(
                n_nodes=int(n_nodes),
                target_g=float(target_g),
                n_slots=int(n_slots),
                warmup_slots=int(warmup_slots),
                epoch_slots=int(epoch_slots),
                n_channels=int(channels),
                profile_name=profile,
                queue_mode=queue_mode,
                gw_obs_mode=gw_obs_mode,
                enable_rayleigh_fading=enable_rayleigh_fading,
                rayleigh_fade_margin_db=rayleigh_fade_margin_db,
                layout=layout,
                reward_variant=reward_variant,
                frame_sizes=_frame_sizes_ui,
                output_dir=output_dir,
            )
            with status:
                st.write("Running phase learning comparison...")
            result = run_phase_compare(cfg)
            _store_result(
                mode=mode,
                image_path=os.path.join(result["output_dir"], "phase_compare.png"),
                table_rows=result["final_rows"],
                table_columns=[
                    "label",
                    "frame_size",
                    "n_states",
                    "success_rate",
                    "collision_rate",
                    "throughput",
                    "fairness",
                    "mean_backlog_per_node",
                    "final_backlog_per_node",
                ],
                queue_mode=queue_mode,
                epoch_data=result["epoch_data"],
                extra_images=[
                    os.path.join(result["output_dir"], "summary_bar.png"),
                    *(([result["channel_g_image"]] if result.get("channel_g_image") else [])),
                ],
                state_variant=DEFAULT_STATE_VARIANT,
            )
            try:
                log_compare(mode, cfg, result["final_rows"], "label", notes=queue_mode)
            except Exception:
                pass

        else:
            cfg = RewardVariantConfig(
                n_nodes=int(n_nodes),
                target_g=float(target_g),
                n_slots=int(n_slots),
                warmup_slots=int(warmup_slots),
                epoch_slots=int(epoch_slots),
                n_channels=int(channels),
                profile_name=profile,
                queue_mode=queue_mode,
                gw_obs_mode=gw_obs_mode,
                enable_rayleigh_fading=enable_rayleigh_fading,
                rayleigh_fade_margin_db=rayleigh_fade_margin_db,
                layout=layout,
                controller_key=controller_key,
                variant_keys=tuple(selected_variant_keys),
                state_variant=state_variant,
                output_dir=output_dir,
            )
            with status:
                st.write("Running reward variant comparison...")
            result = run_reward_variant_compare(cfg)
            _store_result(
                mode=mode,
                image_path=os.path.join(result["output_dir"], "reward_variant_compare.png"),
                table_rows=result["final_rows"],
                table_columns=[
                    "variant_label",
                    "success_rate",
                    "collision_rate",
                    "throughput",
                    "total_attempts",
                    "attempt_load_realized",
                    "mean_backlog_per_node",
                    "fairness",
                ],
                queue_mode=queue_mode,
                epoch_data=result["epoch_data"],
                per_node_image=os.path.join(result["output_dir"], "per_node_breakdown.png"),
                layout_image=os.path.join(result["output_dir"], "node_layout.png"),
                sf_heatmap_image=result.get("sf_heatmap_image"),
                q_per_node=result.get("q_per_node"),
                q_dists_m=result.get("q_dists_m"),
                n_channels=int(channels),
                state_variant=state_variant,
                extra_images=[result["channel_g_image"]] if result.get("channel_g_image") else [],
            )
            try:
                _variant_log_rows = [
                    {**r, "reward_variant": r.get("variant_key", "")}
                    for r in result["final_rows"]
                ]
                log_compare(mode, cfg, _variant_log_rows, "variant_label", notes=queue_mode)
            except Exception:
                pass

        status.update(label="Done", state="complete")
    except Exception as exc:
        status.update(label="Failed", state="error")
        st.exception(exc)


_MODE_IMAGE_NAMES: dict[str, str] = {
    MODE_NODE: "per_node.png",
    MODE_EPOCH: "epoch_timeseries.png",
    MODE_SWEEP: "baseline_comparison.png",
    MODE_VARIANT: "reward_variant_compare.png",
    MODE_STATE: "state_variant_compare.png",
    MODE_ACTION: "action_variant_compare.png",
    MODE_PHASE: "phase_compare.png",
}

last_result = st.session_state.get("last_result")

if not last_result and mode not in (MODE_LOG,) and mode in _MODE_IMAGE_NAMES:
    _output_tag = mode.lower().replace(" ", "_").replace("/", "_")
    _auto_dir = os.path.abspath(os.path.join("outputs", f"gui_{_output_tag}"))
    _auto_img = os.path.join(_auto_dir, _MODE_IMAGE_NAMES[mode])
    if os.path.exists(_auto_img):
        st.info("Showing results from a previous run. Press 'Run Simulation' to refresh.")
        st.image(_auto_img, width="stretch")

if last_result and mode != MODE_LOG:
    st.caption(f"Last run: {last_result['mode']} | queue mode: {last_result['queue_mode']}")

    image_path = last_result["image_path"]
    if os.path.exists(image_path):
        st.image(image_path, width="stretch")
    else:
        st.warning(f"Result image not found: {image_path}")

    table_rows = last_result.get("table_rows", [])
    table_columns = last_result.get("table_columns", [])
    if table_rows:
        st.subheader("Summary Table")
        df = pd.DataFrame(table_rows)
        visible_columns = [column for column in table_columns if column in df.columns]
        if visible_columns:
            renamed = df[visible_columns].rename(columns={"baseline_label": "Baseline", "variant_label": "Reward Variant"})
            st.dataframe(renamed.round(4), width="stretch", hide_index=True)

    per_node_image = last_result.get("per_node_image")
    if per_node_image and os.path.exists(per_node_image):
        st.divider()
        st.subheader("Per-node Breakdown")
        st.image(per_node_image, width="stretch")

    sf_heatmap_image = last_result.get("sf_heatmap_image")
    if sf_heatmap_image and os.path.exists(sf_heatmap_image):
        st.divider()
        st.subheader("SF Selection vs Distance")
        st.image(sf_heatmap_image, width="stretch")

    layout_image = last_result.get("layout_image")
    if layout_image and os.path.exists(layout_image):
        st.divider()
        st.subheader("Node Layout")
        st.image(layout_image, width="stretch")

    for extra_img in last_result.get("extra_images", []):
        if extra_img and os.path.exists(extra_img):
            st.divider()
            st.image(extra_img, width="stretch")

    n_rows = last_result.get("n_rows")
    if n_rows and last_result["mode"] == MODE_SWEEP:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from experiments.style import DEFAULT_MARKERSIZE, series_color, series_linestyle, series_marker
        configure_fonts()

        st.divider()
        st.subheader("N-sweep Summary")
        grouped_n: dict[str, list] = {}
        for row in n_rows:
            grouped_n.setdefault(row["baseline_key"], []).append(row)

        n_runs_label = last_result.get("n_runs", 1)
        runs_note = f" (avg {n_runs_label} runs)" if n_runs_label > 1 else ""

        def _plot_n_metric(metric: str, ylabel: str, ylim=None) -> plt.Figure:
            fig, ax = plt.subplots(figsize=(6, 5))
            for bkey, rows in grouped_n.items():
                rows_sorted = sorted(rows, key=lambda r: r["x_value"])
                xs = [r["x_value"] for r in rows_sorted]
                ys = [r[metric] for r in rows_sorted]
                label = rows_sorted[0]["baseline_label"]
                base_key = rows_sorted[0].get("base_baseline_key", bkey)
                reward_variant = rows_sorted[0].get("reward_variant", "")
                ax.plot(
                    xs, ys,
                    label=label,
                    color=series_color(base_key, reward_variant),
                    linestyle=series_linestyle(base_key, reward_variant),
                    linewidth=1.8,
                    marker=series_marker(base_key, reward_variant),
                    markersize=DEFAULT_MARKERSIZE,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                )
            ax.set_xlabel("Number of nodes (N)", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
            plt.tight_layout()
            return fig

        col_pdr, col_thr = st.columns(2)
        with col_pdr:
            st.caption(f"PDR (Packet Delivery Ratio){runs_note}")
            fig_pdr = _plot_n_metric("success_rate", "PDR (success rate)", ylim=(-0.02, 1.02))
            st.pyplot(fig_pdr)
            plt.close(fig_pdr)
        with col_thr:
            st.caption(f"Throughput (successes / slot){runs_note}")
            fig_thr = _plot_n_metric("throughput", "Throughput (succ/slot)")
            st.pyplot(fig_thr)
            plt.close(fig_thr)

    epoch_data = last_result.get("epoch_data")
    if last_result["mode"] in (MODE_NODE, MODE_VARIANT, MODE_STATE) and epoch_data:
        _render_node_detail(epoch_data)

    q_per_node = last_result.get("q_per_node")
    if q_per_node is not None:
        st.divider()
        st.subheader("Q-table Policy: Preferred Action by Distance")
        _render_q_dist_plot(q_per_node, last_result.get("q_dists_m", []), last_result.get("n_channels", 3))
elif mode == MODE_LOG:
    st.header("Experiment Log")
    _log_rows = read_log()
    if not _log_rows:
        st.info("No experiments recorded yet. Results are saved automatically after each run.")
    else:
        df_log = pd.DataFrame(_log_rows)

        with st.expander("Filters", expanded=True):
            _col1, _col2, _col3 = st.columns(3)
            with _col1:
                _modes = ["(all)"] + sorted(df_log["mode"].dropna().unique().tolist())
                _sel_mode = st.selectbox("Mode", options=_modes, key="log_filter_mode")
            with _col2:
                _profiles = ["(all)"] + sorted(df_log["profile"].dropna().unique().tolist())
                _sel_profile = st.selectbox("Profile", options=_profiles, key="log_filter_profile")
            with _col3:
                _n_nodes_vals = ["(all)"] + sorted(df_log["n_nodes"].dropna().unique().tolist())
                _sel_n = st.selectbox("N nodes", options=_n_nodes_vals, key="log_filter_n")

        _df_filtered = df_log.copy()
        if _sel_mode != "(all)":
            _df_filtered = _df_filtered[_df_filtered["mode"] == _sel_mode]
        if _sel_profile != "(all)":
            _df_filtered = _df_filtered[_df_filtered["profile"] == _sel_profile]
        if _sel_n != "(all)":
            _df_filtered = _df_filtered[_df_filtered["n_nodes"] == _sel_n]

        st.caption(f"{len(_df_filtered)} / {len(df_log)} rows")

        _display_cols = [
            "timestamp", "mode", "n_nodes", "target_g", "n_channels", "profile",
            "reward_variant", "state_variant", "frame_size", "series_label",
            "asr", "throughput", "fairness", "collision_rate",
            "mean_backlog", "final_backlog", "notes",
        ]
        _visible = [c for c in _display_cols if c in _df_filtered.columns]
        st.dataframe(_df_filtered[_visible], width="stretch", hide_index=True)

        _csv_bytes = _df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=_csv_bytes,
            file_name="experiment_log_filtered.csv",
            mime="text/csv",
        )
else:
    st.info("Configure the sidebar, then click Run Simulation.")
    st.markdown(
        """
        **Available modes**

        - `Node Analysis`: per-node ASR, throughput, resource-load scatter and per-node cumulative traces.
        - `Epoch Timeseries`: epoch-wise attempt load, ASR, throughput, collision, and backlog evolution.
        - `G / N Sweep`: sweep offered load `G` and node count `N` for baseline comparison.
        - `Reward Variant Compare`: compare reward variants on decentralized Q-learning.
        - `State Variant Compare`: compare state space variants (s2~s5) on decentralized Q-learning.
        """
    )
