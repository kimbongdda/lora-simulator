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
from experiments.er_compare import ERCompareConfig, run_er_compare
from experiments.per_node_analysis import PerNodeConfig, run_per_node_analysis
from experiments.phase_compare import PhaseCompareConfig, run_phase_compare
from experiments.reward_variant_compare import RewardVariantConfig, run_reward_variant_compare
from experiments.state_variant_compare import StateVariantConfig, run_state_variant_compare
from experiments.optuna_tune import OptunaConfig, OBJECTIVE_METRICS, run_optuna_tune
from experiments.er_sweep import ERSweepConfig, SWEEP_PARAMS, SWEEP_PARAM_LABELS, run_er_sweep
from experiments.style import configure_fonts
from utils.experiment_log import log_compare, read_log


MODE_NODE = "Node Analysis"
MODE_EPOCH = "Epoch Timeseries"
MODE_SWEEP = "G / N Sweep"
MODE_VARIANT = "Reward Variant Compare"
MODE_STATE = "State Variant Compare"
MODE_ACTION = "Action Variant Compare"
MODE_PHASE = "Phase Learning"
MODE_ER = "ER Compare"
MODE_OPTUNA = "Optuna Tune"
MODE_ER_SWEEP = "ER Param Sweep"
MODE_LOG = "Experiment Log"
MODE_SNAPSHOT = "Snapshot Browser"

FIXED_MODES = (MODE_NODE, MODE_EPOCH, MODE_VARIANT, MODE_STATE, MODE_ACTION, MODE_PHASE, MODE_ER, MODE_OPTUNA, MODE_ER_SWEEP)


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


# ── 설정 영속화 ────────────────────────────────────────────────────────────────
import json as _json

_SETTINGS_PATH = os.path.join("outputs", "last_settings.json")


def _load_settings() -> dict:
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as _f:
            return _json.load(_f)
    except Exception:
        return {}


def _save_settings(d: dict) -> None:
    os.makedirs(os.path.dirname(_SETTINGS_PATH), exist_ok=True)
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as _f:
        _json.dump(d, _f, indent=2, ensure_ascii=False)


# ── 프리셋 카트 영속화 ────────────────────────────────────────────────────────
_CART_PATH = os.path.join("outputs", "saved_presets.json")


def _load_cart() -> list:
    try:
        with open(_CART_PATH, encoding="utf-8") as _f:
            return _json.load(_f)
    except Exception:
        return []


def _save_cart(items: list) -> None:
    os.makedirs(os.path.dirname(_CART_PATH), exist_ok=True)
    with open(_CART_PATH, "w", encoding="utf-8") as _f:
        _json.dump(items, _f, indent=2, ensure_ascii=False)


# ── Optuna 실행 기록 영속화 ──────────────────────────────────────────────────
_OPTUNA_HISTORY_PATH = os.path.join("outputs", "optuna_history.json")


def _sanitize_for_json(obj):
    """numpy 등 JSON 비직렬화 타입을 안전하게 변환한다."""
    import math
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    try:
        import numpy as _np
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj


def _load_optuna_history() -> list:
    try:
        with open(_OPTUNA_HISTORY_PATH, encoding="utf-8") as _f:
            return _json.load(_f)
    except Exception:
        return []


def _save_optuna_run(record: dict) -> None:
    import tempfile
    history = _load_optuna_history()
    history.insert(0, _sanitize_for_json(record))
    history = history[:30]
    _out_dir = os.path.dirname(_OPTUNA_HISTORY_PATH)
    os.makedirs(_out_dir, exist_ok=True)
    _fd, _tmp = tempfile.mkstemp(dir=_out_dir, suffix=".tmp")
    try:
        with os.fdopen(_fd, "w", encoding="utf-8") as _f:
            _json.dump(history, _f, indent=2, ensure_ascii=False, default=str)
        os.replace(_tmp, _OPTUNA_HISTORY_PATH)
    except Exception:
        try:
            os.unlink(_tmp)
        except OSError:
            pass
        raise


def _delete_optuna_run(run_id: str) -> None:
    import tempfile
    history = _load_optuna_history()
    history = [r for r in history if r.get("run_id") != run_id]
    _out_dir = os.path.dirname(_OPTUNA_HISTORY_PATH)
    os.makedirs(_out_dir, exist_ok=True)
    _fd, _tmp = tempfile.mkstemp(dir=_out_dir, suffix=".tmp")
    try:
        with os.fdopen(_fd, "w", encoding="utf-8") as _f:
            _json.dump(history, _f, indent=2, ensure_ascii=False, default=str)
        os.replace(_tmp, _OPTUNA_HISTORY_PATH)
    except Exception:
        try:
            os.unlink(_tmp)
        except OSError:
            pass
        raise


def _build_reward_params_from_best(bp: dict) -> "dict | None":
    """Optuna best_params의 r_* 계수를 reward_params dict로 재구성한다."""
    if "r_success_base" not in bp:
        return None
    return {
        "type": "composite",
        "success_base": float(bp.get("r_success_base", 0.5)),
        "success_fair": float(bp.get("r_success_fair", 0.5)),
        "fail": -abs(float(bp.get("r_fail_abs", 1.0))),
        "idle_pkt": float(bp.get("r_idle_pkt", 0.0)),
        "idle_no_pkt": 0.0,
        "retry_coef": 0.0,
        "switch_pen": 0.0,
    }


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


def _make_timeseries_mats(log: list[dict]):
    """epoch log → (xs, asr_mat, thr_mat, n_nodes) numpy arrays."""
    import numpy as np
    xs = [epoch["slot_end"] for epoch in log]
    default_duration = (xs[1] - xs[0]) if len(xs) >= 2 else max(1, int(log[0].get("attempts", 1)))
    durations = [default_duration if i == 0 else max(1, xs[i] - xs[i - 1]) for i in range(len(log))]
    n_nodes = len(log[0]["per_node_successes"])
    n_epochs = len(log)
    suc_mat = np.zeros((n_nodes, n_epochs))
    att_mat = np.zeros((n_nodes, n_epochs))
    for ei, epoch in enumerate(log):
        suc_mat[:, ei] = epoch["per_node_successes"]
        att_mat[:, ei] = epoch["per_node_attempts"]
    cum_suc = np.cumsum(suc_mat, axis=1)
    cum_att = np.cumsum(att_mat, axis=1)
    cum_dur = np.cumsum(np.asarray(durations, dtype=float))
    asr_mat = np.where(cum_att > 0, cum_suc / cum_att, float("nan"))
    thr_mat = cum_suc / cum_dur[None, :]
    return xs, asr_mat, thr_mat, n_nodes


def _save_node_timeseries(epoch_data: dict, snap_dir: str) -> None:
    """epoch_data의 모든 baseline에 대해 시계열 PNG를 snap_dir에 저장한다."""
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    configure_fonts()

    for key, log in epoch_data.items():
        if not log or "per_node_successes" not in log[0]:
            continue
        xs, asr_mat, thr_mat, n_nodes = _make_timeseries_mats(log)
        label = log[0].get("baseline_label", key)
        safe = re.sub(r"[^\w]", "_", key)[:40]
        for mat, suffix, ylabel in [
            (asr_mat, "asr", "Cumulative ASR"),
            (thr_mat, "thr", "Cumulative throughput"),
        ]:
            fig, ax = plt.subplots(figsize=(13, 4))
            for nid in range(n_nodes):
                ax.plot(xs, mat[nid], linewidth=0.7, alpha=0.35)
            ax.plot(xs, np.nanmean(mat, axis=0), color="black", linewidth=2.0, label="Mean")
            ax.set_title(f"Per-node Cumulative {'ASR' if suffix == 'asr' else 'Throughput'} | {label}",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("Slot", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.2)
            if suffix == "asr":
                ax.set_ylim(-0.02, 1.05)
            ax.legend(fontsize=8, framealpha=0.9)
            plt.tight_layout()
            fig.savefig(os.path.join(snap_dir, f"timeseries_{safe}_{suffix}.png"), dpi=120)
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

    xs, asr_plot, thr_plot, n_nodes = _make_timeseries_mats(log)

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

# 저장된 설정을 세션 상태로 복원 (페이지 새로고침 후 첫 실행 시에만 수행)
if "_settings_loaded" not in st.session_state:
    for _k, _v in _load_settings().items():
        if _k not in st.session_state:
            st.session_state[_k] = _v
    st.session_state["_settings_loaded"] = True

# ── 프리셋 카트 초기화 ──────────────────────────────────────────────────────
if "_cart" not in st.session_state:
    st.session_state["_cart"] = _load_cart()

# 프리셋 로드 대기 처리: "Load" 버튼 클릭 후 rerun 시 위젯 값 반영
if "_cart_load_pending" in st.session_state:
    _p = st.session_state.pop("_cart_load_pending")
    st.session_state["s_ql_alpha"]     = float(_p.get("alpha", 0.1))
    st.session_state["s_ql_gamma_q"]   = float(_p.get("gamma_q", 0.9))
    st.session_state["s_ql_eps_min"]   = float(_p.get("eps_min", 0.05))
    st.session_state["s_ql_eps_decay"] = float(_p.get("eps_decay", 0.9995))
    st.session_state["s_reward_variant"] = _p.get("reward_variant", DEFAULT_REWARD_VARIANT)
    _p_psi = float(_p.get("psi", 0.0))
    if _p_psi > 0:
        st.session_state["s_er_enabled"] = True
        st.session_state["s_psi"]        = _p_psi
        st.session_state["s_E0"]         = float(_p.get("E0", 10.0))
        st.session_state["s_W"]          = int(_p.get("W", 20))
        st.session_state["s_er_mode"]    = _p.get("er_mode", "ETD")
        st.session_state["s_mu"]         = float(_p.get("mu", 0.5))
    st.session_state["_loaded_preset_rp"]   = _p.get("reward_params")
    st.session_state["_loaded_preset_name"] = _p.get("name", "")

with st.sidebar:
    st.header("Experiment")
    mode = st.radio(
        "Mode",
        options=[MODE_NODE, MODE_EPOCH, MODE_SWEEP, MODE_VARIANT, MODE_STATE, MODE_ACTION, MODE_PHASE, MODE_ER, MODE_OPTUNA, MODE_ER_SWEEP, MODE_LOG, MODE_SNAPSHOT],
        help=(
            "Node Analysis: fixed (N, G) per-node plots\n"
            "Epoch Timeseries: fixed (N, G) time evolution\n"
            "G / N Sweep: compare baselines over G and N sweeps\n"
            "Reward Variant Compare: compare decentralized Q-learning reward variants at once\n"
            "State Variant Compare: compare Q-learning state space variants at once\n"
            "Action Variant Compare: relative (SF±1) vs absolute (SF direct) action space comparison\n"
            "Phase Learning: slot-phase self-organization comparison by frame_size\n"
            "ER Compare: Q-learning vs ER-ETD vs ER-EM side-by-side epoch timeseries\n"
            "Experiment Log: view cumulative experiment records\n"
            "Snapshot Browser: 저장된 Node Analysis 스냅샷 목록 및 비교"
        ),
        key="s_mode",
    )

    # ── Saved Presets 카트 ──────────────────────────────────────────────────
    _cart = st.session_state.get("_cart", [])
    if _cart:
        st.divider()
        _n_checked = sum(1 for i in range(len(_cart)) if st.session_state.get(f"cart_check_{i}", False))
        _exp_label = f"Saved Presets ({len(_cart)})  ·  {_n_checked} selected" if _n_checked else f"Saved Presets ({len(_cart)})"
        with st.expander(_exp_label, expanded=bool(_n_checked)):
            st.caption("☑ 체크: Node Analysis에서 별도 series로 실행.  Load: 하이퍼파라미터 단일 적용.  ❌: 삭제.")
            for _ci, _item in enumerate(_cart):
                _rp_tag = " · parametric" if _item.get("reward_params") else ""
                _env = _item.get("env", {})
                _env_str = ""
                if _env:
                    _ns = _env.get("n_slots", 0)
                    _ns_k = f"{_ns//1000}k" if _ns >= 1000 else str(_ns)
                    _env_str = (
                        f"N={_env.get('n_nodes','?')} G={_env.get('target_g','?')} "
                        f"ch={_env.get('n_channels','?')} {_env.get('profile','?')} "
                        f"{_env.get('layout','?')} {_ns_k}slots"
                    )
                _cc, _ci_col, _cl, _cr = st.columns([0.5, 3.5, 1, 0.5])
                with _cc:
                    st.checkbox("", key=f"cart_check_{_ci}", label_visibility="collapsed")
                with _ci_col:
                    _md = (
                        f"**{_item['name']}**  \n"
                        f"{_item.get('controller_label','?')}{_rp_tag}  \n"
                        f"ASR `{_item.get('success_rate',0):.3f}` "
                        f"F `{_item.get('fairness',0):.3f}` "
                        f"Thr `{_item.get('throughput',0):.4f}`"
                    )
                    if _env_str:
                        _md += f"  \n<small style='color:#64748b'>{_env_str}</small>"
                    st.markdown(_md, unsafe_allow_html=True)
                    with st.expander("⚙ 파라미터", expanded=False):
                        _rp = _item.get("reward_params")
                        # Q-learning 하이퍼파라미터
                        st.markdown(
                            f"**Q-learning**  \n"
                            f"α `{_item.get('alpha', 0.1):.4f}` · "
                            f"γ `{_item.get('gamma_q', 0.9):.4f}` · "
                            f"ε_min `{_item.get('eps_min', 0.05):.4f}` · "
                            f"ε_decay `{_item.get('eps_decay', 0.9995):.5f}`"
                        )
                        # 보상 함수
                        if _rp:
                            st.markdown(
                                f"**보상** (parametric)  \n"
                                f"base `{_rp.get('success_base', 0):.3f}` · "
                                f"fair `{_rp.get('success_fair', 0):.3f}`  \n"
                                f"fail `{_rp.get('fail', 0):.3f}` · "
                                f"idle `{_rp.get('idle_pkt', 0):.4f}`"
                            )
                        else:
                            st.markdown(f"**보상** variant: `{_item.get('reward_variant', 'base')}`")
                        # ER 파라미터 (psi > 0 이거나 ER 컨트롤러일 때)
                        _psi = _item.get("psi", 0.0)
                        if _psi and _psi > 0:
                            st.markdown(
                                f"**ER**  \n"
                                f"ψ `{_psi:.4f}` · E₀ `{_item.get('E0', 10.0):.2f}` · "
                                f"W `{_item.get('W', 20)}` · "
                                f"`{_item.get('er_mode', 'ETD')}` · "
                                f"μ `{_item.get('mu', 0.5):.3f}`"
                            )
                with _cl:
                    if st.button("Load", key=f"cart_load_{_ci}", use_container_width=True,
                                 help="이 프리셋의 하이퍼파라미터를 사이드바에 단일 적용합니다."):
                        st.session_state["_cart_load_pending"] = _item
                        st.rerun()
                with _cr:
                    if st.button("❌", key=f"cart_rm_{_ci}", use_container_width=True):
                        _new_cart = [x for i, x in enumerate(_cart) if i != _ci]
                        st.session_state["_cart"] = _new_cart
                        _save_cart(_new_cart)
                        st.rerun()
                st.divider()
            if _n_checked:
                st.info(f"Node Analysis 실행 시 체크된 {_n_checked}개 프리셋이 추가 series로 포함됩니다.")

    st.divider()
    st.header("Radio Setup")
    profile = st.selectbox("Profile", options=["short", "medium", "long", "lorasim"], index=3, key="s_profile")
    channels = st.select_slider("Channels", options=[1, 2, 3, 6], value=3, key="s_channels")

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
        key="s_queue_mode",
    )
    gw_obs_mode = st.radio(
        "GW observation mode",
        options=["attempt", "success"],
        index=1,
        help=(
            "attempt: GW estimates load from all transmission attempts (current default)\n"
            "success: GW estimates load from successfully decoded packets only (more realistic)"
        ),
        key="s_gw_obs_mode",
    )

    enable_rayleigh_fading = st.toggle("Rayleigh Fading", value=False,
                                       help="Apply Rayleigh fading (|h|²~Exp(1)) to instantaneous SNR. NLOS 환경 (도심). Rician이 켜지면 무시됨.",
                                       key="s_rayleigh_fading")
    if enable_rayleigh_fading:
        rayleigh_fade_margin_db = st.slider(
            "Fade Margin (dB)", min_value=0.0, max_value=30.0, value=10.0, step=1.0,
            help="Margin added to mean SNR. Higher = less outage. 10 dB → ~9% outage at 2 dB SNR margin",
            key="s_rayleigh_margin",
        )
    else:
        rayleigh_fade_margin_db = 10.0

    enable_rician_fading = st.toggle("Rician Fading", value=False,
                                     help="Apply Rician fading (LOS 성분 포함). K=0→Rayleigh, K→∞→AWGN. Rayleigh보다 우선 적용됨.",
                                     key="s_rician_fading")
    if enable_rician_fading:
        rician_k_factor = st.slider(
            "K-factor", min_value=0.1, max_value=20.0, value=4.0, step=0.5,
            help="LOS/산란 전력 비. K≈1: 도심약LOS, K≈4: 교외(기본), K≈10: 농촌개활지",
            key="s_rician_k",
        )
        rician_fade_margin_db = st.slider(
            "Rician Fade Margin (dB)", min_value=0.0, max_value=20.0, value=5.0, step=1.0,
            help="Rician 페이딩 적용 전 평균 SNR에 더할 마진 (dB). Rayleigh보다 작아도 됨.",
            key="s_rician_margin",
        )
    else:
        rician_k_factor = 4.0
        rician_fade_margin_db = 5.0

    st.divider()
    st.header("Simulation Length")
    if mode in (MODE_EPOCH, MODE_ER):
        default_slots = 30_000
        default_warmup = 0
    elif mode == MODE_SWEEP:
        default_slots = 20_000
        default_warmup = 5_000
    elif mode == MODE_OPTUNA:
        default_slots = 5_000   # 튜닝용: 빠른 탐색 우선 (검증은 나중에 논문 설정으로)
        default_warmup = 1_000
    else:
        default_slots = 20_000
        default_warmup = 5_000

    n_slots = st.number_input("Slots", min_value=1_000, max_value=500_000, value=default_slots, step=1_000, key="s_n_slots")
    warmup_slots = st.number_input("Warmup slots", min_value=0, max_value=100_000, value=default_warmup, step=500, key="s_warmup_slots")
    epoch_slots = st.number_input("Epoch slots", min_value=10, max_value=20_000, value=100, step=10, key="s_epoch_slots")
    if mode == MODE_OPTUNA:
        st.caption("Optuna 튜닝: 5,000슬롯으로도 하이퍼파라미터 순위 식별 충분. 최종 검증은 논문 설정(20,000슬롯)으로.")

    st.divider()
    st.header("Scenario")
    # ── 부하 입력 방식 토글 (G / p_arrival 직접 입력) ──────────────────────────
    _g_input_mode = st.radio(
        "Load input",
        options=["G (normalized)", "p_arrival (direct)"],
        horizontal=True,
        key="s_g_input_mode",
        help="G: 정규화 제공 부하 (G = N·p / N_res).  p_arrival: 슬롯당 패킷 생성 확률을 직접 입력.",
    )
    _n_res_ui = int(channels) * 6  # N_SF=6 고정

    if mode in FIXED_MODES:
        n_nodes = st.slider("N nodes", min_value=10, max_value=200, value=60, step=5, key="s_n_nodes")
        if _g_input_mode == "G (normalized)":
            target_g = st.slider("Target G", min_value=0.1, max_value=10.0, value=3.0, step=0.1, key="s_target_g")
            _p_equiv = min(float(target_g) * _n_res_ui / max(int(n_nodes), 1), 1.0)
            st.caption(f"→ p_arrival ≈ {_p_equiv:.4f}")
        else:
            _p_input = st.number_input(
                "p_arrival", min_value=0.001, max_value=1.0, value=0.05, step=0.001,
                format="%.4f", key="s_p_arrival",
                help="슬롯당 노드별 패킷 생성 확률. 0.001 ~ 1.0",
            )
            target_g = float(_p_input) * max(int(n_nodes), 1) / _n_res_ui
            st.caption(f"→ Equivalent G ≈ {target_g:.3f}")
        if mode in (MODE_NODE, MODE_VARIANT, MODE_STATE, MODE_ACTION, MODE_PHASE, MODE_OPTUNA, MODE_ER_SWEEP):
            layout = st.radio("Node layout", options=["ring", "random"], index=1, key="s_layout")
        else:
            layout = "ring"
            st.caption("Uses ring layout by default for collision-only comparison.")
        n_runs_sweep = 1
    else:
        n_nodes = st.slider("G-sweep fixed N", min_value=10, max_value=200, value=60, step=5, key="s_n_nodes")
        if _g_input_mode == "G (normalized)":
            target_g = st.slider("N-sweep fixed G", min_value=0.1, max_value=10.0, value=3.0, step=0.1, key="s_target_g")
            _p_equiv = min(float(target_g) * _n_res_ui / max(int(n_nodes), 1), 1.0)
            st.caption(f"→ p_arrival ≈ {_p_equiv:.4f}")
        else:
            _p_input = st.number_input(
                "p_arrival", min_value=0.001, max_value=1.0, value=0.05, step=0.001,
                format="%.4f", key="s_p_arrival",
                help="슬롯당 노드별 패킷 생성 확률. 0.001 ~ 1.0",
            )
            target_g = float(_p_input) * max(int(n_nodes), 1) / _n_res_ui
            st.caption(f"→ Equivalent G ≈ {target_g:.3f}")
        layout = st.radio("Node layout", options=["random", "ring"], index=0,
                          help="random: uniform placement inside cell (good for ADR distance diversity). ring: equal distance (controlled experiment)",
                          key="s_layout")
        n_runs_sweep = st.number_input(
            "Runs to average", min_value=1, max_value=30, value=1, step=1,
            help="Number of independent seeds per data point. Results are averaged. More runs = smoother curves but proportionally longer runtime.",
            key="s_n_runs",
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
            key="s_state_variants",
            format_func=lambda k: STATE_VARIANTS[k]["label"],
            help="Select state space variants to compare.",
        )
        state_variant = DEFAULT_STATE_VARIANT

        st.caption("GW Congestion Bins")
        _n_bins_ui = st.number_input(
            "Bin count", min_value=2, max_value=8, value=3, step=1,
            help="Number of GW congestion quantization bins. bins = thresholds + 1.",
            key="s_n_bins",
        )
        _default_thresholds = {2: "1.0", 3: "0.7, 1.3", 4: "0.5, 1.0, 1.5", 5: "0.4, 0.8, 1.2, 1.6"}
        _thresh_str = st.text_input(
            "Bin thresholds (comma-separated)",
            value=_default_thresholds.get(int(_n_bins_ui), "0.7, 1.3"),
            help="Ascending comma-separated thresholds. Count must equal Bin count - 1.",
            key="s_gw_thresh_str",
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
            key="s_action_variants",
        )
        state_variant = st.selectbox(
            "Fixed state variant",
            options=list(STATE_VARIANTS),
            index=list(STATE_VARIANTS).index(DEFAULT_STATE_VARIANT),
            format_func=lambda k: STATE_VARIANTS[k]["label"],
            key="s_state_variant",
        )
        selected_state_variant_keys = []
        gw_thresholds_ui = (0.7, 1.3)
    elif mode == MODE_ER:
        st.caption("ER Compare mode uses decentralized Q-learning. Select a fixed state variant.")
        selected_state_variant_keys = []
        selected_action_variant_keys = []
        gw_thresholds_ui = (0.7, 1.3)
        state_variant = st.selectbox(
            "State variant",
            options=list(STATE_VARIANTS),
            index=list(STATE_VARIANTS).index(DEFAULT_STATE_VARIANT),
            format_func=lambda k: STATE_VARIANTS[k]["label"],
            key="s_state_variant",
        )
    elif mode == MODE_OPTUNA:
        st.caption("Optuna Tune: state variant is fixed. Configure in the Optuna Settings section below.")
        selected_state_variant_keys = []
        selected_action_variant_keys = []
        state_variant = st.selectbox(
            "Fixed state variant",
            options=list(STATE_VARIANTS),
            index=list(STATE_VARIANTS).index(DEFAULT_STATE_VARIANT),
            format_func=lambda k: STATE_VARIANTS[k]["label"],
            key="s_state_variant",
        )
        gw_thresholds_ui = (0.7, 1.3)
        _frame_sizes_ui = (1,)
    elif mode == MODE_ER_SWEEP:
        st.caption("ER Param Sweep: state variant는 아래 설정에서 지정합니다.")
        selected_state_variant_keys = []
        selected_action_variant_keys = []
        state_variant = DEFAULT_STATE_VARIANT
        gw_thresholds_ui = (0.7, 1.3)
        _frame_sizes_ui = (1,)
    elif mode == MODE_PHASE:
        st.caption("Phase Learning mode: state = (phase_bin, sf_idx). No state variant selection.")
        selected_state_variant_keys = []
        selected_action_variant_keys = []
        state_variant = DEFAULT_STATE_VARIANT
        gw_thresholds_ui = (0.7, 1.3)
        _default_frames = "1, 8, 18, 30, 60"
        _frames_str = st.text_input("Frame sizes (comma-separated)", value=_default_frames,
                                    help="Frame sizes to compare. 1 = phase disabled (baseline).",
                                    key="s_frame_sizes")
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
            key="s_state_variant",
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
            key="s_reward_variants",
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
            key="s_reward_variant",
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
            key="s_reward_variant",
        )
        selected_variant_keys = [reward_variant]
    elif mode == MODE_ER:
        controller_key = "decentralized_q_learning"
        st.caption("ER Compare mode: select a fixed reward variant.")
        reward_variant = st.selectbox(
            "Fixed reward variant",
            options=list(REWARD_VARIANTS),
            index=list(REWARD_VARIANTS).index(DEFAULT_REWARD_VARIANT),
            format_func=_reward_label,
            key="s_reward_variant",
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
            key="s_reward_variant",
        )
        selected_variant_keys = [reward_variant]
    elif mode == MODE_OPTUNA:
        controller_key = "decentralized_q_learning"
        st.caption("Optuna Tune: reward variant is part of the search space (configured below).")
        reward_variant = DEFAULT_REWARD_VARIANT
        selected_variant_keys = list(REWARD_VARIANTS.keys())
    elif mode == MODE_ER_SWEEP:
        controller_key = "decentralized_q_learning"
        st.caption("ER Param Sweep: Q-learning 파라미터는 아래 설정에서 고정합니다.")
        reward_variant = DEFAULT_REWARD_VARIANT
        selected_variant_keys = []
    else:
        controller_key = "decentralized_q_learning"
        selected_variant_keys = st.multiselect(
            "Reward variants",
            options=list(REWARD_VARIANTS),
            default=list(REWARD_VARIANTS),
            format_func=lambda key: f"{REWARD_VARIANTS[key]['label']} | {REWARD_VARIANTS[key]['description']}",
            help="If a Q-learning baseline is selected, it will be expanded once per selected reward variant.",
            key="s_reward_variants",
        )
        reward_variant = selected_variant_keys[0] if selected_variant_keys else DEFAULT_REWARD_VARIANT
        st.caption("Selected Q-learning baselines will be duplicated once per chosen reward variant and plotted separately.")

    st.divider()
    st.header("Baselines")
    selected_baselines: list[str] = []
    if mode in (MODE_VARIANT, MODE_STATE, MODE_ACTION, MODE_ER):
        st.caption("This mode compares Q-learning variants only.")
        spec = BASELINE_SPECS["decentralized_q_learning"]
        st.checkbox(spec.label, value=True, disabled=True, key="variant_mode_decentralized_q_learning")
    elif mode == MODE_PHASE:
        st.caption("Phase Learning mode uses the phase Q-learning controller only.")
        st.checkbox("Q-learning (phase)", value=True, disabled=True, key="phase_mode_controller")
    elif mode == MODE_OPTUNA:
        st.caption("Optuna Tune uses decentralized Q-learning only.")
        st.checkbox("Decentralized Q-learning", value=True, disabled=True, key="optuna_mode_q_learning")
    elif mode == MODE_ER_SWEEP:
        st.caption("ER Param Sweep: decentralized Q-learning + ER 사용.")
        st.checkbox("Decentralized Q-learning (ER)", value=True, disabled=True, key="er_sweep_mode_ql")
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
            key="s_dual_mab_b",
        )
    else:
        dual_mab_b = 0.3

    if "kaburaki" in selected_baselines:
        with st.expander("Kaburaki 하이퍼파라미터", expanded=False):
            kaburaki_J = st.number_input(
                "J (오프셋 후보 수)", min_value=1, max_value=20, value=3, step=1,
                help="비-0 오프셋 후보 수. 후보 집합 크기 = J+1. 논문 기본값: 3.",
                key="s_kaburaki_J",
            )
            kaburaki_D_max = st.number_input(
                "D_max (최대 오프셋 슬롯)", min_value=1, max_value=200, value=64, step=1,
                help="오프셋 최댓값(슬롯). 논문 기본값: 64.",
                key="s_kaburaki_D_max",
            )
            kaburaki_alpha = st.number_input(
                "α (학습률)", min_value=0.01, max_value=1.0, value=0.3, step=0.01,
                help="Q-learning 학습률. 논문 기본값: 0.3.",
                key="s_kaburaki_alpha",
            )
            kaburaki_gamma = st.number_input(
                "β (할인율)", min_value=0.0, max_value=1.0, value=0.95, step=0.01,
                help="할인율. 논문 기본값: 0.95.",
                key="s_kaburaki_gamma",
            )
            kaburaki_eps_min = st.number_input(
                "ε_min", min_value=0.0, max_value=0.5, value=0.05, step=0.01,
                help="최소 탐색률. 논문에 명시값 없음. 기본: 0.05.",
                key="s_kaburaki_eps_min",
            )
            st.caption(f"J={kaburaki_J}  D_max={kaburaki_D_max}  α={kaburaki_alpha}  β={kaburaki_gamma}  ε_min={kaburaki_eps_min}")
    else:
        kaburaki_J = 3
        kaburaki_D_max = 64
        kaburaki_alpha = 0.3
        kaburaki_gamma = 0.95
        kaburaki_eps_min = 0.05

    st.divider()
    st.header("Energy Regulation (ER-Q)")

    _er_variant_labels = {
        "etd":     "ER-ETD (에너지 상태 없음)",
        "em":      "ER-EM (에너지 상태 없음)",
        "etd_s13": "ER-ETD + SF+CH (18 states)",
        "etd_s14": "ER-ETD + SF-delta+CH (9 states)",
        "etd_s18": "ER-ETD + SF방향+Eb (9 states)",
        "etd_s17": "ER-ETD + SF+Eb (18 states)",
        "etd_s15": "ER-ETD + SF+Gmax+Eb (54 states)",
        "em_s18":  "ER-EM + SF방향+Eb (9 states)",
        "em_s17":  "ER-EM + SF+Eb (18 states)",
        "em_s15":  "ER-EM + SF+Gmax+Eb (54 states)",
    }

    if mode == MODE_NODE:
        st.caption("Node Analysis: 실행할 ER Q-learning 변형을 선택하세요. 선택한 변형이 기존 베이스라인에 추가됩니다.")
        selected_er_variants = st.multiselect(
            "ER Q-learning 변형",
            options=list(_er_variant_labels),
            default=[],
            format_func=_er_variant_labels.get,
            help="에너지 상태 없음: 기존 ER-Q. SF+Eb / SF+Gmax+Eb: 에너지 빈을 상태에 포함한 변형.",
            key="s_er_variants",
        )
        _er_needs_params = any(v in selected_er_variants for v in ["etd", "em"])
        if _er_needs_params:
            psi = st.slider("ψ (penalty 강도)", min_value=0.05, max_value=2.0, value=0.6, step=0.05,
                            help="ER-ETD / ER-EM에 적용할 penalty 계수.", key="s_psi")
            E0 = st.slider("E₀ (에너지 예산)", min_value=1.0, max_value=50.0, value=10.0, step=1.0,
                           help="슬라이딩 윈도우 W 슬롯 내 허용 누적 에너지.", key="s_E0")
            W = st.number_input("W (윈도우 크기)", min_value=1, max_value=200, value=20, step=5,
                                help="에너지 이력을 집계할 슬롯 수.", key="s_W")
            if "em" in selected_er_variants:
                mu = st.slider("μ (EM 혼합 계수)", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                               help="μ=1 → 누적 비용만. μ=0 → 즉시 비용만.", key="s_mu")
            else:
                mu = 0.5
            st.caption(f"ψ={psi}  E₀={E0}  W={W}" + (f"  μ={mu}" if "em" in selected_er_variants else ""))
        else:
            psi = 0.0; E0 = 10.0; W = 20; mu = 0.5
        er_mode = "ETD"

    elif mode == MODE_OPTUNA:
        selected_er_variants = []
        psi = 0.0; E0 = 10.0; W = 20; mu = 0.5; er_mode = "ETD"
        st.caption("Optuna Tune: ER 파라미터는 사용하지 않습니다.")
    elif mode == MODE_ER_SWEEP:
        selected_er_variants = []
        psi = 0.0; E0 = 10.0; W = 20; mu = 0.5; er_mode = "ETD"
        st.caption("ER Param Sweep: ER 설정은 아래 전용 설정 섹션에서 합니다.")
    elif mode == MODE_ER:
        selected_er_variants = []
        st.caption("ER Compare: baseline / ETD / EM 세 가지를 동시에 실행합니다. 공통 파라미터를 설정하세요.")
        psi = st.slider("ψ (penalty 강도)", min_value=0.05, max_value=2.0, value=0.6, step=0.05,
                        help="ER-ETD / ER-EM 양쪽에 적용할 penalty 계수.", key="s_psi")
        E0 = st.slider("E₀ (에너지 예산)", min_value=1.0, max_value=50.0, value=10.0, step=1.0,
                       help="슬라이딩 윈도우 W 슬롯 내 허용 누적 에너지.", key="s_E0")
        W = st.number_input("W (윈도우 크기)", min_value=1, max_value=200, value=20, step=5,
                            help="에너지 이력을 집계할 슬롯 수.", key="s_W")
        mu = st.slider("μ (EM 혼합 계수)", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                       help="ER-EM 전용. μ=1 → 누적 비용만. μ=0 → 즉시 비용만.", key="s_mu")
        er_mode = "ETD"
        st.caption(f"ψ={psi}  E₀={E0}  W={W}  μ(EM)={mu}")

    else:
        selected_er_variants = []
        er_enabled = st.toggle(
            "Enable Energy Regulation",
            value=False,
            help="psi=0이면 기존 Q-learning과 동일. psi>0이면 누적 에너지가 E0를 초과할 때 Q값에 penalty를 적용해 IDLE로 유도.",
            key="s_er_enabled",
        )
        if er_enabled:
            er_mode = st.radio(
                "ER 방식",
                options=["ETD", "EM"],
                index=0,
                help=(
                    "ETD: Θ(a) = E_k + e(a)  — 누적 에너지 + 액션 비용 합산\n"
                    "EM:  Θ(a) = μ·E_k + (1-μ)·e(a)  — 누적/즉시 비용 가중 혼합"
                ),
                key="s_er_mode",
            )
            psi = st.slider("psi (penalty 강도)", min_value=0.0, max_value=2.0, value=0.6, step=0.05,
                            help="Q값에서 차감할 penalty 계수. 0=규제 없음, 클수록 에너지 절약 강조.", key="s_psi")
            E0 = st.slider("E₀ (에너지 예산)", min_value=1.0, max_value=50.0, value=10.0, step=1.0,
                           help="슬라이딩 윈도우 W 슬롯 내 허용 누적 에너지. 이 값을 초과하면 penalty 발동.", key="s_E0")
            W = st.number_input("W (윈도우 크기)", min_value=1, max_value=200, value=20, step=5,
                                help="에너지 이력을 집계할 슬롯 수.", key="s_W")
            if er_mode == "EM":
                mu = st.slider("μ (EM 혼합 계수)", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                               help="μ=1 → ETD와 동일(누적 비용만). μ=0 → 즉시 비용 e(a)만 사용.", key="s_mu")
            else:
                mu = 0.5
            st.caption(f"{'ER-ETD' if er_mode == 'ETD' else f'ER-EM (μ={mu})'} | ψ={psi} E₀={E0} W={W}")
        else:
            er_mode = "ETD"
            psi = 0.0
            E0 = 10.0
            W = 20
            mu = 0.5

    # ── Q-learning 하이퍼파라미터 (Node Analysis 전용) ──────────────────────
    if mode == MODE_NODE:
        st.divider()
        st.header("Q-learning Hyperparameters")
        _loaded_preset_name = st.session_state.get("_loaded_preset_name", "")
        if _loaded_preset_name:
            st.success(f"Preset loaded: **{_loaded_preset_name}**")
        _expand_ql = bool(_loaded_preset_name)
        with st.expander("Custom hyperparams (alpha, gamma, ε)", expanded=_expand_ql):
            ql_alpha = st.number_input(
                "α (learning rate)", min_value=0.001, max_value=1.0, value=0.1,
                format="%.4f", key="s_ql_alpha",
                help="Q값 업데이트 학습률. 작을수록 안정적, 클수록 빠르게 적응.",
            )
            ql_gamma_q = st.number_input(
                "γ (discount)", min_value=0.1, max_value=0.999, value=0.9,
                format="%.3f", key="s_ql_gamma_q",
                help="미래 보상 할인율. 1에 가까울수록 장기 보상 중시.",
            )
            ql_eps_min = st.number_input(
                "ε_min", min_value=0.001, max_value=0.5, value=0.05,
                format="%.4f", key="s_ql_eps_min",
                help="최소 탐색 확률. 학습이 끝난 뒤에도 이 확률로 무작위 액션.",
            )
            ql_eps_decay = st.number_input(
                "ε_decay", min_value=0.990, max_value=1.0, value=0.9995,
                format="%.6f", key="s_ql_eps_decay",
                help="슬롯마다 ε에 곱하는 감소율. 1에 가까울수록 탐색이 오래 유지.",
            )
            st.caption(f"α={ql_alpha}  γ={ql_gamma_q}  ε_min={ql_eps_min}  ε_decay={ql_eps_decay}")
        _loaded_rp = st.session_state.get("_loaded_preset_rp")
        with st.expander("보상 함수 수치 (커스텀)", expanded=bool(_loaded_rp)):
            _use_custom_rp = st.toggle(
                "커스텀 보상 수치 사용",
                value=bool(_loaded_rp),
                key="s_use_custom_rp",
                help="ON: 아래 수치로 composite 보상 사용. OFF: 위 Reward Variants 선택 사용.",
            )
            if _use_custom_rp:
                _rp_def = _loaded_rp or {}
                _rp_sb = st.number_input(
                    "성공 기본 보상 (success_base)", min_value=0.0, max_value=5.0,
                    value=float(_rp_def.get("success_base", 1.0)),
                    format="%.4f", key="s_rp_success_base",
                    help="전송 성공 시 기본 보상값.",
                )
                _rp_sf = st.number_input(
                    "성공 공정성 보상 (success_fair)", min_value=0.0, max_value=5.0,
                    value=float(_rp_def.get("success_fair", 0.0)),
                    format="%.4f", key="s_rp_success_fair",
                    help="성공 시 공정성 항 계수. r_success = base + fair/(1+ν_i).",
                )
                if _rp_sf > 0:
                    st.warning("⚠ success_fair는 전체 노드 평균 성공 횟수(글로벌 정보)를 사용합니다. 순수 분산 설정에서는 0으로 두는 것이 논문상 안전합니다.", icon="⚠️")
                _rp_fa = st.number_input(
                    "실패 패널티 |fail| (절댓값 입력)", min_value=0.0, max_value=5.0,
                    value=abs(float(_rp_def.get("fail", 1.0))),
                    format="%.4f", key="s_rp_fail_abs",
                    help="전송 실패 시 패널티. 내부적으로 음수(-값)로 적용됨.",
                )
                _rp_ip = st.number_input(
                    "패킷 있을 때 IDLE 보상 (idle_pkt)", min_value=-2.0, max_value=2.0,
                    value=float(_rp_def.get("idle_pkt", 0.0)),
                    format="%.4f", key="s_rp_idle_pkt",
                    help="보낼 패킷이 있는데 IDLE을 선택했을 때의 보상. 음수면 패널티.",
                )
                _loaded_rp = {
                    "type": "composite",
                    "success_base": _rp_sb,
                    "success_fair": _rp_sf,
                    "fail": -abs(_rp_fa),
                    "idle_pkt": _rp_ip,
                    "idle_no_pkt": 0.0,
                    "retry_coef": 0.0,
                    "switch_pen": 0.0,
                }
                st.caption(
                    f"base `{_rp_sb:.4f}` · fair `{_rp_sf:.4f}` · "
                    f"fail `{-abs(_rp_fa):.4f}` · idle `{_rp_ip:.4f}`"
                )
            else:
                _loaded_rp = st.session_state.get("_loaded_preset_rp")
    else:
        ql_alpha = 0.1
        ql_gamma_q = 0.9
        ql_eps_min = 0.05
        ql_eps_decay = 0.9995
        _loaded_rp = None

    # ── Optuna 전용 설정 ──────────────────────────────────────────────
    if mode == MODE_OPTUNA:
        from experiments.optuna_tune import TUNABLE_CONTROLLERS
        import os as _os
        st.divider()
        st.header("Optuna Settings")

        _ctrl_options = list(TUNABLE_CONTROLLERS.keys())
        _ctrl_labels = {k: v["label"] for k, v in TUNABLE_CONTROLLERS.items()}
        optuna_controller_keys = st.multiselect(
            "Tune controllers",
            options=_ctrl_options,
            default=["q_learning"],
            format_func=_ctrl_labels.get,
            help="최적화할 컨트롤러를 선택하세요. 여러 개 선택 시 각각 독립된 연구를 실행합니다.",
            key="s_optuna_controllers",
        )
        if not optuna_controller_keys:
            optuna_controller_keys = ["q_learning"]

        optuna_n_trials = st.number_input(
            "Trials (per controller)", min_value=5, max_value=1000, value=500, step=5,
            help="컨트롤러당 탐색할 trial 수.",
            key="s_optuna_n_trials",
        )
        _max_jobs = max(1, _os.cpu_count() or 1)
        optuna_n_jobs = st.slider(
            "Parallel trials (n_jobs)", min_value=1, max_value=_max_jobs, value=1,
            help=(
                "한 연구 안에서 동시에 실행할 trial 수. "
                f"이 기기 코어 수: {_max_jobs}. "
                "여러 컨트롤러 선택 시 각 연구는 순차 실행됨."
            ),
            key="s_optuna_n_jobs",
        )
        optuna_parallel_studies = False  # Streamlit 내 ProcessPool 데드락, Thread는 GIL로 효과 없음
        optuna_objective = st.selectbox(
            "Objective metric",
            options=list(OBJECTIVE_METRICS.keys()),
            index=0,
            format_func=lambda k: OBJECTIVE_METRICS[k],
            help="최대화할 목적 함수.",
            key="s_optuna_objective",
        )
        if optuna_objective == "weighted_sum":
            with st.expander("Weighted Sum 가중치", expanded=True):
                _w_asr      = st.slider("w_asr (ASR 가중치)",      0.0, 5.0, 1.0, 0.1, key="s_optuna_w_asr")
                _w_thr      = st.slider("w_thr (Throughput 가중치)", 0.0, 5.0, 1.0, 0.1, key="s_optuna_w_thr")
                _w_fairness = st.slider("w_fairness (Fairness 가중치)", 0.0, 5.0, 1.0, 0.1, key="s_optuna_w_fairness")
                _w_total = _w_asr + _w_thr + _w_fairness
                if _w_total > 0:
                    st.caption(
                        f"정규화 후 실제 비율 → ASR: {_w_asr/_w_total:.2f}  "
                        f"Thr: {_w_thr/_w_total:.2f}  "
                        f"Fairness: {_w_fairness/_w_total:.2f}"
                    )
                else:
                    st.warning("가중치 합이 0입니다. 최소 하나는 0보다 크게 설정하세요.")
                    _w_asr = _w_thr = _w_fairness = 1.0
            optuna_w_asr, optuna_w_thr, optuna_w_fairness = _w_asr, _w_thr, _w_fairness
        else:
            optuna_w_asr = optuna_w_thr = optuna_w_fairness = 1.0
        optuna_search_reward = st.toggle(
            "Search reward_variant",
            value=True,
            help="보상 함수 종류를 탐색 공간에 포함할지 여부.",
            key="s_optuna_search_reward",
        )
        optuna_search_parametric_reward = st.toggle(
            "Parametric reward search",
            value=False,
            help="보상 계수(success_base, success_fair, fail, idle_pkt)를 연속값으로 직접 탐색합니다. 활성화 시 reward_variant 선택이 무시됩니다.",
            key="s_optuna_search_parametric_reward",
        )
        if optuna_search_parametric_reward:
            with st.expander("Parametric reward ranges", expanded=True):
                _col_r1, _col_r2 = st.columns(2)
                with _col_r1:
                    optuna_r_success_base_low  = st.number_input("success_base min", value=0.1,  min_value=0.0,  max_value=2.0, format="%.3f", key="s_r_success_base_low")
                    optuna_r_success_fair_low  = st.number_input("success_fair min", value=0.0,  min_value=0.0,  max_value=2.0, format="%.3f", key="s_r_success_fair_low")
                    optuna_r_fail_abs_low      = st.number_input("fail abs min",     value=0.1,  min_value=0.0,  max_value=2.0, format="%.3f", key="s_r_fail_abs_low")
                    optuna_r_idle_pkt_low      = st.number_input("idle_pkt min",     value=-0.1, min_value=-0.5, max_value=0.5, format="%.3f", key="s_r_idle_pkt_low")
                with _col_r2:
                    optuna_r_success_base_high = st.number_input("success_base max", value=1.5,  min_value=0.1,  max_value=5.0, format="%.3f", key="s_r_success_base_high")
                    optuna_r_success_fair_high = st.number_input("success_fair max", value=1.5,  min_value=0.0,  max_value=5.0, format="%.3f", key="s_r_success_fair_high")
                    optuna_r_fail_abs_high     = st.number_input("fail abs max",     value=2.0,  min_value=0.1,  max_value=5.0, format="%.3f", key="s_r_fail_abs_high")
                    optuna_r_idle_pkt_high     = st.number_input("idle_pkt max",     value=0.3,  min_value=-0.5, max_value=1.0, format="%.3f", key="s_r_idle_pkt_high")
        else:
            optuna_r_success_base_low, optuna_r_success_base_high = 0.1, 1.5
            optuna_r_success_fair_low, optuna_r_success_fair_high = 0.0, 1.5
            optuna_r_fail_abs_low,     optuna_r_fail_abs_high     = 0.1, 2.0
            optuna_r_idle_pkt_low,     optuna_r_idle_pkt_high     = -0.1, 0.3
        with st.expander("Q-learning search ranges", expanded=False):
            optuna_alpha_low = st.number_input("alpha min", value=0.01, min_value=0.001, max_value=0.5, format="%.4f", key="s_optuna_alpha_low")
            optuna_alpha_high = st.number_input("alpha max", value=0.5, min_value=0.01, max_value=1.0, format="%.3f", key="s_optuna_alpha_high")
            optuna_gamma_low = st.number_input("gamma_q min", value=0.5, min_value=0.1, max_value=0.99, format="%.2f", key="s_optuna_gamma_low")
            optuna_gamma_high = st.number_input("gamma_q max", value=0.99, min_value=0.5, max_value=0.999, format="%.3f", key="s_optuna_gamma_high")
            optuna_eps_min_low = st.number_input("eps_min min", value=0.01, min_value=0.001, max_value=0.2, format="%.4f", key="s_optuna_eps_min_low")
            optuna_eps_min_high = st.number_input("eps_min max", value=0.2, min_value=0.01, max_value=0.5, format="%.3f", key="s_optuna_eps_min_high")
            optuna_eps_decay_low = st.number_input("eps_decay min", value=0.9990, min_value=0.990, max_value=0.9999, format="%.5f", key="s_optuna_eps_decay_low")
            optuna_eps_decay_high = st.number_input("eps_decay max", value=0.99999, min_value=0.9990, max_value=1.0, format="%.6f", key="s_optuna_eps_decay_high")
        with st.expander("🔒 Fixed parameters (controlled study)", expanded=False):
            optuna_fix_base = st.toggle(
                "전체 고정 — 모든 컨트롤러에 α/γ/ε 고정",
                value=False,
                key="s_optuna_fix_base",
                help="α, γ, ε_min, ε_decay를 모든 컨트롤러에 동일 값으로 고정. ER 메커니즘 단독 비교에 권장.",
            )
            optuna_fix_q_only = st.toggle(
                "Q-learning만 고정 — ER 컨트롤러는 자유 탐색",
                value=False,
                key="s_optuna_fix_q_only",
                help="Q-learning은 고정값 사용, ER-ETD / ER-EM은 α/γ/ε까지 자유롭게 탐색.",
                disabled=optuna_fix_base,
            )
            if optuna_fix_base or optuna_fix_q_only:
                _col_fb1, _col_fb2 = st.columns(2)
                with _col_fb1:
                    optuna_fixed_alpha     = st.number_input("α (fixed)",     value=0.0595, min_value=0.001, max_value=1.0,   format="%.4f", key="s_optuna_fixed_alpha")
                    optuna_fixed_eps_min   = st.number_input("ε_min (fixed)", value=0.0182, min_value=0.001, max_value=0.5,   format="%.4f", key="s_optuna_fixed_eps_min")
                with _col_fb2:
                    optuna_fixed_gamma_q   = st.number_input("γ (fixed)",      value=0.8847, min_value=0.1,   max_value=0.999, format="%.4f", key="s_optuna_fixed_gamma")
                    optuna_fixed_eps_decay = st.number_input("ε_decay (fixed)", value=0.9995, min_value=0.990, max_value=1.0,  format="%.5f", key="s_optuna_fixed_eps_decay")
                _scope = "전체" if optuna_fix_base else "Q-learning만"
                st.caption(f"[{_scope}] α={optuna_fixed_alpha}  γ={optuna_fixed_gamma_q}  ε_min={optuna_fixed_eps_min}  ε_decay={optuna_fixed_eps_decay}")
            else:
                optuna_fixed_alpha = 0.06
                optuna_fixed_gamma_q = 0.88
                optuna_fixed_eps_min = 0.018
                optuna_fixed_eps_decay = 0.9995
            optuna_fix_reward = st.toggle(
                "Fix reward function",
                value=False,
                key="s_optuna_fix_reward",
                help="보상 함수를 고정. reward_variant / parametric 탐색이 무시됩니다.",
            )
            if optuna_fix_reward:
                st.caption("reward 탐색 옵션(위)은 이 설정에 의해 무시됩니다.")
                optuna_fix_reward_numeric = st.toggle(
                    "수치로 직접 고정 (composite reward)",
                    value=False,
                    key="s_optuna_fix_reward_numeric",
                    help="ON: 아래 수치를 사용. OFF: reward_variant 선택 사용.",
                )
                if optuna_fix_reward_numeric:
                    _col_rw1, _col_rw2 = st.columns(2)
                    with _col_rw1:
                        optuna_fixed_r_success_base = st.number_input("success_base", value=1.0, min_value=0.0, max_value=5.0, format="%.4f", key="s_optuna_fixed_r_sb")
                        optuna_fixed_r_fail_abs     = st.number_input("|fail| (절댓값)", value=1.0, min_value=0.0, max_value=5.0, format="%.4f", key="s_optuna_fixed_r_fa")
                    with _col_rw2:
                        optuna_fixed_r_success_fair = st.number_input("success_fair", value=0.0, min_value=0.0, max_value=5.0, format="%.4f", key="s_optuna_fixed_r_sf")
                        optuna_fixed_r_idle_pkt     = st.number_input("idle_pkt", value=0.0, min_value=-2.0, max_value=2.0, format="%.4f", key="s_optuna_fixed_r_ip")
                    st.caption(f"base={optuna_fixed_r_success_base}  fair={optuna_fixed_r_success_fair}  fail={-abs(optuna_fixed_r_fail_abs):.4f}  idle={optuna_fixed_r_idle_pkt}")
                    optuna_fixed_reward_variant = "base"
                else:
                    optuna_fixed_r_success_base = 1.0
                    optuna_fixed_r_success_fair = 0.0
                    optuna_fixed_r_fail_abs = 1.0
                    optuna_fixed_r_idle_pkt = 0.0
                    optuna_fixed_reward_variant = st.selectbox(
                        "Fixed reward variant",
                        options=list(REWARD_VARIANTS.keys()),
                        index=0,
                        key="s_optuna_fixed_reward_variant",
                    )
            else:
                optuna_fix_reward_numeric = False
                optuna_fixed_reward_variant = "base"
                optuna_fixed_r_success_base = 1.0
                optuna_fixed_r_success_fair = 0.0
                optuna_fixed_r_fail_abs = 1.0
                optuna_fixed_r_idle_pkt = 0.0

        if "kaburaki" in optuna_controller_keys:
            with st.expander("Kaburaki search ranges", expanded=False):
                _kc1, _kc2 = st.columns(2)
                with _kc1:
                    optuna_kab_J_low     = st.number_input("J min",     value=1,   min_value=1,  max_value=20,  step=1, key="s_optuna_kab_J_low")
                    optuna_kab_D_max_low = st.number_input("D_max min", value=10,  min_value=1,  max_value=100, step=5, key="s_optuna_kab_D_max_low")
                with _kc2:
                    optuna_kab_J_high     = st.number_input("J max",     value=10,  min_value=1,  max_value=50,  step=1, key="s_optuna_kab_J_high")
                    optuna_kab_D_max_high = st.number_input("D_max max", value=200, min_value=10, max_value=500, step=10, key="s_optuna_kab_D_max_high")
        else:
            optuna_kab_J_low, optuna_kab_J_high = 1, 10
            optuna_kab_D_max_low, optuna_kab_D_max_high = 10, 200

        if any(k in optuna_controller_keys for k in ("er_etd", "er_em")):
            with st.expander("ER search ranges", expanded=False):
                optuna_psi_low = st.number_input("psi min", value=0.05, min_value=0.01, max_value=1.0, format="%.3f", key="s_optuna_psi_low")
                optuna_psi_high = st.number_input("psi max", value=2.0, min_value=0.1, max_value=5.0, format="%.2f", key="s_optuna_psi_high")
                optuna_E0_low = st.number_input("E0 min", value=2.0, min_value=1.0, max_value=20.0, format="%.1f", key="s_optuna_E0_low")
                optuna_E0_high = st.number_input("E0 max", value=30.0, min_value=5.0, max_value=100.0, format="%.1f", key="s_optuna_E0_high")
                optuna_W_low = st.number_input("W min", value=5, min_value=1, max_value=50, step=1, key="s_optuna_W_low")
                optuna_W_high = st.number_input("W max", value=50, min_value=10, max_value=200, step=5, key="s_optuna_W_high")
                if "er_em" in optuna_controller_keys:
                    st.divider()
                    optuna_fix_mu = st.toggle("mu 고정 (ER-EM 전용)", value=False, key="s_optuna_fix_mu",
                                              help="ON: mu를 아래 값으로 고정. OFF: mu_min ~ mu_max 범위에서 탐색.")
                    _c1, _c2 = st.columns(2)
                    with _c1:
                        optuna_mu_low = st.number_input("mu min", value=0.1, min_value=0.0, max_value=1.0,
                                                        format="%.2f", key="s_optuna_mu_low",
                                                        disabled=optuna_fix_mu)
                    with _c2:
                        optuna_mu_high = st.number_input("mu max", value=0.9, min_value=0.0, max_value=1.0,
                                                         format="%.2f", key="s_optuna_mu_high",
                                                         disabled=optuna_fix_mu)
                    optuna_fixed_mu = st.number_input("mu 고정값", value=0.5, min_value=0.0, max_value=1.0,
                                                      format="%.2f", key="s_optuna_fixed_mu",
                                                      disabled=not optuna_fix_mu)
                else:
                    optuna_fix_mu = False
                    optuna_mu_low, optuna_mu_high, optuna_fixed_mu = 0.1, 0.9, 0.5
        else:
            optuna_psi_low, optuna_psi_high = 0.05, 2.0
            optuna_E0_low, optuna_E0_high = 2.0, 30.0
            optuna_W_low, optuna_W_high = 5, 50
            optuna_fix_mu = False
            optuna_mu_low, optuna_mu_high, optuna_fixed_mu = 0.1, 0.9, 0.5

        _n_total = len(optuna_controller_keys) * int(optuna_n_trials)
        st.caption(
            f"총 {_n_total} trials | "
            f"{'병렬 studies' if optuna_parallel_studies and len(optuna_controller_keys) > 1 else '순차 studies'} | "
            f"trial 내 병렬={optuna_n_jobs}"
        )

        # ── 이전 실행 기록 ────────────────────────────────────────────────
        _optuna_hist = _load_optuna_history()
        if _optuna_hist:
            st.divider()
            with st.expander(f"📂 이전 실행 기록 ({len(_optuna_hist)})", expanded=False):
                for _hi, _hrun in enumerate(_optuna_hist):
                    _h_ts  = _hrun.get("timestamp", "")[:16].replace("T", " ")
                    _h_cfg = _hrun.get("config_summary", "")
                    _h_win = _hrun.get("winner_summary", "")
                    _h_id  = _hrun.get("run_id", str(_hi))
                    _h_label = _hrun.get("label", "")
                    _display_name = _h_label if _h_label else _h_ts
                    st.markdown(
                        f"**{_display_name}**  \n"
                        f"<small style='color:#475569'>{_h_cfg}</small>  \n"
                        f"<small style='color:#0f766e'>{_h_win}</small>",
                        unsafe_allow_html=True,
                    )
                    _hc1, _hc2, _hc3 = st.columns([2, 1, 0.6])
                    with _hc1:
                        # key를 run_id 기반으로 — 목록 순서가 바뀌어도 세션값이 꼬이지 않음
                        _widget_key = f"hist_label_{_h_id}"
                        if _widget_key not in st.session_state:
                            st.session_state[_widget_key] = _h_label
                        _new_label = st.text_input(
                            "별칭", placeholder="이름 입력…",
                            key=_widget_key, label_visibility="collapsed",
                        )
                        if _new_label != _h_label:
                            import tempfile as _tmpmod
                            _optuna_hist[_hi]["label"] = _new_label
                            _lbl_dir = os.path.dirname(_OPTUNA_HISTORY_PATH)
                            os.makedirs(_lbl_dir, exist_ok=True)
                            _lfd, _ltmp = _tmpmod.mkstemp(dir=_lbl_dir, suffix=".tmp")
                            try:
                                with os.fdopen(_lfd, "w", encoding="utf-8") as _hf:
                                    _json.dump(_optuna_hist, _hf, indent=2, ensure_ascii=False, default=str)
                                os.replace(_ltmp, _OPTUNA_HISTORY_PATH)
                            except Exception:
                                try:
                                    os.unlink(_ltmp)
                                except OSError:
                                    pass
                    with _hc2:
                        if st.button("Load", key=f"hist_load_{_h_id}", use_container_width=True):
                            _loaded_lr = _hrun.get("last_result", {})
                            if _loaded_lr:
                                st.session_state["last_result"] = _loaded_lr
                                st.rerun()
                    with _hc3:
                        if st.button("🗑", key=f"hist_del_{_h_id}", use_container_width=True,
                                     help="이 기록을 삭제합니다."):
                            _delete_optuna_run(_h_id)
                            st.rerun()
                    if _hi < len(_optuna_hist) - 1:
                        st.markdown("---")
    else:
        optuna_controller_keys = ["q_learning"]
        optuna_n_trials = 30
        optuna_n_jobs = 1
        optuna_parallel_studies = False
        optuna_objective = "asr_x_fairness"
        optuna_search_reward = True
        optuna_alpha_low, optuna_alpha_high = 0.01, 0.5
        optuna_gamma_low, optuna_gamma_high = 0.5, 0.99
        optuna_eps_min_low, optuna_eps_min_high = 0.01, 0.2
        optuna_eps_decay_low, optuna_eps_decay_high = 0.9990, 0.99999
        optuna_psi_low, optuna_psi_high = 0.05, 2.0
        optuna_E0_low, optuna_E0_high = 2.0, 30.0
        optuna_W_low, optuna_W_high = 5, 50
        optuna_fix_mu = False
        optuna_mu_low, optuna_mu_high, optuna_fixed_mu = 0.1, 0.9, 0.5
        optuna_w_asr = optuna_w_thr = optuna_w_fairness = 1.0
        optuna_search_parametric_reward = False
        optuna_fix_base = False
        optuna_fix_q_only = False
        optuna_fixed_alpha = 0.06
        optuna_fixed_gamma_q = 0.88
        optuna_fixed_eps_min = 0.018
        optuna_fixed_eps_decay = 0.9995
        optuna_fix_reward = False
        optuna_fix_reward_numeric = False
        optuna_fixed_reward_variant = "base"
        optuna_fixed_r_success_base = 1.0
        optuna_fixed_r_success_fair = 0.0
        optuna_fixed_r_fail_abs = 1.0
        optuna_fixed_r_idle_pkt = 0.0
        optuna_r_success_base_low, optuna_r_success_base_high = 0.1, 1.5
        optuna_r_success_fair_low, optuna_r_success_fair_high = 0.0, 1.5
        optuna_r_fail_abs_low,     optuna_r_fail_abs_high     = 0.1, 2.0
        optuna_r_idle_pkt_low,     optuna_r_idle_pkt_high     = -0.1, 0.3
        optuna_kab_J_low, optuna_kab_J_high = 1, 10
        optuna_kab_D_max_low, optuna_kab_D_max_high = 10, 200

    # ── ER Sweep 전용 설정 ────────────────────────────────────────────────
    if mode == MODE_ER_SWEEP:
        from experiments.er_sweep import count_total_sims as _sw_count
        st.divider()
        st.header("ER Sweep Settings")

        _sw_er_type = st.radio(
            "ER 컨트롤러",
            options=["etd", "em", "both"],
            format_func=lambda x: {"etd": "ER-ETD", "em": "ER-EM", "both": "ETD + EM 비교"}[x],
            horizontal=True, key="sw_er_type",
        )
        _sw_state_variant = st.selectbox(
            "State variant", options=list(STATE_VARIANTS),
            index=list(STATE_VARIANTS).index(DEFAULT_STATE_VARIANT),
            format_func=lambda k: STATE_VARIANTS[k]["label"], key="sw_state_variant",
        )

        with st.expander("고정 Q-learning 파라미터", expanded=False):
            _sw_alpha     = st.number_input("alpha",     min_value=0.001, max_value=1.0,   value=0.06,   format="%.4f", key="sw_alpha")
            _sw_gamma_q   = st.number_input("gamma",     min_value=0.1,   max_value=0.999, value=0.88,   format="%.3f", key="sw_gamma")
            _sw_eps_min   = st.number_input("eps_min",   min_value=0.001, max_value=0.5,   value=0.018,  format="%.4f", key="sw_eps_min")
            _sw_eps_decay = st.number_input("eps_decay", min_value=0.990, max_value=1.0,   value=0.9995, format="%.6f", key="sw_eps_decay")
            _sw_reward    = st.selectbox(
                "Reward variant", options=list(REWARD_VARIANTS),
                index=list(REWARD_VARIANTS).index(DEFAULT_REWARD_VARIANT),
                format_func=_reward_label, key="sw_reward_variant",
            )
            st.caption("보상 함수 커스텀")
            _sw_use_custom_rp = st.toggle(
                "커스텀 보상 수치 사용",
                value=False, key="sw_use_custom_rp",
                help="ON: 아래 수치로 composite 보상 고정. OFF: 위 Reward variant 사용.",
            )
            if _sw_use_custom_rp:
                _sw_rp_sb = st.number_input(
                    "성공 기본 보상 (success_base)", min_value=0.0, max_value=5.0,
                    value=1.0, format="%.4f", key="sw_rp_success_base",
                )
                _sw_rp_sf = st.number_input(
                    "성공 공정성 보상 (success_fair)", min_value=0.0, max_value=5.0,
                    value=0.0, format="%.4f", key="sw_rp_success_fair",
                )
                if _sw_rp_sf > 0:
                    st.warning("success_fair는 글로벌 정보를 사용합니다. 순수 분산 설정에서는 0 권장.", icon="⚠️")
                _sw_rp_fa = st.number_input(
                    "실패 패널티 |fail| (절댓값 입력)", min_value=0.0, max_value=5.0,
                    value=1.0, format="%.4f", key="sw_rp_fail_abs",
                )
                _sw_rp_ip = st.number_input(
                    "패킷 있을 때 IDLE 보상 (idle_pkt)", min_value=-2.0, max_value=2.0,
                    value=0.0, format="%.4f", key="sw_rp_idle_pkt",
                )
                st.caption(
                    f"base `{_sw_rp_sb:.3f}` · fair `{_sw_rp_sf:.3f}` · "
                    f"fail `-{_sw_rp_fa:.3f}` · idle `{_sw_rp_ip:.4f}`"
                )
            else:
                _sw_rp_sb = 1.0; _sw_rp_sf = 0.0; _sw_rp_fa = 1.0; _sw_rp_ip = 0.0

        st.divider()
        _sw_mode = st.radio(
            "스윕 모드",
            options=["single", "all4", "grid2d"],
            format_func={"single": "단일 파라미터", "all4": "4개 전부 (자동)", "grid2d": "2D 그리드 (히트맵)"}.get,
            key="sw_mode",
        )
        _sw_n_repeats = st.number_input(
            "반복 횟수 (포인트당 평균)", min_value=1, max_value=30, value=1, step=1,
            help="각 파라미터 포인트를 다른 seed로 N번 실행 후 평균. 많을수록 정확하지만 N배 느림.",
            key="sw_n_repeats",
        )
        # "both" 모드에서 single 스윕은 공유 파라미터(psi/E0/W)만 허용; mu는 고정값으로 사용
        _sw_param_opts = SWEEP_PARAMS if _sw_er_type == "em" else [p for p in SWEEP_PARAMS if p != "mu"]

        # ── single 모드 ────────────────────────────────────────────────
        if _sw_mode == "single":
            st.subheader("스윕 파라미터")
            _sw_param = st.selectbox(
                "스윕할 파라미터", options=_sw_param_opts,
                format_func=lambda k: SWEEP_PARAM_LABELS.get(k, k), key="sw_sweep_param",
            )
            _c1, _c2, _c3 = st.columns(3)
            with _c1:
                _sw_min   = st.number_input("Min",   value=5.0  if _sw_param == "W" else 0.05, min_value=0.0, format="%.4g", key="sw_min")
            with _c2:
                _sw_max   = st.number_input("Max",   value=80.0 if _sw_param == "W" else 2.0,  min_value=0.0, format="%.4g", key="sw_max")
            with _c3:
                _sw_steps = st.number_input("Steps", min_value=2, max_value=50, value=10, step=1, key="sw_steps")
            _sw_log = st.toggle("Log scale X", value=False, key="sw_log_scale")
            # defaults for unused all4 / grid2d fields
            _sw_param_2 = "E0"; _sw_min_2 = 2.0; _sw_max_2 = 30.0; _sw_steps_2 = 8; _sw_log_2 = False
            _sw_psi_min=0.05; _sw_psi_max=2.0; _sw_psi_steps=10; _sw_psi_log=False
            _sw_E0_min=2.0;  _sw_E0_max=30.0; _sw_E0_steps=10; _sw_E0_log=False
            _sw_W_min=5.0;   _sw_W_max=80.0;  _sw_W_steps=10;  _sw_W_log=False
            _sw_mu_min=0.0;  _sw_mu_max=1.0;  _sw_mu_steps=10; _sw_mu_log=False

            st.subheader("고정 ER 파라미터")
            _sw_fixed_psi = 0.5 if _sw_param == "psi" else st.number_input(
                "고정 psi", value=0.5, min_value=0.0, max_value=5.0, format="%.3f", key="sw_fixed_psi")
            _sw_fixed_E0  = 10.0 if _sw_param == "E0" else st.number_input(
                "고정 E0",  value=10.0, min_value=0.5, max_value=200.0, format="%.1f", key="sw_fixed_E0")
            _sw_fixed_W   = 20   if _sw_param == "W"  else int(st.number_input(
                "고정 W",   value=20, min_value=1, max_value=500, step=5, key="sw_fixed_W"))
            if _sw_er_type in ("em", "both"):
                _sw_fixed_mu = 0.5 if _sw_param == "mu" else st.number_input(
                    "고정 mu", value=0.5, min_value=0.0, max_value=1.0, format="%.3f", key="sw_fixed_mu")
            else:
                _sw_fixed_mu = 0.5
            if _sw_param == "psi": st.caption("psi: 스윕 중")
            if _sw_param == "E0":  st.caption("E0: 스윕 중")
            if _sw_param == "W":   st.caption("W: 스윕 중")
            if _sw_param == "mu":  st.caption("mu: 스윕 중")

        # ── all4 모드 ──────────────────────────────────────────────────
        elif _sw_mode == "all4":
            st.subheader("파라미터별 스윕 범위")
            _all4_param_defs = [
                ("psi", 0.05, 2.0), ("E0", 2.0, 30.0), ("W", 5.0, 80.0),
            ] + ([("mu", 0.0, 1.0)] if _sw_er_type in ("em", "both") else [])

            _sw_all4_ranges: dict = {}
            for _p, _def_lo, _def_hi in _all4_param_defs:
                with st.expander(f"{_p}  [{_def_lo:.4g} ~ {_def_hi:.4g}]", expanded=False):
                    _ac1, _ac2, _ac3 = st.columns(3)
                    with _ac1: _a_lo = st.number_input("Min", value=_def_lo, min_value=0.0, format="%.4g", key=f"sw_a4_{_p}_min")
                    with _ac2: _a_hi = st.number_input("Max", value=_def_hi, min_value=0.0, format="%.4g", key=f"sw_a4_{_p}_max")
                    with _ac3: _a_n  = st.number_input("Steps", min_value=2, max_value=50, value=10, step=1, key=f"sw_a4_{_p}_steps")
                    _a_log = st.toggle("Log scale", value=False, key=f"sw_a4_{_p}_log")
                    _sw_all4_ranges[_p] = (_a_lo, _a_hi, _a_n, _a_log)

            _sw_psi_min,  _sw_psi_max,  _sw_psi_steps,  _sw_psi_log  = _sw_all4_ranges.get("psi", (0.05,2.0,10,False))
            _sw_E0_min,   _sw_E0_max,   _sw_E0_steps,   _sw_E0_log   = _sw_all4_ranges.get("E0",  (2.0,30.0,10,False))
            _sw_W_min,    _sw_W_max,    _sw_W_steps,     _sw_W_log    = _sw_all4_ranges.get("W",   (5.0,80.0,10,False))
            _sw_mu_min,   _sw_mu_max,   _sw_mu_steps,    _sw_mu_log   = _sw_all4_ranges.get("mu",  (0.0,1.0,10,False))

            st.subheader("고정 ER 파라미터 (다른 파라미터 스윕 시 사용)")
            _sw_fixed_psi = st.number_input("고정 psi", value=0.5, min_value=0.0, max_value=5.0,   format="%.3f", key="sw_fixed_psi")
            _sw_fixed_E0  = st.number_input("고정 E0",  value=10.0, min_value=0.5, max_value=200.0, format="%.1f",  key="sw_fixed_E0")
            _sw_fixed_W   = int(st.number_input("고정 W", value=20, min_value=1, max_value=500, step=5, key="sw_fixed_W"))
            if _sw_er_type in ("em", "both"):
                _sw_fixed_mu = st.number_input("고정 mu", value=0.5, min_value=0.0, max_value=1.0, format="%.3f", key="sw_fixed_mu")
            else:
                _sw_fixed_mu = 0.5
            # unused single/grid2d fields
            _sw_param = "psi"; _sw_min = 0.05; _sw_max = 2.0; _sw_steps = 10; _sw_log = False
            _sw_param_2 = "E0"; _sw_min_2 = 2.0; _sw_max_2 = 30.0; _sw_steps_2 = 8; _sw_log_2 = False

        # ── grid2d 모드 ────────────────────────────────────────────────
        else:
            st.subheader("파라미터 1 (X축)")
            _sw_param = st.selectbox(
                "파라미터 1", options=_sw_param_opts,
                format_func=lambda k: SWEEP_PARAM_LABELS.get(k, k), key="sw_sweep_param",
            )
            _c1, _c2, _c3 = st.columns(3)
            with _c1: _sw_min   = st.number_input("Min 1", value=5.0 if _sw_param=="W" else 0.05, min_value=0.0, format="%.4g", key="sw_min")
            with _c2: _sw_max   = st.number_input("Max 1", value=80.0 if _sw_param=="W" else 2.0,  min_value=0.0, format="%.4g", key="sw_max")
            with _c3: _sw_steps = st.number_input("Steps 1", min_value=2, max_value=30, value=8, step=1, key="sw_steps")
            _sw_log = st.toggle("Log scale (축 1)", value=False, key="sw_log_scale")

            st.subheader("파라미터 2 (Y축)")
            _param2_opts = [p for p in _sw_param_opts if p != _sw_param]
            _sw_param_2 = st.selectbox(
                "파라미터 2", options=_param2_opts,
                format_func=lambda k: SWEEP_PARAM_LABELS.get(k, k), key="sw_sweep_param_2",
            )
            _c4, _c5, _c6 = st.columns(3)
            with _c4: _sw_min_2   = st.number_input("Min 2", value=5.0 if _sw_param_2=="W" else 2.0, min_value=0.0, format="%.4g", key="sw_min_2")
            with _c5: _sw_max_2   = st.number_input("Max 2", value=80.0 if _sw_param_2=="W" else 30.0, min_value=0.0, format="%.4g", key="sw_max_2")
            with _c6: _sw_steps_2 = st.number_input("Steps 2", min_value=2, max_value=30, value=8, step=1, key="sw_steps_2")
            _sw_log_2 = st.toggle("Log scale (축 2)", value=False, key="sw_log_scale_2")

            _est_sims = int(_sw_steps) * int(_sw_steps_2) * int(_sw_n_repeats)
            st.warning(f"예상 실행 수: {_est_sims}회 ({int(_sw_steps)}×{int(_sw_steps_2)}×{int(_sw_n_repeats)})", icon="⚠️")

            st.subheader("고정 ER 파라미터")
            _active_params = {_sw_param, _sw_param_2}
            _sw_fixed_psi = 0.5  if "psi" in _active_params else st.number_input("고정 psi", value=0.5, min_value=0.0, max_value=5.0, format="%.3f", key="sw_fixed_psi")
            _sw_fixed_E0  = 10.0 if "E0"  in _active_params else st.number_input("고정 E0",  value=10.0, min_value=0.5, max_value=200.0, format="%.1f", key="sw_fixed_E0")
            _sw_fixed_W   = 20   if "W"   in _active_params else int(st.number_input("고정 W", value=20, min_value=1, max_value=500, step=5, key="sw_fixed_W"))
            if _sw_er_type in ("em", "both"):
                _sw_fixed_mu = 0.5 if "mu" in _active_params else st.number_input("고정 mu", value=0.5, min_value=0.0, max_value=1.0, format="%.3f", key="sw_fixed_mu")
            else:
                _sw_fixed_mu = 0.5
            for _ap in _active_params:
                st.caption(f"{_ap}: 스윕 중")
            # unused all4 fields
            _sw_psi_min=0.05; _sw_psi_max=2.0; _sw_psi_steps=10; _sw_psi_log=False
            _sw_E0_min=2.0;  _sw_E0_max=30.0; _sw_E0_steps=10; _sw_E0_log=False
            _sw_W_min=5.0;   _sw_W_max=80.0;  _sw_W_steps=10;  _sw_W_log=False
            _sw_mu_min=0.0;  _sw_mu_max=1.0;  _sw_mu_steps=10; _sw_mu_log=False

        # ── 예상 실행 수 표시 (single/all4) ──────────────────────────
        if _sw_mode != "grid2d":
            try:
                _est_cfg = ERSweepConfig(
                    sweep_mode=_sw_mode, n_repeats=int(_sw_n_repeats), er_type=_sw_er_type,
                    sweep_param=_sw_param, sweep_min=float(_sw_min), sweep_max=float(_sw_max),
                    sweep_n_steps=int(_sw_steps), sweep_log_scale=bool(_sw_log),
                    psi_min=float(_sw_psi_min), psi_max=float(_sw_psi_max), psi_steps=int(_sw_psi_steps), psi_log=bool(_sw_psi_log),
                    E0_min=float(_sw_E0_min),   E0_max=float(_sw_E0_max),   E0_steps=int(_sw_E0_steps),   E0_log=bool(_sw_E0_log),
                    W_min=float(_sw_W_min),      W_max=float(_sw_W_max),     W_steps=int(_sw_W_steps),     W_log=bool(_sw_W_log),
                    mu_min=float(_sw_mu_min),    mu_max=float(_sw_mu_max),   mu_steps=int(_sw_mu_steps),   mu_log=bool(_sw_mu_log),
                )
                st.caption(f"예상 실행 수: **{_sw_count(_est_cfg)}회**")
            except Exception:
                pass
    else:
        _sw_er_type = "etd"; _sw_state_variant = DEFAULT_STATE_VARIANT
        _sw_alpha = 0.06; _sw_gamma_q = 0.88; _sw_eps_min = 0.018; _sw_eps_decay = 0.9995
        _sw_reward = DEFAULT_REWARD_VARIANT
        _sw_use_custom_rp = False
        _sw_rp_sb = 1.0; _sw_rp_sf = 0.0; _sw_rp_fa = 1.0; _sw_rp_ip = 0.0
        _sw_mode = "single"; _sw_n_repeats = 1
        _sw_param = "psi"; _sw_min = 0.05; _sw_max = 2.0; _sw_steps = 10; _sw_log = False
        _sw_param_2 = "E0"; _sw_min_2 = 2.0; _sw_max_2 = 30.0; _sw_steps_2 = 8; _sw_log_2 = False
        _sw_psi_min=0.05; _sw_psi_max=2.0; _sw_psi_steps=10; _sw_psi_log=False
        _sw_E0_min=2.0;  _sw_E0_max=30.0; _sw_E0_steps=10; _sw_E0_log=False
        _sw_W_min=5.0;   _sw_W_max=80.0;  _sw_W_steps=10;  _sw_W_log=False
        _sw_mu_min=0.0;  _sw_mu_max=1.0;  _sw_mu_steps=10; _sw_mu_log=False
        _sw_fixed_psi = 0.5; _sw_fixed_E0 = 10.0; _sw_fixed_W = 20; _sw_fixed_mu = 0.5

    if mode != MODE_SNAPSHOT:
        run_clicked = st.button("Run Simulation", type="primary", width="stretch")
    else:
        run_clicked = False


# 카트에서 체크된 프리셋 수집 (사이드바 렌더 후 세션 상태에서 읽음)
_cart_for_run = st.session_state.get("_cart", [])
_selected_presets = [
    _cart_for_run[i] for i in range(len(_cart_for_run))
    if st.session_state.get(f"cart_check_{i}", False)
]

if run_clicked:
    # 현재 설정을 파일로 저장 (새로고침 후 복원에 사용)
    _settings_to_save = {k: v for k, v in st.session_state.items()
                         if k.startswith("s_") or k.startswith("baseline_")}
    _save_settings(_settings_to_save)

    output_tag = mode.lower().replace(" ", "_").replace("/", "_")
    output_dir = os.path.join("outputs", f"gui_{output_tag}")
    status = st.status("Running simulation...", expanded=True)

    try:
        if mode == MODE_LOG:
            st.stop()
        if mode not in (MODE_VARIANT, MODE_STATE, MODE_ACTION, MODE_PHASE, MODE_ER, MODE_OPTUNA, MODE_ER_SWEEP) and not selected_baselines:
            st.error("Select at least one baseline.")
            st.stop()
        _ql_baseline_keys = {"decentralized_q_learning", "q_learning_abs", "q_learning_rel_sfch", "q_learning_abs_sfch"}
        if mode not in (MODE_VARIANT, MODE_STATE, MODE_ACTION, MODE_PHASE, MODE_ER, MODE_OPTUNA, MODE_ER_SWEEP) and _ql_baseline_keys.intersection(selected_baselines) and not selected_variant_keys:
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

        if mode == MODE_OPTUNA:
            optuna_cfg = OptunaConfig(
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
                enable_rician_fading=enable_rician_fading,
                rician_k_factor=rician_k_factor,
                rician_fade_margin_db=rician_fade_margin_db,
                layout=layout,
                state_variant=state_variant,
                controller_keys=tuple(optuna_controller_keys),
                n_trials=int(optuna_n_trials),
                n_jobs=int(optuna_n_jobs),
                parallel_studies=bool(optuna_parallel_studies),
                objective_metric=optuna_objective,
                search_reward_variant=optuna_search_reward,
                alpha_low=float(optuna_alpha_low),
                alpha_high=float(optuna_alpha_high),
                gamma_low=float(optuna_gamma_low),
                gamma_high=float(optuna_gamma_high),
                eps_min_low=float(optuna_eps_min_low),
                eps_min_high=float(optuna_eps_min_high),
                eps_decay_low=float(optuna_eps_decay_low),
                eps_decay_high=float(optuna_eps_decay_high),
                psi_low=float(optuna_psi_low),
                psi_high=float(optuna_psi_high),
                E0_low=float(optuna_E0_low),
                E0_high=float(optuna_E0_high),
                W_low=int(optuna_W_low),
                W_high=int(optuna_W_high),
                mu_low=float(optuna_mu_low),
                mu_high=float(optuna_mu_high),
                fix_mu=bool(optuna_fix_mu),
                fixed_mu=float(optuna_fixed_mu),
                kab_J_low=int(optuna_kab_J_low),
                kab_J_high=int(optuna_kab_J_high),
                kab_D_max_low=int(optuna_kab_D_max_low),
                kab_D_max_high=int(optuna_kab_D_max_high),
                w_asr=float(optuna_w_asr),
                w_thr=float(optuna_w_thr),
                w_fairness=float(optuna_w_fairness),
                search_parametric_reward=bool(optuna_search_parametric_reward),
                r_success_base_low=float(optuna_r_success_base_low),
                r_success_base_high=float(optuna_r_success_base_high),
                r_success_fair_low=float(optuna_r_success_fair_low),
                r_success_fair_high=float(optuna_r_success_fair_high),
                r_fail_abs_low=float(optuna_r_fail_abs_low),
                r_fail_abs_high=float(optuna_r_fail_abs_high),
                r_idle_pkt_low=float(optuna_r_idle_pkt_low),
                r_idle_pkt_high=float(optuna_r_idle_pkt_high),
                fix_base_params=bool(optuna_fix_base),
                fix_q_learning_only=bool(optuna_fix_q_only),
                fixed_alpha=float(optuna_fixed_alpha),
                fixed_gamma_q=float(optuna_fixed_gamma_q),
                fixed_eps_min=float(optuna_fixed_eps_min),
                fixed_eps_decay=float(optuna_fixed_eps_decay),
                fix_reward=bool(optuna_fix_reward),
                fix_reward_numeric=bool(optuna_fix_reward_numeric),
                fixed_reward_variant=str(optuna_fixed_reward_variant),
                fixed_r_success_base=float(optuna_fixed_r_success_base),
                fixed_r_success_fair=float(optuna_fixed_r_success_fair),
                fixed_r_fail_abs=float(optuna_fixed_r_fail_abs),
                fixed_r_idle_pkt=float(optuna_fixed_r_idle_pkt),
                output_dir=output_dir,
            )

            _n_studies = len(optuna_controller_keys)
            _progress_bars = {
                k: st.progress(0.0, text=f"{TUNABLE_CONTROLLERS[k]['label']}…")
                for k in optuna_controller_keys
            }

            def _optuna_progress(ctrl_key: str, trial_num: int, n_trials: int, best_value: float) -> None:
                if ctrl_key in _progress_bars:
                    label = TUNABLE_CONTROLLERS[ctrl_key]["label"]
                    _progress_bars[ctrl_key].progress(
                        trial_num / n_trials,
                        text=f"{label}: {trial_num}/{n_trials} | best={best_value:.4f}",
                    )

            def _study_done(ctrl_key: str, res: dict) -> None:
                if ctrl_key in _progress_bars:
                    label = TUNABLE_CONTROLLERS[ctrl_key]["label"]
                    best_val = res["best_params"].get("objective_value", 0.0)
                    _progress_bars[ctrl_key].progress(
                        1.0,
                        text=f"✅ {label} 완료 | best={best_val:.4f}",
                    )

            with status:
                st.write(f"Running Optuna: {_n_studies}개 컨트롤러 × {optuna_n_trials} trials (순차 실행)...")
            result = run_optuna_tune(
                optuna_cfg,
                progress_callback=_optuna_progress,
                study_complete_callback=_study_done,
            )

            # ── 결과 표시 ──────────────────────────────────────────────
            all_sub = result["results"]
            _all_trial_rows = []
            _all_fig_paths = []

            # winner 결정 (objective_value 최대)
            _winner_key = max(
                all_sub,
                key=lambda k: all_sub[k]["best_params"].get("objective_value", -1),
            )
            _winner = all_sub[_winner_key]["best_params"]
            _obj_label = OBJECTIVE_METRICS.get(optuna_objective, optuna_objective)
            _metric_skip = {"objective_value", "objective_metric", "label", "controller",
                            "mean_backlog_per_node", "final_backlog_per_node"}
            _hparam_keys = ["reward_variant", "alpha", "gamma_q", "eps_min", "eps_decay",
                            "psi", "E0", "W", "mu",
                            "r_success_base", "r_success_fair", "r_fail_abs", "r_idle_pkt"]
            _detail_keys = [("success_rate", "ASR"), ("fairness", "Fair(S)"),
                            ("fairness_thr", "Fair(Thr)"),
                            ("throughput", "Throughput"), ("collision_rate", "Collision")]

            # 우승자 카드
            _hp_str = "  |  ".join(
                f"{k} = {_winner[k]:.5g}" if isinstance(_winner.get(k), float) else f"{k} = {_winner.get(k)}"
                for k in _hparam_keys if _winner.get(k) is not None
            )
            _dm_str = "  |  ".join(
                f"{lbl} = {_winner.get(mk, 0):.4f}" for mk, lbl in _detail_keys
            )
            st.success(
                f"**Winner: {all_sub[_winner_key]['label']}**\n\n"
                f"{_obj_label} = **{_winner.get('objective_value', 0):.4f}**\n\n"
                f"{_dm_str}\n\n"
                f"Params: {_hp_str}"
            )

            # 컨트롤러별 요약 (expander)
            with st.expander("Per-controller best results", expanded=False):
                _summary_rows = []
                for k, sub in all_sub.items():
                    bp = sub["best_params"]
                    row = {"controller": sub["label"]}
                    for mk, lbl in _detail_keys:
                        row[lbl] = round(bp.get(mk, 0.0) or 0.0, 4)
                    row[_obj_label] = round(bp.get("objective_value", 0.0), 4)
                    for hk in _hparam_keys:
                        v = bp.get(hk)
                        if v is not None:
                            row[hk] = f"{v:.5g}" if isinstance(v, float) else v
                    _summary_rows.append(row)
                st.dataframe(pd.DataFrame(_summary_rows), hide_index=True, width="stretch")

            for k, sub in all_sub.items():
                _all_trial_rows.extend(sub["trial_rows"])
                _all_fig_paths.extend(sub["fig_paths"])

            if result.get("comparison_fig"):
                _all_fig_paths.insert(0, result["comparison_fig"])

            _main_img = _all_fig_paths[0] if _all_fig_paths else os.path.join(output_dir, "comparison.png")
            _trial_cols = ["trial", "controller", "value",
                           "success_rate", "fairness", "fairness_thr", "throughput", "collision_rate",
                           "reward_variant", "alpha", "gamma_q", "eps_min", "eps_decay",
                           "psi", "E0", "W", "mu",
                           "r_success_base", "r_success_fair", "r_fail_abs", "r_idle_pkt"]
            _store_result(
                mode=mode,
                image_path=_main_img,
                table_rows=_all_trial_rows,
                table_columns=_trial_cols,
                queue_mode=queue_mode,
                extra_images=_all_fig_paths[1:],
                optuna_best_params={k: sub["best_params"] for k, sub in all_sub.items()},
                optuna_env={
                    "n_nodes": int(n_nodes),
                    "target_g": float(target_g),
                    "n_slots": int(n_slots),
                    "warmup_slots": int(warmup_slots),
                    "n_channels": int(channels),
                    "profile": profile,
                    "queue_mode": queue_mode,
                    "layout": layout,
                    "state_variant": state_variant,
                },
            )

            # ── 실행 기록 자동 저장 ────────────────────────────────────
            import datetime as _dt_h
            _h_winner_bp = all_sub[_winner_key]["best_params"]
            _save_optuna_run({
                "run_id": _dt_h.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "timestamp": _dt_h.datetime.now().isoformat(timespec="seconds"),
                "label": "",
                "config_summary": (
                    f"N={n_nodes} G={target_g} ch={channels} {profile} {layout} | "
                    f"{','.join(optuna_controller_keys)} | "
                    f"{optuna_n_trials} trials | {optuna_objective}"
                    + (" | fixed-all" if optuna_fix_base else " | fixed-ql" if optuna_fix_q_only else "")
                ),
                "winner_summary": (
                    f"🏆 {all_sub[_winner_key]['label']} | "
                    f"ASR={_h_winner_bp.get('success_rate', 0):.3f} "
                    f"Fair={_h_winner_bp.get('fairness', 0):.3f} "
                    f"Thr={_h_winner_bp.get('throughput', 0):.2f} "
                    f"Obj={_h_winner_bp.get('objective_value', 0):.4f}"
                ),
                "last_result": st.session_state.get("last_result", {}),
            })

        elif mode == MODE_ER_SWEEP:
            sweep_cfg = ERSweepConfig(
                n_nodes=int(n_nodes),
                target_g=float(target_g),
                n_slots=int(n_slots),
                warmup_slots=int(warmup_slots),
                epoch_slots=int(epoch_slots),
                n_channels=int(channels),
                profile_name=profile,
                seed=42,
                layout=layout,
                queue_mode=queue_mode,
                gw_obs_mode=gw_obs_mode,
                enable_rayleigh_fading=enable_rayleigh_fading,
                rayleigh_fade_margin_db=rayleigh_fade_margin_db,
                enable_rician_fading=enable_rician_fading,
                rician_k_factor=rician_k_factor,
                rician_fade_margin_db=rician_fade_margin_db,
                er_type=_sw_er_type,
                state_variant=_sw_state_variant,
                alpha=float(_sw_alpha),
                gamma_q=float(_sw_gamma_q),
                eps_min=float(_sw_eps_min),
                eps_decay=float(_sw_eps_decay),
                reward_variant=_sw_reward,
                use_custom_reward=bool(_sw_use_custom_rp),
                r_success_base=float(_sw_rp_sb),
                r_success_fair=float(_sw_rp_sf),
                r_fail_abs=float(_sw_rp_fa),
                r_idle_pkt=float(_sw_rp_ip),
                # sweep mode
                sweep_mode=_sw_mode,
                n_repeats=int(_sw_n_repeats),
                # single / grid2d - param 1
                sweep_param=_sw_param,
                sweep_n_steps=int(_sw_steps),
                sweep_log_scale=bool(_sw_log),
                sweep_min=float(_sw_min),
                sweep_max=float(_sw_max),
                # grid2d - param 2
                sweep_param_2=_sw_param_2,
                sweep_n_steps_2=int(_sw_steps_2),
                sweep_log_scale_2=bool(_sw_log_2),
                sweep_min_2=float(_sw_min_2),
                sweep_max_2=float(_sw_max_2),
                # all4 - per-param ranges
                psi_min=float(_sw_psi_min), psi_max=float(_sw_psi_max),
                psi_steps=int(_sw_psi_steps), psi_log=bool(_sw_psi_log),
                E0_min=float(_sw_E0_min),   E0_max=float(_sw_E0_max),
                E0_steps=int(_sw_E0_steps), E0_log=bool(_sw_E0_log),
                W_min=float(_sw_W_min),     W_max=float(_sw_W_max),
                W_steps=int(_sw_W_steps),   W_log=bool(_sw_W_log),
                mu_min=float(_sw_mu_min),   mu_max=float(_sw_mu_max),
                mu_steps=int(_sw_mu_steps), mu_log=bool(_sw_mu_log),
                # fixed ER params
                fixed_psi=float(_sw_fixed_psi),
                fixed_E0=float(_sw_fixed_E0),
                fixed_W=int(_sw_fixed_W),
                fixed_mu=float(_sw_fixed_mu),
                output_dir=output_dir,
            )

            from experiments.er_sweep import count_total_sims as _sw_count_total
            _sw_total = _sw_count_total(sweep_cfg)
            _sw_pb = st.progress(0.0, text=f"0 / {_sw_total} 실행 중…")

            def _sw_progress(cur: int, total: int, asr: float) -> None:
                _sw_pb.progress(cur / max(total, 1),
                                text=f"{cur} / {total} 완료  |  last ASR={asr:.3f}")

            _mode_label = {"single": "단일 파라미터", "all4": "4개 전부", "grid2d": "2D 그리드"}.get(_sw_mode, _sw_mode)
            with status:
                st.write(f"ER Sweep ({_mode_label}): 총 {_sw_total}회 실행 예정…")

            sweep_result = run_er_sweep(sweep_cfg, progress_callback=_sw_progress)
            _sw_pb.progress(1.0)

            _sw_rows = sweep_result["sweep_rows"]
            _best = max(_sw_rows, key=lambda r: r["success_rate"])

            if _sw_mode == "all4":
                st.success(
                    f"**4-Sweep 완료** — 전체 {len(_sw_rows)}포인트  |  "
                    f"최고 ASR = **{_best['success_rate']:.4f}**  "
                    f"(param={_best.get('sweep_param','?')}  val={_best['sweep_value']:.4g})"
                )
            elif _sw_mode == "grid2d":
                st.success(
                    f"**Grid 완료** — {_sw_param} × {_sw_param_2}  |  "
                    f"최고 ASR = **{_best['success_rate']:.4f}**  "
                    f"({_sw_param}={_best.get('psi' if _sw_param=='psi' else _sw_param, '?'):.4g})"
                )
            else:
                st.success(
                    f"**Best ASR = {_best['success_rate']:.4f}**  at  "
                    f"{_sw_param} = {_best['sweep_value']:.4g}  |  "
                    f"Fairness={_best['fairness']:.4f}  Thr={_best['throughput']:.4f}"
                )

            _sw_table_cols = ["step", "sweep_param", "sweep_value", "psi", "E0", "W",
                              "success_rate", "throughput", "fairness", "collision_rate"]
            if _sw_er_type in ("em", "both"):
                _sw_table_cols.insert(6, "mu")
            if _sw_mode == "grid2d":
                _sw_table_cols.insert(3, "sweep_param_2")
                _sw_table_cols.insert(4, "sweep_value_2")
            _store_result(
                mode=mode,
                image_path=sweep_result["image_path"],
                extra_images=sweep_result.get("extra_images", []),
                table_rows=_sw_rows,
                table_columns=_sw_table_cols,
                queue_mode=queue_mode,
            )

        elif mode == MODE_NODE:
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
                enable_rician_fading=enable_rician_fading,
                rician_k_factor=rician_k_factor,
                rician_fade_margin_db=rician_fade_margin_db,
                dual_mab_b=dual_mab_b,
                kaburaki_J=int(kaburaki_J),
                kaburaki_D_max=int(kaburaki_D_max),
                kaburaki_alpha=float(kaburaki_alpha),
                kaburaki_gamma=float(kaburaki_gamma),
                kaburaki_eps_min=float(kaburaki_eps_min),
                psi=float(psi),
                E0=float(E0),
                W=int(W),
                er_mode=er_mode,
                mu=float(mu),
                er_q_variants=tuple(selected_er_variants),
                layout=layout,
                reward_variant=reward_variant,
                reward_variants=tuple(selected_variant_keys),
                state_variant=state_variant,
                output_dir=output_dir,
                baseline_keys=tuple(selected_baselines),
                alpha=float(ql_alpha),
                gamma_q=float(ql_gamma_q),
                eps_min=float(ql_eps_min),
                eps_decay=float(ql_eps_decay),
                reward_params=_loaded_rp if _loaded_rp else None,
                custom_ql_presets=tuple(_selected_presets),
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
                    "attempts",
                    "system_success_rate",
                    "system_collision_rate",
                    "system_throughput",
                    "mae",
                    "elr",
                    "energy_efficiency",
                    "fairness",
                    "fairness_asr",
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
                node_cfg={
                    "n_nodes": int(n_nodes),
                    "target_g": float(target_g),
                    "layout": layout,
                    "profile": profile,
                    "n_slots": int(n_slots),
                    "state_variant": state_variant,
                    "baselines": list(selected_baselines),
                    "er_variants": list(selected_er_variants),
                    "er_mode": er_mode,
                    "psi": float(psi),
                    "E0": float(E0),
                    "W": int(W),
                    "mu": float(mu),
                    "output_dir": result["output_dir"],
                },
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
                        "fairness_asr": mr.get("fairness_asr"),
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
                enable_rician_fading=enable_rician_fading,
                rician_k_factor=rician_k_factor,
                rician_fade_margin_db=rician_fade_margin_db,
                dual_mab_b=dual_mab_b,
                psi=float(psi),
                E0=float(E0),
                W=int(W),
                er_mode=er_mode,
                mu=float(mu),
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
                enable_rician_fading=enable_rician_fading,
                rician_k_factor=rician_k_factor,
                rician_fade_margin_db=rician_fade_margin_db,
                dual_mab_b=dual_mab_b,
                psi=float(psi),
                E0=float(E0),
                W=int(W),
                er_mode=er_mode,
                mu=float(mu),
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
                enable_rician_fading=enable_rician_fading,
                rician_k_factor=rician_k_factor,
                rician_fade_margin_db=rician_fade_margin_db,
                gw_thresholds=gw_thresholds_ui,
                layout=layout,
                controller_key=controller_key,
                reward_variant=reward_variant,
                state_variant_keys=tuple(selected_state_variant_keys),
                psi=float(psi),
                E0=float(E0),
                W=int(W),
                er_mode=er_mode,
                mu=float(mu),
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
                enable_rician_fading=enable_rician_fading,
                rician_k_factor=rician_k_factor,
                rician_fade_margin_db=rician_fade_margin_db,
                gw_thresholds=gw_thresholds_ui,
                layout=layout,
                controller_key=controller_key,
                reward_variant=reward_variant,
                state_variant=state_variant,
                action_series_keys=tuple(selected_action_variant_keys),
                psi=float(psi),
                E0=float(E0),
                W=int(W),
                er_mode=er_mode,
                mu=float(mu),
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

        elif mode == MODE_ER:
            cfg = ERCompareConfig(
                n_nodes=int(n_nodes),
                target_g=float(target_g),
                n_slots=int(n_slots),
                warmup_slots=int(warmup_slots),
                epoch_slots=int(epoch_slots),
                n_channels=int(channels),
                profile_name=profile,
                queue_mode=queue_mode,
                gw_obs_mode=gw_obs_mode,
                reward_variant=reward_variant,
                state_variant=state_variant,
                psi=float(psi),
                E0=float(E0),
                W=int(W),
                mu=float(mu),
                output_dir=output_dir,
            )
            with status:
                st.write("Running ER Compare (baseline / ETD / EM)...")
            result = run_er_compare(cfg)
            _store_result(
                mode=mode,
                image_path=os.path.join(result["output_dir"], "er_compare.png"),
                table_rows=[],
                table_columns=[],
                queue_mode=queue_mode,
            )

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
                enable_rician_fading=enable_rician_fading,
                rician_k_factor=rician_k_factor,
                rician_fade_margin_db=rician_fade_margin_db,
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
                enable_rician_fading=enable_rician_fading,
                rician_k_factor=rician_k_factor,
                rician_fade_margin_db=rician_fade_margin_db,
                layout=layout,
                controller_key=controller_key,
                variant_keys=tuple(selected_variant_keys),
                state_variant=state_variant,
                psi=float(psi),
                E0=float(E0),
                W=int(W),
                er_mode=er_mode,
                mu=float(mu),
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
    MODE_ER: "er_compare.png",
    MODE_OPTUNA: "optuna_history.png",
    MODE_ER_SWEEP: "er_sweep.png",
}

last_result = st.session_state.get("last_result")

if not last_result and mode not in (MODE_LOG,) and mode in _MODE_IMAGE_NAMES:
    _output_tag = mode.lower().replace(" ", "_").replace("/", "_")
    _auto_dir = os.path.abspath(os.path.join("outputs", f"gui_{_output_tag}"))
    _auto_img = os.path.join(_auto_dir, _MODE_IMAGE_NAMES[mode])
    if os.path.exists(_auto_img):
        st.info("Showing results from a previous run. Press 'Run Simulation' to refresh.")
        st.image(_auto_img, width="stretch")

if last_result and mode not in (MODE_LOG, MODE_SNAPSHOT):
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

        # Optuna Tune: 컨트롤러 필터 + 지표 필터
        if last_result.get("mode") == MODE_OPTUNA and "controller" in df.columns:
            _ctrl_options = sorted(df["controller"].dropna().unique().tolist())
            _col_f1, _col_f2 = st.columns([1, 2])
            with _col_f1:
                _sel_ctrls = st.multiselect(
                    "Controller",
                    options=_ctrl_options,
                    default=_ctrl_options,
                    key="optuna_filter_ctrl",
                )
            with _col_f2:
                _metric_cols = ["value", "success_rate", "fairness", "fairness_thr", "throughput", "collision_rate"]
                _available_metrics = [c for c in _metric_cols if c in df.columns]
                _sel_metrics = st.multiselect(
                    "표시할 지표",
                    options=_available_metrics,
                    default=_available_metrics,
                    key="optuna_filter_metrics",
                )
            if _sel_ctrls:
                df = df[df["controller"].isin(_sel_ctrls)]
            # 선택한 지표 + 하이퍼파라미터 컬럼 구성
            _hparam_cols = ["trial", "controller", "reward_variant",
                            "alpha", "gamma_q", "eps_min", "eps_decay",
                            "psi", "E0", "W", "mu",
                            "r_success_base", "r_success_fair", "r_fail_abs", "r_idle_pkt"]
            _core_metric_cols = [c for c in ["value", "success_rate", "fairness", "fairness_thr"] if c in df.columns]
            _show_cols = (
                [c for c in _hparam_cols if c in df.columns]
                + _core_metric_cols
                + [c for c in _sel_metrics if c in df.columns and c not in _hparam_cols and c not in _core_metric_cols]
            )
            df = df[_show_cols]
            # 가장 좋은 trial 상단 정렬
            if "value" in df.columns:
                df = df.sort_values("value", ascending=False)

        if last_result.get("mode") == MODE_OPTUNA:
            display_df = df  # _show_cols 필터 이미 적용됨
        else:
            visible_columns = [column for column in table_columns if column in df.columns]
            display_df = df[visible_columns] if visible_columns else df
        _rename_map = {
            "baseline_label": "Baseline",
            "variant_label": "Reward Variant",
            "success_rate": "ASR",
            "fairness": "Fair(S)",
            "fairness_thr": "Fair(Thr)",
            "throughput": "Throughput",
            "collision_rate": "Collision",
            "value": "Objective",
        }

        # ── Optuna: 체크박스로 행 선택 → 카트 담기 ───────────────────────
        if last_result.get("mode") == MODE_OPTUNA:
            import datetime as _dt
            from experiments.optuna_tune import TUNABLE_CONTROLLERS as _TC
            # 원본 df(필터·정렬된)를 인덱스 리셋 후 체크박스 컬럼 추가
            _edit_df = display_df.reset_index(drop=True).copy()
            _edit_df.insert(0, "🛒", False)
            # 컬럼 구조가 바뀌면 data_editor 캐시 초기화 (stale state 방지)
            _cur_editor_cols = tuple(_edit_df.columns)
            if st.session_state.get("_optuna_editor_cols") != _cur_editor_cols:
                st.session_state["_optuna_editor_cols"] = _cur_editor_cols
                st.session_state.pop("optuna_trial_editor", None)
            _col_cfg = {"🛒": st.column_config.CheckboxColumn("🛒", help="카트에 담을 행 선택", width="small")}
            _col_cfg.update({
                c: st.column_config.NumberColumn(label=_rename_map.get(c, c), format="%.4f")
                for c in _edit_df.columns if _edit_df[c].dtype == float and c != "🛒"
            })
            _edited = st.data_editor(
                _edit_df,
                column_config=_col_cfg,
                disabled=[c for c in _edit_df.columns if c != "🛒"],
                use_container_width=True,
                hide_index=True,
                key="optuna_trial_editor",
            )
            _selected_rows = _edited[_edited["🛒"] == True]
            _n_sel = len(_selected_rows)
            _c_btn, _c_info = st.columns([1, 3])
            with _c_btn:
                _add_clicked = st.button(
                    f"Add {_n_sel} to Cart" if _n_sel else "Add to Cart",
                    disabled=(_n_sel == 0),
                    type="primary",
                    use_container_width=True,
                )
            with _c_info:
                if _n_sel:
                    st.caption(f"{_n_sel}개 행 선택됨. 버튼을 눌러 프리셋으로 저장하세요.")
                else:
                    st.caption("🛒 열의 체크박스를 클릭해 원하는 trial을 선택하세요.")
            if _add_clicked and _n_sel:
                _cur_cart = st.session_state.get("_cart", [])
                _added_names = []

                def _sf(v, d=0.0):
                    """NaN/None 안전 float 변환."""
                    try:
                        r = float(v)
                        return r if r == r else d  # NaN → default
                    except (TypeError, ValueError):
                        return d

                def _si(v, d=0):
                    """NaN/None 안전 int 변환."""
                    try:
                        r = float(v)
                        return int(r) if r == r else d
                    except (TypeError, ValueError):
                        return d

                _run_env = last_result.get("optuna_env", {})
                for _, _row in _selected_rows.iterrows():
                    _ck = str(_row.get("controller", "q_learning"))
                    _ctrl_label = _TC.get(_ck, {}).get("label", _ck)
                    _rp = _build_reward_params_from_best(dict(_row))
                    _obj = _sf(_row.get("value") if pd.notna(_row.get("value")) else _row.get("Objective"))
                    _name = f"Trial#{_si(_row.get('trial'))} {_ctrl_label} | Obj={_obj:.4f}"
                    _item = {
                        "id": str(_dt.datetime.now().timestamp()),
                        "name": _name,
                        "controller_key": _ck,
                        "controller_label": _ctrl_label,
                        "alpha":     _sf(_row.get("alpha"),     0.1),
                        "gamma_q":   _sf(_row.get("gamma_q"),   0.9),
                        "eps_min":   _sf(_row.get("eps_min"),   0.05),
                        "eps_decay": _sf(_row.get("eps_decay"), 0.9995),
                        "reward_variant": str(_row.get("reward_variant") or DEFAULT_REWARD_VARIANT),
                        "reward_params":  _rp,
                        "psi":    _sf(_row.get("psi"),  0.0),
                        "E0":     _sf(_row.get("E0"),   10.0),
                        "W":      _si(_row.get("W"),    20),
                        "er_mode":  str(_row.get("er_mode") or "ETD"),
                        "mu":     _sf(_row.get("mu"),   0.5),
                        "objective_value": _obj,
                        "success_rate":    _sf(_row.get("success_rate")),
                        "fairness":        _sf(_row.get("fairness")),
                        "throughput":      _sf(_row.get("throughput")),
                        "added_at": _dt.datetime.now().isoformat(timespec="seconds"),
                        "env": _run_env,
                    }
                    _cur_cart.append(_item)
                    _added_names.append(_name)
                st.session_state["_cart"] = _cur_cart
                _save_cart(_cur_cart)
                st.success(f"카트에 추가됨: {', '.join(_added_names)}")
                st.rerun()
        else:
            renamed = display_df.rename(columns=_rename_map)
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

    # ── Node Analysis 스냅샷 저장 ──────────────────────────────────────────
    if last_result.get("mode") == MODE_NODE and last_result.get("node_cfg"):
        st.divider()
        _ncfg = last_result["node_cfg"]

        def _make_snapshot_name(ncfg: dict, label: str) -> str:
            import re
            bl = ncfg.get("baselines", [])
            er = ncfg.get("er_variants", [])
            _bl_abbr = "+".join(
                k.replace("decentralized_q_learning", "ql")
                 .replace("q_learning_abs_sfch", "ql_abs_sfch")
                 .replace("q_learning_rel_sfch", "ql_rel_sfch")
                 .replace("q_learning_abs", "ql_abs")
                 .replace("er_etd", "etd")
                 .replace("er_em", "em")
                 .replace("pure_aloha", "aloha")
                 .replace("adr_like", "adr")
                for k in (bl + er)
            ) or "none"
            base = (
                f"N{ncfg.get('n_nodes','?')}"
                f"_G{ncfg.get('target_g', 0):.1f}"
                f"_{ncfg.get('layout','?')}"
                f"_{ncfg.get('profile','?')}"
                f"_{_bl_abbr}"
                f"_{ncfg.get('state_variant','').replace('_','')}"
            )
            if label.strip():
                safe = re.sub(r"[^\w\-]", "_", label.strip())[:30]
                base = f"{base}_{safe}"
            return base

        _snap_label = st.text_input(
            "저장 레이블 (선택)",
            value="",
            placeholder="예: er_etd_controlled_study",
            key="s_node_snap_label",
            help="폴더명 끝에 붙는 설명. 비워도 됩니다.",
        )
        _snap_name = _make_snapshot_name(_ncfg, _snap_label)
        st.caption(f"저장 폴더명: `{_snap_name}/`")

        if st.button("💾 결과 저장", key="btn_node_snapshot"):
            import shutil, json as _json
            _snap_dir = os.path.join("outputs", "node_snapshots", _snap_name)
            os.makedirs(_snap_dir, exist_ok=True)

            # 그래프 복사
            _imgs = {
                "per_node.png": last_result.get("image_path"),
                "node_layout.png": last_result.get("layout_image"),
                "sf_heatmap.png": last_result.get("sf_heatmap_image"),
            }
            for i, ei in enumerate(last_result.get("extra_images", [])):
                _imgs[f"extra_{i}.png"] = ei
            for _fname, _src in _imgs.items():
                if _src and os.path.exists(_src):
                    shutil.copy2(_src, os.path.join(_snap_dir, _fname))

            # 수치 CSV 저장
            _trows = last_result.get("table_rows", [])
            if _trows:
                import csv as _csv
                _csv_path = os.path.join(_snap_dir, "summary.csv")
                with open(_csv_path, "w", newline="", encoding="utf-8") as _f:
                    _w = _csv.DictWriter(_f, fieldnames=list(_trows[0].keys()))
                    _w.writeheader()
                    _w.writerows(_trows)

            # 환경 설정 JSON
            _json_cfg = dict(_ncfg)
            _json_cfg["snapshot_label"] = _snap_label
            with open(os.path.join(_snap_dir, "config.json"), "w", encoding="utf-8") as _f:
                _json.dump(_json_cfg, _f, ensure_ascii=False, indent=2)

            # 시계열 그래프 저장
            _snap_epoch = last_result.get("epoch_data")
            if _snap_epoch:
                _save_node_timeseries(_snap_epoch, _snap_dir)

            _er_info = ""
            if _ncfg.get("psi", 0.0) > 0:
                _show_mu_snap = _ncfg.get("er_mode") == "EM" or "em" in _ncfg.get("er_variants", [])
                _er_info = (
                    f"  |  ψ={_ncfg['psi']}  E₀={_ncfg['E0']}  W={_ncfg['W']}"
                    + (f"  μ={_ncfg['mu']}" if _show_mu_snap else "")
                )
            st.success(f"저장 완료: `outputs/node_snapshots/{_snap_name}/`{_er_info}")

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
elif mode == MODE_SNAPSHOT:
    import json as _snap_json
    import glob as _glob

    _snap_root = os.path.join("outputs", "node_snapshots")
    _snap_dirs = sorted(
        [d for d in _glob.glob(os.path.join(_snap_root, "*")) if os.path.isdir(d)],
        key=os.path.getmtime,
        reverse=True,
    )

    if not _snap_dirs:
        st.info(f"`{_snap_root}/` 에 저장된 스냅샷이 없습니다. Node Analysis 결과 화면에서 💾 결과 저장을 눌러 저장하세요.")
    else:
        # ── 인덱스 테이블 ────────────────────────────────────────────────────
        _index_rows = []
        for _sd in _snap_dirs:
            _cfg_path = os.path.join(_sd, "config.json")
            _cfg = {}
            if os.path.exists(_cfg_path):
                try:
                    with open(_cfg_path, encoding="utf-8") as _f:
                        _cfg = _snap_json.load(_f)
                except Exception:
                    pass
            _bl = ", ".join(_cfg.get("baselines", []) + _cfg.get("er_variants", []))
            _psi = _cfg.get("psi", 0.0)
            _index_rows.append({
                "폴더명": os.path.basename(_sd),
                "N": _cfg.get("n_nodes", "?"),
                "G": _cfg.get("target_g", "?"),
                "layout": _cfg.get("layout", "?"),
                "profile": _cfg.get("profile", "?"),
                "slots": _cfg.get("n_slots", "?"),
                "state": _cfg.get("state_variant", "?"),
                "baselines": _bl,
                "ψ": _psi if _psi else "—",
                "E₀": _cfg.get("E0", "—") if _psi else "—",
                "W": _cfg.get("W", "—") if _psi else "—",
                "μ": _cfg.get("mu", "—") if (_cfg.get("er_mode") == "EM" or "em" in _cfg.get("er_variants", [])) else "—",
            })

        st.subheader(f"저장된 스냅샷 ({len(_snap_dirs)}개)")
        _idx_df = pd.DataFrame(_index_rows)
        st.dataframe(_idx_df, width="stretch", hide_index=True)

        st.divider()

        # ── 스냅샷 선택 후 상세 보기 ─────────────────────────────────────────
        _snap_names = [os.path.basename(d) for d in _snap_dirs]
        _sel = st.selectbox("상세 보기", options=_snap_names, key="s_snap_select")
        _sel_dir = os.path.join(_snap_root, _sel)

        # config
        _sel_cfg_path = os.path.join(_sel_dir, "config.json")
        if os.path.exists(_sel_cfg_path):
            with open(_sel_cfg_path, encoding="utf-8") as _f:
                _sel_cfg = _snap_json.load(_f)
            with st.expander("환경 설정 (config.json)", expanded=True):
                _cfg_disp = {k: v for k, v in _sel_cfg.items() if k != "output_dir"}
                st.json(_cfg_disp)

        # summary CSV
        _sel_csv = os.path.join(_sel_dir, "summary.csv")
        if os.path.exists(_sel_csv):
            st.subheader("수치 요약")
            _sdf = pd.read_csv(_sel_csv)
            st.dataframe(_sdf.round(4), width="stretch", hide_index=True)
            _csv_dl = _sdf.to_csv(index=False).encode("utf-8")
            st.download_button("CSV 다운로드", _csv_dl, file_name=f"{_sel}_summary.csv", mime="text/csv")

        # 그래프
        _img_map = {
            "per_node.png": "분석 그래프",
            "node_layout.png": "노드 배치",
            "sf_heatmap.png": "SF 히트맵",
        }
        for _ifname, _ititle in _img_map.items():
            _ipath = os.path.join(_sel_dir, _ifname)
            if os.path.exists(_ipath):
                st.divider()
                st.subheader(_ititle)
                st.image(_ipath, width="stretch")

        # extra 이미지
        for _ei in sorted(_glob.glob(os.path.join(_sel_dir, "extra_*.png"))):
            st.divider()
            st.image(_ei, width="stretch")

        # 시계열 그래프 (저장 시점에 생성된 timeseries_*.png)
        _ts_files = sorted(_glob.glob(os.path.join(_sel_dir, "timeseries_*.png")))
        if _ts_files:
            st.divider()
            st.subheader("Per-node Cumulative Time-Series")
            for _ts in _ts_files:
                st.image(_ts, width="stretch")

        # 삭제 버튼
        st.divider()
        if st.button("🗑 이 스냅샷 삭제", key="btn_snap_delete", type="secondary"):
            import shutil as _snap_shutil
            _snap_shutil.rmtree(_sel_dir, ignore_errors=True)
            st.success(f"`{_sel}` 삭제 완료")
            st.rerun()

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
