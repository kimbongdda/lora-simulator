"""ER 파라미터 스윕 실험.

세 가지 스윕 모드:
  single  — psi / E0 / W / mu 중 하나를 1D 스윕 (라인 플롯)
  all4    — 4개 파라미터 각각 순서대로 1D 스윕 (총 3-4회 자동 실행)
  grid2d  — 두 파라미터를 2D 그리드로 스윕 (히트맵)

각 그리드 포인트에서 n_repeats 번 다른 seed로 실행 후 평균을 냄.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from typing import Callable

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.q_learning import DEFAULT_STATE_VARIANT, DecentralizedQLearningController
from env import ScenarioConfig, run_simulation
from experiments.style import configure_fonts
from utils.io import ensure_dir, write_csv_rows, write_json

SWEEP_PARAMS = ["psi", "E0", "W", "mu"]

SWEEP_PARAM_LABELS = {
    "psi": "psi (penalty strength)",
    "E0":  "E0 (energy budget)",
    "W":   "W (window size)",
    "mu":  "mu (EM mix coefficient)",
}

METRICS = [
    ("success_rate",   "ASR",                "#1f77b4"),
    ("throughput",     "Throughput",         "#2ca02c"),
    ("fairness",       "Jain FI (성공수)",   "#ff7f0e"),
    ("fairness_asr",   "Jain FI (ASR)",      "#9467bd"),
    ("fairness_thr",   "Jain FI (Thr)",      "#8c564b"),
    ("collision_rate", "Collision Rate",     "#d62728"),
]


@dataclasses.dataclass(frozen=True)
class ERSweepConfig:
    # 시나리오
    n_nodes: int = 60
    target_g: float = 2.0
    n_slots: int = 15_000
    warmup_slots: int = 3_000
    epoch_slots: int = 500
    n_channels: int = 3
    profile_name: str = "short"
    seed: int = 42
    layout: str = "random"
    queue_mode: str = "accumulate"
    gw_obs_mode: str = "attempt"
    enable_rayleigh_fading: bool = False
    rayleigh_fade_margin_db: float = 10.0
    enable_rician_fading: bool = False
    rician_k_factor: float = 4.0
    rician_fade_margin_db: float = 5.0
    # ER 컨트롤러 타입: "etd" | "em" | "both"
    er_type: str = "etd"
    # 고정 Q-learning 파라미터
    state_variant: str = DEFAULT_STATE_VARIANT
    alpha: float = 0.06
    gamma_q: float = 0.88
    eps_min: float = 0.018
    eps_decay: float = 0.9995
    reward_variant: str = "base"
    # ── 스윕 모드: "single" | "all4" | "grid2d" ──────────────────────────
    sweep_mode: str = "single"
    # 포인트당 반복 횟수 (seed 다르게 실행 후 평균)
    n_repeats: int = 1
    # single / grid2d: 첫 번째(주) 파라미터
    sweep_param: str = "psi"
    sweep_n_steps: int = 10
    sweep_log_scale: bool = False
    sweep_min: float = 0.05
    sweep_max: float = 2.0
    # grid2d: 두 번째 파라미터
    sweep_param_2: str = "E0"
    sweep_n_steps_2: int = 8
    sweep_log_scale_2: bool = False
    sweep_min_2: float = 2.0
    sweep_max_2: float = 30.0
    # all4: 파라미터별 스윕 범위
    psi_min: float = 0.05;  psi_max: float = 2.0;  psi_steps: int = 10;  psi_log: bool = False
    E0_min:  float = 2.0;   E0_max:  float = 30.0; E0_steps:  int = 10;  E0_log:  bool = False
    W_min:   float = 5.0;   W_max:   float = 80.0; W_steps:   int = 10;  W_log:   bool = False
    mu_min:  float = 0.0;   mu_max:  float = 1.0;  mu_steps:  int = 10;  mu_log:  bool = False
    # 고정 ER 파라미터 (스윕되지 않는 것)
    fixed_psi: float = 0.5
    fixed_E0: float = 10.0
    fixed_W: int = 20
    fixed_mu: float = 0.5
    # 커스텀 보상 함수 (use_custom_reward=True이면 reward_variant 무시)
    use_custom_reward: bool = False
    r_success_base: float = 1.0
    r_fail_abs: float = 1.0    # 절댓값; 내부에서 음수로 적용
    r_idle_pkt: float = 0.0
    output_dir: str = os.path.join("outputs", "er_sweep")


# ── 내부 유틸 ─────────────────────────────────────────────────────────────

def _gen_values(lo: float, hi: float, n: int, log_scale: bool, is_int: bool = False) -> list:
    n = max(2, n)
    if log_scale:
        arr = np.logspace(np.log10(max(lo, 1e-9)), np.log10(hi), n)
    else:
        arr = np.linspace(lo, hi, n)
    vals = arr.tolist()
    if is_int:
        vals = sorted({int(round(v)) for v in vals})
    return vals


def _run_point(
    config: ERSweepConfig,
    psi: float, E0: float, W: int, mu: float,
    seed_base: int,
) -> dict:
    """n_repeats번 실행 후 메트릭 평균 (+ std) 반환."""
    er_mode = "EM" if config.er_type == "em" else "ETD"
    keys = ("success_rate", "throughput", "fairness", "fairness_asr", "fairness_thr", "collision_rate", "mean_backlog_per_node")
    accum: dict[str, list] = {k: [] for k in keys}

    for rep in range(max(1, config.n_repeats)):
        scenario = ScenarioConfig(
            n_nodes=config.n_nodes,
            target_g=config.target_g,
            n_slots=config.n_slots,
            warmup_slots=config.warmup_slots,
            epoch_slots=config.epoch_slots,
            n_channels=config.n_channels,
            profile_name=config.profile_name,
            seed=seed_base + rep * 31,
            layout_seed=config.seed,
            queue_mode=config.queue_mode,
            gw_obs_mode=config.gw_obs_mode,
            enable_rayleigh_fading=config.enable_rayleigh_fading,
            rayleigh_fade_margin_db=config.rayleigh_fade_margin_db,
            enable_rician_fading=config.enable_rician_fading,
            rician_k_factor=config.rician_k_factor,
            rician_fade_margin_db=config.rician_fade_margin_db,
            layout=config.layout,
        )
        if config.use_custom_reward:
            _reward_params = {
                "type": "composite",
                "success_base": config.r_success_base,
                "fail": -abs(config.r_fail_abs),
                "idle_pkt": config.r_idle_pkt,
                "idle_no_pkt": 0.0,
                "retry_coef": 0.0,
                "switch_pen": 0.0,
            }
            _reward_variant = "base"
        else:
            _reward_params = None
            _reward_variant = config.reward_variant
        ctrl = DecentralizedQLearningController(
            alpha=config.alpha,
            gamma_q=config.gamma_q,
            epsilon=1.0,
            eps_min=config.eps_min,
            eps_decay=config.eps_decay,
            reward_variant=_reward_variant,
            reward_params=_reward_params,
            state_variant=config.state_variant,
            psi=float(psi),
            E0=float(E0),
            W=int(W),
            er_mode=er_mode,
            mu=float(mu),
        )
        res = run_simulation(scenario, ctrl)
        for k in keys:
            accum[k].append(float(res.get(k, 0.0)))

    out: dict = {}
    for k, vals in accum.items():
        out[k] = float(np.mean(vals))
        if len(vals) > 1:
            out[k + "_std"] = float(np.std(vals, ddof=1))
    return out


def count_total_sims(config: ERSweepConfig) -> int:
    n_rep = max(1, config.n_repeats)
    if config.er_type == "both":
        if config.sweep_mode == "single":
            n = len(_gen_values(config.sweep_min, config.sweep_max, config.sweep_n_steps,
                                config.sweep_log_scale, config.sweep_param == "W"))
            return 2 * n * n_rep
        elif config.sweep_mode == "all4":
            total = 0
            for p in ["psi", "E0", "W", "psi", "E0", "W", "mu"]:  # ETD×3 + EM×4
                lo, hi = getattr(config, f"{p}_min"), getattr(config, f"{p}_max")
                steps, log = getattr(config, f"{p}_steps"), getattr(config, f"{p}_log")
                total += len(_gen_values(lo, hi, steps, log, p == "W"))
            return total * n_rep
        else:  # grid2d
            n1 = len(_gen_values(config.sweep_min,   config.sweep_max,   config.sweep_n_steps,
                                 config.sweep_log_scale,   config.sweep_param   == "W"))
            n2 = len(_gen_values(config.sweep_min_2, config.sweep_max_2, config.sweep_n_steps_2,
                                 config.sweep_log_scale_2, config.sweep_param_2 == "W"))
            return 2 * n1 * n2 * n_rep

    if config.sweep_mode == "single":
        n = len(_gen_values(config.sweep_min, config.sweep_max, config.sweep_n_steps,
                            config.sweep_log_scale, config.sweep_param == "W"))
        return n * n_rep
    elif config.sweep_mode == "all4":
        params = _all4_params(config)
        total = 0
        for p in params:
            lo, hi = getattr(config, f"{p}_min"), getattr(config, f"{p}_max")
            steps, log = getattr(config, f"{p}_steps"), getattr(config, f"{p}_log")
            total += len(_gen_values(lo, hi, steps, log, p == "W"))
        return total * n_rep
    else:  # grid2d
        n1 = len(_gen_values(config.sweep_min,   config.sweep_max,   config.sweep_n_steps,
                             config.sweep_log_scale,   config.sweep_param   == "W"))
        n2 = len(_gen_values(config.sweep_min_2, config.sweep_max_2, config.sweep_n_steps_2,
                             config.sweep_log_scale_2, config.sweep_param_2 == "W"))
        return n1 * n2 * n_rep


def _all4_params(config: ERSweepConfig) -> list[str]:
    return ["psi", "E0", "W", "mu"] if config.er_type == "em" else ["psi", "E0", "W"]


# ── 단일 1D 스윕 (내부용) ──────────────────────────────────────────────────

def _build_sweep_args(config: ERSweepConfig, sweep_vals: list, seed_offset: int):
    """스윕 포인트별 (psi, E0, W, mu, seed_base) 인자 리스트 생성."""
    args = []
    for i, val in enumerate(sweep_vals):
        psi = val             if config.sweep_param == "psi" else config.fixed_psi
        E0  = val             if config.sweep_param == "E0"  else config.fixed_E0
        W   = int(round(val)) if config.sweep_param == "W"   else config.fixed_W
        mu  = val             if config.sweep_param == "mu"  else config.fixed_mu
        seed_base = config.seed + seed_offset + i * 1009
        args.append((psi, E0, W, mu, seed_base))
    return args


def _run_single_sweep(
    config: ERSweepConfig,
    seed_offset: int,
    advance_cb: Callable[[float], None],
) -> tuple[list[dict], list]:
    is_int = config.sweep_param == "W"
    sweep_vals = _gen_values(config.sweep_min, config.sweep_max,
                             config.sweep_n_steps, config.sweep_log_scale, is_int)
    point_args = _build_sweep_args(config, sweep_vals, seed_offset)
    rows: list[dict] = []

    for i, (val, args) in enumerate(zip(sweep_vals, point_args)):
        psi, E0, W, mu, seed_base = args
        metrics = _run_point(config, psi, E0, W, mu, seed_base)
        row = {"step": i + 1, "sweep_param": config.sweep_param,
               "sweep_value": val, "psi": psi, "E0": E0, "W": W, "mu": mu}
        row.update(metrics)
        rows.append(row)
        advance_cb(metrics["success_rate"])

    return rows, sweep_vals


# ── 플롯 함수 ─────────────────────────────────────────────────────────────

def _plot_single(sweep_vals: list, rows: list[dict], config: ERSweepConfig,
                 out_path: str, sweep_param: str | None = None) -> None:
    """2×2 라인 플롯. n_repeats>1이면 에러바 표시."""
    sp = sweep_param or config.sweep_param
    param_label = SWEEP_PARAM_LABELS.get(sp, sp)
    er_label = "ER-EM" if config.er_type == "em" else "ER-ETD"

    fixed_parts = [
        f"{k}={v}" for k, v in [("psi", config.fixed_psi), ("E0", config.fixed_E0),
                                  ("W", config.fixed_W), ("mu", config.fixed_mu)]
        if k != sp and not (k == "mu" and config.er_type == "etd")
    ]
    scale_tag = " [log]" if (sweep_param == sp and config.sweep_log_scale) or \
                            (sweep_param is None and config.sweep_log_scale) else ""
    title = (
        f"{er_label}  sweep: {sp}{scale_tag}  [{min(sweep_vals):.4g}, {max(sweep_vals):.4g}]"
        f"  n={len(sweep_vals)} × {config.n_repeats}rep\n"
        f"fixed: {'  '.join(fixed_parts)}   |   "
        f"N={config.n_nodes}  G={config.target_g}  ch={config.n_channels}  "
        f"slots={config.n_slots}"
    )
    xs = [r["sweep_value"] for r in rows]
    use_log_x = config.sweep_log_scale and sweep_param is None or \
                (sweep_param == config.sweep_param and config.sweep_log_scale)

    # detect log_x properly
    if sweep_param is None:
        use_log_x = config.sweep_log_scale
    else:
        use_log_x = False  # all4 sub-calls pass sweep_param explicitly; log handled per sub

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(title, fontsize=10, fontweight="bold")

    _higher_better = ("success_rate", "fairness", "fairness_asr", "fairness_thr", "throughput")
    _ylim_01 = ("success_rate", "fairness", "fairness_asr", "fairness_thr", "collision_rate")

    for ax, (col, label, color) in zip(axes.flat, METRICS):
        ys = [r[col] for r in rows]
        yerr_key = col + "_std"
        yerr = [r.get(yerr_key, 0.0) for r in rows] if config.n_repeats > 1 else None

        if yerr and any(e > 0 for e in yerr):
            ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color=color,
                        linewidth=1.8, markersize=5, capsize=3, zorder=3)
        else:
            ax.plot(xs, ys, "o-", color=color, linewidth=1.8, markersize=5, zorder=3)

        best_idx = int(np.argmax(ys)) if col in _higher_better else int(np.argmin(ys))
        bx, by = xs[best_idx], ys[best_idx]
        ax.axvline(bx, color=color, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.annotate(f"{bx:.3g}", xy=(bx, by), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color=color)

        ax.set_xlabel(param_label, fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25)
        if col in _ylim_01:
            ax.set_ylim(-0.02, 1.05)
        if use_log_x:
            ax.set_xscale("log")

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_grid2d(
    grid: np.ndarray,      # shape (n2, n1, 4) — metric index last
    vals1: list, vals2: list,
    config: ERSweepConfig,
    out_path: str,
) -> None:
    """2D 히트맵 (4 메트릭 2×2)."""
    er_label = "ER-EM" if config.er_type == "em" else "ER-ETD"
    p1, p2 = config.sweep_param, config.sweep_param_2
    title = (
        f"{er_label}  grid: {p1} × {p2}  |  "
        f"N={config.n_nodes}  G={config.target_g}  "
        f"slots={config.n_slots}  ×{config.n_repeats}rep"
    )
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    cmaps = ["Blues", "Greens", "Oranges", "Purples", "YlOrBr", "Reds"]
    _higher_better = ("success_rate", "fairness", "fairness_asr", "fairness_thr", "throughput")
    _ylim_01 = ("success_rate", "fairness", "fairness_asr", "fairness_thr", "collision_rate")
    for idx, (ax, (col, label, _), cmap) in enumerate(zip(axes.flat, METRICS, cmaps)):
        data = grid[:, :, idx]  # shape (n2, n1)
        vmin, vmax = data.min(), data.max()
        if col in _ylim_01:
            vmin, vmax = 0.0, 1.0

        im = ax.pcolormesh(
            vals1, vals2, data,
            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # mark best cell
        if col in _higher_better:
            best_flat = int(np.argmax(data))
        else:
            best_flat = int(np.argmin(data))
        bi, bj = np.unravel_index(best_flat, data.shape)
        ax.plot(vals1[bj], vals2[bi], "w*", markersize=10, zorder=5,
                label=f"best={data[bi, bj]:.3f}")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)

        ax.set_xlabel(SWEEP_PARAM_LABELS.get(p1, p1), fontsize=9)
        ax.set_ylabel(SWEEP_PARAM_LABELS.get(p2, p2), fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        if config.sweep_log_scale:
            ax.set_xscale("log")
        if config.sweep_log_scale_2:
            ax.set_yscale("log")

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_all4_summary(param_results: list[tuple], config: ERSweepConfig, out_path: str) -> None:
    """all4 모드: 4개 파라미터의 ASR 곡선을 한 figure에 4패널로."""
    n_params = len(param_results)
    ncols = min(n_params, 4)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    er_label = "ER-EM" if config.er_type == "em" else "ER-ETD"
    fig.suptitle(
        f"{er_label} — All-4 Sweep Summary  |  ASR  |  "
        f"N={config.n_nodes}  G={config.target_g}  ×{config.n_repeats}rep",
        fontsize=11, fontweight="bold",
    )
    colors = {"psi": "#1f77b4", "E0": "#ff7f0e", "W": "#2ca02c", "mu": "#d62728"}
    for ax, (param, rows, vals) in zip(axes, param_results):
        col_c = colors.get(param, "#888888")
        xs = [r["sweep_value"] for r in rows]
        ys = [r["success_rate"] for r in rows]
        yerr = [r.get("success_rate_std", 0.0) for r in rows]
        if config.n_repeats > 1 and any(e > 0 for e in yerr):
            ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color=col_c,
                        linewidth=1.8, markersize=5, capsize=3)
        else:
            ax.plot(xs, ys, "o-", color=col_c, linewidth=1.8, markersize=5)
        ax.set_xlabel(SWEEP_PARAM_LABELS.get(param, param), fontsize=9)
        ax.set_ylabel("ASR", fontsize=9)
        ax.set_title(f"sweep {param}", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
        p_log = getattr(config, f"{param}_log", False)
        if p_log:
            ax.set_xscale("log")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_single_compare(
    sweep_vals: list, rows_etd: list[dict], rows_em: list[dict],
    config: ERSweepConfig, out_path: str, sweep_param: str | None = None,
) -> None:
    """ETD와 EM을 같은 패널에 겹쳐서 비교하는 라인 플롯."""
    sp = sweep_param or config.sweep_param
    param_label = SWEEP_PARAM_LABELS.get(sp, sp)

    fixed_parts = [
        f"{k}={v}" for k, v in [("psi", config.fixed_psi), ("E0", config.fixed_E0),
                                  ("W", config.fixed_W), ("mu", config.fixed_mu)]
        if k != sp
    ]
    title = (
        f"ER-ETD vs ER-EM  sweep: {sp}  [{min(sweep_vals):.4g}, {max(sweep_vals):.4g}]"
        f"  n={len(sweep_vals)} × {config.n_repeats}rep\n"
        f"fixed: {'  '.join(fixed_parts)}   |   "
        f"N={config.n_nodes}  G={config.target_g}  ch={config.n_channels}  slots={config.n_slots}"
    )
    xs = [r["sweep_value"] for r in rows_etd]

    _higher_better = ("success_rate", "fairness", "fairness_asr", "fairness_thr", "throughput")
    _ylim_01 = ("success_rate", "fairness", "fairness_asr", "fairness_thr", "collision_rate")

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(title, fontsize=10, fontweight="bold")

    for ax, (col, label, color) in zip(axes.flat, METRICS):
        ys_etd = [r[col] for r in rows_etd]
        ys_em  = [r[col] for r in rows_em]
        ax.plot(xs, ys_etd, "o-",  color=color, linewidth=1.8, markersize=5,
                label="ETD", zorder=3)
        ax.plot(xs, ys_em,  "s--", color=color, linewidth=1.6, markersize=5,
                label="EM", alpha=0.65, zorder=3)
        ax.set_xlabel(param_label, fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, framealpha=0.7)
        if col in _ylim_01:
            ax.set_ylim(-0.02, 1.05)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_all4_summary_compare(
    param_results_etd: list[tuple], param_results_em: list[tuple],
    config: ERSweepConfig, out_path: str,
) -> None:
    """all4 both 모드: ETD vs EM ASR 곡선 비교 (파라미터별)."""
    etd_dict = {p: (rows, vals) for p, rows, vals in param_results_etd}
    em_dict  = {p: (rows, vals) for p, rows, vals in param_results_em}
    all_params = ["psi", "E0", "W", "mu"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(
        f"ER-ETD vs ER-EM — All-4 Sweep Summary  |  ASR  |  "
        f"N={config.n_nodes}  G={config.target_g}  ×{config.n_repeats}rep",
        fontsize=11, fontweight="bold",
    )
    colors = {"psi": "#1f77b4", "E0": "#ff7f0e", "W": "#2ca02c", "mu": "#d62728"}

    for ax, param in zip(axes, all_params):
        col_c = colors.get(param, "#888888")
        p_log = getattr(config, f"{param}_log", False)

        if param in etd_dict:
            rows_e, _ = etd_dict[param]
            xs_e = [r["sweep_value"] for r in rows_e]
            ys_e = [r["success_rate"] for r in rows_e]
            ax.plot(xs_e, ys_e, "o-", color=col_c, linewidth=1.8, markersize=5, label="ETD")

        if param in em_dict:
            rows_m, _ = em_dict[param]
            xs_m = [r["sweep_value"] for r in rows_m]
            ys_m = [r["success_rate"] for r in rows_m]
            ax.plot(xs_m, ys_m, "s--", color=col_c, linewidth=1.6, markersize=5,
                    alpha=0.65, label="EM")

        ax.set_xlabel(SWEEP_PARAM_LABELS.get(param, param), fontsize=9)
        ax.set_ylabel("ASR", fontsize=9)
        ax.set_title(f"sweep {param}", fontsize=10, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, framealpha=0.7)
        if p_log:
            ax.set_xscale("log")

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 공개 진입점 ───────────────────────────────────────────────────────────

def run_er_sweep(
    config: ERSweepConfig | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> dict:
    """
    Returns dict with keys:
      image_path, extra_images, sweep_rows, output_dir
    """
    config = config or ERSweepConfig()
    configure_fonts()
    out_dir = os.path.abspath(config.output_dir)
    ensure_dir(out_dir)

    total_sims = count_total_sims(config)
    _sim_counter = [0]

    def _advance(asr: float) -> None:
        _sim_counter[0] += max(1, config.n_repeats)
        if progress_callback:
            progress_callback(min(_sim_counter[0], total_sims), total_sims, asr)

    # ── single ──────────────────────────────────────────────────────────
    if config.sweep_mode == "single":
        if config.er_type == "both":
            cfg_etd = dataclasses.replace(config, er_type="etd")
            cfg_em  = dataclasses.replace(config, er_type="em")
            rows_etd, vals = _run_single_sweep(cfg_etd, seed_offset=0,       advance_cb=_advance)
            rows_em,  _    = _run_single_sweep(cfg_em,  seed_offset=500_003, advance_cb=_advance)
            write_csv_rows(os.path.join(out_dir, "er_sweep_etd.csv"), rows_etd)
            write_csv_rows(os.path.join(out_dir, "er_sweep_em.csv"),  rows_em)
            write_json(os.path.join(out_dir, "metadata.json"), {"config": dataclasses.asdict(config)})
            out_img = os.path.join(out_dir, "er_sweep_compare.png")
            _plot_single_compare(vals, rows_etd, rows_em, config, out_img)
            return {"image_path": out_img, "extra_images": [],
                    "sweep_rows": rows_etd + rows_em, "output_dir": out_dir}

        rows, vals = _run_single_sweep(config, seed_offset=0, advance_cb=_advance)
        write_csv_rows(os.path.join(out_dir, "er_sweep.csv"), rows)
        write_json(os.path.join(out_dir, "metadata.json"), {"config": dataclasses.asdict(config)})
        out_img = os.path.join(out_dir, "er_sweep.png")
        _plot_single(vals, rows, config, out_img)
        return {"image_path": out_img, "extra_images": [], "sweep_rows": rows, "output_dir": out_dir}

    # ── all4 ────────────────────────────────────────────────────────────
    elif config.sweep_mode == "all4":
        if config.er_type == "both":
            cfg_etd = dataclasses.replace(config, er_type="etd")
            cfg_em  = dataclasses.replace(config, er_type="em")
            param_results_etd: list[tuple] = []
            param_results_em:  list[tuple] = []
            all_rows: list[dict] = []
            fig_paths: list[str] = []

            for p_idx, param in enumerate(["psi", "E0", "W"]):
                lo, hi = getattr(config, f"{param}_min"), getattr(config, f"{param}_max")
                steps, log = getattr(config, f"{param}_steps"), getattr(config, f"{param}_log")
                sub = dataclasses.replace(cfg_etd, sweep_param=param,
                                          sweep_min=float(lo), sweep_max=float(hi),
                                          sweep_n_steps=int(steps), sweep_log_scale=bool(log))
                rows, vals = _run_single_sweep(sub, seed_offset=p_idx * 100_003, advance_cb=_advance)
                param_results_etd.append((param, rows, vals))
                all_rows.extend(rows)

            for p_idx, param in enumerate(["psi", "E0", "W", "mu"]):
                lo, hi = getattr(config, f"{param}_min"), getattr(config, f"{param}_max")
                steps, log = getattr(config, f"{param}_steps"), getattr(config, f"{param}_log")
                sub = dataclasses.replace(cfg_em, sweep_param=param,
                                          sweep_min=float(lo), sweep_max=float(hi),
                                          sweep_n_steps=int(steps), sweep_log_scale=bool(log))
                rows, vals = _run_single_sweep(sub, seed_offset=400_003 + p_idx * 100_003, advance_cb=_advance)
                param_results_em.append((param, rows, vals))
                all_rows.extend(rows)

            etd_dict = {p: (r, v) for p, r, v in param_results_etd}
            em_dict  = {p: (r, v) for p, r, v in param_results_em}

            for param in ["psi", "E0", "W"]:
                rows_e, vals_e = etd_dict[param]
                rows_m, _      = em_dict[param]
                sub_cfg = dataclasses.replace(config, sweep_param=param)
                img = os.path.join(out_dir, f"er_sweep_{param}_compare.png")
                _plot_single_compare(vals_e, rows_e, rows_m, sub_cfg, img, sweep_param=param)
                fig_paths.append(img)

            rows_mu, vals_mu = em_dict["mu"]
            sub_mu = dataclasses.replace(cfg_em, sweep_param="mu",
                                          sweep_min=config.mu_min, sweep_max=config.mu_max,
                                          sweep_n_steps=config.mu_steps, sweep_log_scale=config.mu_log)
            img_mu = os.path.join(out_dir, "er_sweep_mu_em.png")
            _plot_single(vals_mu, rows_mu, sub_mu, img_mu, sweep_param="mu")
            fig_paths.append(img_mu)

            summary_path = os.path.join(out_dir, "er_sweep_all4_compare_summary.png")
            _plot_all4_summary_compare(param_results_etd, param_results_em, config, summary_path)
            fig_paths.insert(0, summary_path)

            write_csv_rows(os.path.join(out_dir, "er_sweep_all4_both.csv"), all_rows)
            write_json(os.path.join(out_dir, "metadata.json"), {"config": dataclasses.asdict(config)})
            return {"image_path": fig_paths[0], "extra_images": fig_paths[1:],
                    "sweep_rows": all_rows, "output_dir": out_dir}

        params = _all4_params(config)
        param_results: list[tuple] = []
        all_rows: list[dict] = []
        fig_paths: list[str] = []

        for p_idx, param in enumerate(params):
            lo    = getattr(config, f"{param}_min")
            hi    = getattr(config, f"{param}_max")
            steps = getattr(config, f"{param}_steps")
            log   = getattr(config, f"{param}_log")
            sub = dataclasses.replace(
                config,
                sweep_param=param,
                sweep_min=float(lo), sweep_max=float(hi),
                sweep_n_steps=int(steps), sweep_log_scale=bool(log),
            )
            rows, vals = _run_single_sweep(
                sub, seed_offset=p_idx * 100_003, advance_cb=_advance
            )
            param_results.append((param, rows, vals))
            all_rows.extend(rows)

            img_path = os.path.join(out_dir, f"er_sweep_{param}.png")
            _plot_single(vals, rows, sub, img_path, sweep_param=param)
            fig_paths.append(img_path)

        # summary figure (ASR overview)
        summary_path = os.path.join(out_dir, "er_sweep_all4_summary.png")
        _plot_all4_summary(param_results, config, summary_path)
        fig_paths.insert(0, summary_path)

        write_csv_rows(os.path.join(out_dir, "er_sweep_all4.csv"), all_rows)
        write_json(os.path.join(out_dir, "metadata.json"), {"config": dataclasses.asdict(config)})
        return {
            "image_path": fig_paths[0],
            "extra_images": fig_paths[1:],
            "sweep_rows": all_rows,
            "output_dir": out_dir,
        }

    # ── grid2d ──────────────────────────────────────────────────────────
    else:
        if config.er_type == "both":
            results = {}
            for tag, er_t, seed_off in [("etd", "etd", 0), ("em", "em", 700_003)]:
                cfg_t = dataclasses.replace(config, er_type=er_t)
                p1, p2 = cfg_t.sweep_param, cfg_t.sweep_param_2
                vals1 = _gen_values(cfg_t.sweep_min,   cfg_t.sweep_max,   cfg_t.sweep_n_steps,
                                    cfg_t.sweep_log_scale,   p1 == "W")
                vals2 = _gen_values(cfg_t.sweep_min_2, cfg_t.sweep_max_2, cfg_t.sweep_n_steps_2,
                                    cfg_t.sweep_log_scale_2, p2 == "W")
                n1, n2 = len(vals1), len(vals2)
                grid = np.zeros((n2, n1, len(METRICS)), dtype=float)
                all_rows_t: list[dict] = []
                global_pt = 0
                for j, v2 in enumerate(vals2):
                    for i, v1 in enumerate(vals1):
                        psi = v1 if p1 == "psi" else (v2 if p2 == "psi" else cfg_t.fixed_psi)
                        E0  = v1 if p1 == "E0"  else (v2 if p2 == "E0"  else cfg_t.fixed_E0)
                        W   = int(round(v1 if p1 == "W" else (v2 if p2 == "W" else cfg_t.fixed_W)))
                        mu  = v1 if p1 == "mu"  else (v2 if p2 == "mu"  else cfg_t.fixed_mu)
                        metrics = _run_point(cfg_t, psi, E0, W, mu, cfg_t.seed + seed_off + global_pt * 1009)
                        global_pt += 1
                        _advance(metrics["success_rate"])
                        for m_idx, (col, _, _) in enumerate(METRICS):
                            grid[j, i, m_idx] = metrics[col]
                        row = {"tag": tag, "step": global_pt, "sweep_param": p1, "sweep_value": v1,
                               "sweep_param_2": p2, "sweep_value_2": v2,
                               "psi": psi, "E0": E0, "W": W, "mu": mu}
                        row.update(metrics)
                        all_rows_t.append(row)
                img_t = os.path.join(out_dir, f"er_sweep_grid2d_{tag}.png")
                _plot_grid2d(grid, vals1, vals2, cfg_t, img_t)
                write_csv_rows(os.path.join(out_dir, f"er_sweep_grid2d_{tag}.csv"), all_rows_t)
                results[tag] = img_t

            write_json(os.path.join(out_dir, "metadata.json"), {"config": dataclasses.asdict(config)})
            return {"image_path": results["etd"], "extra_images": [results["em"]],
                    "sweep_rows": [], "output_dir": out_dir}

        p1, p2 = config.sweep_param, config.sweep_param_2
        vals1 = _gen_values(config.sweep_min,   config.sweep_max,   config.sweep_n_steps,
                            config.sweep_log_scale,   p1 == "W")
        vals2 = _gen_values(config.sweep_min_2, config.sweep_max_2, config.sweep_n_steps_2,
                            config.sweep_log_scale_2, p2 == "W")
        n1, n2 = len(vals1), len(vals2)

        # grid[j, i, metric_idx]  — j=dim2, i=dim1
        grid = np.zeros((n2, n1, len(METRICS)), dtype=float)
        all_rows: list[dict] = []
        global_pt = 0

        for j, v2 in enumerate(vals2):
            for i, v1 in enumerate(vals1):
                psi = v1 if p1 == "psi" else (v2 if p2 == "psi" else config.fixed_psi)
                E0  = v1 if p1 == "E0"  else (v2 if p2 == "E0"  else config.fixed_E0)
                W   = int(round(v1 if p1 == "W" else (v2 if p2 == "W" else config.fixed_W)))
                mu  = v1 if p1 == "mu"  else (v2 if p2 == "mu"  else config.fixed_mu)

                seed_base = config.seed + global_pt * 1009
                metrics = _run_point(config, psi, E0, W, mu, seed_base)
                global_pt += 1
                _advance(metrics["success_rate"])

                for m_idx, (col, _, _) in enumerate(METRICS):
                    grid[j, i, m_idx] = metrics[col]

                row = {"step": global_pt, "sweep_param": p1, "sweep_value": v1,
                       "sweep_param_2": p2, "sweep_value_2": v2,
                       "psi": psi, "E0": E0, "W": W, "mu": mu}
                row.update(metrics)
                all_rows.append(row)

        out_img = os.path.join(out_dir, "er_sweep_grid2d.png")
        _plot_grid2d(grid, vals1, vals2, config, out_img)
        write_csv_rows(os.path.join(out_dir, "er_sweep_grid2d.csv"), all_rows)
        write_json(os.path.join(out_dir, "metadata.json"), {"config": dataclasses.asdict(config)})
        return {
            "image_path": out_img,
            "extra_images": [],
            "sweep_rows": all_rows,
            "output_dir": out_dir,
        }


if __name__ == "__main__":
    result = run_er_sweep()
    print(f"Saved to: {result['output_dir']}")
