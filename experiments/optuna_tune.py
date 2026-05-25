"""Optuna 기반 Q-learning 하이퍼파라미터 + 보상 함수 자동 탐색.

지원 컨트롤러
-------------
q_learning  — 기본 Q-learning (psi=0)
er_etd      — ER-ETD (psi, E0, W 추가 탐색)
er_em       — ER-EM  (psi, E0, W, mu 추가 탐색)
lr_rl       — LR-RL (Hong et al. 2023) V-table 베이스라인
kaburaki    — Kaburaki (2021) 타이밍 오프셋 Q-learning

병렬 실행
---------
parallel_studies=True 이면 ThreadPoolExecutor로 여러 컨트롤러 연구를 동시에 돌린다.
numpy가 GIL을 해제하므로 멀티스레드에서도 실제 속도 향상이 있다.

n_jobs > 1 이면 Optuna 내부에서 한 연구 안의 trial들도 병렬 실행한다.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _worker_init(project_root: str) -> None:
    """ProcessPoolExecutor worker 초기화: 프로젝트 루트를 sys.path에 등록."""
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from agents.q_learning import (
    DEFAULT_STATE_VARIANT,
    REWARD_VARIANTS,
    DecentralizedQLearningController,
)
from baselines.kaburaki import make_kaburaki_controller
from baselines.lr_rl import make_lr_rl_controller
from env import ScenarioConfig, run_simulation
from utils.io import ensure_dir, write_csv_rows, write_json

# ── 지원 컨트롤러 정의 ────────────────────────────────────────────────────
TUNABLE_CONTROLLERS: dict[str, dict] = {
    "q_learning": {
        "label": "Q-learning (no ER)",
        "er_mode": "ETD",
        "psi": 0.0,
        "search_er": False,
    },
    "er_etd": {
        "label": "ER-ETD",
        "er_mode": "ETD",
        "search_er": True,
    },
    "er_em": {
        "label": "ER-EM",
        "er_mode": "EM",
        "search_er": True,
        "search_mu": True,
    },
    "lr_rl": {
        "label": "LR-RL (Hong 2023)",
        "search_er": False,
    },
    "kaburaki": {
        "label": "Kaburaki (2021)",
        "search_er": False,
    },
}

OBJECTIVE_METRICS: dict[str, str] = {
    "asr_x_fairness":       "ASR × Fairness(S)",
    "asr_x_fairness_thr":   "ASR × Fairness(Thr)",
    "asr_thr_fairness":     "ASR × Thr_norm × Fairness(S)",
    "asr_thr_fairness_thr": "ASR × Thr_norm × Fairness(Thr)",
    "weighted_sum":         "Weighted Sum (w·ASR + w·Thr_norm + w·Fairness(Thr))",
    "throughput":           "Throughput (normalized)",
    "success_rate":         "ASR",
    "fairness":             "Fairness(S) — Jain on success counts",
    "fairness_thr":         "Fairness(Thr) — Jain on per-node throughput",
    "thr_x_fairness":       "Thr_norm × Fairness(S)",
    "thr_x_fairness_thr":   "Thr_norm × Fairness(Thr)",
}


@dataclasses.dataclass(frozen=True)
class OptunaConfig:
    n_nodes: int = 60
    target_g: float = 2.0
    n_slots: int = 15_000
    warmup_slots: int = 3_000
    epoch_slots: int = 500
    n_channels: int = 3
    profile_name: str = "short"
    seed: int = 42
    queue_mode: str = "accumulate"
    gw_obs_mode: str = "attempt"
    enable_rayleigh_fading: bool = False
    rayleigh_fade_margin_db: float = 10.0
    enable_rician_fading: bool = False
    rician_k_factor: float = 4.0
    rician_fade_margin_db: float = 5.0
    layout: str = "random"
    state_variant: str = DEFAULT_STATE_VARIANT
    # 탐색할 컨트롤러 목록
    controller_keys: tuple[str, ...] = ("q_learning",)
    # 탐색 설정
    n_trials: int = 500
    n_jobs: int = 1          # 연구 내 병렬 trial 수 (1=직렬, -1=코어 전체)
    parallel_studies: bool = False  # True: 여러 컨트롤러 연구를 동시에 실행
    objective_metric: str = "asr_x_fairness"
    search_reward_variant: bool = True
    # Q-learning 탐색 범위
    alpha_low: float = 0.01
    alpha_high: float = 0.5
    gamma_low: float = 0.5
    gamma_high: float = 0.99
    eps_min_low: float = 0.01
    eps_min_high: float = 0.2
    eps_decay_low: float = 0.9990
    eps_decay_high: float = 0.99999
    # ER 탐색 범위
    psi_low: float = 0.05
    psi_high: float = 2.0
    E0_low: float = 2.0
    E0_high: float = 30.0
    W_low: int = 5
    W_high: int = 50
    mu_low: float = 0.1
    mu_high: float = 0.9
    fix_mu: bool = False
    fixed_mu: float = 0.5
    # Kaburaki 탐색 범위
    kab_J_low: int = 1
    kab_J_high: int = 10
    kab_D_max_low: int = 10
    kab_D_max_high: int = 200
    # weighted_sum 가중치 (합이 1이 되도록 정규화해서 사용)
    w_asr: float = 1.0
    w_thr: float = 1.0
    w_fairness: float = 1.0
    # parametric reward search: reward_variant 범주 선택 대신 계수를 직접 탐색
    search_parametric_reward: bool = False
    r_success_base_low: float = 0.1
    r_success_base_high: float = 1.5
    r_success_fair_low: float = 0.0
    r_success_fair_high: float = 1.5
    r_fail_abs_low: float = 0.1
    r_fail_abs_high: float = 2.0
    r_idle_pkt_low: float = -0.1
    r_idle_pkt_high: float = 0.3
    # 고정 파라미터 모드 (controlled study: ER 메커니즘만 비교)
    fix_base_params: bool = False       # 전체 컨트롤러에 α/γ/ε 고정
    fix_q_learning_only: bool = False   # q_learning 컨트롤러만 고정 (ER은 자유 탐색)
    fixed_alpha: float = 0.06
    fixed_gamma_q: float = 0.88
    fixed_eps_min: float = 0.018
    fixed_eps_decay: float = 0.9995
    fix_reward: bool = False
    fixed_reward_variant: str = "base"
    # fix_reward=True 일 때 수치를 직접 고정하는 경우 (fix_reward_numeric=True)
    fix_reward_numeric: bool = False
    fixed_r_success_base: float = 1.0
    fixed_r_success_fair: float = 0.0
    fixed_r_fail_abs: float = 1.0
    fixed_r_idle_pkt: float = 0.0
    output_dir: str = os.path.join("outputs", "optuna_tune")


# ── 공통 유틸 ─────────────────────────────────────────────────────────────

def _compute_objective(result: dict, metric_key: str, config: "OptunaConfig | None" = None) -> float:
    asr      = float(result.get("success_rate",  0.0))
    thr      = float(result.get("throughput",    0.0))
    jain_s   = float(result.get("fairness",      0.0))   # Jain on success counts (active only)
    jain_thr = float(result.get("fairness_thr",  0.0))   # Jain on per-node thr   (all N nodes)

    # throughput 정규화: 이론 최대 = N_SF × N_ch (모든 자원 매 슬롯 성공) → [0, 1]
    _n_ch = float(config.n_channels) if config is not None else 3.0
    _max_thr = 6.0 * _n_ch
    thr_norm = min(thr / _max_thr, 1.0) if _max_thr > 0 else thr

    if metric_key == "throughput":
        return thr_norm
    if metric_key == "success_rate":
        return asr
    if metric_key == "fairness":
        return jain_s
    if metric_key == "fairness_thr":
        return jain_thr
    if metric_key == "thr_x_fairness":
        return thr_norm * jain_s
    if metric_key == "thr_x_fairness_thr":
        return thr_norm * jain_thr
    if metric_key == "asr_x_fairness":
        return asr * jain_s
    if metric_key == "asr_x_fairness_thr":
        return asr * jain_thr
    if metric_key == "asr_thr_fairness":
        return asr * thr_norm * jain_s
    if metric_key == "asr_thr_fairness_thr":
        return asr * thr_norm * jain_thr
    if metric_key == "weighted_sum":
        total_w = config.w_asr + config.w_thr + config.w_fairness if config else 3.0
        total_w = total_w if total_w > 0 else 1.0
        wa = (config.w_asr      if config else 1.0) / total_w
        wt = (config.w_thr      if config else 1.0) / total_w
        wf = (config.w_fairness  if config else 1.0) / total_w
        return wa * asr + wt * thr_norm + wf * jain_thr   # fairness_thr: 전체 노드 기준
    return asr * jain_thr


def _make_objective(config: OptunaConfig, controller_key: str):
    """controller_key에 맞는 Optuna objective 클로저를 반환한다."""
    ctrl_def = TUNABLE_CONTROLLERS[controller_key]
    reward_keys = list(REWARD_VARIANTS.keys())

    def objective(trial) -> float:
        trial_seed = config.seed + trial.number * 997
        scenario = ScenarioConfig(
            n_nodes=config.n_nodes,
            target_g=config.target_g,
            n_slots=config.n_slots,
            warmup_slots=config.warmup_slots,
            epoch_slots=config.epoch_slots,
            n_channels=config.n_channels,
            profile_name=config.profile_name,
            seed=trial_seed,
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

        # ── LR-RL ────────────────────────────────────────────────────────
        if controller_key == "lr_rl":
            alpha     = trial.suggest_float("alpha",     config.alpha_low,     config.alpha_high,     log=True)
            gamma_q   = trial.suggest_float("gamma_q",   config.gamma_low,     config.gamma_high)
            eps_min   = trial.suggest_float("eps_min",   config.eps_min_low,   config.eps_min_high,   log=True)
            eps_decay = trial.suggest_float("eps_decay", config.eps_decay_low, config.eps_decay_high, log=True)
            controller = make_lr_rl_controller(
                alpha=alpha, gamma=gamma_q, eps_start=1.0, eps_min=eps_min, eps_decay=eps_decay,
            )

        # ── Kaburaki ─────────────────────────────────────────────────────
        elif controller_key == "kaburaki":
            alpha     = trial.suggest_float("alpha",     config.alpha_low,     config.alpha_high,     log=True)
            gamma_q   = trial.suggest_float("gamma_q",   config.gamma_low,     config.gamma_high)
            eps_min   = trial.suggest_float("eps_min",   config.eps_min_low,   config.eps_min_high,   log=True)
            kab_J     = trial.suggest_int("kab_J",       config.kab_J_low,     config.kab_J_high)
            kab_D_max = trial.suggest_int("kab_D_max",   config.kab_D_max_low, config.kab_D_max_high)
            controller = make_kaburaki_controller(
                J=kab_J, D_max=kab_D_max, alpha=alpha, gamma=gamma_q, eps_min=eps_min,
            )

        # ── Q-learning / ER-Q ─────────────────────────────────────────────
        else:
            _fix = config.fix_base_params or (config.fix_q_learning_only and controller_key == "q_learning")
            if _fix:
                alpha     = config.fixed_alpha
                gamma_q   = config.fixed_gamma_q
                eps_min   = config.fixed_eps_min
                eps_decay = config.fixed_eps_decay
            else:
                alpha     = trial.suggest_float("alpha",     config.alpha_low,     config.alpha_high,     log=True)
                gamma_q   = trial.suggest_float("gamma_q",   config.gamma_low,     config.gamma_high)
                eps_min   = trial.suggest_float("eps_min",   config.eps_min_low,   config.eps_min_high,   log=True)
                eps_decay = trial.suggest_float("eps_decay", config.eps_decay_low, config.eps_decay_high, log=True)

            # reward 설정: fix_reward_numeric > fix_reward(variant) > parametric > categorical > fixed
            trial_reward_params: dict | None = None
            reward_variant = "base"
            if config.fix_reward and config.fix_reward_numeric:
                trial_reward_params = {
                    "type": "composite",
                    "success_base": config.fixed_r_success_base,
                    "success_fair": config.fixed_r_success_fair,
                    "fail": -abs(config.fixed_r_fail_abs),
                    "idle_pkt": config.fixed_r_idle_pkt,
                    "idle_no_pkt": 0.0,
                    "retry_coef": 0.0,
                    "switch_pen": 0.0,
                }
            elif config.fix_reward:
                reward_variant = config.fixed_reward_variant
            elif config.search_parametric_reward:
                r_success_base = trial.suggest_float("r_success_base", config.r_success_base_low, config.r_success_base_high)
                r_success_fair = trial.suggest_float("r_success_fair", config.r_success_fair_low, config.r_success_fair_high)
                r_fail_abs     = trial.suggest_float("r_fail_abs",     config.r_fail_abs_low,     config.r_fail_abs_high)
                r_idle_pkt     = trial.suggest_float("r_idle_pkt",     config.r_idle_pkt_low,     config.r_idle_pkt_high)
                trial_reward_params = {
                    "type": "composite",
                    "success_base": r_success_base,
                    "success_fair": r_success_fair,
                    "fail": -abs(r_fail_abs),
                    "idle_pkt": r_idle_pkt,
                    "idle_no_pkt": 0.0,
                    "retry_coef": 0.0,
                    "switch_pen": 0.0,
                }
            elif config.search_reward_variant:
                reward_variant = trial.suggest_categorical("reward_variant", reward_keys)

            psi = 0.0; E0 = 10.0; W = 20; mu = 0.5
            er_mode = ctrl_def["er_mode"]

            if ctrl_def["search_er"]:
                psi = trial.suggest_float("psi", config.psi_low, config.psi_high, log=True)
                E0  = trial.suggest_float("E0",  config.E0_low,  config.E0_high)
                W   = trial.suggest_int("W",     config.W_low,   config.W_high)
            if ctrl_def.get("search_mu"):
                if config.fix_mu:
                    mu = config.fixed_mu
                else:
                    mu = trial.suggest_float("mu", config.mu_low, config.mu_high)

            controller = DecentralizedQLearningController(
                alpha=alpha,
                gamma_q=gamma_q,
                epsilon=1.0,
                eps_min=eps_min,
                eps_decay=eps_decay,
                reward_variant=reward_variant,
                reward_params=trial_reward_params,
                state_variant=config.state_variant,
                psi=psi,
                E0=E0,
                W=W,
                er_mode=er_mode,
                mu=mu,
            )

        sim_result = run_simulation(scenario, controller)
        # 세부 지표를 user_attr에 저장 → 나중에 winner 요약에 활용
        for key in ("success_rate", "fairness", "fairness_thr", "throughput", "collision_rate",
                    "mean_backlog_per_node", "final_backlog_per_node"):
            trial.set_user_attr(key, float(sim_result.get(key, 0.0)))
        return _compute_objective(sim_result, config.objective_metric, config)

    return objective


def _run_one_study(
    config: OptunaConfig,
    controller_key: str,
    progress_callback: Callable[[str, int, int, float], None] | None = None,
) -> dict:
    """단일 컨트롤러에 대한 Optuna 연구를 실행하고 결과 dict을 반환한다."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        raise ImportError("optuna가 설치되지 않았습니다: pip install optuna") from e

    ctrl_label = TUNABLE_CONTROLLERS[controller_key]["label"]
    study_dir = os.path.join(os.path.abspath(config.output_dir), controller_key)
    ensure_dir(study_dir)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        study_name=f"qlrn_{controller_key}",
    )

    trial_rows: list[dict] = []
    _METRIC_KEYS = ("success_rate", "fairness", "fairness_thr", "throughput", "collision_rate",
                    "mean_backlog_per_node", "final_backlog_per_node")
    _lock = __import__("threading").Lock()

    def _after_trial(study, trial) -> None:
        row = {"trial": trial.number, "controller": controller_key, "value": trial.value}
        row.update(trial.params)
        row.update({k: trial.user_attrs.get(k, None) for k in _METRIC_KEYS})
        with _lock:
            trial_rows.append(row)
        if progress_callback:
            progress_callback(controller_key, trial.number + 1, config.n_trials, study.best_value)

    objective = _make_objective(config, controller_key)
    study.optimize(objective, n_trials=config.n_trials, n_jobs=config.n_jobs, callbacks=[_after_trial])

    best = study.best_trial
    best_params = dict(best.params)
    best_params.update({
        "controller": controller_key,
        "label": ctrl_label,
        "objective_value": best.value,
        "objective_metric": config.objective_metric,
    })
    # best trial의 세부 지표도 best_params에 포함
    best_params.update({k: best.user_attrs.get(k, None) for k in _METRIC_KEYS})

    write_csv_rows(os.path.join(study_dir, "trials.csv"), trial_rows)
    write_json(os.path.join(study_dir, "best_params.json"), best_params)

    print(f"[{ctrl_label}] best trial #{best.number}  value={best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    fig_paths = _plot_study(study, trial_rows, study_dir, config, controller_key)
    return {
        "controller_key": controller_key,
        "label": ctrl_label,
        "best_params": best_params,
        "trial_rows": trial_rows,
        "fig_paths": fig_paths,
        "study_dir": study_dir,
    }


# ── 메인 진입점 ───────────────────────────────────────────────────────────

def run_optuna_tune(
    config: OptunaConfig | None = None,
    progress_callback: Callable[[str, int, int, float], None] | None = None,
    study_complete_callback: Callable[[str, dict], None] | None = None,
) -> dict:
    """
    config.controller_keys 에 지정된 컨트롤러들을 탐색한다.

    progress_callback(controller_key, trial_num, n_trials, best_value)
        — 순차 모드(n_jobs=1)에서 trial 완료마다 호출됨.
        — 병렬 모드에서는 worker 스레드에서 호출되므로 Streamlit UI 업데이트 불가, 무시됨.

    study_complete_callback(controller_key, result_dict)
        — 순차/병렬 공통, 각 study가 완료될 때마다 메인 스레드에서 호출됨.
        — 병렬 모드 진행 표시에 사용.

    반환 dict:
        "results": {controller_key: {...}}
        "output_dir": str
        "comparison_fig": str  (여러 컨트롤러일 때 비교 그래프 경로)
    """
    config = config or OptunaConfig()
    ensure_dir(os.path.abspath(config.output_dir))

    keys = list(config.controller_keys)

    # parallel_studies는 비활성화: ThreadPoolExecutor는 GIL로 효과 없고,
    # ProcessPoolExecutor는 Streamlit 프로세스 안에서 spawn 시 데드락 발생.
    # 속도 향상은 n_jobs > 1 (Optuna 내부 trial 병렬)로 얻는 것이 올바른 방법.
    all_results = {}
    for k in keys:
        res = _run_one_study(config, k, progress_callback)
        all_results[k] = res
        if study_complete_callback:
            study_complete_callback(k, res)

    comparison_fig = _plot_comparison(all_results, config)

    return {
        "results": all_results,
        "output_dir": os.path.abspath(config.output_dir),
        "comparison_fig": comparison_fig,
    }


# ── 플롯 ──────────────────────────────────────────────────────────────────

def _plot_study(study, trial_rows: list[dict], out_dir: str, config: OptunaConfig, ctrl_key: str) -> list[str]:
    paths: list[str] = []
    metric_label = OBJECTIVE_METRICS.get(config.objective_metric, config.objective_metric)
    ctrl_label = TUNABLE_CONTROLLERS[ctrl_key]["label"]

    # 1) 최적화 히스토리 (내장 플롯)
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
        )
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        plot_optimization_history(study, ax=ax1)
        ax1.set_title(f"Optimization History — {ctrl_label}", fontsize=11)
        p1 = os.path.join(out_dir, "optuna_history.png")
        fig1.savefig(p1, dpi=140, bbox_inches="tight")
        plt.close(fig1)
        paths.append(p1)

        if len(study.trials) >= 5:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            plot_param_importances(study, ax=ax2)
            ax2.set_title(f"Hyperparameter Importance — {ctrl_label}", fontsize=11)
            p2 = os.path.join(out_dir, "optuna_importance.png")
            fig2.savefig(p2, dpi=140, bbox_inches="tight")
            plt.close(fig2)
            paths.append(p2)
    except Exception:
        pass

    # 2) 연속 파라미터 scatter
    numeric_params = ["alpha", "gamma_q", "eps_min", "eps_decay", "psi", "E0", "W", "mu",
                      "kab_J", "kab_D_max",
                      "r_success_base", "r_success_fair", "r_fail_abs", "r_idle_pkt"]
    available = [p for p in numeric_params if any(p in r for r in trial_rows)]
    if available:
        ncols = min(len(available), 4)
        nrows = (len(available) + 3) // 4
        fig3, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes_flat = np.array(axes).flatten() if nrows * ncols > 1 else [axes]
        for ax, param in zip(axes_flat, available):
            xs = [r[param] for r in trial_rows if param in r]
            ys = [r["value"] for r in trial_rows if param in r]
            best_x = study.best_trial.params.get(param)
            ax.scatter(xs, ys, s=28, alpha=0.55, color="#0072b2")
            if best_x is not None:
                ax.scatter([best_x], [study.best_value], s=110, color="#d55e00", zorder=5, label="best")
            if param in ("alpha", "eps_min", "eps_decay", "psi"):
                ax.set_xscale("log")
            ax.set_xlabel(param, fontsize=9)
            ax.set_ylabel(metric_label, fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
        for ax in axes_flat[len(available):]:
            ax.set_visible(False)
        fig3.suptitle(f"{ctrl_label} | {metric_label} | N={config.n_nodes} G={config.target_g}", fontsize=10, fontweight="bold")
        plt.tight_layout()
        p3 = os.path.join(out_dir, "optuna_scatter.png")
        fig3.savefig(p3, dpi=140, bbox_inches="tight")
        plt.close(fig3)
        paths.append(p3)

    # 3) reward_variant 별 성능
    if config.search_reward_variant and trial_rows and "reward_variant" in trial_rows[0]:
        from collections import defaultdict
        by_variant: dict[str, list[float]] = defaultdict(list)
        for r in trial_rows:
            by_variant[r["reward_variant"]].append(r["value"])
        labels = list(by_variant.keys())
        means = [np.mean(by_variant[k]) for k in labels]
        maxes = [np.max(by_variant[k]) for k in labels]
        x_pos = list(range(len(labels)))
        fig4, ax4 = plt.subplots(figsize=(max(5, len(labels) * 1.2), 4))
        ax4.bar(x_pos, means, alpha=0.7, color="#009e73", label="mean")
        ax4.scatter(x_pos, maxes, color="#d55e00", zorder=5, s=60, label="best")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax4.set_ylabel(metric_label, fontsize=9)
        ax4.set_title(f"Reward Variant — {ctrl_label}", fontsize=10)
        ax4.legend(fontsize=8)
        ax4.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        p4 = os.path.join(out_dir, "optuna_reward_variant.png")
        fig4.savefig(p4, dpi=140, bbox_inches="tight")
        plt.close(fig4)
        paths.append(p4)

    return paths


def _plot_comparison(all_results: dict[str, dict], config: OptunaConfig) -> str:
    """
    멀티패널 종합 비교 그래프.

    Row 1: 수렴 곡선 | 멀티 지표 그룹 바 (ASR / Fairness / Throughput / Collision)
    Row 2: 파라미터 비교표 (heatmap 형식)
    """
    metric_label = OBJECTIVE_METRICS.get(config.objective_metric, config.objective_metric)
    colors = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#f0e442", "#56b4e9"]

    # 우승자: objective_value 기준
    winner_key = max(
        all_results,
        key=lambda k: all_results[k]["best_params"].get("objective_value", -1),
    )

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.45, wspace=0.35)
    ax_conv = fig.add_subplot(gs[0, 0])
    ax_multi = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    winner_label = all_results[winner_key]["label"]
    fig.suptitle(
        f"Optuna Tune — 종합 결과  |  N={config.n_nodes}  G={config.target_g}  |  {metric_label}  |  Winner: {winner_label}",
        fontsize=12, fontweight="bold",
    )

    # ── 1) 수렴 곡선 ──────────────────────────────────────────────────────
    for i, (key, res) in enumerate(all_results.items()):
        rows = res["trial_rows"]
        if not rows:
            continue
        col = colors[i % len(colors)]
        vals = [r["value"] for r in rows]
        cum_best = [max(vals[:j + 1]) for j in range(len(vals))]
        lw = 2.4 if key == winner_key else 1.6
        zorder = 3 if key == winner_key else 2
        ax_conv.plot(
            range(1, len(cum_best) + 1), cum_best,
            label=res["label"], color=col, linewidth=lw, zorder=zorder,
        )
    ax_conv.set_xlabel("Trial", fontsize=10)
    ax_conv.set_ylabel(f"Cumulative best  {metric_label}", fontsize=10)
    ax_conv.set_title("Convergence", fontsize=10)
    ax_conv.grid(True, alpha=0.25)
    ax_conv.legend(fontsize=9)

    # ── 2) 멀티 지표 그룹 바 ─────────────────────────────────────────────
    detail_metrics = [
        ("success_rate",   "ASR",           (0.0, 1.0)),
        ("fairness",       "Fair(S)",        (0.0, 1.0)),
        ("fairness_thr",   "Fair(Thr)",      (0.0, 1.0)),
        ("throughput",     "Throughput",     None),
        ("collision_rate", "Collision",      (0.0, 1.0)),
    ]
    keys_list = list(all_results.keys())
    n_ctrl = len(keys_list)
    n_metrics = len(detail_metrics)
    bar_width = 0.7 / n_ctrl
    x_base = np.arange(n_metrics)

    for i, key in enumerate(keys_list):
        bp = all_results[key]["best_params"]
        vals = [bp.get(mk, 0.0) or 0.0 for mk, *_ in detail_metrics]
        offset = (i - n_ctrl / 2 + 0.5) * bar_width
        col = colors[i % len(colors)]
        bars = ax_multi.bar(
            x_base + offset, vals, bar_width * 0.92,
            color=col, alpha=0.85,
            label=all_results[key]["label"],
            linewidth=1.4 if key == winner_key else 0,
            edgecolor="black" if key == winner_key else col,
        )
        for bar, v in zip(bars, vals):
            ax_multi.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7,
            )

    ax_multi.set_xticks(x_base)
    ax_multi.set_xticklabels([lbl for _, lbl, _ in detail_metrics], fontsize=10)
    ax_multi.set_title("Best Trial — Detailed Metrics", fontsize=10)
    ax_multi.set_ylabel("Value", fontsize=10)
    ax_multi.legend(fontsize=8, loc="upper right")
    ax_multi.grid(True, axis="y", alpha=0.25)
    ax_multi.set_ylim(bottom=0)

    # ── 3) 파라미터 비교표 ────────────────────────────────────────────────
    param_keys = ["reward_variant", "alpha", "gamma_q", "eps_min", "eps_decay",
                  "psi", "E0", "W", "mu", "kab_J", "kab_D_max",
                  "success_rate", "fairness", "fairness_thr", "throughput", "collision_rate",
                  "objective_value"]
    ctrl_labels = [all_results[k]["label"] for k in keys_list]

    rows_data, row_labels = [], []
    for pk in param_keys:
        row = [all_results[k]["best_params"].get(pk) for k in keys_list]
        if all(v is None for v in row):
            continue
        row_labels.append(pk)
        rows_data.append(row)

    if rows_data:
        ax_table.axis("off")
        cell_text = []
        for row in rows_data:
            cell_text.append([
                f"{v:.4g}" if isinstance(v, float) else (str(v) if v is not None else "—")
                for v in row
            ])

        # winner 열 강조색
        col_colors = []
        for k in keys_list:
            col_colors.append("#fff9c4" if k == winner_key else "white")

        tbl = ax_table.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=ctrl_labels,
            cellLoc="center",
            loc="center",
            colColours=col_colors,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 1.35)

        # objective_value 행 배경색으로 강조
        obj_row_idx = row_labels.index("objective_value") + 1  # +1: header offset
        for col_idx in range(len(keys_list)):
            tbl[obj_row_idx, col_idx].set_facecolor("#d4edda")

        ax_table.set_title(
            f"Best Hyperparameters & Metrics  (winner: {winner_label}, 노란 열)",
            fontsize=10, pad=8,
        )

    out_path = os.path.join(os.path.abspath(config.output_dir), "comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison figure: {out_path}")
    return out_path


def main() -> None:
    result = run_optuna_tune(OptunaConfig(
        controller_keys=("q_learning", "er_etd"),
        n_trials=500,
        parallel_studies=True,
    ))
    for key, res in result["results"].items():
        print(f"{key}: best={res['best_params']['objective_value']:.4f}")


if __name__ == "__main__":
    main()
