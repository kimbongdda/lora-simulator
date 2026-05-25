"""충돌 중심 LoRaWAN 설정에서 노드별 분해 분석을 수행한다."""

from __future__ import annotations

import dataclasses
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents.q_learning import DecentralizedQLearningController
from baselines import BASELINE_ORDER, expand_run_specs, get_baseline_specs, make_controller
from env import ScenarioConfig, run_simulation
from experiments.plots import plot_channel_g_timeseries, plot_sf_distance_heatmap
from experiments.style import series_color, series_linestyle, series_marker
from utils.io import ensure_dir, write_csv_rows, write_json

# 카트 프리셋 전용 색상 (기존 BASELINE_COLORS와 겹치지 않는 7색)
_CART_PRESET_COLORS = [
    "#003f5c", "#58508d", "#bc5090", "#ff6361",
    "#ffa600", "#488f31", "#00b0be",
]


# ER 변형 정의
# psi_override=None → config.psi 사용, mu_override=None → config.mu 사용
# state_variant=None → config.state_variant 사용 (에너지 상태 없음)
_ER_VARIANT_DEFS: dict[str, dict] = {
    "etd": {
        "label": "Q-lrn ER-ETD",
        "color_key": "q_learning_er_etd",
        "psi_override": None,
        "er_mode": "ETD",
        "mu_override": 0.5,
        "state_variant": None,        # config.state_variant 사용 (에너지 빈 없음)
        "action_variant": None,
    },
    "em": {
        "label": "Q-lrn ER-EM",
        "color_key": "q_learning_er_em",
        "psi_override": None,
        "er_mode": "EM",
        "mu_override": None,
        "state_variant": None,
        "action_variant": None,
    },
    "etd_s18": {
        "label": "Q-lrn ER-ETD (SF방향+Eb)",
        "color_key": "q_learning_er_etd_s18",
        "psi_override": None,
        "er_mode": "ETD",
        "mu_override": 0.5,
        "state_variant": "s18_sf_delta_eb",  # SF_delta(3)×Eb(3) = 9 states
        "action_variant": None,
    },
    "etd_s17": {
        "label": "Q-lrn ER-ETD (SF+Eb)",
        "color_key": "q_learning_er_etd_s17",
        "psi_override": None,
        "er_mode": "ETD",
        "mu_override": 0.5,
        "state_variant": "s17_sf_eb", # SF(6)×Eb(3) = 18 states
        "action_variant": None,
    },
    "etd_s15": {
        "label": "Q-lrn ER-ETD (SF+Gmax+Eb)",
        "color_key": "q_learning_er_etd_s15",
        "psi_override": None,
        "er_mode": "ETD",
        "mu_override": 0.5,
        "state_variant": "s15_sf_gmax_eb",  # SF(6)×Gmax(3)×Eb(3) = 54 states
        "action_variant": None,
    },
    "em_s18": {
        "label": "Q-lrn ER-EM (SF방향+Eb)",
        "color_key": "q_learning_er_em_s18",
        "psi_override": None,
        "er_mode": "EM",
        "mu_override": None,
        "state_variant": "s18_sf_delta_eb",
        "action_variant": None,
    },
    "em_s17": {
        "label": "Q-lrn ER-EM (SF+Eb)",
        "color_key": "q_learning_er_em_s17",
        "psi_override": None,
        "er_mode": "EM",
        "mu_override": None,
        "state_variant": "s17_sf_eb",
        "action_variant": None,
    },
    "em_s15": {
        "label": "Q-lrn ER-EM (SF+Gmax+Eb)",
        "color_key": "q_learning_er_em_s15",
        "psi_override": None,
        "er_mode": "EM",
        "mu_override": None,
        "state_variant": "s15_sf_gmax_eb",
        "action_variant": None,
    },
    "etd_s13": {
        "label": "Q-lrn ER-ETD (SF+CH)",
        "color_key": "q_learning_er_etd_s13",
        "psi_override": None,
        "er_mode": "ETD",
        "mu_override": 0.5,
        "state_variant": "s13_sf_ch",      # SF(6)×CH(3) = 18 states
        "action_variant": "absolute",
    },
    "etd_s14": {
        "label": "Q-lrn ER-ETD (SF-delta+CH)",
        "color_key": "q_learning_er_etd_s14",
        "psi_override": None,
        "er_mode": "ETD",
        "mu_override": 0.5,
        "state_variant": "s14_sf_delta_ch",  # SF_delta(3)×CH(3) = 9 states
        "action_variant": "relative",
    },
}


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
    enable_rician_fading: bool = False
    rician_k_factor: float = 4.0
    rician_fade_margin_db: float = 5.0
    dual_mab_b: float = 0.3
    kaburaki_J: int = 3
    kaburaki_D_max: int = 64
    kaburaki_alpha: float = 0.3
    kaburaki_gamma: float = 0.95
    kaburaki_eps_min: float = 0.05
    psi: float = 0.0
    E0: float = 10.0
    W: int = 20
    er_mode: str = "ETD"
    mu: float = 0.5
    er_q_variants: tuple[str, ...] = ()   # ("etd","em","etd_s18","etd_s17","etd_s15","em_s18","em_s17","em_s15") 중 선택
    layout: str = "random"
    reward_variant: str = "v0_current"
    reward_variants: tuple[str, ...] = ()
    state_variant: str = "s0_full"
    output_dir: str = os.path.join("outputs", "per_node_analysis")
    baseline_keys: tuple[str, ...] = tuple(BASELINE_ORDER)
    # Q-learning 하이퍼파라미터 (Optuna 프리셋 로드 시 사용)
    alpha: float = 0.1
    gamma_q: float = 0.9
    eps_min: float = 0.05
    eps_decay: float = 0.9995
    reward_params: object = dataclasses.field(default=None, hash=False, compare=False)
    # 카트에서 선택한 다중 프리셋 — 각각 별도 series로 실행
    custom_ql_presets: object = dataclasses.field(default=(), hash=False, compare=False)


def _resolved_reward_variants(config: PerNodeConfig) -> tuple[str, ...]:
    return tuple(config.reward_variants) if config.reward_variants else (config.reward_variant,)


def _extract_energy_meta(result: dict, n_nodes: int) -> dict:
    """result dict에서 에너지 관련 summary 지표를 추출한다."""
    return {
        "mae": result.get("mae", 0.0),
        "elr": result.get("elr", 0.0),
        "energy_efficiency": result.get("energy_efficiency", 0.0),
        "total_energy": result.get("total_energy", 0.0),
    }


def _run_all(config: PerNodeConfig):
    rows = []
    meta_rows = []
    epoch_data: dict[str, list[dict]] = {}
    node_positions: dict[str, tuple] = {}
    sf_data: dict[str, tuple] = {}
    extra_colors: dict[str, str] = {}   # series_key → hex color (카트 프리셋 전용)
    last_q_table_data = None
    last_q_per_node = None
    last_q_dists_m = None
    run_specs = expand_run_specs(config.baseline_keys, _resolved_reward_variants(config))

    def _process_result(result, series_key, base_key, label, reward_variant_str, run_idx):
        nonlocal last_q_table_data, last_q_per_node, last_q_dists_m

        if result.get("q_table_data") is not None:
            last_q_table_data = result["q_table_data"]
        if result.get("q_per_node") is not None:
            last_q_per_node = result["q_per_node"]
            last_q_dists_m = result["dists_m"]

        annotated_epoch_log = []
        for ep in result["epoch_log"]:
            row = dict(ep)
            row["baseline_key"] = series_key
            row["base_baseline_key"] = base_key
            row["baseline_label"] = label
            row["reward_variant"] = reward_variant_str
            annotated_epoch_log.append(row)
        epoch_data[series_key] = annotated_epoch_log
        node_positions[series_key] = (result["xs"], result["ys"], result["dists_m"])
        sf_data[series_key] = (
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
        per_energy_total = np.asarray(result.get("per_node_energy_total", [0.0] * config.n_nodes), dtype=np.float64)
        per_energy_fail = np.asarray(result.get("per_node_energy_fail", [0.0] * config.n_nodes), dtype=np.float64)
        n_resources = config.n_channels * 6

        for node_id in range(config.n_nodes):
            attempts = per_attempts[node_id]
            successes = per_success[node_id]
            collisions = per_collisions[node_id]
            link_failures = per_link_failures[node_id]
            rows.append(
                {
                    "baseline_key": series_key,
                    "base_baseline_key": base_key,
                    "baseline_label": label,
                    "reward_variant": reward_variant_str,
                    "node_id": node_id,
                    "attempts": int(attempts),
                    "successes": int(successes),
                    "collisions": int(collisions),
                    "link_failures": int(link_failures),
                    "asr": successes / attempts if attempts > 0 else 0.0,
                    "throughput": successes / measured if measured > 0 else 0.0,
                    "resource_load": attempts / (n_resources * measured) if measured > 0 else 0.0,
                    "energy_total": float(per_energy_total[node_id]),
                    "energy_fail": float(per_energy_fail[node_id]),
                }
            )

        energy_meta = _extract_energy_meta(result, config.n_nodes)
        meta_rows.append(
            {
                "baseline_key": series_key,
                "base_baseline_key": base_key,
                "baseline_label": label,
                "reward_variant": reward_variant_str,
                "system_success_rate": result["success_rate"],
                "system_collision_rate": result["collision_rate"],
                "system_throughput": result["throughput"],
                "fairness": result["fairness"],
                "fairness_asr": result.get("fairness_asr", 0.0),
                "fairness_thr": result.get("fairness_thr", 0.0),
                "mean_backlog_per_node": result["mean_backlog_per_node"],
                "final_backlog_per_node": result.get("final_backlog_per_node", 0.0),
                "attempts": result["attempts"],
                **energy_meta,
            }
        )

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
            layout=config.layout,
        )
        _is_ql = run_spec.base_key == "decentralized_q_learning"
        controller = make_controller(
            run_spec.base_key, run_spec.reward_variant or config.reward_variant,
            config.state_variant, dual_mab_b=config.dual_mab_b,
            psi=0.0, E0=config.E0, W=config.W, er_mode=config.er_mode, mu=config.mu,
            alpha=config.alpha if _is_ql else 0.1,
            gamma_q=config.gamma_q if _is_ql else 0.9,
            eps_min=config.eps_min if _is_ql else 0.05,
            eps_decay=config.eps_decay if _is_ql else 0.9995,
            reward_params=config.reward_params if _is_ql else None,
            kaburaki_J=config.kaburaki_J,
            kaburaki_D_max=config.kaburaki_D_max,
            kaburaki_alpha=config.kaburaki_alpha,
            kaburaki_gamma=config.kaburaki_gamma,
            kaburaki_eps_min=config.kaburaki_eps_min,
        )
        result = run_simulation(scenario, controller)
        _process_result(result, run_spec.series_key, run_spec.base_key, run_spec.label,
                        run_spec.reward_variant or "", idx)
        print(
            f"  {run_spec.label:<42} "
            f"ASR={result['success_rate']:.3f}  MAE={result.get('mae', 0.0):.2f}  "
            f"ELR={result.get('elr', 0.0):.3f}  EE={result.get('energy_efficiency', 0.0):.4f}"
        )

    # ER Q-learning 변형 추가 실행
    # decentralized_q_learning baseline과 동일한 시드를 써야 공정 비교가 되고,
    # ψ=0인 "baseline" 변형이 regular Q-learning과 완전히 같은 결과를 내야 한다.
    _dql_idx = next(
        (i for i, rs in enumerate(run_specs) if rs.base_key == "decentralized_q_learning"),
        None,
    )
    _er_seed = config.seed + 10_000 * ((_dql_idx + 1) if _dql_idx is not None else 1)

    for v_idx, variant_key in enumerate(config.er_q_variants):
        vdef = _ER_VARIANT_DEFS.get(variant_key)
        if vdef is None:
            continue
        psi_v = vdef["psi_override"] if vdef["psi_override"] is not None else config.psi
        er_mode_v = vdef["er_mode"]
        mu_v = vdef["mu_override"] if vdef["mu_override"] is not None else config.mu
        state_v = vdef.get("state_variant") or config.state_variant
        action_v = vdef.get("action_variant") or "relative"
        color_key = vdef["color_key"]
        label = vdef["label"]
        series_key = f"er_q_{variant_key}"

        scenario = ScenarioConfig(
            n_nodes=config.n_nodes,
            target_g=config.target_g,
            n_slots=config.n_slots,
            warmup_slots=config.warmup_slots,
            epoch_slots=config.epoch_slots,
            n_channels=config.n_channels,
            fixed_distance_ratio=config.fixed_distance_ratio,
            profile_name=config.profile_name,
            seed=_er_seed,          # 모든 ER 변형이 Q-learning baseline과 동일 시드
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
        controller = make_controller(
            "decentralized_q_learning", config.reward_variant, state_v,
            action_variant=action_v,
            psi=psi_v, E0=config.E0, W=config.W, er_mode=er_mode_v, mu=mu_v,
        )
        result = run_simulation(scenario, controller)
        _process_result(result, series_key, color_key, label, "", v_idx)
        print(
            f"  {label:<42} "
            f"ASR={result['success_rate']:.3f}  MAE={result.get('mae', 0.0):.2f}  "
            f"ELR={result.get('elr', 0.0):.3f}  EE={result.get('energy_efficiency', 0.0):.4f}"
        )

    # ── 카트 프리셋 추가 실행 ──────────────────────────────────────────────────
    _dql_idx = next(
        (i for i, rs in enumerate(run_specs) if rs.base_key == "decentralized_q_learning"),
        None,
    )
    _preset_seed = config.seed + 10_000 * ((_dql_idx + 1) if _dql_idx is not None else 1)

    for p_idx, preset in enumerate(config.custom_ql_presets or ()):
        series_key = f"cart_ql_{p_idx}"
        color = _CART_PRESET_COLORS[p_idx % len(_CART_PRESET_COLORS)]
        extra_colors[series_key] = color
        label = preset.get("name", f"Preset {p_idx + 1}")

        scenario = ScenarioConfig(
            n_nodes=config.n_nodes,
            target_g=config.target_g,
            n_slots=config.n_slots,
            warmup_slots=config.warmup_slots,
            epoch_slots=config.epoch_slots,
            n_channels=config.n_channels,
            fixed_distance_ratio=config.fixed_distance_ratio,
            profile_name=config.profile_name,
            seed=_preset_seed,
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
        controller = DecentralizedQLearningController(
            alpha=float(preset.get("alpha", 0.1)),
            gamma_q=float(preset.get("gamma_q", 0.9)),
            epsilon=1.0,
            eps_min=float(preset.get("eps_min", 0.05)),
            eps_decay=float(preset.get("eps_decay", 0.9995)),
            reward_variant=preset.get("reward_variant", "base"),
            reward_params=preset.get("reward_params"),
            psi=float(preset.get("psi", 0.0)),
            E0=float(preset.get("E0", 10.0)),
            W=int(preset.get("W") or 20),
            er_mode=preset.get("er_mode", "ETD"),
            mu=float(preset.get("mu", 0.5)),
            state_variant=config.state_variant,
        )
        result = run_simulation(scenario, controller)
        _process_result(result, series_key, series_key, label, "", len(run_specs) + p_idx)
        print(
            f"  [Cart] {label:<40} "
            f"ASR={result['success_rate']:.3f}  "
            f"Fairness={result['fairness']:.3f}  "
            f"Thr={result['throughput']:.4f}"
        )

    return rows, meta_rows, epoch_data, node_positions, sf_data, extra_colors, last_q_table_data, last_q_per_node, last_q_dists_m, run_specs


def _plot(rows: list[dict], epoch_data: dict[str, list[dict]], meta_rows: list[dict], out_path: str, config: PerNodeConfig, extra_colors: dict | None = None) -> None:
    def _clr(base_key: str, rv: str = "") -> str:
        if extra_colors and base_key in extra_colors:
            return extra_colors[base_key]
        return series_color(base_key, rv)

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

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    p_gen_equiv = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    fig.suptitle(
        f"Node Analysis | N={config.n_nodes} | G={config.target_g} | p_gen≈{p_gen_equiv:.3f} | "
        f"profile={config.profile_name} | channels={config.n_channels}",
        fontsize=12,
        fontweight="bold",
    )

    # ── Row 0: 노드별 산점도 ──────────────────────────────────────────────────
    scatter_metrics = [
        ("asr", "Per-node ASR", "Success rate (per node)"),
        ("throughput", "Per-node Throughput", "Successes / measured slot"),
        ("resource_load", "Per-node Resource Load", "Attempts / (resource × slot)"),
    ]

    for col, (metric, title, ylabel) in enumerate(scatter_metrics):
        ax = axes[0, col]
        for key, data in grouped.items():
            color = _clr(data["base_key"], data["reward_variant"])
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

    # ── Row 1: 에폭 시계열 + Failure Breakdown ────────────────────────────────
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
            color = _clr(base_key, reward_variant)
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
    labels_bar = []
    suc_frac = []
    coll_frac = []
    link_frac = []
    total_attempts_bar = []
    for key, data in grouped.items():
        key_rows = [row for row in rows if row["baseline_key"] == key]
        successes = sum(row["successes"] for row in key_rows)
        collisions = sum(row["collisions"] for row in key_rows)
        link_failures = sum(row["link_failures"] for row in key_rows)
        total = successes + collisions + link_failures
        labels_bar.append(data["label"])
        suc_frac.append(successes / total if total else 0.0)
        coll_frac.append(collisions / total if total else 0.0)
        link_frac.append(link_failures / total if total else 0.0)
        total_attempts_bar.append(total)
    x_idx = list(range(len(labels_bar)))
    ax.bar(x_idx, suc_frac, label="Success", color="#2ca02c", alpha=0.85)
    ax.bar(x_idx, coll_frac, bottom=suc_frac, label="Collision", color="#d62728", alpha=0.85)
    ax.bar(x_idx, link_frac, bottom=[s + c for s, c in zip(suc_frac, coll_frac)], label="Link Fail", color="#ff7f0e", alpha=0.85)
    for xi, total in zip(x_idx, total_attempts_bar):
        ax.text(xi, 1.02, f"{total:,}", ha="center", va="bottom", fontsize=6.5, color="#333333", rotation=45)
    ax.set_title("Failure Cause Breakdown  (top: total attempts)", fontsize=10)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels_bar, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of attempts", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)

    # ── Row 2: 에너지 지표 막대 차트 ─────────────────────────────────────────
    energy_labels = [mr["baseline_label"] for mr in meta_rows]
    bar_colors = [_clr(mr["base_baseline_key"], mr.get("reward_variant", "")) for mr in meta_rows]
    x_e = list(range(len(energy_labels)))

    ax = axes[2, 0]
    vals_mae = [mr.get("mae", 0.0) for mr in meta_rows]
    bars = ax.bar(x_e, vals_mae, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, vals_mae):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals_mae) * 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_e)
    ax.set_xticklabels(energy_labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("MAE — Mean Access Energy", fontsize=11)
    ax.set_ylabel("Total energy / active nodes", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[2, 1]
    vals_elr = [mr.get("elr", 0.0) for mr in meta_rows]
    bars = ax.bar(x_e, vals_elr, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, vals_elr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_e)
    ax.set_xticklabels(energy_labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("ELR — Energy Loss Rate", fontsize=11)
    ax.set_ylabel("Failed energy / total energy", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[2, 2]
    vals_ee = [mr.get("energy_efficiency", 0.0) for mr in meta_rows]
    bars = ax.bar(x_e, vals_ee, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, vals_ee):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals_ee) * 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_e)
    ax.set_xticklabels(energy_labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("EE — Energy Efficiency", fontsize=11)
    ax.set_ylabel("Successes / total energy", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    # ── Row 3: 공정성 지표 ─────────────────────────────────────────────────────
    ax = axes[3, 0]
    vals_f = [mr.get("fairness", 0.0) for mr in meta_rows]
    bars = ax.bar(x_e, vals_f, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, vals_f):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_e)
    ax.set_xticklabels(energy_labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("Jain's FI (success counts)", fontsize=11)
    ax.set_ylabel("Fairness index [0–1]", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[3, 1]
    vals_fa = [mr.get("fairness_asr", 0.0) for mr in meta_rows]
    bars = ax.bar(x_e, vals_fa, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, vals_fa):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_e)
    ax.set_xticklabels(energy_labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("Jain's FI on per-node ASR", fontsize=11)
    ax.set_ylabel("Fairness index [0–1]", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[3, 2]
    vals_ft = [mr.get("fairness_thr", 0.0) for mr in meta_rows]
    bars = ax.bar(x_e, vals_ft, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, vals_ft):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_e)
    ax.set_xticklabels(energy_labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("Jain's FI (Thr, all nodes)", fontsize=11)
    ax.set_ylabel("Fairness index [0–1]", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
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

    rows, meta_rows, epoch_data, node_positions, sf_data, extra_colors, last_q_table_data, last_q_per_node, last_q_dists_m, run_specs = _run_all(config)
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
                "mae": "total_energy / active_nodes (Mean Access Energy)",
                "elr": "fail_energy / total_energy (Energy Loss Rate)",
                "energy_efficiency": "successes / total_energy (Energy Efficiency)",
            },
        },
    )
    p_gen_equiv = config.target_g * config.n_channels * 6 / max(config.n_nodes, 1)
    _plot(rows, epoch_data, meta_rows, os.path.join(out_dir, "per_node.png"), config, extra_colors=extra_colors)
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
        k: (
            extra_colors.get(epoch_data[k][0]["base_baseline_key"])
            or series_color(epoch_data[k][0]["base_baseline_key"], epoch_data[k][0].get("reward_variant", ""))
        ) if epoch_data[k] else "#888888"
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
