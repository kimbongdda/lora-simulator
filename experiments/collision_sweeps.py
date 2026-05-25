"""충돌 중심 LoRaWAN 설정에서 기준선 비교 스윕을 수행한다."""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import os
from types import SimpleNamespace

from baselines import BASELINE_ORDER, expand_run_specs, get_baseline_specs, make_controller
from env import PROFILE_SHORT, ScenarioConfig, run_simulation
from experiments.plotting import plot_summary_figure
from utils.io import ensure_dir, write_csv_rows, write_json


DEFAULT_G_VALUES = tuple(round(0.1 * idx, 2) for idx in range(1, 31))
DEFAULT_N_VALUES = (1,) + tuple(range(5, 156, 5))


@dataclasses.dataclass(frozen=True)
class ComparisonConfig:
    profile_name: str = PROFILE_SHORT.name
    n_channels: int = 3
    fixed_distance_ratio: float = 0.5
    n_slots: int = 20_000
    warmup_slots: int = 5_000
    epoch_slots: int = 1_000
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
    g_values: tuple[float, ...] = DEFAULT_G_VALUES
    n_values: tuple[int, ...] = DEFAULT_N_VALUES
    g_ref_n: int = 60
    n_ref_g: float = 1.0
    output_dir: str = os.path.join("outputs", "baseline_comparison")
    workers: int | None = None
    layout: str = "random"
    baseline_keys: tuple[str, ...] = BASELINE_ORDER
    n_runs: int = 1


def _resolved_reward_variants(config: ComparisonConfig) -> tuple[str, ...]:
    return tuple(config.reward_variants) if config.reward_variants else (config.reward_variant,)


def _job_seed(base_seed: int, sweep_kind: str, run_spec, x_value: float) -> int:
    sweep_offset = 100_000 if sweep_kind == "g" else 200_000
    base_offset = (BASELINE_ORDER.index(run_spec.base_key) + 1) * 10_000
    variant_offset = 0
    if run_spec.reward_variant:
        variant_offset = (sum(ord(ch) for ch in run_spec.reward_variant) % 97) * 100
    x_offset = int(round(float(x_value) * 10.0)) if sweep_kind == "g" else int(x_value)
    return int(base_seed + sweep_offset + base_offset + variant_offset + x_offset)


def _nearest_g_value(target_g: float, g_values: tuple[float, ...]) -> float:
    return min(g_values, key=lambda value: abs(float(value) - float(target_g)))


def _run_one(job):
    config_dict, sweep_kind, run_spec_dict, x_value = job
    config = ComparisonConfig(**config_dict)
    run_spec = SimpleNamespace(**run_spec_dict)

    n_nodes = config.g_ref_n if sweep_kind == "g" else int(x_value)
    if sweep_kind == "g":
        target_g = float(x_value)
    else:
        # N-sweep: 모든 노드가 매 슬롯 패킷을 생성 (p_arrival = 1.0)
        # p_arrival = G * N_res / N = 1.0  →  G = N / N_res
        n_res = 6 * config.n_channels
        target_g = float(x_value) / n_res
    reward_variant = run_spec.reward_variant or config.reward_variant

    scenario = ScenarioConfig(
        n_nodes=n_nodes,
        target_g=target_g,
        n_slots=config.n_slots,
        warmup_slots=config.warmup_slots,
        epoch_slots=config.epoch_slots,
        n_channels=config.n_channels,
        fixed_distance_ratio=config.fixed_distance_ratio,
        profile_name=config.profile_name,
        seed=_job_seed(config.seed, sweep_kind, run_spec, x_value),
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
    controller = make_controller(run_spec.base_key, reward_variant, config.state_variant, dual_mab_b=config.dual_mab_b, psi=config.psi, E0=config.E0, W=config.W, er_mode=config.er_mode, mu=config.mu)
    result = run_simulation(scenario, controller)

    row = {
        "sweep": sweep_kind,
        "x_value": float(x_value),
        "baseline_key": run_spec.series_key,
        "base_baseline_key": run_spec.base_key,
        "baseline_label": run_spec.label,
        "profile_name": result["profile_name"],
        "n_nodes": result["n_nodes"],
        "target_g": result["target_g"],
        "raw_p_arrival": result["raw_p_arrival"],
        "p_arrival": result["p_arrival"],
        "p_arrival_is_clipped": result["p_arrival_is_clipped"],
        "n_resources": result["n_resources"],
        "attempts": result["attempts"],
        "successes": result["successes"],
        "collisions": result["collisions"],
        "link_failures": result["link_failures"],
        "success_rate": result["success_rate"],
        "collision_rate": result["collision_rate"],
        "throughput": result["throughput"],
        "fairness": result["fairness"],
        "arrival_load_realized": result["arrival_load_realized"],
        "attempt_load_realized": result["attempt_load_realized"],
        "mean_backlog_total": result["mean_backlog_total"],
        "mean_backlog_per_node": result["mean_backlog_per_node"],
        "max_backlog_total": result["max_backlog_total"],
        "final_backlog_total": result["final_backlog_total"],
        "final_backlog_per_node": result["final_backlog_per_node"],
        "reward_variant": run_spec.reward_variant or "",
    }
    for sf_offset, fraction in enumerate(result["sf_fractions"], start=7):
        row[f"sf{sf_offset}_fraction"] = fraction

    result["baseline_key"] = run_spec.series_key
    result["base_baseline_key"] = run_spec.base_key
    result["baseline_label"] = run_spec.label
    result["reward_variant"] = run_spec.reward_variant or ""
    return row, result


def _run_jobs(config: ComparisonConfig, jobs: list[tuple]) -> tuple[list[dict], dict[tuple[str, float], dict]]:
    rows: list[dict] = []
    full_results: dict[tuple[str, float], dict] = {}
    workers = config.workers or max(1, mp.cpu_count() - 1)

    if workers == 1:
        iterator = map(_run_one, jobs)
        for row, result in iterator:
            rows.append(row)
            full_results[(row["baseline_key"], row["x_value"])] = result
    else:
        # Windows + Streamlit 환경에서는 spawn 컨텍스트를 명시해야 한다.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for row, result in pool.imap_unordered(_run_one, jobs):
                rows.append(row)
                full_results[(row["baseline_key"], row["x_value"])] = result

    rows.sort(key=lambda rec: (rec["baseline_key"], rec["x_value"]))
    return rows, full_results


def _write_reference_epochs(out_dir: str, run_specs, g_results: dict[tuple[str, float], dict], g_ref: float) -> None:
    epoch_rows = []
    for run_spec in run_specs:
        result = g_results[(run_spec.series_key, g_ref)]
        for epoch in result["epoch_log"]:
            row = dict(epoch)
            row["baseline_key"] = run_spec.series_key
            row["base_baseline_key"] = run_spec.base_key
            row["baseline_label"] = run_spec.label
            row["reward_variant"] = run_spec.reward_variant or ""
            row["reference_sweep"] = "g"
            epoch_rows.append(row)
    if epoch_rows:
        write_csv_rows(os.path.join(out_dir, "reference_epoch_log.csv"), epoch_rows)


def _average_rows(rows: list[dict]) -> dict:
    avg = dict(rows[0])
    n = len(rows)
    for col in list(avg):
        try:
            avg[col] = sum(float(r[col]) for r in rows) / n
        except (TypeError, ValueError):
            pass
    return avg


def _average_all_rows(all_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in all_rows:
        groups.setdefault((row["baseline_key"], row["x_value"]), []).append(row)
    result = [_average_rows(g) for g in groups.values()]
    result.sort(key=lambda r: (r["baseline_key"], r["x_value"]))
    return result


def run_comparison(config: ComparisonConfig | None = None) -> dict:
    config = config or ComparisonConfig()
    out_dir = os.path.abspath(config.output_dir)
    ensure_dir(out_dir)

    reward_variants = _resolved_reward_variants(config)
    run_specs = expand_run_specs(config.baseline_keys, reward_variants)

    n_runs = max(1, config.n_runs)
    all_g_rows: list[dict] = []
    all_n_rows: list[dict] = []
    g_results: dict[tuple[str, float], dict] = {}

    for run_idx in range(n_runs):
        run_seed = config.seed + run_idx * 1_000_000
        run_cfg = dataclasses.replace(config, seed=run_seed, n_runs=1)
        run_dict = dataclasses.asdict(run_cfg)

        g_jobs = [
            (run_dict, "g", dataclasses.asdict(run_spec), float(g_value))
            for run_spec in run_specs
            for g_value in config.g_values
        ]
        n_jobs = [
            (run_dict, "n", dataclasses.asdict(run_spec), float(n_value))
            for run_spec in run_specs
            for n_value in config.n_values
        ]

        g_rows_run, g_results_run = _run_jobs(config, g_jobs)
        n_rows_run, _ = _run_jobs(config, n_jobs)

        all_g_rows.extend(g_rows_run)
        all_n_rows.extend(n_rows_run)
        if run_idx == 0:
            g_results = g_results_run

    g_rows = _average_all_rows(all_g_rows) if n_runs > 1 else all_g_rows
    n_rows = _average_all_rows(all_n_rows) if n_runs > 1 else all_n_rows

    write_csv_rows(os.path.join(out_dir, "g_sweep.csv"), g_rows)
    write_csv_rows(os.path.join(out_dir, "n_sweep.csv"), n_rows)

    reference_g = _nearest_g_value(config.n_ref_g, config.g_values)
    representative = g_results[(run_specs[0].series_key, float(reference_g))]
    figure_meta = {
        "profile_name": config.profile_name,
        "fixed_radius_m": representative["fixed_radius_m"],
        "n_channels": config.n_channels,
        "n_resources": representative["n_resources"],
        "n_slots": config.n_slots,
        "warmup_slots": config.warmup_slots,
        "g_ref_n": config.g_ref_n,
        "n_ref_g": config.n_ref_g,
        "n_sweep_label": "p_arr = 1.0 (saturated)",
    }
    plot_summary_figure(
        g_rows=g_rows,
        n_rows=n_rows,
        out_path=os.path.join(out_dir, "baseline_comparison.png"),
        meta=figure_meta,
    )

    _write_reference_epochs(out_dir=out_dir, run_specs=run_specs, g_results=g_results, g_ref=reference_g)

    write_json(
        os.path.join(out_dir, "metadata.json"),
        {
            "config": dataclasses.asdict(config),
            "run_specs": [dataclasses.asdict(run_spec) for run_spec in run_specs],
            "baselines": [dataclasses.asdict(spec) for spec in get_baseline_specs() if spec.key in config.baseline_keys],
            "notes": {
                "g_definition": "G is defined against modeled SF x channel resources: G = N * p_arrival / (N_SF * n_channels).",
                "scenario": "All nodes are placed on the same ring at R_fixed, removing distance variation while keeping the probabilistic link model intact.",
                "queue_model": "Each node uses an accumulating packet queue. New arrivals are admitted even when packets are already waiting.",
                "reward_variants": "Q-learning baselines are expanded once per selected reward variant and plotted as separate series.",
            },
        },
    )

    return {"output_dir": out_dir, "g_rows": g_rows, "n_rows": n_rows}


def main() -> None:
    result = run_comparison()
    print(f"Outputs saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
