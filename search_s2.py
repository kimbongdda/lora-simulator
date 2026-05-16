"""s2_compact 최적 설정 탐색.

조건: N=60, G=3.0, fresh, success, random layout, s2_compact
탐색 대상: reward_variant × bin_count/thresholds
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import dataclasses
from itertools import product

from agents.q_learning import REWARD_VARIANTS
from baselines.catalog import make_controller
from env import ScenarioConfig, run_simulation
from experiments.state_variant_compare import StateVariantConfig, run_state_variant_compare

# ── 고정 조건 ──────────────────────────────────────────────
N_NODES   = 60
TARGET_G  = 3.0
N_SLOTS   = 15_000
WARMUP    = 3_000
EPOCH     = 500
N_CH      = 3
PROFILE   = "short"
SEED      = 42
QUEUE     = "fresh"
OBS       = "success"
LAYOUT    = "random"
SV_KEY    = "s2_compact"

# ── 탐색 공간 ──────────────────────────────────────────────
REWARD_KEYS = list(REWARD_VARIANTS.keys())

# success 모드: ch_g = successes / (window * N_SF), 물리적 상한 = 1.0
# → 모든 threshold는 (0, 1) 범위 내에 있어야 의미 있음
BIN_CONFIGS: list[tuple[int, tuple[float, ...]]] = [
    (2, (0.3,)),
    (2, (0.5,)),
    (2, (0.7,)),
    (3, (0.2, 0.5)),
    (3, (0.3, 0.6)),
    (3, (0.4, 0.7)),
    (3, (0.5, 0.8)),
    (4, (0.2, 0.4, 0.7)),
    (4, (0.3, 0.5, 0.8)),
    (4, (0.2, 0.5, 0.8)),
]

def run_one(reward_key: str, n_bins: int, thresholds: tuple[float, ...], idx: int) -> dict:
    scenario = ScenarioConfig(
        n_nodes=N_NODES,
        target_g=TARGET_G,
        n_slots=N_SLOTS,
        warmup_slots=WARMUP,
        epoch_slots=EPOCH,
        n_channels=N_CH,
        fixed_distance_ratio=0.5,
        profile_name=PROFILE,
        seed=SEED + idx * 1000,
        layout_seed=SEED,
        queue_mode=QUEUE,
        gw_obs_mode=OBS,
        gw_thresholds=thresholds,
        layout=LAYOUT,
    )
    controller = make_controller(
        "decentralized_q_learning",
        reward_variant=reward_key,
        state_variant=SV_KEY,
    )
    result = run_simulation(scenario, controller)
    return {
        "reward": reward_key,
        "n_bins": n_bins,
        "thresholds": thresholds,
        "asr":       result["success_rate"],
        "thr":       result["throughput"],
        "fairness":  result["fairness"],
        "collision": result["collision_rate"],
        "backlog":   result["final_backlog_per_node"],
        # 종합 점수: thr 0.5 + fairness 0.3 + asr 0.2
        "score": 0.5 * result["throughput"] + 0.3 * result["fairness"] + 0.2 * result["success_rate"],
    }


def main():
    combos = [(r, nb, th) for r in REWARD_KEYS for nb, th in BIN_CONFIGS]
    total = len(combos)
    print(f"총 {total}개 조합 탐색 시작 (N={N_NODES}, G={TARGET_G}, {QUEUE}/{OBS}, {LAYOUT})\n")

    results = []
    for i, (reward_key, n_bins, thresholds) in enumerate(combos):
        th_str = "/".join(f"{t:.1f}" for t in thresholds)
        print(f"[{i+1:>3}/{total}] reward={reward_key:<20} bins={n_bins}  th=({th_str})", end="  ", flush=True)
        try:
            r = run_one(reward_key, n_bins, thresholds, i)
            results.append(r)
            print(f"ASR={r['asr']:.3f}  thr={r['thr']:.4f}  fair={r['fairness']:.3f}  score={r['score']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")

    if not results:
        print("결과 없음")
        return

    results.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "="*90)
    print(f"{'RANK':<5} {'REWARD':<22} {'BINS':<5} {'THRESHOLDS':<20} {'ASR':>6} {'THR':>8} {'FAIR':>7} {'SCORE':>8}")
    print("-"*90)
    for rank, r in enumerate(results[:15], 1):
        th_str = "(" + ", ".join(f"{t:.1f}" for t in r["thresholds"]) + ")"
        print(f"{rank:<5} {r['reward']:<22} {r['n_bins']:<5} {th_str:<20} "
              f"{r['asr']:>6.3f} {r['thr']:>8.4f} {r['fairness']:>7.3f} {r['score']:>8.4f}")

    best = results[0]
    th_str = "(" + ", ".join(f"{t:.1f}" for t in best["thresholds"]) + ")"
    print(f"\n최적: reward={best['reward']}  bins={best['n_bins']}  thresholds={th_str}")
    print(f"      ASR={best['asr']:.4f}  throughput={best['thr']:.4f}  fairness={best['fairness']:.4f}")


if __name__ == "__main__":
    main()
