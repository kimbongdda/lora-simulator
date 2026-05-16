"""간단한 예제: 기준선마다 짧은 비교 시나리오를 한 번씩 실행한다."""

from __future__ import annotations

from baselines import BASELINE_ORDER, make_controller
from env import ScenarioConfig, run_simulation


def main() -> None:
    scenario = ScenarioConfig(
        n_nodes=40,
        target_g=1.0,
        n_slots=4_000,
        warmup_slots=500,
        epoch_slots=500,
        n_channels=3,
        fixed_distance_ratio=0.5,
        profile_name="short",
        seed=7,
    )

    print("Short smoke example")
    print("=" * 60)
    print(f"N={scenario.n_nodes}, G={scenario.target_g}, channels={scenario.n_channels}")
    print(f"slots={scenario.n_slots}, warmup={scenario.warmup_slots}, p_arrival={scenario.p_arrival:.4f}")
    print("=" * 60)

    for baseline_key in BASELINE_ORDER:
        result = run_simulation(scenario, make_controller(baseline_key))
        print(
            f"{result['baseline_label']:<12} "
            f"success={result['success_rate']:.3f} "
            f"collision={result['collision_rate']:.3f} "
            f"throughput={result['throughput']:.3f} "
            f"Garr={result['arrival_load_realized']:.3f} "
            f"Gatt={result['attempt_load_realized']:.3f} "
            f"backlog/node={result['mean_backlog_per_node']:.2f} "
            f"fairness={result['fairness']:.3f}"
        )


if __name__ == "__main__":
    main()
