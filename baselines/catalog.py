"""이름이 있는 기준선 등록표와 실행 사양 확장 보조 함수."""

from __future__ import annotations

from dataclasses import dataclass

from agents.q_learning import DEFAULT_ACTION_VARIANT, REWARD_VARIANTS

from .adr_like import make_adr_like_controller
from .dual_mab import make_dual_mab_controller
from .kaburaki import make_kaburaki_controller
from .lora_mab import make_lora_mab_controller
from .pure_aloha import make_pure_aloha_controller
from .q_learning import make_decentralized_q_learning_controller
from .retry_aware import make_retry_aware_controller
from .phase_q_learning import make_phase_q_learning_controller
from .thompson_mab import make_thompson_mab_controller
from .lr_rl import make_lr_rl_controller, make_lr_rl_idle_controller


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    label: str
    description: str
    grounding: str


@dataclass(frozen=True)
class BaselineRunSpec:
    series_key: str
    base_key: str
    label: str
    reward_variant: str | None = None


BASELINE_ORDER = (
    "pure_aloha",
    "adr_like",
    "retry_aware",
    "lora_mab",
    "lora_mab_neg",
    "lora_mab_idle",
    "lora_mab_idle_neg",
    "thompson_mab",
    "thompson_mab_idle",
    "dual_mab",
    "dual_mab_no_acb",
    "decentralized_q_learning",
    "q_learning_abs",
    "q_learning_rel_sfch",
    "q_learning_abs_sfch",
    "kaburaki",
    "lr_rl",
    "lr_rl_idle",
)


BASELINE_SPECS = {
    "pure_aloha": BaselineSpec(
        key="pure_aloha",
        label="Pure ALOHA",
        description="Fixed-rule baseline with random channel selection and no learning.",
        grounding="Contention-only baseline with no learning or adaptation.",
    ),
    "adr_like": BaselineSpec(
        key="adr_like",
        label="ADR-like",
        description="Static SF rule with random channel selection, inspired by LoRaWAN ADR.",
        grounding="LoRaWAN ADR-inspired fixed data-rate assignment.",
    ),
    "retry_aware": BaselineSpec(
        key="retry_aware",
        label="Retry-aware",
        description="Heuristic backoff controller that reacts to retry count.",
        grounding="Slotted ALOHA collision-mitigation heuristic with no learning.",
    ),
    "lora_mab": BaselineSpec(
        key="lora_mab",
        label="LoRa-MAB (EXP3)",
        description="EXP3 multi-armed bandit: each node independently learns the best (SF, channel) arm.",
        grounding="LoRa-MAB (IEEE GLOBECOM Workshops 2019): stateless distributed MAB baseline.",
    ),
    "lora_mab_neg": BaselineSpec(
        key="lora_mab_neg",
        label="LoRa-MAB (EXP3, fail=-1)",
        description="EXP3 MAB with negative reward on failure to accelerate avoidance of bad arms.",
        grounding="LoRa-MAB variant: success=+1, failure=-1.",
    ),
    "lora_mab_idle": BaselineSpec(
        key="lora_mab_idle",
        label="LoRa-MAB+IDLE (EXP3)",
        description="EXP3 with an additional IDLE arm — agent can learn to defer transmission under congestion.",
        grounding="LoRa-MAB extended with IDLE action for congestion-aware MAB comparison.",
    ),
    "lora_mab_idle_neg": BaselineSpec(
        key="lora_mab_idle_neg",
        label="LoRa-MAB+IDLE (EXP3, fail=-1)",
        description="EXP3 with IDLE arm and negative failure reward.",
        grounding="LoRa-MAB extended with IDLE action and failure=-1 reward.",
    ),
    "thompson_mab": BaselineSpec(
        key="thompson_mab",
        label="Thompson Sampling MAB",
        description="Beta-Bernoulli Thompson Sampling per node. Handles sparse rewards better than EXP3.",
        grounding="Thompson Sampling: Beta posterior per arm, samples to balance explore/exploit.",
    ),
    "thompson_mab_idle": BaselineSpec(
        key="thompson_mab_idle",
        label="Thompson Sampling MAB+IDLE",
        description="Thompson Sampling with additional IDLE arm for congestion-aware deferral.",
        grounding="Thompson Sampling extended with IDLE action.",
    ),
    "dual_mab": BaselineSpec(
        key="dual_mab",
        label="Dual-MAB (ACB b=0.3)",
        description="Resource-MAB (ε-greedy EMA) + Backoff-MAB with ACB barring (b=0.3).",
        grounding="Dual-MAB: decentralized joint access-control and resource-selection learning.",
    ),
    "dual_mab_no_acb": BaselineSpec(
        key="dual_mab_no_acb",
        label="Dual-MAB (no ACB)",
        description="Resource-MAB only, no ACB barring (b=0). Backoff-MAB inactive.",
        grounding="Dual-MAB with b=0: pure resource selection baseline for ablation.",
    ),
    "decentralized_q_learning": BaselineSpec(
        key="decentralized_q_learning",
        label="Q-learning (decentralized)",
        description="Independent Q-table per node.",
        grounding="Independent tabular Q-learning with per-channel congestion feedback.",
    ),
    "q_learning_abs": BaselineSpec(
        key="q_learning_abs",
        label="Q-learning (절대 SF+채널혼잡)",
        description="독립 Q-테이블, 절대 SF 액션: IDLE + SF0~5 × 채널. |A|=1+6×N_ch. 상태: SF(6)×G_ch0(3)×G_ch1(3)×G_ch2(3)=162.",
        grounding="Absolute SF action variant with per-channel congestion state (s7_sf_gch) — agent sees which channels are congested.",
    ),
    "q_learning_rel_sfch": BaselineSpec(
        key="q_learning_rel_sfch",
        label="Q-learning (SF-delta+CH)",
        description="독립 Q-테이블, 상대 SF 액션(±1) × 채널. 상태: SF_delta(3)×CH(3)=9. |A|=10.",
        grounding="Relative SF delta action with (SF_delta, channel) state (s14_sf_delta_ch) — minimal channel-aware relative action baseline.",
    ),
    "q_learning_abs_sfch": BaselineSpec(
        key="q_learning_abs_sfch",
        label="Q-learning (SF+CH)",
        description="독립 Q-테이블, 절대 SF 액션 × 채널. 상태: SF(6)×CH(3)=18. |A|=19.",
        grounding="Absolute SF action with (SF, channel) state (s13_sf_ch) — direct SF+channel selection without congestion signal.",
    ),
    "kaburaki": BaselineSpec(
        key="kaburaki",
        label="Kaburaki (offset Q-lrn)",
        description="타이밍 오프셋 Q-learning. 각 노드가 J+1개 오프셋 후보 중 ε-greedy로 선택 후 해당 슬롯만큼 대기 후 고정 SF/채널로 전송.",
        grounding="Kaburaki et al. (2021): Q-learning-based timing offset control for LoRa networks.",
    ),
    "lr_rl": BaselineSpec(
        key="lr_rl",
        label="LR-RL (V-table)",
        description="Stateless V-table TD(0). 각 노드가 (SF, 채널) 18개 조합의 가치를 학습. 보상은 PCR·패킷 비율로 정규화. 패킷 도착 시 IDLE 없이 반드시 전송.",
        grounding="Hong et al. (2023) IEEE IoT J.: RL-based SF allocation, (SF, ch) joint 선택으로 확장.",
    ),
    "lr_rl_idle": BaselineSpec(
        key="lr_rl_idle",
        label="LR-RL+IDLE (V-table)",
        description="LR-RL V-table에 IDLE arm 1개 추가 (총 19 엔트리). 전송 arm이 모두 음수로 수렴할 때 자연스럽게 IDLE 선택. IDLE 업데이트: R=0, γ·max(전송 arms).",
        grounding="LR-RL 확장: IDLE arm 추가로 고부하 환경에서 자발적 혼잡 회피 가능.",
    ),
}


def get_baseline_specs():
    """사전 정의된 기준선 설명 목록을 순서대로 반환한다."""
    return [BASELINE_SPECS[key] for key in BASELINE_ORDER]


def expand_run_specs(
    baseline_keys: tuple[str, ...] | list[str],
    reward_variants: tuple[str, ...] | list[str] = ("v0_current",),
) -> list[BaselineRunSpec]:
    """선택한 기준선을 실제 실행 가능한 시계열 묶음으로 확장한다.

    Q-learning 기준선은 선택한 reward variant마다 하나씩 복제해,
    같은 그림이나 표에서 서로 비교할 수 있게 만든다.
    """
    reward_variants = tuple(key for key in reward_variants if key in REWARD_VARIANTS) or ("v0_current",)
    run_specs: list[BaselineRunSpec] = []

    for baseline_key in baseline_keys:
        spec = BASELINE_SPECS[baseline_key]
        if baseline_key in ("decentralized_q_learning", "q_learning_abs", "q_learning_rel_sfch", "q_learning_abs_sfch"):
            for variant_key in reward_variants:
                variant_label = REWARD_VARIANTS[variant_key].get("plot_label", REWARD_VARIANTS[variant_key]["label"])
                run_specs.append(
                    BaselineRunSpec(
                        series_key=f"{baseline_key}__{variant_key}",
                        base_key=baseline_key,
                        label=f"{spec.label} | {variant_label}",
                        reward_variant=variant_key,
                    )
                )
        else:
            run_specs.append(
                BaselineRunSpec(
                    series_key=baseline_key,
                    base_key=baseline_key,
                    label=spec.label,
                    reward_variant=None,
                )
            )

    return run_specs


def make_controller(
    baseline_key: str,
    reward_variant: str = "v0_current",
    state_variant: str = "s0_full",
    action_variant: str = DEFAULT_ACTION_VARIANT,
    dual_mab_b: float = 0.3,
    psi: float = 0.0,
    E0: float = 10.0,
    W: int = 20,
    energy_cost: dict | None = None,
    er_mode: str = "ETD",
    mu: float = 0.5,
    alpha: float = 0.1,
    gamma_q: float = 0.9,
    eps_min: float = 0.05,
    eps_decay: float = 0.9995,
    reward_params: "dict | None" = None,
    kaburaki_J: int = 3,
    kaburaki_D_max: int = 64,
    kaburaki_alpha: float = 0.3,
    kaburaki_gamma: float = 0.95,
    kaburaki_eps_min: float = 0.05,
):
    """기준선 키에 맞는 컨트롤러 인스턴스를 만든다."""
    if baseline_key == "pure_aloha":
        return make_pure_aloha_controller()
    if baseline_key == "adr_like":
        return make_adr_like_controller(margin_db=1.0)
    if baseline_key == "retry_aware":
        return make_retry_aware_controller(margin_db=1.0, base_window=2, max_window=64)
    if baseline_key == "lora_mab":
        return make_lora_mab_controller(eta=0.1, with_idle=False, fail_reward=0.0)
    if baseline_key == "lora_mab_neg":
        return make_lora_mab_controller(eta=0.1, with_idle=False, fail_reward=-1.0)
    if baseline_key == "lora_mab_idle":
        return make_lora_mab_controller(eta=0.1, with_idle=True, fail_reward=0.0)
    if baseline_key == "lora_mab_idle_neg":
        return make_lora_mab_controller(eta=0.1, with_idle=True, fail_reward=-1.0)
    if baseline_key == "decentralized_q_learning":
        return make_decentralized_q_learning_controller(
            alpha=alpha,
            gamma_q=gamma_q,
            epsilon=1.0,
            eps_min=eps_min,
            eps_decay=eps_decay,
            reward_variant=reward_variant,
            reward_params=reward_params,
            state_variant=state_variant,
            action_variant=action_variant,
            psi=psi,
            E0=E0,
            W=W,
            energy_cost=energy_cost,
            er_mode=er_mode,
            mu=mu,
        )
    if baseline_key == "phase_q_learning":
        return make_phase_q_learning_controller(
            reward_variant=reward_variant,
        )
    if baseline_key == "thompson_mab":
        return make_thompson_mab_controller(with_idle=False)
    if baseline_key == "thompson_mab_idle":
        return make_thompson_mab_controller(with_idle=True)
    if baseline_key == "dual_mab":
        return make_dual_mab_controller(b=dual_mab_b)
    if baseline_key == "dual_mab_no_acb":
        return make_dual_mab_controller(b=0.0)
    if baseline_key == "kaburaki":
        return make_kaburaki_controller(
            J=kaburaki_J, D_max=kaburaki_D_max,
            alpha=kaburaki_alpha, gamma=kaburaki_gamma, eps_min=kaburaki_eps_min,
        )
    if baseline_key == "lr_rl":
        return make_lr_rl_controller()
    if baseline_key == "lr_rl_idle":
        return make_lr_rl_idle_controller()
    if baseline_key == "q_learning_abs":
        return make_decentralized_q_learning_controller(
            alpha=0.1,
            gamma_q=0.9,
            epsilon=1.0,
            eps_min=0.05,
            eps_decay=0.9995,
            reward_variant=reward_variant,
            state_variant="s7_sf_gch",   # SF(6) × 채널별 G bin — 채널 혼잡도 직접 관측
            action_variant="absolute",
            psi=psi,
            E0=E0,
            W=W,
            energy_cost=energy_cost,
            er_mode=er_mode,
            mu=mu,
        )
    if baseline_key == "q_learning_rel_sfch":
        return make_decentralized_q_learning_controller(
            alpha=alpha,
            gamma_q=gamma_q,
            epsilon=1.0,
            eps_min=eps_min,
            eps_decay=eps_decay,
            reward_variant=reward_variant,
            state_variant="s14_sf_delta_ch",  # SF_delta(3) × CH(3) = 9 states
            action_variant="relative",
            psi=psi,
            E0=E0,
            W=W,
            energy_cost=energy_cost,
            er_mode=er_mode,
            mu=mu,
            reward_params=reward_params,
        )
    if baseline_key == "q_learning_abs_sfch":
        return make_decentralized_q_learning_controller(
            alpha=alpha,
            gamma_q=gamma_q,
            epsilon=1.0,
            eps_min=eps_min,
            eps_decay=eps_decay,
            reward_variant=reward_variant,
            state_variant="s13_sf_ch",    # SF(6) × CH(3) = 18 states
            action_variant="absolute",
            psi=psi,
            E0=E0,
            W=W,
            energy_cost=energy_cost,
            er_mode=er_mode,
            mu=mu,
            reward_params=reward_params,
        )
    raise KeyError(f"Unknown baseline key: {baseline_key}")
