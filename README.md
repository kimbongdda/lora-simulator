# LoRa Simulator

LoRaWAN 충돌 중심 시뮬레이터. 다수의 노드가 슬롯 ALOHA 방식으로 게이트웨이에 패킷을 전송하는 환경에서 다양한 MAC 제어 알고리즘의 성능을 비교한다.

채널 모델은 LoRaSIM 논문 (Bor et al., 2016) 기반이며, Streamlit GUI와 CLI 두 가지 인터페이스를 제공한다.

---

## 주요 기능

- **SF × 채널 직교 자원** 기반 충돌 시뮬레이션 (SF7~SF12, 채널 1/2/3/6)
- **Rayleigh fading** 옵션 (`|h|² ~ Exp(1)` 순시 SNR)
- **4가지 라디오 프로파일**: short (도시), medium (교외), long (농촌), lorasim (논문 기반)
- 에폭 단위 시계열, 노드별 분석, G·N 스윕 실험 모드
- **Optuna 하이퍼파라미터 자동 탐색**: TPE 샘플러, 멀티 컨트롤러 병렬 스터디
- **파라메트릭 보상 탐색**: 보상 계수를 연속값으로 최적화 (`r_success_base`, `r_fail_abs` 등)
- **프리셋 카트 시스템**: Optuna 최적 결과를 JSON으로 저장, Node Analysis에서 다중 프리셋 동시 비교
- 실험 결과 자동 CSV 로그 저장

---

## 지원 알고리즘

| 알고리즘 | 유형 | 설명 |
|---------|------|------|
| Pure ALOHA | Rule-based | 랜덤 채널 선택, 학습 없음 |
| ADR-like | Rule-based | SNR 기반 고정 SF 배정 (LoRaWAN ADR 모방) |
| Retry-aware | Heuristic | 재시도 횟수 기반 지수 백오프 |
| LoRa-MAB (EXP3) | MAB | EXP3 알고리즘으로 최적 (SF, 채널) 학습 |
| Thompson Sampling MAB | MAB | Beta-Bernoulli 사후 분포 기반 탐색/활용 균형 |
| Dual-MAB | MAB | Resource-MAB + ACB Backoff-MAB 이중 구조 |
| Q-learning (decentralized) | RL | 노드별 독립 Q-테이블, 다양한 상태·보상 변형 지원 |
| Q-learning (절대 SF) | RL | 절대 SF 액션 변형, 채널별 혼잡도 상태 |
| **ER-Q (ETD/EM)** | **RL + 에너지 규제** | Q-값에 누적 에너지 패널티를 적용해 과도한 전송을 억제. ETD(누적+즉시) 및 EM(가중 혼합) 두 가지 추정 방식 지원 |

---

## 보상 변형 (Q-learning)

| 변형 키 | 타입 | 설계 목표 |
|---------|------|---------|
| `base` | standard | 기준선. 성공=+1, 실패=-1, IDLE(패킷 있음)=-0.02 |
| `fair` | composite | 공정성 강화. 성공 보상 = 0.3 + 0.7/(1+ν) → [0.3, 1.0] |
| `explore` | composite | 탐색 허용. 성공 보상 = 0.6 + 0.4/(1+ν), 실패=-0.7 |
| `congestion` | standard | 혼잡 회피. IDLE 보상=+0.10으로 전송 포기 장려 |

파라메트릭 보상 탐색(Optuna)으로 위 계수를 연속값으로 자동 최적화할 수 있다.

---

## 설치

Python 3.10 이상 권장.

```bash
git clone https://github.com/kimbongdda/lora-simulator.git
cd lora-simulator

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 실행

### GUI (Streamlit)

```bash
streamlit run app.py
```

브라우저에서 사이드바로 시나리오를 설정하고 **Run Simulation** 버튼을 클릭한다.

**실험 모드**

| 모드 | 설명 |
|------|------|
| Node Analysis | 노드별 ASR, 처리량, SF 히트맵. 프리셋 카트에서 불러온 Q-learning 설정을 추가 시리즈로 동시 실행 |
| Epoch Timeseries | 에폭 단위 학습 곡선 (충돌률, 처리량, 백로그) |
| G / N Sweep | 제공 부하 G 및 노드 수 N 스윕 비교 |
| Reward Variant Compare | Q-learning 보상 설계 변형 비교 (base/fair/explore/congestion) |
| State Variant Compare | Q-learning 상태 공간 변형 비교 |
| Action Variant Compare | 상대 SF vs 절대 SF 액션 공간 비교 |
| Phase Learning | 슬롯 페이즈 자기조직화 비교 |
| **ER Compare** | **ER-Q ETD/EM vs 기준 Q-learning 에폭 시계열 비교** |
| **Optuna Tune** | **Optuna TPE로 Q-learning 하이퍼파라미터 자동 탐색. 파라메트릭 보상 탐색 지원** |
| Experiment Log | 누적 실험 결과 조회 및 CSV 다운로드 |

### Optuna Tune 사용법

1. 사이드바에서 **Optuna Tune** 모드 선택
2. 탐색할 컨트롤러(decentralized Q-learning 등), 트라이얼 수, 목적 지표 설정
3. `Search parametric reward` 옵션 활성화 시 보상 계수를 연속값으로 탐색
4. **Run Optuna Search** 클릭 → 결과 테이블에서 원하는 행의 🛒 체크박스 선택
5. **Add N to Cart** 버튼으로 프리셋 카트에 저장
6. Node Analysis 모드로 전환 → 사이드바 **Preset Cart** 패널에서 프리셋 체크 후 실행

### CLI

```bash
python example_usage.py
```

모든 베이스라인을 고정 시나리오(N=40, G=1.0, 채널=3)로 한 번씩 실행하고 결과를 터미널에 출력한다.

---

## 주요 설정 파라미터

`ScenarioConfig`에서 설정한다.

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `n_nodes` | — | 노드 수 |
| `target_g` | — | 목표 제공 부하 G (SF×채널 수로 정규화) |
| `n_slots` | — | 총 시뮬레이션 슬롯 수 |
| `warmup_slots` | — | 워밍업 구간 (측정에서 제외) |
| `n_channels` | 3 | 채널 수 |
| `profile_name` | `"short"` | 라디오 프로파일 |
| `layout` | `"ring"` | 노드 배치: `"ring"` (고정 거리) / `"random"` (균등 무작위) |
| `queue_mode` | `"accumulate"` | `"fresh"`: 슬롯마다 패킷 교체 / `"accumulate"`: 큐 누적 |
| `enable_rayleigh_fading` | `False` | Rayleigh fading 활성화 여부 |
| `psi` (ψ) | `0.0` | ER-Q 에너지 패널티 계수 (0이면 비활성) |
| `E0` | `10.0` | ER-Q 에너지 허용 기준값 (초과분에만 패널티 부과) |
| `W` | `20` | ER-Q 에너지 슬라이딩 윈도우 길이 (슬롯) |
| `mu` (μ) | `0.5` | ER-EM 모드의 누적/즉시 에너지 가중 혼합 비율 |

---

## 라디오 프로파일 및 SF별 최대 도달 거리

경로손실 모델: `L(d) = L(d₀) + 10γ log₁₀(d/d₀)`

| 프로파일 | d₀ | L(d₀) | γ | SF12 최대 반경 |
|---------|-----|--------|---|--------------|
| short (도시) | 1 m | 40 dB | 3.5 | ~140 m |
| medium (교외) | 1 m | 37 dB | 3.0 | ~590 m |
| long (농촌) | 1 m | 34 dB | 2.5 | ~3.7 km |
| **lorasim** | 40 m | 127.41 dB | 2.08 | **~545 m** |

**lorasim 프로파일 SF별 최대 도달 거리** (SNR 임계값 + 2 dB 마진 기준)

| SF | 최대 거리 |
|----|---------|
| SF7 | ~136 m |
| SF8 | ~180 m |
| SF9 | ~238 m |
| SF10 | ~313 m |
| SF11 | ~413 m |
| SF12 | ~545 m |

---

## 프로젝트 구조

```
lora-simulator/
├── app.py                      # Streamlit GUI 진입점 (10가지 실험 모드)
├── example_usage.py            # CLI 예제
├── outputs/
│   └── saved_presets.json      # 프리셋 카트 영구 저장소
├── env/
│   ├── simulator.py            # 슬롯 단위 시뮬레이션 루프
│   ├── channel.py              # 자원 정의, G 추정
│   ├── link.py                 # 경로손실 모델, SNR 임계값
│   ├── collision.py            # 충돌 판정
│   ├── layout.py               # 노드 배치
│   └── types.py                # 공통 데이터 타입
├── agents/
│   └── q_learning.py           # Q-learning 컨트롤러 (상태·보상 변형, ER-Q, reward_params)
├── baselines/
│   ├── catalog.py              # 베이스라인 등록 및 팩토리 (make_controller)
│   ├── pure_aloha.py
│   ├── adr_like.py
│   ├── retry_aware.py
│   ├── lora_mab.py             # EXP3 MAB
│   ├── thompson_mab.py
│   ├── dual_mab.py
│   └── q_learning.py           # make_decentralized_q_learning_controller
├── experiments/                # 실험 모드별 실행 스크립트 및 플롯
│   ├── per_node_analysis.py    # 노드별 분석 (다중 프리셋 시리즈 지원)
│   ├── epoch_timeseries.py     # 에폭 시계열
│   ├── collision_sweeps.py     # G/N 스윕
│   ├── reward_variant_compare.py
│   ├── state_variant_compare.py
│   ├── er_compare.py           # ER-Q (ETD/EM) vs 기준 Q-learning 비교
│   ├── optuna_tune.py          # Optuna 하이퍼파라미터 탐색 (OptunaConfig, 파라메트릭 보상)
│   └── style.py                # 시리즈별 색상/마커
└── utils/                      # 지표 계산, 실험 로그
```

---

## 참고 문헌

- M. Bor, U. Roedig, T. Voigt, J. M. Alonso, "Do LoRa Low-Power Wide-Area Networks Scale?", *ACM MSWiM*, 2016. (LoRaSIM 채널 모델)
- S. Bonnefoi et al., "Multi-Armed Bandit Model for Contention Resolution in LoRa Networks," *IEEE GLOBECOM Workshops*, 2019. (LoRa-MAB)

---

## 라이선스

MIT
