# LoRaWAN 충돌 집중 시뮬레이터 — 프로젝트 개요

> 최종 수정: 2026-04-14

---

## 목차

1. [프로젝트 목적](#1-프로젝트-목적)
2. [디렉터리 구조](#2-디렉터리-구조)
3. [env/ — 환경 코어](#3-env--환경-코어)
4. [agents/ — 컨트롤러](#4-agents--컨트롤러)
5. [baselines/ — 베이스라인 레지스트리](#5-baselines--베이스라인-레지스트리)
6. [experiments/ — 실험 모듈](#6-experiments--실험-모듈)
7. [app.py — Streamlit GUI](#7-apppy--streamlit-gui)
8. [Q-learning 상세](#8-q-learning-상세)
9. [시뮬레이션 루프 흐름](#9-시뮬레이션-루프-흐름)
10. [주요 설계 결정 및 버그 수정 이력](#10-주요-설계-결정-및-버그-수정-이력)

---

## 1. 프로젝트 목적

슬롯 단위 LoRaWAN 환경에서 **충돌 최소화 + 처리량 최대화**를 목표로 여러 MAC 전략(Pure ALOHA, ADR-like, Retry-Aware, Q-learning)을 비교하는 시뮬레이터.

핵심 특징:
- SF7~SF12 × n_channels 개 직교 리소스 (기본 3채널 → 18개)
- 충돌 = 동일 (SF, 채널)을 같은 슬롯에 두 노드 이상 사용
- 링크 실패 = 충돌 없어도 SNR 마진 < 0 dB 시 결정론적 실패 (하드 임계값)
- 독립 분산 Q-learning (노드별 별도 Q-table, 공유 없음)
- 게이트웨이 → 노드 방향의 채널 혼잡도 피드백 (G bin)

---

## 2. 디렉터리 구조

```
lora_env/
├── app.py                          # Streamlit GUI 진입점
├── env/
│   ├── channel.py                  # 리소스 정의, G 양자화
│   ├── collision.py                # 충돌/링크 실패 판정
│   ├── layout.py                   # 노드 배치 (ring / random)
│   ├── link.py                     # SNR 모델, ADR SF 계산
│   ├── simulator.py                # 메인 시뮬레이션 루프
│   ├── types.py                    # ScenarioConfig, NodeState, 액션/결과 타입
│   └── __init__.py
├── agents/
│   ├── rule_based.py               # PureAloha, AdrLike, RetryAware
│   ├── q_learning.py               # Q-learning 컨트롤러, 보상/상태 인코딩
│   └── __init__.py
├── baselines/
│   ├── catalog.py                  # BaselineSpec, make_controller, expand_run_specs
│   ├── adr_like.py                 # AdrLikeController 팩토리
│   ├── pure_aloha.py               # PureAlohaController 팩토리
│   ├── retry_aware.py              # RetryAwareController 팩토리
│   ├── decentralized_q_learning.py # DecentralizedQLearningController 팩토리
│   ├── q_learning.py               # make_decentralized_q_learning_controller
│   └── __init__.py
├── experiments/
│   ├── collision_sweeps.py         # G-sweep / N-sweep 비교 실험
│   ├── epoch_timeseries.py         # epoch별 시계열 진단
│   ├── per_node_analysis.py        # 노드별 ASR / 처리량 분석
│   ├── reward_variant_compare.py   # 보상 함수 변형 비교 (v6~v9)
│   ├── state_variant_compare.py    # 상태 공간 변형 비교 (s2~s7), configurable bins
│   ├── plots.py                    # 공통 플롯 헬퍼 (SF 히트맵, Q-table 히트맵)
│   ├── plotting.py                 # collision_sweeps 전용 요약 그래프
│   ├── style.py                    # 시리즈별 색상/마커/선 스타일
│   └── __init__.py
├── utils/
│   ├── io.py                       # CSV/JSON 저장 헬퍼
│   ├── metrics.py                  # Jain's fairness index
│   └── __init__.py
└── docs/
    └── PROJECT_OVERVIEW_KO.md      # 이 파일
```

---

## 3. env/ — 환경 코어

### env/channel.py

LoRaWAN 리소스 구조와 게이트웨이 피드백 양자화 담당.

| 상수 | 값 | 의미 |
|---|---|---|
| `N_SF` | 6 | SF7~SF12 |
| `G_LOW_THRESHOLD` | 0.7 | G 낮음/중간 경계 |
| `G_HIGH_THRESHOLD` | 1.3 | G 중간/높음 경계 |
| `GATEWAY_WINDOW` | 50 | G 추정 롤링 윈도우 (슬롯) |

**G (offered load) 정의:**

```
G = N * p_arrival / (N_SF * n_channels)
```

전체 리소스 용량 대비 슬롯당 전송 시도 비율. G=1이 slotted ALOHA 처리량 최대점.

**`quantize_g(g_val, thresholds=(0.7, 1.3))`** → G bin 정수 반환:

```
bin 수 = len(thresholds) + 1
```

기본값 `(0.7, 1.3)` 기준 3-bin:
- 0 (Low): G < 0.7 — 여유 있음
- 1 (Mid): 0.7 ≤ G < 1.3 — 적정 부하
- 2 (High): G ≥ 1.3 — 과부하, 충돌 다수

`thresholds`를 바꾸면 bin 수가 달라지며, `ScenarioConfig.gw_thresholds`로 실험별 설정 가능.

### env/collision.py

`evaluate_transmissions(transmissions, rng)` — 한 슬롯의 전송 결과 판정.

1. 동일 (SF, 채널)에 2개 이상 → **충돌** (`OUTCOME_FAIL_COLLISION`)
2. 단독 전송 → SNR 마진 기반 **하드 임계값** 판정:
   - `margin_db = mean_snr_db − SNR_THRESH[sf_idx]`
   - `margin_db ≥ 0` → `OUTCOME_SUCCESS`
   - `margin_db < 0` → `OUTCOME_FAIL_LINK`

> 이전에는 시그모이드 `1/(1+exp(−margin_db))` 기반 확률로 판정했으나,  
> 결정론적 하드 임계값 모델로 교체. `rng` 는 충돌 판정에만 사용됨.

### env/layout.py

- `place_nodes_random(n, radius, seed)` — 원 내 균등 랜덤 배치
- `place_nodes_on_ring(n, radius, seed)` — 고정 반경 링 배치

### env/link.py

**경로손실 모델: 로그-거리 (Log-distance Path Loss)**

```
LPL(d) = LPL(d0) + 10 · η · log10(d / d0)
RSSI   = P_TX − GL − LPL(d)
SNR    = RSSI − NOISE_FLOOR
```

| 상수 | 값 | 설명 |
|---|---|---|
| `P_TX_DBM` | 14.0 dBm | 송신 전력 |
| `NOISE_FLOOR_DBM` | −119.0 dBm | 잡음 바닥 |
| `GL_DB` | 0.0 dB | 안테나 이득/손실 |

**라디오 프로파일 (`RadioProfile`):**

| 이름 | d0 | lpl_d0 | η | 출처 |
|---|---|---|---|---|
| `short` | 1 m | 40 dB | 3.5 | 임의 (도심 단거리) |
| `medium` | 1 m | 37 dB | 3.0 | 임의 (교외 중거리) |
| `long` | 1 m | 34 dB | 2.5 | 임의 (농촌 장거리) |
| `lorasim` | 40 m | 127.41 dB | 2.08 | **논문 기반** (LoRaSim) |

> `lorasim` 프로파일은 LoRaSim 논문에서 가져온 실측 기반 파라미터.  
> 나머지 세 프로파일은 특정 논문을 캘리브레이션한 것이 아닌 경험적 값.

**주요 함수:**

- `compute_cell_radius(profile)` — 프로파일별 최대 도달 거리 계산
- `compute_mean_snr(dist_m, profile)` — 거리 → 평균 SNR (dB)
- `adr_static_sf_index(mean_snr_db, margin_db)` — SNR 마진 기반 SF 결정
- `simple_distance_sf_index(dist_m, cell_radius_m)` — 거리 비율 기반 초기 SF
- `link_success(margin_db)` — `margin_db ≥ 0` 이면 `True`, 미만이면 `False`

### env/types.py

**`ScenarioConfig`** (frozen dataclass):

| 필드 | 기본값 | 설명 |
|---|---|---|
| `n_nodes` | — | 노드 수 |
| `target_g` | — | 목표 offered load |
| `n_slots` | — | 총 슬롯 수 |
| `warmup_slots` | — | 워밍업 슬롯 (측정 제외) |
| `epoch_slots` | — | 에폭 길이 (슬롯) |
| `n_channels` | 3 | LoRa 채널 수 |
| `fixed_distance_ratio` | 0.5 | 링 배치 시 반경 비율 |
| `profile_name` | "short" | 링크 프로파일 |
| `seed` | 42 | 난수 시드 |
| `queue_mode` | "accumulate" | 큐 모드: `"accumulate"` or `"fresh"` |
| `layout` | "ring" | 노드 배치: `"ring"` or `"random"` |
| `gw_obs_mode` | "attempt" | GW 부하 추정 모드: `"attempt"` or `"success"` |
| `gw_thresholds` | `(0.7, 1.3)` | GW 혼잡도 양자화 bin 경계값. `len+1 = bin 수`. |

파생 프로퍼티: `n_resources`, `p_arrival`, `measured_slots`

**`queue_mode` 차이:**
- `"accumulate"` — 미전송 패킷 큐에 누적. 큐 발산 가능성 있음.
- `"fresh"` — 매 슬롯 0 or 1 패킷. 미전송 패킷 드롭. G 이론 정의와 정확히 일치.

> `retry_count`(연속 실패 횟수)는 **패킷 경계에서 리셋하지 않음**. 전송 성공 시에만 0으로 초기화 (accumulate/fresh 모드 공통).

**`NodeState`** (mutable dataclass):

| 필드 | 설명 |
|---|---|
| `sf_idx` | 현재 SF 인덱스 (0=SF7, 5=SF12) |
| `channel_idx` | 현재 채널 인덱스 |
| `queue_len` | 현재 큐 길이 |
| `retry_count` | 현재 패킷 재시도 횟수 |
| `backoff_slots` | 남은 백오프 슬롯 수 |
| `last_outcome` | 직전 슬롯 결과 (0=IDLE, 1=성공, 2=충돌, 3=링크실패) |
| `dist_m` | 게이트웨이까지 거리 (m) |
| `mean_snr_db` | 평균 SNR (dB, 시뮬레이터 내부 정보) |
| `attempts_measured` | 측정 구간 전송 시도 수 |
| `successes_measured` | 측정 구간 성공 수 |

**`ControllerAction`:**

| 필드 | 설명 |
|---|---|
| `transmit` | 이번 슬롯 전송 여부 |
| `sf_idx` | 선택한 SF |
| `channel_idx` | 선택한 채널 |
| `prev_channel_idx` | 이전 채널 (채널 전환 패널티 계산용) |
| `internal_action` | Q-learning 내부 액션 ID |
| `state` | Q-learning 상태 튜플 |
| `retry_count_before` | 액션 선택 시점 retry_count |

### env/simulator.py

`run_simulation(config, controller)` — 핵심 루프.

**루프 순서 (슬롯당):**
1. 패킷 생성 (`p_arrival` 확률로 큐 증가 or fresh 모드에서 교체)
2. `fresh` 모드 시 `retry_count = 0` 리셋
3. 백오프 감소 (`backoff_slots -= 1`, 음수 방지)
4. 각 노드: `has_packet & backoff_slots == 0` 이면 `choose_action()` 호출
5. 채널별 전송 횟수 집계 → 게이트웨이 G 추정값 갱신 (`_gateway_info`)
6. `evaluate_transmissions()` 로 충돌/성공/링크실패 판정
7. 각 노드 결과 반영 (성공 시 큐 감소, 실패 시 retry_count 증가)
8. `controller.observe(node, action, outcome, slot, gateway_info=_gateway_info)` 호출
9. 에폭 경계 도달 시 epoch_log 저장 후 카운터 리셋
10. `controller.end_slot(slot)` 호출

**반환 dict 주요 키:**

| 키 | 내용 |
|---|---|
| `success_rate` | 측정 구간 ASR (성공/시도) |
| `collision_rate` | 측정 구간 충돌률 |
| `throughput` | 측정 구간 슬롯당 성공 패킷 수 |
| `fairness` | Jain's fairness index |
| `per_node_success` | 노드별 성공 횟수 리스트 |
| `per_node_collisions` | 노드별 충돌 횟수 리스트 |
| `per_node_link_failures` | 노드별 링크실패 횟수 리스트 |
| `attempts` | 측정 구간 총 전송 시도 횟수 ($A_{\rm total}$) |
| `sf_usage` | (n_nodes, 6) ndarray, SF별 전송 횟수 |
| `epoch_log` | 에폭별 통계 리스트 |
| `xs`, `ys`, `dists_m` | 노드 좌표 및 거리 |
| `q_table_data` | Q-table 평균 배열 (Q-learning 시), 그 외 None |

---

## 4. agents/ — 컨트롤러

모든 컨트롤러는 동일한 인터페이스를 구현:

```python
controller.begin_run(nodes, config, rng)
action = controller.choose_action(node, slot, gateway_info)
controller.observe(node, action, outcome, slot, gateway_info=...)
controller.end_slot(slot)
```

### agents/rule_based.py

**`_BaseController`** — 공통 기반 클래스.  
`observe(**_)` 는 `**_`로 `gateway_info` 키워드 인자를 무시.

---

**`PureAlohaController`**

매 슬롯 SF(0~5)와 채널을 균등 랜덤 선택. 패킷 있으면 무조건 전송.  
학습 없음. 18개 리소스를 균등하게 사용 → 충돌 기준선.

---

**`AdrLikeController`**

`begin_run` 시 각 노드의 `mean_snr_db`를 기반으로 `adr_static_sf_index(margin_db=1.0)` 로 노드별 SF 고정. 채널은 랜덤.  
학습 없음. SF는 거리 기반으로 분산됨 (먼 노드 = 높은 SF).

---

**`RetryAwareController`**

AdrLike와 동일한 SF 초기화. 실패 후 **지수적 백오프** 적용:

```
exponent = clamp(retry_count - 1, 0, 5)
window = min(max_window, base_window * 2^exponent)
backoff_slots ~ Uniform[0, window]
```

기본값: `base_window=2`, `max_window=64`.  
성공 시 백오프 즉시 해제 (`backoff_slots = 0`).

---

### agents/q_learning.py

자세한 내용은 [8. Q-learning 상세](#8-q-learning-상세) 참조.

---

## 5. baselines/ — 베이스라인 레지스트리

### baselines/catalog.py

**`BASELINE_ORDER`** — 고정 실행 순서:
```python
("pure_aloha", "adr_like", "retry_aware", "decentralized_q_learning")
```

**`make_controller(baseline_key, reward_variant, state_variant)`** — 컨트롤러 팩토리.  
Q-learning 기본 하이퍼파라미터: `alpha=0.1, gamma=0.9, epsilon=1.0, eps_min=0.05, eps_decay=0.9995`  
기본값: `reward_variant="v6_signal"`, `state_variant="s0_full"`

**`expand_run_specs(baseline_keys, reward_variants)`** — Q-learning을 각 보상 함수 변형별로 복제한 `BaselineRunSpec` 리스트 반환. 논 Q-learning 베이스라인은 단일 스펙으로 그대로 포함.

---

## 6. experiments/ — 실험 모듈

### experiments/collision_sweeps.py — G/N 스윕 비교

**`ComparisonConfig`** 주요 파라미터:

| 필드 | 기본값 |
|---|---|
| `g_values` | 0.1 ~ 3.0 (30단계) |
| `n_values` | 10 ~ 155 (5 단위) |
| `n_slots` | 20,000 |
| `warmup_slots` | 5,000 |

출력: G-sweep 비교 그래프, N-sweep 비교 그래프, CSV, JSON

---

### experiments/epoch_timeseries.py — epoch별 시계열

**`TimeseriesConfig`** 기본값: `n_slots=30,000`, `warmup_slots=0`, `epoch_slots=500`

워밍업 없이 전체 구간 관찰. Q-learning 수렴 과정 시각화에 적합.

출력 그래프 (2행 3열):
- 실현 G_att, ASR, Throughput
- Mean backlog, Collision rate, Mean backlog (log scale)

---

### experiments/per_node_analysis.py — 노드별 분석

**`PerNodeConfig`** 기본값: `n_nodes=60`, `target_g=1.0`, `n_slots=20,000`

각 베이스라인에 대해 노드별 ASR, 처리량, 리소스 부하를 계산.

출력:
- `per_node.png` — 노드별 산점도 + epoch 시계열 + 실패 원인 분해 바 차트
- `node_layout.png` — 노드 좌표 + ASR 색상 맵
- `sf_heatmap.png` — 거리 vs SF 선택 히트맵
- `per_node.csv`, `system_summary.csv`, `metadata.json`

반환 dict에 `q_table_data` 포함 (Q-learning 사용 시).

---

### experiments/reward_variant_compare.py — 보상 함수 비교

**`RewardVariantConfig`** 기본값: `n_nodes=60`, `target_g=1.0`, `variant_keys=전체 v6~v9`

동일 조건에서 보상 함수 변형(v6~v9)별 Q-learning 성능 비교.

출력:
- `reward_variant_compare.png` — ASR/충돌률/처리량/attempts/backlog 시계열 비교 (2×3 레이아웃)
- `reward_variant_summary_bar.png` — 변형별 요약 막대 그래프
- `per_node.png`, `node_layout.png` — 노드별 분석 그래프
- `sf_heatmap.png` — SF 선택 히트맵
- CSV, JSON

---

### experiments/state_variant_compare.py — 상태 공간 비교

**`StateVariantConfig`** 기본값: `n_nodes=60`, `target_g=1.0`, `variant_keys=전체 s2~s7`

주요 파라미터:
- `state_variant_keys` — 비교할 변형 키 튜플 (기본: s2_compact~s7_sf_gch)
- `gw_thresholds` — bin 경계값 튜플. 실험별로 bin 수·범위 변경 가능 (기본: `(0.7, 1.3)`)

동일 조건에서 상태 공간 변형별 Q-learning 성능 비교. bin 수는 `len(gw_thresholds)+1`로 자동 결정.

출력:
- `state_variant_compare.png` — ASR/충돌률/처리량/attempts/backlog 시계열 비교 (2×3 레이아웃)
- `state_variant_summary_bar.png` — 변형별 요약 막대 그래프
- CSV, JSON

---

### experiments/plots.py — 공통 플롯 헬퍼

**`plot_sf_distance_heatmap(sf_data, labels, keys, suptitle, out_path)`**  
거리 구간(12 bin) × SF(7~12) 히트맵. 각 거리 bin에서 SF 선택 비율을 색상으로 표시.

**`plot_q_table_heatmap(q_table_data, state_variant, suptitle, out_path)`**  
- `s2_compact`: 4행(failure_level) × 3열(G bin) 패널 구조. 각 패널 = SF × 액션 히트맵 + 수치 주석.
- 기타 변형: 상태 × 액션 raw 히트맵.

라벨: `FAILURE_LEVEL_LABELS = ["OK/fresh", "fail(r=1)", "fail(r=2~3)", "fail(r=4+)"]`  
G bin: `G_BIN_LABELS = ["G Low", "G Mid", "G High"]`

---

## 7. app.py — Streamlit GUI

실행: `python -m streamlit run app.py`

**5가지 실험 모드:**

| 모드 | 설명 |
|---|---|
| Node Analysis | 고정 (N, G)에서 모든 베이스라인 실행, 노드별 분석 |
| Epoch Timeseries | epoch별 수렴 과정 시계열 |
| Reward Variant Compare | 보상 함수 변형 v6~v9 성능 비교 |
| State Variant Compare | 상태 공간 변형 s2~s7 성능 비교, bin 수/경계값 설정 가능 |
| Collision Sweeps | G-sweep / N-sweep 비교 |

**공통 파라미터:** N nodes, Target G, Slots, Warmup, Epoch slots, Channels, Queue mode, Layout, Seed, 베이스라인 선택

**Q-learning 전용 파라미터:**
- `Reward Variant` — v6_signal / v7_asymmetric / v8_fairness / v9_log_thr 중 선택
- `State Variant` — s0_full / s1_realistic / s2_compact / s3_no_sf / s4_no_fl / s5_no_gmax / s6_sf_only / s7_sf_gch 중 선택
- `GW Bin count / Thresholds` — State Variant Compare 모드: bin 수(2~8)와 경계값(쉼표 구분) 입력 가능

**결과 표시 순서 (Node Analysis / Reward Variant Compare):**
1. 시스템 요약 메트릭 테이블
2. per_node.png (노드별 산점도 + 시계열 + 실패 분해)
3. node_layout.png (노드 좌표 + ASR 색상 맵)
4. sf_heatmap.png (거리 vs SF 히트맵)
5. Q-table 뷰어 (Q-learning + s2_compact 시 구조적 패널, 기타 시 raw 히트맵)

**Q-table 뷰어 (`_render_q_table`):**
- `s2_compact`: G bin 3개 탭 → 탭마다 failure_level 4개 패널 → 각 패널 = SF × 액션 히트맵 (Streamlit plotly)
- 기타: 단일 상태 × 액션 히트맵

---

## 8. Q-learning 상세

### 액션 공간

`n_actions = 1 + 3 * n_channels`  
3채널 기준 10개 액션:

| 액션 ID | 의미 |
|---|---|
| 0 | IDLE — 이번 슬롯 전송 포기 |
| 1~3 | SF 유지, 채널 0/1/2로 전송 |
| 4~6 | SF+1 (더 강한 SF), 채널 0/1/2로 전송 |
| 7~9 | SF-1 (더 빠른 SF), 채널 0/1/2로 전송 |

### 상태 공간 (STATE_VARIANTS)

**s0_full** — 기본 (34,992 상태, n_channels=3):

```
(has_packet, last_result_bin, retry_bin, sf_idx, snr_margin_bin, channel_idx, g_ch0, g_ch1, g_ch2)
```

- `snr_margin_bin` 포함 → 시뮬레이터 내부 정보 (실제 노드는 측정 불가)

**s1_realistic** — snr_margin_bin 제거 (11,664 상태):

```
(has_packet, last_result_bin, retry_bin, sf_idx, channel_idx, g_ch0, g_ch1, g_ch2)
```

**s2_compact** — SF×FL×Gmax (기본 72 상태):

```
(sf_idx, failure_level, gateway_g_max)
```

4가지 최적화:
1. `has_packet` 제거 — `choose_action`은 패킷 있을 때만 호출됨
2. `channel_idx` 제거 — 액션에 이미 채널이 인코딩됨
3. `last_result_bin + retry_bin` → `failure_level` 4단계 통합
4. `per_channel_g_bins (3개)` → `gateway_g_max (최대 bin 1개)`

**상태 변형 목록 (bin 수 = n_bins, 기본 n_bins=3):**

| 변형 키 | 포함 차원 | 상태 수 공식 | 기본값 |
|---|---|---|---|
| `s2_compact` | sf_idx, failure_level, gateway_g_max | 6×4×n_bins | 72 |
| `s3_no_sf` | failure_level, gateway_g_max | 4×n_bins | 12 |
| `s4_no_fl` | sf_idx, gateway_g_max | 6×n_bins | 18 |
| `s5_no_gmax` | sf_idx, failure_level | 6×4 | 24 |
| `s6_sf_only` | sf_idx | 6 | 6 |
| `s7_sf_gch` | sf_idx, bin(G_ch0), …, bin(G_chN) | 6×n_bins^n_channels | 162 |

> `n_bins = len(gw_thresholds) + 1` — `ScenarioConfig.gw_thresholds`로 실험별 조정 가능.  
> `s5_no_gmax` / `s6_sf_only`는 G bin 미사용이므로 bin 수 변경 영향 없음.

**failure_level 상세:**

| 값 | 조건 | 의미 |
|---|---|---|
| 0 | last_outcome==SUCCESS or retry_count==0 | 성공했거나 fresh 상태 |
| 1 | retry_count == 1 | 처음 실패 |
| 2 | retry_count 2~3 | 반복 실패 |
| 3 | retry_count >= 4 | 누적 실패 |

### 보상 함수 (REWARD_VARIANTS)

| 변형 | 타입 | 특징 |
|---|---|---|
| `v6_signal` | standard | 기준선. success=+1.0, fail=-1.0, idle_pkt=-0.02. retry_coef=0, switch_pen=0 |
| `v7_asymmetric` | composite | 성공 = 0.5 + 0.5/(1+ν) → [0.5, 1.0]. 처리량·공정성 균형 |
| `v8_fairness` | composite | 성공 = 0.3 + 0.7/(1+ν) → [0.3, 1.0]. 공정성 우선 |
| `v9_log_thr` | composite | 성공 = 0.6 + 0.4/(1+ν) → [0.6, 1.0]. fail=-0.7, 탐색 허용 |

**composite 보상 설명:** `ν_i = n_i^ep / max(1, n̄^ep)` (에폭 내 성공 횟수 정규화)  
성공 많은 노드는 보상이 `r_base`로 수렴, 성공 적은 노드는 `r_base + r_fair`에 가까워져 공정성 향상.

**`compute_reward` 파라미터:**
- `outcome` — 슬롯 결과
- `retry_count_before` — 액션 선택 시점의 retry_count
- `channel_changed` — 채널 전환 여부 (switch_pen 적용 조건)
- `has_packet` — IDLE 보상 분기에 사용
- `variant` — 보상 함수 키
- `node_success_count` — 에폭 내 이 노드 성공 횟수 (composite 타입 전용)
- `mean_success_count` — 에폭 내 전체 노드 평균 성공 횟수 (composite 타입 전용)

### 학습 알고리즘

ε-greedy 탐색 + 단순 TD(0) Q-update:

```
Q(x, a) += alpha * (r + gamma * max_a' Q(x', a') - Q(x, a))
```

ε 감쇠: 매 슬롯 `epsilon *= eps_decay` (분산형은 노드별 독립 감쇠)

**observe() 호출 시점:**
- 전송한 경우: 결과 판정 후 호출 (outcome = SUCCESS / COLLISION / LINK_FAIL)
- 전송 안 한 경우 (백오프 또는 큐 없음): outcome = IDLE로 즉시 호출
- `gateway_info` 를 keyword 인자로 받아 `next_state` 인코딩에 **현재 슬롯** G bin 사용

**`get_mean_q_array()`** — 전체 노드 Q-table 평균을 `(states_list, ndarray)` 튜플로 반환.  
GUI Q-table 뷰어에 사용.

### DecentralizedQLearningController

`shared_table = False` → 노드별 독립 Q-table + 독립 ε.  
`_node_q_tables: dict[node_id, defaultdict]`  
`_node_epsilons: dict[node_id, float]`

---

## 9. 시뮬레이션 루프 흐름

```
for slot in range(n_slots):
    is_measured = slot >= warmup_slots

    # 패킷 생성
    for node in nodes:
        if fresh: node.queue_len = (1 if arrived else 0)  # retry_count 리셋 없음
        else:     if arrived: node.queue_len += 1
        if node.backoff_slots > 0: node.backoff_slots -= 1

    # 액션 결정
    actions = []
    for node in nodes:
        if not node.has_packet or node.backoff_slots > 0:
            action = ControllerAction(transmit=False, ...)  # 강제 IDLE
        else:
            action = controller.choose_action(node, slot, _gateway_info)

    # G 추정 갱신 (현재 슬롯 전송 기반)
    _ch_history.append(ch_counts)
    _gateway_info = compute_g_bins(_ch_history)

    # 충돌 판정
    outcomes = evaluate_transmissions(transmissions, rng)

    # 결과 반영 + observe
    for node, action in zip(nodes, actions):
        if not action.transmit:
            controller.observe(node, action, OUTCOME_IDLE, slot, gateway_info=_gateway_info)
            continue
        outcome = outcomes[node.node_id]
        if outcome == SUCCESS: node.queue_len -= 1; node.retry_count = 0
        else: node.retry_count += 1
        controller.observe(node, action, outcome, slot, gateway_info=_gateway_info)

    controller.end_slot(slot)  # ε 감쇠
```

---

## 10. 주요 설계 결정 및 버그 수정 이력

### tx_probability 수정
기존 `tx_probability = config.p_arrival` → 패킷 있어도 `p_arrival` 확률로만 전송 → 실효 전송률 = `p_arrival²`.  
수정: `tx_probability = 1.0`. G 제어는 패킷 생성 확률만 담당.

### Pure ALOHA SF 랜덤화
기존: 초기화 시 거리 기반 SF 고정 → 링 배치에서 전 노드 동일 SF → 실효 리소스 3개.  
수정: 매 슬롯 SF/채널 완전 랜덤 → 18개 리소스 균등 사용.

### fresh 모드 retry_count 의미론 변경
기존: fresh 모드 패킷 생성 시 `node.retry_count = 0` 리셋 → Q-learning 상태의 `failure_level`이 항상 0 → 실패 이력 기반 학습 불가.  
수정: `retry_count`를 **연속 실패 횟수**로 재정의. 패킷 경계에서 리셋하지 않고 **전송 성공 시에만** 0으로 초기화. accumulate/fresh 모드 모두 동일한 의미론 적용 → fresh 모드에서도 `failure_level` 유효.

### IDLE 학습 누락
전송하지 않은 노드에 `controller.observe()` 미호출 → IDLE 액션이 Q-table에서 학습 안 됨.  
수정: 강제 IDLE 노드에도 `outcome=OUTCOME_IDLE`로 `observe()` 호출.

### next_state G bin 시차 버그
`observe()` 내 `next_state` 인코딩이 이전 슬롯 G bin 사용 → TD 타겟 부정확.  
수정: `observe()`에 `gateway_info` 파라미터 추가, 현재 슬롯 G bin으로 `next_state` 인코딩.

### 노드별 SF 초기화 (AdrLike / RetryAware)
기존: 모든 노드 동일 SF(전역 평균 SNR 기반) → 동일 SF에 집중.  
수정: 노드별 개별 SNR 기반 SF 할당 → SF 분산.

### 보상 함수 v1~v4 충돌 패널티 약화 문제
v1~v4 에서 collision=-0.5 (v0의 절반) → v0 대비 충돌 억제 신호 약함.  
수정: 모든 변형 collision 최소 -0.8 이상으로 강화.

### 링크 성공 모델 하드 임계값 전환
기존: 시그모이드 `1/(1+exp(−k·margin_db))` → 확률적 성공 판정 (매 슬롯 `rng.random()` 호출).  
수정: `margin_db ≥ 0` 이면 무조건 성공, 미만이면 무조건 링크실패. 완전 결정론적.  
이유: 섀도잉 없는 로그-거리 모델과 일관성 확보. SNR 마진이 거리만으로 결정되므로 확률 모델은 불필요한 노이즈.

### lorasim 프로파일 추가
기존 세 프로파일(`short/medium/long`)은 임의 파라미터 기반.  
추가: `lorasim` 프로파일 (`d0=40m, lpl_d0=127.41dB, η=2.08`) — LoRaSim 논문 기반 실측값.
