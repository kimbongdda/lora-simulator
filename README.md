# LoRa Simulator

LoRaWAN 충돌 중심 시뮬레이터. 다수의 노드가 슬롯 ALOHA 방식으로 게이트웨이에 패킷을 전송하는 환경에서 다양한 MAC 제어 알고리즘의 성능을 비교한다.

채널 모델은 LoRaSIM 논문 (Bor et al., 2016) 기반이며, Streamlit GUI와 CLI 두 가지 인터페이스를 제공한다.

---

## 주요 기능

- **SF × 채널 직교 자원** 기반 충돌 시뮬레이션 (SF7~SF12, 채널 1/2/3/6)
- **Rayleigh fading** 옵션 (`|h|² ~ Exp(1)` 순시 SNR)
- **4가지 라디오 프로파일**: short (도시), medium (교외), long (농촌), lorasim (논문 기반)
- 에폭 단위 시계열, 노드별 분석, G·N 스윕 실험 모드
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
| Node Analysis | 노드별 ASR, 처리량, SF 히트맵 |
| Epoch Timeseries | 에폭 단위 학습 곡선 (충돌률, 처리량, 백로그) |
| G / N Sweep | 제공 부하 G 및 노드 수 N 스윕 비교 |
| Reward Variant Compare | Q-learning 보상 설계 변형 비교 |
| State Variant Compare | Q-learning 상태 공간 변형 비교 |
| Action Variant Compare | 상대 SF vs 절대 SF 액션 공간 비교 |
| Phase Learning | 슬롯 페이즈 자기조직화 비교 |
| Experiment Log | 누적 실험 결과 조회 및 CSV 다운로드 |

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
├── app.py                  # Streamlit GUI 진입점
├── example_usage.py        # CLI 예제
├── env/
│   ├── simulator.py        # 슬롯 단위 시뮬레이션 루프
│   ├── channel.py          # 자원 정의, G 추정
│   ├── link.py             # 경로손실 모델, SNR 임계값
│   ├── collision.py        # 충돌 판정
│   ├── layout.py           # 노드 배치
│   └── types.py            # 공통 데이터 타입
├── agents/
│   └── q_learning.py       # Q-learning 컨트롤러 (상태·보상 변형 포함)
├── baselines/
│   ├── catalog.py          # 베이스라인 등록 및 팩토리
│   ├── pure_aloha.py
│   ├── adr_like.py
│   ├── retry_aware.py
│   ├── lora_mab.py         # EXP3 MAB
│   ├── thompson_mab.py
│   └── dual_mab.py
├── experiments/            # 실험 모드별 실행 스크립트 및 플롯
└── utils/                  # 지표 계산, 실험 로그
```

---

## 참고 문헌

- M. Bor, U. Roedig, T. Voigt, J. M. Alonso, "Do LoRa Low-Power Wide-Area Networks Scale?", *ACM MSWiM*, 2016. (LoRaSIM 채널 모델)
- S. Bonnefoi et al., "Multi-Armed Bandit Model for Contention Resolution in LoRa Networks," *IEEE GLOBECOM Workshops*, 2019. (LoRa-MAB)

---

## 라이선스

MIT
