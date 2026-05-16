# System Model: Collision-Focused Slotted LoRaWAN Simulator

> 최종 수정: 2026-04-26

---

## 목차

0. [노테이션 요약](#0-노테이션-요약)
1. [네트워크 구성](#1-네트워크-구성)
2. [무선 채널 모델](#2-무선-채널-모델)
3. [노드 배치 모델](#3-노드-배치-모델)
4. [트래픽 모델](#4-트래픽-모델)
5. [리소스 구조 및 제공 부하](#5-리소스-구조-및-제공-부하)
6. [전송 결정 모델](#6-전송-결정-모델)
7. [충돌 및 링크 실패 모델](#7-충돌-및-링크-실패-모델)
8. [게이트웨이 피드백 모델](#8-게이트웨이-피드백-모델)
9. [성능 지표](#9-성능-지표)
10. [MAC 전략 모델](#10-mac-전략-모델) — Pure ALOHA · ADR-like · Retry-Aware · EXP3-MAB · Thompson MAB · Dual-MAB
11. [Q-learning 컨트롤러](#11-q-learning-컨트롤러)
12. [시뮬레이션 구현 및 최적화](#12-시뮬레이션-구현-및-최적화)
13. [모델 단순화 가정 및 한계](#13-모델-단순화-가정-및-한계)
14. [시뮬레이션 파라미터 요약](#14-시뮬레이션-파라미터-요약)

---

## 0. 노테이션 요약

### 0.1 기본 인덱스 (전체 문서 고정)

| 기호 | 의미 | 범위 |
|---|---|---|
| $i$ | 노드(ED) 인덱스 | $\{1, \ldots, N\}$ |
| $t$ | 시간 슬롯 인덱스 | $\{0, 1, \ldots, T_{\rm total}-1\}$ |
| $k$ | SF 인덱스 | $\{0,\ldots,5\}$ (SF7=0, SF12=5) |
| $c$ | 채널 인덱스 | $\{0,\ldots,N_{\rm ch}-1\}$ |
| $g$ | 게이트웨이 인덱스 | 현재 단일 GW → $g = 0$ 고정 |

### 0.2 집합 및 스칼라 상수

| 기호 | 정의 | 값/단위 |
|---|---|---|
| $N$ | 단말 노드 수 | 양의 정수 |
| $N_{\rm SF}$ | SF 종류 수 | 6 |
| $N_{\rm ch}$ | 직교 채널 수 | 설정값 (기본 3) |
| $N_{\rm res}$ | 총 직교 자원 수 | $N_{\rm SF} \times N_{\rm ch}$ |
| $T_{\rm warm}$ | 워밍업 슬롯 수 | — |
| $T_{\rm total}$ | 전체 슬롯 수 | — |
| $T_{\rm meas}$ | 측정 슬롯 수 | $T_{\rm total} - T_{\rm warm}$ |
| $T_{\rm epoch}$ | 에폭 길이 | 슬롯 수 |

### 0.3 채널 및 링크

| 기호 | 정의 | 단위 |
|---|---|---|
| $d_i$ | 노드 $i$와 GW 간 거리 | m |
| $d_0$ | 경로손실 기준 거리 | m |
| $\eta$ | 경로손실 지수 | 무차원 |
| $\mathrm{LPL}(d)$ | 거리 $d$에서의 경로손실 | dB |
| $P_{\rm tx}$ | 송신 전력 | dBm |
| $G_L$ | 안테나 이득/손실 합산 | dB |
| $N_{\rm floor}$ | 잡음 바닥 | dBm |
| $P_{\rm rx}(d)$ | 거리 $d$에서의 수신 전력 | dBm |
| $\mathrm{SNR}(d)$ | 거리 $d$에서의 평균 SNR | dB |
| $\mathrm{SNR}_{\rm inst}$ | 순시 SNR (Rayleigh 포락선 적용 후) | dB |
| $\gamma_{\rm rf}$ | Rayleigh 페이딩 전력 이득 ($\sim \mathrm{Exp}(1)$) | 무차원 |
| $\Delta_{\rm fade}$ | 페이드 마진 | dB |
| $k_i(t)$ | 슬롯 $t$에서 노드 $i$가 선택한 SF 인덱스 | $\{0,\ldots,5\}$ |
| $\theta_k$ | SF 인덱스 $k$의 수신 감도 임계값 | dB |
| $\mathrm{margin}_i(t)$ | 노드 $i$의 슬롯 $t$ SNR 여유분 | dB |
| $R_{\rm cell}$ | SF12 ($k=5$) 기준 셀 반경 | m |
| $\delta_{\rm edge}$ | 셀 가장자리 마진 | dB |

### 0.4 트래픽 및 큐

| 기호 | 정의 |
|---|---|
| $G$ | 목표 제공 부하, 직교 자원 단위로 정규화 |
| $p_{\rm arr}$ | 슬롯당 노드별 패킷 도착 확률 |
| $\Lambda_i(t)$ | 슬롯 $t$에서 노드 $i$의 패킷 도착 지시자 ($\in \{0,1\}$) |
| $q_i(t)$ | 슬롯 $t$ 시작 시 노드 $i$의 큐 길이 (패킷 수) |
| $m_i(t)$ | 슬롯 $t$에서 노드 $i$의 현재 연속 실패 횟수 (성공 시에만 리셋, 큐 모드 무관) |
| $b_i(t)$ | 슬롯 $t$에서 노드 $i$의 잔여 백오프 카운터 |
| $c_i(t)$ | 슬롯 $t$에서 노드 $i$가 선택한 채널 인덱스 |

### 0.5 성능 지표

| 기호 | 정의 |
|---|---|
| $S_i$ | 노드 $i$의 측정 구간 총 성공 횟수 |
| $A_i$ | 노드 $i$의 측정 구간 총 전송 시도 횟수 |
| $C_i$ | 노드 $i$의 측정 구간 총 충돌 횟수 |
| $A_{\rm total}$ | 측정 구간 총 전송 시도 횟수 ($= \sum_i A_i$) |
| $\mathrm{ASR}$ | 패킷 성공률 |
| $G_{\rm att}$ | 실현 제공 부하 |
| $\bar{Q}$ | 측정 구간 평균 노드별 백로그 |
| $\mathcal{F}$ | Jain's Fairness Index |

### 0.6 게이트웨이 피드백

| 기호 | 정의 |
|---|---|
| $n_c(\tau)$ | 슬롯 $\tau$의 채널 $c$ 기여 카운트 (모드에 따라 정의) |
| $\hat{G}_c(t)$ | 슬롯 $t$에서 채널 $c$의 추정 제공 부하 |
| $W_{\rm gw}$ | GW 부하 추정 롤링 윈도우 길이 (슬롯) |
| $\mathrm{bin}(\hat{G}_c)$ | 추정 부하 양자화 값 ($\in \{0,1,2\}$) |
| $\hat{g}_{\rm max}(t)$ | 슬롯 $t$에서 채널 전체의 최대 양자화 bin |

### 0.7 Q-learning

| 기호 | 정의 |
|---|---|
| $\mathcal{S}$ | 상태 공간 |
| $\mathcal{A}$ | 액션 공간 |
| $x_t$ | 슬롯 $t$에서의 Q-learning 상태 인덱스 ($\in \mathcal{S}$) |
| $a_t$ | 슬롯 $t$에서 선택한 액션 ($\in \mathcal{A}$) |
| $Q_i(x, a)$ | 노드 $i$의 Q-테이블 값 |
| $r_t$ | 슬롯 $t$에서의 보상 |
| $\alpha$ | Q-learning 학습률 |
| $\gamma$ | Q-learning 할인율 |
| $\epsilon_i(t)$ | 슬롯 $t$에서 노드 $i$의 탐색률 |
| $f_{\rm level}$ | 상태 벡터의 실패 레벨 차원 (연속 실패 횟수 $m_i$의 이산화) |
| $\hat{g}_{\rm max}$ | 상태 벡터의 최대 혼잡도 bin 차원 |
| $n_i^{\rm ep}(t)$ | 현재 에폭 내 노드 $i$의 누적 성공 횟수 (에폭 경계마다 초기화) |
| $\bar{n}^{\rm ep}(t)$ | 현재 에폭 내 전체 노드 평균 성공 횟수 |
| $\nu_i(t)$ | 에폭 내 성공 횟수 정규화값 ($= n_i^{\rm ep} / \max(1, \bar{n}^{\rm ep})$) |

---

## 1. 네트워크 구성

본 시뮬레이터는 단일 게이트웨이($g=0$)와 $N$개의 단말 노드(End Device, ED)로 구성된 단일-셀 LoRaWAN 업링크 시나리오를 모델링한다.

- 게이트웨이는 원점 $(0, 0)$에 고정 배치된다.
- 각 노드 $i \in \{1, \ldots, N\}$는 시뮬레이션 시작 시 배치되며, 이후 위치는 변하지 않는다(정적 노드).
- 모든 노드는 LoRa 변조를 사용하며, SF 인덱스 $k \in \{0,\ldots,5\}$ (SF7=0, SF12=5) 중 하나와 채널 $c \in \{0,\ldots,N_{\rm ch}-1\}$ 중 하나를 선택하여 업링크 전송을 수행한다.
- 다운링크 및 ACK는 없으며, 전송 성공 여부는 시뮬레이터 내부에서 결정론적으로 판정된다.

---

## 2. 무선 채널 모델

### 2.1 경로손실 모델

로그-거리 경로손실 모델(Log-distance Path Loss Model)을 사용한다:

$$
\mathrm{LPL}(d) = \mathrm{LPL}(d_0) + 10\,\eta \log_{10}\!\left(\frac{d}{d_0}\right) \quad [\mathrm{dB}]
$$

여기서 $d$는 노드와 게이트웨이 사이의 거리(m), $d_0$는 기준 거리(m), $\eta$는 경로손실 지수이다.

수신 전력(RSSI) 및 신호 대 잡음비(SNR)는 다음과 같이 계산한다:

$$
P_{\rm rx}(d) = P_{\rm tx} - G_L - \mathrm{LPL}(d) \quad [\mathrm{dBm}]
$$

$$
\mathrm{SNR}(d) = P_{\rm rx}(d) - N_{\rm floor} \quad [\mathrm{dB}]
$$

| 파라미터 | 기호 | 값 |
|---|---|---|
| 송신 전력 | $P_{\rm tx}$ | 14.0 dBm |
| 안테나 이득/손실 | $G_L$ | 0.0 dB |
| 잡음 바닥 | $N_{\rm floor}$ | −119.0 dBm |

> **비고:** 기본적으로 섀도잉 및 소규모 페이딩을 포함하지 않는다. 동일 거리의 노드들은 동일한 결정론적 SNR을 가진다. 소규모 페이딩은 [섹션 2.5](#25-rayleigh-페이딩-모델-선택적)에서 선택적으로 활성화할 수 있다.

### 2.2 라디오 프로파일

경로손실 파라미터는 아래 프로파일 중 하나로 지정된다:

| 프로파일 | $d_0$ (m) | $\mathrm{LPL}(d_0)$ (dB) | $\eta$ | 비고 |
|---|---|---|---|---|
| `short`   | 1   | 40.0   | 3.5  | 임의 (도심 단거리) |
| `medium`  | 1   | 37.0   | 3.0  | 임의 (교외 중거리) |
| `long`    | 1   | 34.0   | 2.5  | 임의 (농촌 장거리) |
| `lorasim` | 40  | 127.41 | 2.08 | LoRaSim 논문 기반 실측값 |

### 2.3 SF별 수신 감도 임계값

LoRa 복조기의 SF별 최소 SNR 임계값은 다음과 같이 정의된다:

| SF | 인덱스 $k$ | $\theta_k$ (dB) |
|---|---|---|
| SF7  | 0 | −7.5  |
| SF8  | 1 | −10.0 |
| SF9  | 2 | −12.5 |
| SF10 | 3 | −15.0 |
| SF11 | 4 | −17.5 |
| SF12 | 5 | −20.0 |

### 2.4 셀 반경

SF 인덱스 $k \in \{0,\ldots,5\}$를 사용할 때, 셀 가장자리 마진 $\delta_{\rm edge}$ (기본값 2.0 dB)를 포함한 최대 커버리지 반경은 다음과 같이 계산된다:

$$
R_{\rm cell}(k) = d_0 \cdot 10^{\,\dfrac{L_{\rm max}(k) - \mathrm{LPL}(d_0)}{10\,\eta}}
$$

$$
L_{\rm max}(k) = P_{\rm tx} - G_L - \bigl(N_{\rm floor} + \theta_k + \delta_{\rm edge}\bigr)
$$

이하 단순 $R_{\rm cell}$로 표기할 때는 $k = 5$ (SF12) 기준값을 의미한다:

$$
R_{\rm cell} \;:=\; R_{\rm cell}(5)
$$

### 2.5 Rayleigh 페이딩 모델 (선택적)

`enable_rayleigh_fading = True`로 활성화하면, 단독 전송 판정 시 결정론적 평균 SNR에 Rayleigh 페이딩 포락선(envelope)을 적용한다.

Rayleigh 채널의 전력 이득 $\gamma_{\rm rf} \sim \mathrm{Exp}(1)$ (평균 1 지수분포, 슬롯마다 독립 샘플링):

$$
\mathrm{SNR}_{\rm inst}(d_i) = \mathrm{SNR}(d_i) + \Delta_{\rm fade} + 10\log_{10}(\gamma_{\rm rf})
$$

여기서 $\Delta_{\rm fade}$ = `rayleigh_fade_margin_db` (기본값 10.0 dB)은 평균 SNR에 더해지는 운용 마진 오프셋이다. 이 마진은 실제 시스템에서 링크 예산에 확보하는 여유분에 해당하며, 극단적인 페이딩 이벤트로 인한 과도한 아웃레이지를 완화한다.

링크 성공 조건은 순시 SNR 기준으로 평가된다:

$$
\text{결과}_i(t) = \begin{cases}
\text{SUCCESS}    & \mathrm{SNR}_{\rm inst}(d_i) \geq \theta_{k_i(t)} \\
\text{FAIL\_LINK} & \mathrm{SNR}_{\rm inst}(d_i) < \theta_{k_i(t)}
\end{cases}
$$

**페이드 마진에 따른 아웃레이지 확률 근사** (평균 SNR 마진이 충분히 클 때):

$$
P_{\rm out} \approx 1 - \exp\!\left(-10^{-\Delta_{\rm fade}/10}\right)
$$

| $\Delta_{\rm fade}$ (dB) | $P_{\rm out}$ (근사) |
|---|---|
| 0 dB  | 63.2% |
| 3 dB  | 39.3% |
| 5 dB  | 26.4% |
| 10 dB | 9.5%  |
| 20 dB | 1.0%  |

기본값 $\Delta_{\rm fade} = 10\,\mathrm{dB}$에서 아웃레이지 약 9.5%. 비활성화 시 결정론적 하드 임계값 모델로 동작한다 ([섹션 7.2](#72-링크-성공-조건-하드-임계값-모델) 참조).

> **구현:** 배치 경로(`DecentralizedQLearningController`)에서는 `np.random.exponential(1.0, size=N_solo)`로 벡터 샘플링, 표준 경로에서는 `rng.expovariate(1.0)`으로 스칼라 샘플링한다. 두 경로 모두 `rayleigh_fade_margin_db`를 동일하게 적용한다.

---

## 3. 노드 배치 모델

### 3.1 배치 시드 분리

복수의 베이스라인을 동일한 조건에서 비교하기 위해, 노드 배치에 사용하는 난수 시드(`layout_seed`)와 시뮬레이션 동작에 사용하는 시드(`seed`)를 분리한다.

$$
\text{layout\_seed} = \text{const}, \quad \text{seed}_b = \text{base\_seed} + 10000 \times b
$$

베이스라인 $b$마다 시뮬레이션 RNG는 독립적이나, 노드 위치는 동일하게 유지된다.

### 3.2 링 배치 (Ring Layout)

모든 노드를 게이트웨이로부터 동일한 거리 $r_{\rm ring}$의 원 위에 배치한다:

$$
r_{\rm ring} = \rho \cdot R_{\rm cell}, \quad \rho \in (0, 1]
$$

$\rho$는 `fixed_distance_ratio` 파라미터 (기본값 0.5)이다. 각 노드의 각도 $\phi_i$는 균등 분포에서 무작위로 샘플링된다:

$$
\phi_i \sim \mathcal{U}(0,\, 2\pi), \quad i = 1, \ldots, N
$$

링 배치에서 모든 노드는 동일한 경로손실을 가지므로 SNR이 균일하게 분포된다. 거리 기반 SF 다양성을 활용하는 전략에게는 불리하므로, **통제 실험 목적** 이외에는 사용을 권장하지 않는다.

### 3.3 무작위 배치 (Random Layout)

노드를 반경 $R_{\rm cell}$의 원판 내에 균일하게 배치한다:

$$
r_i = R_{\rm cell}\sqrt{u_i}, \quad \phi_i = 2\pi v_i, \quad u_i,\, v_i \sim \mathcal{U}(0, 1)
$$

$$
x_i^{\rm pos} = r_i \cos\phi_i, \quad y_i^{\rm pos} = r_i \sin\phi_i
$$

이 변환은 면적에 비례한 균일 분포를 보장한다. 노드 간 거리 이질성이 확보되므로 **기본 배치 방식**으로 사용된다.

> **비고:** $x_i^{\rm pos}$, $y_i^{\rm pos}$는 노드 $i$의 위치 좌표로, Q-learning 상태 인덱스 $x_t$와 구분하기 위해 위첨자 pos를 사용한다.

---

## 4. 트래픽 모델

### 4.1 패킷 도착 과정

각 노드에서 패킷 도착은 독립적인 베르누이 과정을 따른다. 슬롯 $t$에서 노드 $i$의 패킷 도착 지시자 $\Lambda_i(t) \in \{0, 1\}$은 다음 확률을 따른다:

$$
\Pr\bigl[\Lambda_i(t) = 1\bigr] = p_{\rm arr} = \min\!\left(\frac{G \cdot N_{\rm res}}{N},\; 1\right)
$$

여기서 $G$는 목표 제공 부하, $N_{\rm res} = N_{\rm SF} \times N_{\rm ch}$는 총 직교 자원 수이다.

### 4.2 큐 모드

두 가지 큐 운영 방식이 지원된다:

**Accumulate 모드:**

$$
q_i(t) = q_i(t-1) + \Lambda_i(t)
$$

미전송 패킷은 큐에 누적된다. 연속 실패 횟수 $m_i(t)$는 동일 패킷에 대해 누적된다.

**Fresh 모드:**

$$
q_i(t) = \Lambda_i(t)
$$

매 슬롯마다 이전 패킷을 버리고 새 패킷을 0 또는 1개 받는다. 단, $m_i(t)$(**연속 실패 횟수**)는 패킷 경계에서 리셋하지 않는다. 전송 성공 시에만 $m_i \leftarrow 0$으로 초기화된다. 이는 accumulate 모드와 동일한 의미론이며, fresh 모드에서도 Q-learning 상태의 `failure_level` 차원이 유효한 정보를 갖게 한다.

---

## 5. 리소스 구조 및 제공 부하

### 5.1 직교 자원

LoRa의 다중 SF와 다중 채널이 만드는 직교 자원의 총 수는:

$$
N_{\rm res} = N_{\rm SF} \times N_{\rm ch} = 6 \times N_{\rm ch}
$$

기본 설정 $N_{\rm ch} = 3$에서 $N_{\rm res} = 18$개의 직교 자원이 존재한다.

> **ToA 단순화 주의:** 실제 LoRa에서 SF별 ToA는 SF7 ≈ 41 ms에서 SF12 ≈ 1318 ms로 최대 32배 차이가 난다. 본 시뮬레이터는 모든 SF가 동일하게 슬롯 1개를 점유한다고 가정한다. 자세한 내용은 [섹션 13.1](#131-toa-단순화)을 참조하라.

### 5.2 제공 부하 정의

$$
G = \frac{N \cdot p_{\rm arr}}{N_{\rm res}}
$$

$G = 1$은 슬롯당 자원 1개당 평균 1회 전송 시도에 해당하며, Slotted ALOHA의 이론적 처리량 최대점이다.

---

## 6. 전송 결정 모델

### 6.1 슬롯 구조

시뮬레이터는 이산 시간 슬롯 모델을 사용한다. 각 슬롯 $t$에서 노드 $i$는 다음 순서로 동작한다:

1. 패킷 도착: $\Lambda_i(t)$ 판정 → $q_i(t)$ 갱신
2. 백오프 감소: $b_i(t) \leftarrow \max\bigl(0,\; b_i(t-1) - 1\bigr)$
3. 전송 가능 조건: $q_i(t) > 0$ 이고 $b_i(t) = 0$ 인 경우에만 컨트롤러 호출
4. SF $k_i(t) \in \{0,\ldots,5\}$ 및 채널 $c_i(t) \in \{0,\ldots,N_{\rm ch}-1\}$ 선택
5. 충돌 및 링크 실패 판정
6. 게이트웨이 부하 추정 갱신
7. 결과 반영 및 학습

### 6.2 SF 초기화

시뮬레이션 시작 시 각 노드의 초기 SF 인덱스는 균등 분포에서 무작위로 샘플링된다:

$$
k_i(0) \sim \mathcal{U}\{0, 1, 2, 3, 4, 5\}
$$

ADR-like 및 Retry-Aware 컨트롤러는 `begin_run()` 단계에서 노드별 SNR 기반의 정적 SF로 덮어쓴다. Q-learning 컨트롤러는 랜덤 초기 SF에서 시작하여 탐색을 통해 최적 SF를 학습한다.

---

## 7. 충돌 및 링크 실패 모델

### 7.1 충돌 조건

동일 슬롯 $t$에서 동일 자원 $(k, c)$를 사용하는 노드가 2개 이상인 경우, 해당 자원의 모든 전송은 충돌로 판정된다:

$$
\text{COLLISION on } (k,c) \iff \bigl|\{i : k_i(t) = k,\; c_i(t) = c\}\bigr| \geq 2
$$

충돌 판정은 링크 품질과 무관하게 우선 적용된다.

> **단순화:** 실제 LoRa 수신기는 캡처 효과(capture effect)를 통해 신호 강도가 충분히 강한 패킷은 충돌 중에도 복조할 수 있다. 본 시뮬레이터는 이를 모델링하지 않는다.

### 7.2 링크 성공 조건 (하드 임계값 모델)

단독 전송(충돌 없음)에 대해 하드 임계값 기반의 결정론적 링크 성공 판정을 적용한다:

$$
\mathrm{margin}_i(t) = \mathrm{SNR}(d_i) - \theta_{k_i(t)}
$$

$$
\text{결과}_i(t) = \begin{cases}
\text{SUCCESS}    & \mathrm{margin}_i(t) \geq 0 \\
\text{FAIL\_LINK} & \mathrm{margin}_i(t) < 0
\end{cases}
$$

> **비고:** 이전 구현에서는 시그모이드 함수 $P_{\rm success} = 1/(1 + e^{-\mathrm{margin}})$를 사용한 확률적 모델을 적용하였으나, 섀도잉이 없는 결정론적 경로손실 모델과의 일관성을 위해 하드 임계값 모델로 교체하였다.

### 7.3 결과 코드 요약

| 코드 | 의미 | 조건 |
|---|---|---|
| `OUTCOME_IDLE` (0) | 전송 없음 | $q_i(t) = 0$ 또는 $b_i(t) > 0$ |
| `OUTCOME_SUCCESS` (1) | 성공 | 단독 전송 + $\mathrm{margin}_i(t) \geq 0$ |
| `OUTCOME_FAIL_COLLISION` (2) | 충돌 실패 | 동일 $(k,c)$ 복수 전송 |
| `OUTCOME_FAIL_LINK` (3) | 링크 실패 | 단독 전송 + $\mathrm{margin}_i(t) < 0$ |

노드는 ACK 부재 시 실패로 인식하나, **충돌 실패와 링크 실패를 구분하지 못한다**.

---

## 8. 게이트웨이 피드백 모델

### 8.1 부하 추정 방식

게이트웨이($g=0$)는 각 채널의 혼잡도를 추정하여 모든 노드에 브로드캐스트한다고 가정한다. 채널 $c$의 추정 제공 부하는 길이 $W_{\rm gw}$ 슬롯의 롤링 윈도우로 계산된다:

$$
\hat{G}_c(t) = \frac{1}{W_{\rm gw} \cdot N_{\rm SF}} \sum_{\tau=t-W_{\rm gw}+1}^{t} n_c(\tau)
$$

$W_{\rm gw} = 50$ (기본값)은 에폭 길이 $T_{\rm epoch}$와 독립적인 파라미터이다.

### 8.2 GW 관측 모드 (gw\_obs\_mode)

$n_c(\tau)$의 정의에 따라 두 모드로 분기된다:

**`attempt` 모드 (기본값) — GW가 캐리어 센싱으로 전체 트래픽을 파악한다고 가정:**

$$
n_c^{\rm att}(\tau) = \bigl|\{i : c_i(\tau) = c,\; \text{전송 시도}\}\bigr|
$$

충돌·링크 실패 패킷을 포함하므로 실제 채널 부하를 반영한다.

**`success` 모드 — GW가 디코딩 가능한 정보만 사용:**

$$
n_c^{\rm suc}(\tau) = \bigl|\{i : c_i(\tau) = c,\; \text{OUTCOME\_SUCCESS}\}\bigr|
$$

$n_c^{\rm suc}(\tau) \leq n_c^{\rm att}(\tau)$이므로 혼잡할수록 $\hat{G}_c$가 과소추정된다.

### 8.3 부하 양자화

추정 부하 $\hat{G}_c(t)$는 3단계 bin으로 양자화된다:

$$
\mathrm{bin}(\hat{G}_c) = \begin{cases}
0 \;\text{(Low)}  & \hat{G}_c < \ell_{\rm lo} \\
1 \;\text{(Mid)}  & \ell_{\rm lo} \leq \hat{G}_c \leq \ell_{\rm hi} \\
2 \;\text{(High)} & \hat{G}_c > \ell_{\rm hi}
\end{cases}
$$

**모드별 임계값:**

| 모드 | $\ell_{\rm lo}$ | $\ell_{\rm hi}$ | 상한 | 설계 기준 |
|---|---|---|---|---|
| `attempt` | 0.7 | 1.3 | 무제한 | $G=1$ 처리량 피크를 Mid 중심에 배치 |
| `success` | 0.33 | 0.67 | 1.0 | $[0,1]$ 균등 3분할, $1/e \approx 0.37$이 Mid 내 위치 |

Q-learning 상태 인코딩에 사용하는 최대 혼잡도 bin:

$$
\hat{g}_{\rm max}(t) = \max_{c \in \{0,\ldots,N_{\rm ch}-1\}} \mathrm{bin}\!\bigl(\hat{G}_c(t)\bigr) \;\in\; \{0,1,2\}
$$

---

## 9. 성능 지표

측정 구간 $[T_{\rm warm},\, T_{\rm total})$, 즉 $T_{\rm meas} = T_{\rm total} - T_{\rm warm}$ 슬롯에서 다음 지표를 수집한다.

### 9.1 패킷 성공률 (ASR)

$$
\mathrm{ASR} = \frac{\displaystyle\sum_{i=1}^{N} S_i}{\displaystyle\sum_{i=1}^{N} A_i}
$$

### 9.2 처리량

$$
\mathrm{Throughput} = \frac{\displaystyle\sum_{i=1}^{N} S_i}{T_{\rm meas}} \quad [\text{packets/slot}]
$$

### 9.3 충돌률

$$
\mathrm{CR} = \frac{\displaystyle\sum_{i=1}^{N} C_i}{\displaystyle\sum_{i=1}^{N} A_i}
$$

### 9.4 총 전송 시도 횟수

$$
A_{\rm total} = \sum_{i=1}^{N} A_i
$$

정책별 전송 적극성(aggressiveness)을 직접 비교하는 지표이다.

### 9.5 실현 제공 부하

$$
G_{\rm att} = \frac{A_{\rm total}}{N_{\rm res} \cdot T_{\rm meas}}
$$

### 9.6 평균 백로그

$$
\bar{Q} = \frac{1}{T_{\rm meas} \cdot N} \sum_{t=T_{\rm warm}}^{T_{\rm total}-1} \sum_{i=1}^{N} q_i(t) \quad [\text{packets/node}]
$$

### 9.7 공정성 (Jain's Fairness Index)

$$
\mathcal{F} = \frac{\left(\displaystyle\sum_{i=1}^{N} S_i\right)^2}{N \cdot \displaystyle\sum_{i=1}^{N} S_i^2} \;\in\; \left[\frac{1}{N},\; 1\right]
$$

$\mathcal{F} = 1$은 완전 공정, $\mathcal{F} = 1/N$은 완전 불공정을 의미한다.

### 9.8 Min/Max 비율

$$
\mathrm{MinMax} = \frac{\min_i S_i}{\max_i S_i}
$$

최악 노드와 최선 노드 간 성공 횟수 비율로, $\mathcal{F}$보다 극단값에 민감하다.

---

## 10. MAC 전략 모델

### 10.1 Pure ALOHA

매 슬롯마다 SF와 채널을 균등 무작위 선택한다:

$$
k_i(t) \sim \mathcal{U}\{0,\ldots,5\}, \quad c_i(t) \sim \mathcal{U}\{0,\ldots,N_{\rm ch}-1\}
$$

패킷이 있으면 무조건 전송한다. 학습 없음. 무작위 배치 환경에서 SNR이 낮은 먼 노드는 유효 SF가 제한되어 구조적으로 낮은 공정성을 보인다.

### 10.2 ADR-like

시뮬레이션 시작 시 각 노드의 $\mathrm{SNR}(d_i)$에 기반하여 SF를 정적으로 할당하고 고정한다:

$$
k_i^* = \min\bigl\{k \in \{0,\ldots,5\} : \mathrm{SNR}(d_i) \geq \theta_k + \delta_{\rm adr}\bigr\}
$$

$\delta_{\rm adr} = 1.0$ dB. 채널 $c_i(t)$는 매 슬롯 균등 무작위 선택. SF 다양성을 통해 Pure ALOHA 대비 충돌을 감소시키며, 거리 이질성이 있는 무작위 배치에서 공정성이 향상된다.

> **주의:** 링 배치에서는 모든 노드가 동일한 SNR을 가지므로 전체 노드에 동일한 $k^*$가 할당되어, Pure ALOHA보다 오히려 성능이 저하될 수 있다.

### 10.3 Retry-Aware (지수 백오프)

ADR-like와 동일한 SF 초기화 후, 실패 시 지수적 백오프를 적용한다:

$$
b_i(t) \sim \mathcal{U}\bigl\{0,\; W(m_i)\bigr\}
$$

$$
W(m) = \min\!\left(W_{\rm max},\; W_0 \cdot 2^{\max(m-1,\, 0)}\right)
$$

$W_0 = 2$ (기본 윈도우), $W_{\rm max} = 64$ (최대 윈도우), $m_i = m_i(t)$: 현재 연속 실패 횟수. 성공 시 즉시 $b_i = 0$ 초기화.

### 10.4 EXP3 MAB (LoRa-MAB)

각 노드는 $M = N_{\rm SF} \times N_{\rm ch}$개의 자원 팔(arm)에 대해 EXP3(Exponential-weight algorithm for Exploration and Exploitation) 알고리즘으로 독립적인 가중치를 유지한다.

**선택 확률 및 가중치 갱신:**

$$
\pi_{i,a}(t) = \frac{w_{i,a}(t)}{\displaystyle\sum_{a'} w_{i,a'}(t)}, \quad a_t \sim \pi_i(t)
$$

$$
w_{i,a}(t+1) = w_{i,a}(t) \cdot \exp\!\left(\frac{\eta_{\rm exp3} \cdot \hat{r}_t}{\pi_{i,a_t}(t)}\right), \quad \hat{r}_t = r_t \cdot \mathbf{1}[a = a_t]
$$

$\eta_{\rm exp3} = 0.1$ (기본값). 가중치는 팔을 선택했을 때만 갱신되며, 중요도 샘플링(importance sampling) 보정을 통해 비선택 팔의 추정 편향을 제거한다.

| 변형 키 | IDLE 팔 | 실패 보상 $r_{\rm fail}$ | 설명 |
|---|---|---|---|
| `lora_mab`          | 없음 | 0.0  | 기본 EXP3, 실패 시 무보상 |
| `lora_mab_neg`      | 없음 | −1.0 | 실패 페널티로 나쁜 팔 회피 가속 |
| `lora_mab_idle`     | 있음 | 0.0  | IDLE 팔 추가: 혼잡 시 전송 포기 학습 |
| `lora_mab_idle_neg` | 있음 | −1.0 | IDLE + 실패 페널티 조합 |

성공 보상 = +1 고정. 기반 알고리즘: LoRa-MAB (IEEE GLOBECOM Workshops 2019).

### 10.5 Thompson Sampling MAB

각 팔 $a$에 대해 Beta 사후분포를 유지하고, 매 슬롯 샘플링으로 가장 높은 값을 보인 팔을 선택한다:

$$
\hat{\mu}_{i,a}(t) \sim \mathrm{Beta}\!\bigl(\alpha_{i,a}(t),\; \beta_{i,a}(t)\bigr), \quad a_t = \arg\max_a \hat{\mu}_{i,a}(t)
$$

**파라미터 갱신:**

$$
(\alpha_{i,a},\, \beta_{i,a}) \leftarrow \begin{cases}
(\alpha_{i,a} + 1,\; \beta_{i,a})     & \text{결과} = \text{SUCCESS} \\
(\alpha_{i,a},\; \beta_{i,a} + 1)     & \text{결과} \in \{\text{COLLISION},\, \text{LINK\_FAIL}\}
\end{cases}
$$

초기값 $\alpha = \beta = 1$ (균등 사전분포, uniform prior). Beta-Bernoulli 켤레 쌍을 이용하므로 갱신이 닫힌형(closed-form)으로 처리된다. 희소 보상(sparse reward) 환경에서 EXP3 대비 수렴이 빠르다.

| 변형 키 | IDLE 팔 |
|---|---|
| `thompson_mab`      | 없음 |
| `thompson_mab_idle` | 있음 |

### 10.6 Dual-MAB (ACB + Resource-MAB + Backoff-MAB)

접근 제어와 자원 선택을 두 단계의 독립적인 MAB로 분리하여 분산 학습한다.

#### ACB (Access Class Barring)

슬롯마다 균등 확률 $u \sim \mathcal{U}(0,1)$를 추출하여 차단 여부를 결정한다:

$$
\text{barred}_i(t) = \begin{cases}
\text{True}  & u < b \\
\text{False} & \text{otherwise}
\end{cases}
$$

$b \in [0,1]$: ACB 차단 확률 파라미터. 대시보드 슬라이더로 조정 가능 (기본값 0.3). $b = 0$이면 ACB 비활성화 (`dual_mab_no_acb` 변형에 해당).

#### Resource-MAB ($\varepsilon$-greedy EMA)

차단되지 않은 노드는 $M = N_{\rm SF} \times N_{\rm ch}$개 팔 중 $\varepsilon$-greedy로 선택하고, 지수이동평균(EMA)으로 가중치를 갱신한다:

$$
Q_{i,a}^{\rm res}(t+1) = \begin{cases}
(1-\alpha_{\rm ema})\,Q_{i,a}^{\rm res}(t) + \alpha_{\rm ema}\cdot r_t & a = a_t \\
Q_{i,a}^{\rm res}(t) & a \neq a_t
\end{cases}
$$

#### Backoff-MAB

이전 전송 결과에 따라 백오프 길이를 학습하는 보조 MAB. ACB와 함께 네트워크 부하를 분산적으로 제어한다.

| 변형 키 | ACB $b$ | 비고 |
|---|---|---|
| `dual_mab`        | 0.3 (기본, 조정 가능) | 대시보드 슬라이더로 $b \in [0, 1]$ 설정 |
| `dual_mab_no_acb` | 0.0 (고정)           | 순수 Resource-MAB, ablation 기준선 |

---

## 11. Q-learning 컨트롤러

### 11.1 분산형 구조

각 노드 $i$는 독립적인 Q-테이블 $Q_i(x, a)$와 탐색 파라미터 $\epsilon_i$를 유지한다. 노드 간 Q-테이블 공유 없음. $x \in \mathcal{S}$는 Q-learning 상태 인덱스이며, $a \in \mathcal{A}$는 액션이다.

### 11.2 액션 공간

두 가지 액션 변형(`action_variant`)이 지원된다.

#### relative (기본값) — SF 상대 변화 × 채널

$$
|\mathcal{A}_{\rm rel}| = 1 + 3 \times N_{\rm ch}
$$

$N_{\rm ch} = 3$일 때 $|\mathcal{A}_{\rm rel}| = 10$:

| 액션 $a$ | 의미 |
|---|---|
| 0 | IDLE (전송 포기) |
| $1 \sim N_{\rm ch}$ | SF 유지 ($k_i$ 불변), 채널 $c = 0 \sim N_{\rm ch}-1$ |
| $N_{\rm ch}+1 \sim 2N_{\rm ch}$ | $k_i \leftarrow k_i + 1$, 채널 $c = 0 \sim N_{\rm ch}-1$ |
| $2N_{\rm ch}+1 \sim 3N_{\rm ch}$ | $k_i \leftarrow k_i - 1$, 채널 $c = 0 \sim N_{\rm ch}-1$ |

SF 변화 후 $k_i(t)$는 $\{0, \ldots, 5\}$ 범위로 클램핑된다.

#### absolute — SF 절대 선택 × 채널

$$
|\mathcal{A}_{\rm abs}| = 1 + N_{\rm SF} \times N_{\rm ch} = 1 + 6 \times N_{\rm ch}
$$

$N_{\rm ch} = 3$일 때 $|\mathcal{A}_{\rm abs}| = 19$. SF를 직접 지정하므로 SF 선택 수렴 속도는 빠르지만 탐색 공간이 크다.

#### Phase Learning 대기 액션 (PhaseQLearningController 전용)

`frame_size > 3`일 때 3가지 대기(wait) 액션이 추가된다:

$$
|\mathcal{A}_{\rm phase}| = 1 + 3 \times N_{\rm ch} + 3 \quad (\text{frame\_size} > 3)
$$

| 추가 액션 | 대기 슬롯 수 |
|---|---|
| wait\_short | $\lfloor F/4 \rfloor$ |
| wait\_medium | $\lfloor F/2 \rfloor$ |
| wait\_long | $F - 1$ |

$F$: `frame_size`. 노드는 대기 액션 선택 시 해당 슬롯만큼 `backoff_arr`에 등록되어 전송을 지연한다. 대기 중 보상은 0(중립).

### 11.3 상태 공간 변형

상태 벡터는 아래 후보 차원의 조합으로 구성된다:

| 차원 기호 | 의미 | 범위 | 크기 |
|---|---|---|---|
| $k_i(t)$ | 현재 SF 인덱스 | $\{0,\ldots,5\}$ | 6 |
| $f_{\rm level}$ | 연속 실패 레벨 | $\{0,1,2,3\}$ | 4 |
| $\hat{g}_{\rm max}(t)$ | 채널 전체 최대 혼잡도 bin | $\{0,\ldots,n_{\rm bins}-1\}$ | $n_{\rm bins}$ |
| $\Delta k_i$ | 직전 SF 변화 방향 | $\{-1, 0, +1\}$ | 3 |
| $c_i(t)$ | 현재 채널 인덱스 | $\{0,\ldots,N_{\rm ch}-1\}$ | $N_{\rm ch}$ |
| $\mathrm{bin}(\hat{G}_c)$ | 채널 $c$별 혼잡도 bin | $\{0,\ldots,n_{\rm bins}-1\}$ | $n_{\rm bins}^{N_{\rm ch}}$ |

**실패 레벨 $f_{\rm level}$** 이산화:

$$
f_{\rm level} = \begin{cases}
0 & m_i(t) = 0 \;\text{(직전 성공 또는 전송 없음)} \\
1 & m_i(t) = 1 \\
2 & 2 \leq m_i(t) \leq 3 \\
3 & m_i(t) \geq 4
\end{cases}
$$

**SF 변화 방향 $\Delta k_i$**: 직전 슬롯에 선택한 SF 변화량 $k_i(t) - k_i(t-1)$. IDLE이면 0. 상태 인코딩 시 $\Delta k_i + 1 \in \{0,1,2\}$로 변환한다.

**상태 변형 전체 목록** ($n_{\rm bins}=3$, $N_{\rm ch}=3$ 기준):

| 변형 키 | 포함 차원 | $|\mathcal{S}|$ | 플랫 인덱스 $x$ |
|---|---|---|---|
| `s2_compact` | $k_i$, $f_{\rm level}$, $\hat{g}_{\rm max}$ | $6 \times 4 \times 3 = 72$ | $k \cdot 12 + f \cdot 3 + g$ |
| `s3_no_sf` | $f_{\rm level}$, $\hat{g}_{\rm max}$ | $4 \times 3 = 12$ | $f \cdot 3 + g$ |
| `s4_no_fl` | $k_i$, $\hat{g}_{\rm max}$ | $6 \times 3 = 18$ | $k \cdot 3 + g$ |
| `s5_no_gmax` | $k_i$, $f_{\rm level}$ | $6 \times 4 = 24$ | $k \cdot 4 + f$ |
| `s6_sf_only` | $k_i$ | $6$ | $k$ |
| `s7_sf_gch` | $k_i$, $\hat{G}_{c=0..2}$ | $6 \times 3^3 = 162$ | $k \cdot 27 + g_0 \cdot 9 + g_1 \cdot 3 + g_2$ |
| `s8_sf_delta` | $\Delta k_i$, $f_{\rm level}$, $\hat{g}_{\rm max}$ | $3 \times 4 \times 3 = 36$ | $(\Delta k{+}1) \cdot 12 + f \cdot 3 + g$ |
| `s9_sf_delta_gmax` | $\Delta k_i$, $\hat{g}_{\rm max}$ | $3 \times 3 = 9$ | $(\Delta k{+}1) \cdot 3 + g$ |
| `s10_sf_delta_fl` | $\Delta k_i$, $f_{\rm level}$ | $3 \times 4 = 12$ | $(\Delta k{+}1) \cdot 4 + f$ |
| `s11_sf_delta_only` | $\Delta k_i$ | $3$ | $\Delta k + 1$ |
| `s12_sf_ch_gmax` | $k_i$, $c_i$, $\hat{g}_{\rm max}$ | $6 \times 3 \times 3 = 54$ | $k \cdot (N_{\rm ch} \cdot n_{\rm bins}) + c \cdot n_{\rm bins} + g$ |
| `s13_sf_ch` | $k_i$, $c_i$ | $6 \times 3 = 18$ | $k \cdot N_{\rm ch} + c$ |
| `s14_sf_delta_ch` | $\Delta k_i$, $c_i$ | $3 \times 3 = 9$ | $(\Delta k{+}1) \cdot N_{\rm ch} + c$ |

**설계 원칙 메모:**

- `s2_compact`: SF + 실패이력 + GW부하 — 기본 추천 변형. 채널 정보 없음.
- `s8~s11` (sf\_delta 계열): SF 절대값 대신 *변화 방향*을 상태로 사용. SF가 같아도 올라가는 중인지 내려가는 중인지 구분.
- `s12_sf_ch_gmax`: SF + 채널 + GW부하. 채널 전환 전략과 혼잡도 회피를 동시에 학습.
- `s13_sf_ch`: SF + 채널만. GW피드백 없이 순수 채널 선택 학습. 18 상태로 가장 빠르게 수렴.
- `s14_sf_delta_ch`: SF변화방향 + 채널. 9 상태로 최소 크기 채널 인식 변형.
- `s7_sf_gch`: 채널 *별* 혼잡도를 개별 차원으로 사용 — 상태 수 162개. 어느 채널이 혼잡한지 직접 식별 가능하지만 수렴 속도 느림.
- `s3_no_sf`: SF를 제거하면 거리 이질 환경에서 먼 노드/가까운 노드가 동일 상태를 공유 → 암묵적 평등화 효과 가능.

### 11.4 보상 함수 변형

노드는 실패 원인(충돌/링크 실패)을 구분할 수 없으므로 단일 실패 페널티를 사용한다.

#### v6\_signal (기본값) — 대칭 신호 보상

$$
r_t^{v6} = \begin{cases}
+1.0   & \text{결과}_i(t) = \text{SUCCESS} \\
-1.0   & \text{결과}_i(t) \in \{\text{FAIL\_COLLISION},\, \text{FAIL\_LINK}\} \\
-0.02  & \text{결과}_i(t) = \text{IDLE},\; q_i(t) > 0 \\
\phantom{-}0.0  & \text{결과}_i(t) = \text{IDLE},\; q_i(t) = 0
\end{cases}
$$

#### v7\_asymmetric, v8\_fairness, v9\_log\_thr — 복합(Composite) 보상

v7/v8/v9는 처리량·공정성을 동시에 최적화하는 **복합 보상(composite reward)**을 사용한다:

$$
r_t^{\rm cmp} = \begin{cases}
r_{\rm base} + \dfrac{r_{\rm fair}}{1 + \nu_i(t)} & \text{결과}_i(t) = \text{SUCCESS} \\[8pt]
r_{\rm fail}  & \text{결과}_i(t) \in \{\text{FAIL\_COLLISION},\, \text{FAIL\_LINK}\} \\[2pt]
-0.02  & \text{결과}_i(t) = \text{IDLE},\; q_i(t) > 0 \\
\phantom{-}0.0  & \text{결과}_i(t) = \text{IDLE},\; q_i(t) = 0
\end{cases}
$$

여기서 $\nu_i(t)$는 에폭 내 성공 횟수의 전체 평균 대비 정규화값이다:

$$
\nu_i(t) = \frac{n_i^{\rm ep}(t)}{\max\!\bigl(1,\; \bar{n}^{\rm ep}(t)\bigr)}, \quad \bar{n}^{\rm ep}(t) = \frac{1}{N}\sum_{j=1}^{N} n_j^{\rm ep}(t)
$$

$n_i^{\rm ep}(t)$: 현재 에폭 시작 이후 노드 $i$의 누적 성공 횟수 (에폭 경계 $T_{\rm epoch}$마다 초기화).

**설계 원리:**

- $\nu_i = 0$: 에폭 내 첫 번째 성공 → 보상 = $r_{\rm base} + r_{\rm fair}$ (최대)
- $\nu_i = 1$: 노드 $i$가 평균과 동일 → 보상 = $r_{\rm base} + r_{\rm fair}/2$
- $\nu_i \to \infty$: 과도한 성공 → 보상 → $r_{\rm base}$ (기저값으로 수렴)

기저값 $r_{\rm base} > 0$이 항상 존재하므로, 장기 시뮬레이션에서도 보상이 0으로 붕괴하지 않는다.

**변형별 파라미터:**

| 변형 | $r_{\rm base}$ | $r_{\rm fair}$ | $r_{\rm fail}$ | 성공 보상 범위 | 설계 목표 |
|---|---|---|---|---|---|
| `v7_asymmetric` | 0.5 | 0.5 | −1.0 | $[0.5,\; 1.0]$ | ASR·처리량·공정성 균형 |
| `v8_fairness`   | 0.3 | 0.7 | −1.0 | $[0.3,\; 1.0]$ | 공정성 우선 |
| `v9_log_thr`    | 0.6 | 0.4 | −0.7 | $[0.6,\; 1.0]$ | 탐색 허용, 처리량 중심 |

#### v10\_phase\_idle — 전략적 IDLE 특화 (standard 타입)

Phase Learning 및 혼잡 회피 학습에 최적화된 변형. IDLE을 양수 보상으로 장려하여 노드가 혼잡한 슬롯을 능동적으로 회피하도록 유도한다.

$$
r_t^{v10} = \begin{cases}
+1.0   & \text{결과}_i(t) = \text{SUCCESS} \\
-0.7   & \text{결과}_i(t) \in \{\text{FAIL\_COLLISION},\, \text{FAIL\_LINK}\} \\
+0.10  & \text{결과}_i(t) = \text{IDLE},\; q_i(t) > 0 \quad \text{(패킷 있음에도 대기)} \\
\phantom{+}0.0  & \text{결과}_i(t) = \text{IDLE},\; q_i(t) = 0
\end{cases}
$$

`idle_pkt = +0.10`: 패킷이 있어도 IDLE을 선택하면 양수 보상 → 불필요한 충돌 회피 학습 강화.  
`fail = −0.7`: 페널티 완화로 실패를 지나치게 두려워하지 않아 탐색 유지.

### 11.5 Q-update (TD(0))

$$
Q_i(x_t, a_t) \;\leftarrow\; Q_i(x_t, a_t) + \alpha \Bigl[ r_t + \gamma \max_{a'} Q_i(x_{t+1}, a') - Q_i(x_t, a_t) \Bigr]
$$

| 하이퍼파라미터 | 기호 | 기본값 |
|---|---|---|
| 학습률 | $\alpha$ | 0.1 |
| 할인율 | $\gamma$ | 0.9 |
| 초기 탐색률 | $\epsilon_0$ | 1.0 |
| 최소 탐색률 | $\epsilon_{\rm min}$ | 0.05 |
| 탐색률 감쇠 계수 | $\epsilon_{\rm decay}$ | 0.9995 |

탐색률 감쇠는 매 슬롯 노드별 독립적으로 적용된다:

$$
\epsilon_i(t+1) = \max\!\bigl(\epsilon_{\rm min},\; \epsilon_i(t) \cdot \epsilon_{\rm decay}\bigr)
$$

### 11.6 다음 상태 인코딩 시점

TD 업데이트에 사용하는 다음 상태 $x_{t+1}$은 **슬롯 $t$ 결과 판정 완료 후** 갱신된 $\hat{g}_{\rm max}(t)$를 사용하여 인코딩된다. GW 부하 추정은 충돌·링크 판정이 완료된 이후에 수행되며, 그 결과가 `gateway_info`로 전달된다. 이전 슬롯의 $\hat{g}_{\rm max}$를 사용할 경우 TD 타겟에 1-슬롯 시차 오류가 발생한다.

---

## 12. 시뮬레이션 구현 및 최적화

### 12.1 벡터화 배치 경로

`DecentralizedQLearningController`가 **배치 경로 지원 상태 변형** 중 하나를 사용하는 경우, 시뮬레이터는 per-node Python 루프 대신 numpy 기반 배치 경로를 활성화한다 (`SUPPORTS_BATCH = True`).

배치 경로를 지원하는 상태 변형 전체 목록:

`s2_compact`, `s3_no_sf`, `s4_no_fl`, `s5_no_gmax`, `s6_sf_only`, `s7_sf_gch`, `s8_sf_delta`, `s9_sf_delta_gmax`, `s10_sf_delta_fl`, `s11_sf_delta_only`, `s12_sf_ch_gmax`, `s13_sf_ch`, `s14_sf_delta_ch`

**배치 경로의 주요 최적화:**

| 연산 | 표준 경로 | 배치 경로 |
|---|---|---|
| Q-테이블 자료구조 | `defaultdict(dict)` × $N$ | dense `ndarray` $(N,\, |\mathcal{S}|,\, |\mathcal{A}|)$ |
| 상태 인코딩 | tuple 생성 × $N$ | 정수 산술 벡터 연산 |
| $\epsilon$-greedy 탐색 | Python `random()` × $N$ | `np.random` 1회 호출 |
| TD 업데이트 | Python 스칼라 루프 × $N$ | numpy 인덱싱 + 벡터 연산 |
| 패킷 도착 | `rng.random()` × $N$ | `np_rng.random(N)` 1회 |
| 충돌 판정 | Python `Counter` | `np.unique` |
| 통계 집계 | Python 조건문 루프 | numpy boolean indexing |

`_encode_batch()` 메서드는 `state_variant` 키를 기준으로 디스패치되어 각 변형에 맞는 플랫 인덱스 $x_t$를 반환한다.

**속도 향상:** $N=60$, $T_{\rm total}=20{,}000$ 기준 표준 경로 대비 약 **3.2배** 향상.

### 12.2 G/N 스윕 병렬화

G/N 스윕 실험은 각 (베이스라인, $G$값) 조합을 독립적인 잡(job)으로 분리하여 멀티프로세싱으로 병렬 실행한다. Windows 환경의 Streamlit과의 호환성을 위해 `spawn` 컨텍스트를 명시한다.

$$
\text{speedup} \approx \min(P_{\rm cpu},\; N_{\rm jobs})
$$

$P_{\rm cpu}$: 가용 CPU 코어 수. $N=60$, 30-point G-sweep 기준 실측 **약 4.6배** 향상.

---

## 13. 모델 단순화 가정 및 한계

### 13.1 ToA 단순화

실제 LoRa 전송에서 SF별 ToA는 페이로드 크기와 대역폭에 따라 크게 달라진다. 예시: 50 byte 페이로드, BW = 125 kHz:

| SF | 인덱스 $k$ | ToA (근사) | SF7 대비 배율 |
|---|---|---|---|
| SF7  | 0 | ~41 ms   | 1× |
| SF8  | 1 | ~72 ms   | ~1.8× |
| SF9  | 2 | ~144 ms  | ~3.5× |
| SF10 | 3 | ~247 ms  | ~6× |
| SF11 | 4 | ~495 ms  | ~12× |
| SF12 | 5 | ~1318 ms | ~32× |

**본 시뮬레이터의 가정:** 모든 SF ($k=0,\ldots,5$)에서 전송은 동일하게 **슬롯 1개**를 점유한다.

**논문 작성 시:** "슬롯 길이를 SF12 기준 최대 ToA로 설정한다"고 명시하면 된다. Limitation으로 명시 권장.

**영향:** SF12 ($k=5$) 노드의 채널 점유 불이익이 과소평가된다. 실제 환경에서 SF12 노드는 본 시뮬레이션보다 더 불공정한 상황에 처할 가능성이 있다.

### 13.2 기타 단순화 가정

| 가정 | 내용 |
|---|---|
| 섀도잉 없음 | 로그-정규 섀도잉 미포함. 경로손실은 결정론적 |
| 소규모 페이딩 선택적 | `enable_rayleigh_fading=False`(기본)이면 결정론적 SNR. 활성화 시 $\gamma_{\rm rf} \sim \mathrm{Exp}(1)$ 적용, `rayleigh_fade_margin_db`(기본 10 dB)로 운용 마진 확보 |
| 캡처 효과 없음 | 충돌 시 신호 강도와 무관하게 모든 패킷 실패 |
| 단일 게이트웨이 | $g=0$ 고정. 복수 GW 및 다이버시티 수신 미지원 |
| 완전 직교 SF | $k$가 다르면 충돌 없음 (실제로는 near-far 간섭 가능) |
| 다운링크/ACK 없음 | 노드는 성공 여부를 외부 신호 없이 판단 불가 |
| 정적 노드 | 이동성 없음 |
| 동일 송신 전력 | TX power 제어 미지원 ($P_{\rm tx} = 14$ dBm 고정) |

---

## 14. 시뮬레이션 파라미터 요약

| 파라미터 | 기호/키 | 기본값 | 설명 |
|---|---|---|---|
| 노드 수 | $N$ | 60 | 단말 노드 수 |
| 목표 제공 부하 | $G$ | 1.0 | 정규화 제공 부하 |
| 채널 수 | $N_{\rm ch}$ | 3 | 직교 채널 수 ($c \in \{0,\ldots,N_{\rm ch}-1\}$) |
| 총 슬롯 수 | $T_{\rm total}$ | 20,000 | 전체 시뮬레이션 길이 |
| 워밍업 슬롯 | $T_{\rm warm}$ | 5,000 | 측정 제외 초기 구간 |
| 에폭 슬롯 | $T_{\rm epoch}$ | 500 | 시계열 집계 및 카운터 초기화 주기 |
| 거리 비율 | $\rho$ | 0.5 | 링 배치 반경 비율 |
| 큐 모드 | `queue_mode` | `accumulate` | 패킷 큐 운영 방식 |
| 노드 배치 | `layout` | `random` | 노드 공간 배치 방식 |
| GW 관측 모드 | `gw_obs_mode` | `attempt` | GW 부하 추정 기반 패킷 범위 |
| GW 윈도우 | $W_{\rm gw}$ | 50 | 부하 추정 롤링 윈도우 (슬롯) |
| 셀 가장자리 마진 | $\delta_{\rm edge}$ | 2.0 dB | 셀 반경 계산 여유분 |
| ADR 마진 | $\delta_{\rm adr}$ | 1.0 dB | ADR-like SF 할당 여유분 |
| 라디오 프로파일 | `profile_name` | `short` | 경로손실 파라미터 집합 ($\eta$, $\mathrm{LPL}(d_0)$, $d_0$) |
| 배치 시드 | `layout_seed` | 42 | 모든 베이스라인 공유, 재현성 보장 |
| 시뮬레이션 시드 | `seed` | 베이스라인별 독립 | 패킷 도착·충돌 판정 RNG |
| 상태 변형 | `state_variant` | `s2_compact` | Q-learning 상태 공간 구성 |
| 보상 변형 | `reward_variant` | `v6_signal` | Q-learning 보상 함수 |
| Rayleigh 페이딩 | `enable_rayleigh_fading` | False | 소규모 페이딩 활성화 여부 |
| 페이드 마진 | `rayleigh_fade_margin_db` ($\Delta_{\rm fade}$) | 10.0 dB | Rayleigh 페이딩 평균 SNR 오프셋 (대시보드 슬라이더 0–30 dB) |
| Dual-MAB ACB 차단 확률 | `dual_mab_b` ($b$) | 0.3 | Dual-MAB Access Class Barring 파라미터 (대시보드 슬라이더 0–1) |
