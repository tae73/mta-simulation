# Part 1: 데이터 생성 프로세스 (DGP) 설계 방법론

---
**방법론 문서 시리즈**
- **Part 1: DGP 설계** (현재 문서)
- [Part 2: Attribution 방법론](./Methodology_02_Attribution_Methods.md)
- [Part 3: 실험 설계 및 결과](./Methodology_03_Experimental_Design.md)
- [프로젝트 계획서](./Marketing_Attribution_Project_Plan.md)
---

## 1. 개요 및 연구 동기

### 1.1 MTA 데이터의 구조적 한계

Multi-Touch Attribution(MTA) 연구에서 가장 근본적인 문제는 **ground truth의 부재**이다. 실 데이터에서는 각 마케팅 채널의 "진정한 기여도"를 알 수 없으므로, 방법론 간 정확도 비교가 원칙적으로 불가능하다.

현존하는 공개 MTA 데이터셋을 진단하면:

| 데이터셋 | 규모 | 한계 |
|---------|------|------|
| Criteo Attribution (2018) | 16.5M events | 모든 feature 해시/익명화 → 도메인 해석 불가 |
| GA4 BigQuery Public | 해석 가능 | traffic_source가 first-touch 고정 → 세션별 시퀀스 재구성 불가 |
| CriteoPrivateAd (2025) | 대규모 | uuid 일 단위 리셋 → 멀티데이 여정 구성 불가 |
| Hillstrom / Criteo Uplift | 인과추론용 | treatment가 단일(binary) → 멀티채널 attribution 시나리오 부적합 |

**"해석 가능한 채널명 + ground truth + 충분한 규모"를 동시에 갖춘 공개 MTA 데이터셋은 존재하지 않는다.**

### 1.2 시뮬레이션 접근의 장점

직접 설계한 DGP(Data Generation Process)를 사용하면:

1. **Ground truth 보유**: 각 채널의 전환 기여도를 정확히 알고 있으므로, 방법론 간 정량 비교가 가능하다.
2. **파라미터 통제**: 시간 감쇠, 교차 영향, confounding 강도 등을 독립적으로 조절하여 방법론의 강건성을 실험할 수 있다.
3. **해석 가능성**: Display, Paid Search, Email 등 실무에서 사용하는 채널명으로 결과를 직관적으로 해석할 수 있다.
4. **규모 확장**: 1K~100K 유저 규모를 자유롭게 조절하여 데이터 규모 민감도를 테스트할 수 있다.

### 1.3 학술적 기반

본 DGP는 세 편의 학술 논문을 통합한다:

| 구성 요소 | 채택 소스 | 구체적 활용 |
|----------|----------|-----------|
| 기본 골격 (sequential dependence, user heterogeneity) | Du et al. (2019) | 유저-채널-시간 구조, 세그먼트별 이질성 |
| 시간 감쇠 함수 (temporal decay) | Shender et al. (2023) | 채널별 차별화된 $f(t - t_j)$ |
| Poisson 기반 전환 모델 | Shender et al. (2023) | Log-linear intensity → survival analysis 연결 |
| 채널 간 교차 영향 (cross-influence) | CDA (2025) | 시너지/대체 효과를 전이 확률과 전환 확률에 반영 |

---

## 2. 전환 강도 모델 (Conversion Intensity Model)

### 2.1 핵심 수식: Log-Linear Poisson Intensity

유저 $i$의 시점 $t$에서의 전환 강도(conversion intensity)는 다음 log-linear 모델로 정의된다:

$$\log(\lambda_i(t)) = \underbrace{\alpha_0}_{\text{baseline}} + \underbrace{\sum_{j=1}^{J_i} \beta_{k(j)} \cdot \exp\left(-\frac{\Delta t_j}{\tau_{k(j)} \times 24}\right)}_{\text{채널 효과 + 시간 감쇠}} + \underbrace{\sum_{(s,t) \in \mathcal{C}} \delta_{st} \cdot f_s(\Delta t) \cdot \mathbb{1}[s \prec t]}_{\text{교차 채널 영향}} + \underbrace{\eta_{\text{seg}(i)}}_{\text{유저 이질성}}$$

여기서:

| 기호 | 의미 | 범위 |
|------|------|------|
| $\alpha_0$ | 기저 전환 강도 (baseline) | $-5.625$ (보정값) |
| $\beta_k$ | 채널 $k$의 전환 효과 계수 | $[0.3, 1.2]$ |
| $\tau_k$ | 채널 $k$의 시간 감쇠 반감기 (일) | $[1, 14]$ |
| $\Delta t_j$ | 관측 시점에서 터치포인트 $j$까지의 경과 시간 (시간) | $\geq 0$ |
| $\delta_{st}$ | 소스 채널 $s$ → 타깃 채널 $t$의 시너지 강도 | $[0.2, 0.4]$ |
| $\eta_{\text{seg}}$ | 유저 세그먼트별 이질성 효과 | $[-0.3, 0.5]$ |

전환 확률은 Poisson process의 실현(realization)으로 계산된다:

$$P(\text{conversion}) = 1 - \exp(-\lambda_i(t)) = 1 - \exp\left(-\exp\left(\log(\lambda_i(t))\right)\right)$$

> **구현**: `part1_simulation/dgp/conversion_model.py` — `compute_log_intensity()` (line 94), `intensity_to_conversion_prob()` (line 138)

### 2.2 채널 효과 항 (Channel Effect Term)

각 터치포인트 $j$의 채널 $k(j)$는 관측 시점까지의 시간 경과에 따라 지수적으로 감쇠하는 효과를 가진다:

$$\text{channel\_effect} = \sum_{j=1}^{J_i} \beta_{k(j)} \cdot \exp\left(-\frac{t_{\text{obs}} - t_j}{\tau_{k(j)} \times 24}\right)$$

시간 감쇠 함수 $f_k(\Delta t) = \exp(-\Delta t / (\tau_k \times 24))$는 Shender et al. (2023)의 프레임워크에서 채택하였다. 이 설계의 핵심은 **채널별 차별화된 감쇠 속도**이다:

- **Paid Search** ($\tau = 1$일): 구매 의도가 명확한 검색 광고는 노출 직후 효과가 급속히 감소
- **Display** ($\tau = 14$일): 브랜드 인지 광고는 장기간에 걸쳐 서서히 효과가 지속
- **Email** ($\tau = 5$일): 프로모션 이메일은 중간 정도의 유효 기간

이 차이는 마케팅 실무의 직관과 일치한다. Display 광고를 본 후 2주 뒤에도 브랜드를 기억할 수 있지만, 검색 광고의 클릭 유도 효과는 당일에 집중된다.

### 2.3 교차 채널 영향 항 (Cross-Channel Influence Term)

CDA (2025) 프레임워크에 기반한 교차 채널 시너지 효과이다:

$$\text{cross\_influence} = \sum_{(s,t) \in \mathcal{C}} \delta_{st} \cdot \exp\left(-\frac{t_{\text{obs}} - t_s^{\text{first}}}{\tau_s \times 24}\right) \cdot \mathbb{1}[\text{pos}(s) < \text{pos}(t)]$$

**활성화 조건**: 소스 채널 $s$가 타깃 채널 $t$보다 **먼저** 여정에 등장해야 한다. 또한 시너지 강도는 소스 채널의 시간 감쇠를 따라 감소한다.

| 소스 → 타깃 | $\delta$ | 마케팅 근거 |
|------------|---------|-----------|
| Display → Paid Search | 0.4 | Display 광고로 브랜드 인지 → 이후 검색 시 클릭률 상승 |
| Social → Email | 0.3 | Social 콘텐츠로 관심 유발 → Email 구독/오픈률 상승 |
| Organic Search → Direct | 0.2 | 검색으로 사이트 발견 → 이후 직접 방문 (URL 기억) |

이 세 쌍은 마케팅에서 흔히 관찰되는 **보조 효과(assist effect)**를 모델링한다. Display 광고가 직접 전환을 만들지 못하더라도($\beta = 0.3$으로 낮음), Paid Search의 전환을 시너지($\delta = 0.4$)로 증폭시킨다.

> **구현**: `part1_simulation/dgp/conversion_model.py` — `compute_cross_influence_bonus()` (line 43)

### 2.4 유저 이질성 항 (User Heterogeneity Term)

Du et al. (2019)의 유저 이질성 프레임워크를 세그먼트(segment) 수준으로 구현한다:

$$\text{heterogeneity} = \eta_{\text{seg}(i)}$$

이 항은 **동일한 채널 여정을 겪더라도 유저 특성에 따라 전환 확률이 다른** 현실을 반영한다. 동시에, 이질성이 **채널 선택과 상관**되어 있으므로 **selection bias (confounding)**를 생성한다.

| 세그먼트 | $\eta$ | 전환 성향 | 선호 채널 | Confounding 구조 |
|---------|--------|----------|----------|-----------------|
| New | $-0.3$ | 낮음 | Display, Social | 인지 채널 ↔ 낮은 전환율 |
| Exploratory | $0.0$ | 중간 | Organic Search | 중립 |
| Loyal | $+0.5$ | 높음 | Email, Direct | 직접 채널 ↔ 높은 전환율 |

**핵심 confounding 구조**: Loyal 유저는 Email/Direct를 선호하면서 동시에 전환 성향이 높다($\eta = +0.5$). 따라서 Email과 Direct의 전환율이 높게 관측되지만, 이 중 상당 부분은 채널 효과가 아닌 **유저 특성 효과**이다. Causal inference 방법론은 이 confounding을 보정해야 하며, correlational 방법론은 이를 분리하지 못한다.

---

## 3. 유저 여정 생성 파이프라인

전체 DGP는 5단계 파이프라인으로 구성된다:

> **구현**: `part1_simulation/dgp/generate_data.py` — `generate_all_journeys()` (line 312)

### 3.1 Step 1: 세그먼트 할당

100,000명의 유저를 3개 세그먼트에 다항분포(Multinomial)로 할당한다:

$$\text{segment\_counts} \sim \text{Multinomial}(N, \pi)$$

여기서 $\pi = (0.5, 0.3, 0.2)$는 (New, Exploratory, Loyal)의 비율이다.

**실제 생성 결과:**

| 세그먼트 | 비율 | 실제 유저 수 | 전환율 |
|---------|------|-----------|-------|
| New | 50% | 49,707 | 1.76% |
| Exploratory | 30% | 30,243 | 3.08% |
| Loyal | 20% | 20,050 | 2.49% |

Exploratory의 전환율이 Loyal보다 높은 것은, Loyal이 여정 길이가 짧아($\text{Geometric}(0.5) + 1$) 채널 효과 누적이 적기 때문이다. $\eta$의 이점이 짧은 여정으로 상쇄된다.

> **구현**: `part1_simulation/dgp/user_segments.py` — `assign_segments()` (line 60)

### 3.2 Step 2: 여정 길이 분포

각 세그먼트별로 Geometric 분포 기반의 여정 길이를 생성한다:

$$L_i = \min\left(\max\left(1,\ \text{Geometric}(p_{\text{seg}}) + \text{offset}_{\text{seg}}\right),\ 20\right)$$

| 세그먼트 | $p$ | offset | 기대 여정 길이 | 특성 |
|---------|-----|--------|-------------|------|
| New | 0.25 | 1 | $1/0.25 + 1 = 5$ | 다양한 탐색, 중-장 여정 |
| Exploratory | 0.20 | 2 | $1/0.20 + 2 = 7$ | 가장 긴 여정, 넓은 채널 탐색 |
| Loyal | 0.50 | 1 | $1/0.50 + 1 = 3$ | 짧고 직접적인 여정 |

상한은 20 터치포인트로 설정하였다. 이는 현실적인 디지털 마케팅 여정의 상한선을 반영한다.

**실제 생성 결과:**
- 평균 여정 길이: 5.18
- 중앙값: 4.0
- 왜도(skewness): 1.78 (right-skewed, 현실적)

> **구현**: `part1_simulation/dgp/user_segments.py` — `generate_journey_length()` (line 20)

### 3.3 Step 3: 위치 의존적 채널 전이 행렬

채널 시퀀스는 **위치 의존적 1차 Markov chain**으로 생성된다. 여정 내 현재 위치 비율($\text{position\_ratio} = \text{step} / \text{total\_length}$)에 따라 세 개의 서로 다른 $7 \times 7$ 전이 확률 행렬을 사용한다.

**설계 근거**: 실제 마케팅 퍼널은 **인지(Awareness) → 고려(Consideration) → 전환(Conversion)** 단계를 거친다. 여정 초반에는 Display/Social 등 upper-funnel 채널이 지배적이고, 후반에는 Paid Search/Direct 등 lower-funnel 채널로 수렴한다.

#### Early 단계 ($\text{position} \leq 30\%$)

Upper-funnel 채널(Display, Social)이 높은 유지율과 교차 전이율을 보인다. Organic Search와 Referral이 탐색 대상으로 등장한다.

|  | Display | Social | Organic | Paid | Email | Referral | Direct |
|--|---------|--------|---------|------|-------|----------|--------|
| **Display** | 0.10 | 0.25 | 0.30 | 0.08 | 0.07 | 0.15 | 0.05 |
| **Social** | 0.20 | 0.10 | 0.25 | 0.08 | 0.12 | 0.18 | 0.07 |
| **Organic** | 0.15 | 0.20 | 0.15 | 0.12 | 0.10 | 0.18 | 0.10 |
| **Paid** | 0.10 | 0.12 | 0.20 | 0.15 | 0.13 | 0.15 | 0.15 |
| **Email** | 0.12 | 0.18 | 0.20 | 0.10 | 0.15 | 0.15 | 0.10 |
| **Referral** | 0.15 | 0.22 | 0.25 | 0.08 | 0.10 | 0.10 | 0.10 |
| **Direct** | 0.12 | 0.15 | 0.25 | 0.12 | 0.13 | 0.13 | 0.10 |

#### Mid 단계 ($30\% < \text{position} \leq 70\%$)

Organic Search가 중심 허브가 된다. Email과 Referral이 중간 퍼널 채널로 비중 증가. Paid Search가 점차 등장한다.

|  | Display | Social | Organic | Paid | Email | Referral | Direct |
|--|---------|--------|---------|------|-------|----------|--------|
| **Display** | 0.08 | 0.15 | 0.25 | 0.18 | 0.12 | 0.12 | 0.10 |
| **Social** | 0.12 | 0.08 | 0.22 | 0.15 | 0.18 | 0.15 | 0.10 |
| **Organic** | 0.10 | 0.12 | 0.10 | 0.22 | 0.15 | 0.16 | 0.15 |
| **Paid** | 0.05 | 0.08 | 0.15 | 0.18 | 0.18 | 0.12 | 0.24 |
| **Email** | 0.08 | 0.10 | 0.18 | 0.20 | 0.12 | 0.12 | 0.20 |
| **Referral** | 0.10 | 0.15 | 0.20 | 0.18 | 0.15 | 0.07 | 0.15 |
| **Direct** | 0.08 | 0.10 | 0.18 | 0.22 | 0.18 | 0.12 | 0.12 |

#### Late 단계 ($\text{position} > 70\%$)

전환 채널(Paid Search, Email, Direct)이 지배적이다. Display/Social의 비중이 크게 감소한다.

|  | Display | Social | Organic | Paid | Email | Referral | Direct |
|--|---------|--------|---------|------|-------|----------|--------|
| **Display** | 0.05 | 0.08 | 0.12 | 0.30 | 0.20 | 0.08 | 0.17 |
| **Social** | 0.05 | 0.05 | 0.10 | 0.28 | 0.25 | 0.10 | 0.17 |
| **Organic** | 0.03 | 0.05 | 0.07 | 0.30 | 0.20 | 0.10 | 0.25 |
| **Paid** | 0.02 | 0.03 | 0.05 | 0.20 | 0.25 | 0.08 | 0.37 |
| **Email** | 0.03 | 0.05 | 0.08 | 0.28 | 0.12 | 0.07 | 0.37 |
| **Referral** | 0.03 | 0.07 | 0.10 | 0.28 | 0.22 | 0.05 | 0.25 |
| **Direct** | 0.02 | 0.03 | 0.05 | 0.30 | 0.25 | 0.05 | 0.30 |

> **구현**: `part1_simulation/dgp/channel_config.py` — `_build_early_matrix()` (line 33), `_build_mid_matrix()` (line 59), `_build_late_matrix()` (line 85)

### 3.4 Step 4: 타임스탬프 생성

터치포인트 간 시간 간격은 지수분포에서 생성된다:

$$\Delta t_{j} \sim \text{Exponential}(\lambda = 48\text{h})$$

첫 터치포인트는 $t_0 = 0$이며, 이후 누적합으로 단조 증가하는 타임스탬프를 생성한다:

$$t_j = \sum_{l=1}^{j} \Delta t_l$$

평균 48시간(2일) 간격은 디지털 마케팅에서 일반적인 유저 활동 빈도를 반영한다.

> **구현**: `part1_simulation/dgp/generate_data.py` — `assign_timestamps()` (line 88)

### 3.5 Step 5: 전환 결정

각 유저의 마지막 터치포인트 시점에서 log-intensity를 계산하고, Bernoulli 추출로 전환 여부를 결정한다:

$$\text{converted}_i \sim \text{Bernoulli}\left(1 - \exp\left(-\exp\left(\log(\lambda_i(t_{\text{last}}))\right)\right)\right)$$

> **구현**: `part1_simulation/dgp/generate_data.py` — `compute_conversions()` (line 132)

---

## 4. $\alpha_0$ 보정 (Calibration)

### 4.1 문제

$\alpha_0$는 기저 전환 강도로서, 전체 전환율을 직접적으로 결정한다. DGP 파라미터($\beta$, $\tau$, $\delta$, $\eta$)를 먼저 설정한 후, 원하는 전환율(2~3%)이 달성되도록 $\alpha_0$를 보정해야 한다.

### 4.2 Binary Search 알고리즘

$$\alpha_0^* = \arg\min_{\alpha_0 \in [-10, 0]} \left| \text{conv\_rate}(\alpha_0) - 0.025 \right|$$

1. 탐색 범위: $[\alpha_0^{\text{lo}}, \alpha_0^{\text{hi}}] = [-10.0, 0.0]$
2. 각 반복에서 $n_{\text{cal}} = 5{,}000$ 유저로 DGP를 실행하여 전환율 추정
3. 전환율이 $[0.02, 0.03]$ 범위에 진입하면 종료
4. 최대 20회 반복

### 4.3 보정 결과

$$\alpha_0^* = -5.625$$

이 값에서 100,000 유저 대상 전환율은 **2.305%** (2,305명 전환)로, 목표 범위 내에 수렴하였다.

$\alpha_0 = -5.625$의 의미: 채널 효과, 교차 영향, 유저 이질성이 모두 0일 때의 전환 확률은 $1 - \exp(-\exp(-5.625)) = 0.36\%$이다. 이 **기저 전환율(base conversion rate)**은 광고 없이도 발생하는 자연 전환을 나타낸다.

> **구현**: `part1_simulation/dgp/generate_data.py` — `calibrate_alpha_0()` (line 244)

---

## 5. 7개 채널 정의

각 채널의 DGP 파라미터와 마케팅 근거를 정리한다.

| 채널 | $\beta$ (전환 효과) | $\tau$ (반감기, 일) | Funnel | 마케팅 근거 |
|------|------|------|--------|-----------|
| **Display** | 0.3 | 14 | Upper | 브랜드 인지 광고. 직접 전환 효과 약하나 장기 기억 지속. Display→PaidSearch 시너지($\delta=0.4$)의 소스 역할 |
| **Social** | 0.4 | 3 | Upper | SNS 광고/콘텐츠. Display보다 약간 강한 효과이나 빠른 피로도(3일). Social→Email 시너지($\delta=0.3$)의 소스 |
| **Organic Search** | 0.5 | 7 | Mid | 자연 검색. 적극적 정보 탐색 의도 반영. 중간 강도, 중간 지속. Organic→Direct 시너지($\delta=0.2$)의 소스 |
| **Paid Search** | 1.2 | 1 | Lower | 검색 광고. **가장 강한 직접 전환 효과**. 구매 의도 명확, 효과 즉시 소멸. Display→PaidSearch 시너지의 타깃 |
| **Email** | 0.8 | 5 | Mid | 마케팅 이메일. 높은 효과(기존 관계), 중간 지속. Social→Email 시너지의 타깃 |
| **Referral** | 0.5 | 7 | Mid | 추천/제휴 트래픽. 신뢰도 높으나 빈도 낮음. Organic Search와 동일 파라미터 |
| **Direct** | 0.7 | 2 | Lower | 직접 방문(URL/북마크). 구매 의도 높음, 빠른 감쇠. Organic→Direct 시너지의 타깃 |

**설계 원칙:**
- $\beta$ 크기 순서: Paid Search(1.2) > Email(0.8) > Direct(0.7) > Organic Search = Referral(0.5) > Social(0.4) > Display(0.3)
- 이 순서는 마케팅 퍼널의 하단으로 갈수록 전환 효과가 강해지는 실무 직관을 반영
- 반감기는 채널 특성을 반영: 인지 채널(Display)은 길고, 전환 채널(Paid Search)은 짧음

---

## 6. Ground Truth 계산

Ground truth는 두 가지 독립적인 방법으로 계산한다. 이를 통해 "정답"의 정의 자체에 대한 민감도를 검토할 수 있다.

### 6.1 Ground Truth A: 강도 분해 (Intensity Decomposition)

**원리**: 전환한 유저의 log-intensity를 각 항의 기여분으로 분해한다.

각 전환 유저 $i$에 대해:

1. **채널 효과 분해**: 터치포인트 $j$의 채널 $k$에 대해 $\beta_k \cdot f_k(\Delta t_j)$를 해당 채널에 할당
2. **교차 영향 분배**: $\delta_{st} \cdot f_s(\Delta t)$를 소스와 타깃에 $\beta$ 비율로 분배
   - 소스 할당: $\delta_{st} \cdot f_s(\Delta t) \times \frac{\beta_s}{\beta_s + \beta_t}$
   - 타깃 할당: $\delta_{st} \cdot f_s(\Delta t) \times \frac{\beta_t}{\beta_s + \beta_t}$
3. **이질성 분배**: $|\eta|$를 각 채널의 기여 비중에 비례하여 분배
4. **음수 클램핑**: 각 채널 기여도의 하한을 0으로 설정 ($\eta < 0$인 New 세그먼트에서 발생 가능)
5. **정규화**: 전체 전환 유저의 합산 기여도를 채널별로 집계 후 합 = 1.0으로 정규화

> **구현**: `part1_simulation/evaluation/ground_truth.py` — `_decompose_user_intensity()` (line 37), `compute_ground_truth_intensity()` (line 97)

### 6.2 Ground Truth B: 반사실적 Shapley (Counterfactual Shapley)

**원리**: 7개 채널의 모든 부분집합(coalition) $S \subseteq \{1, ..., 7\}$에 대해 DGP를 재실행하여 전환율을 계산하고, 이로부터 Shapley value를 산출한다.

$$v(S) = \frac{1}{N} \sum_{i=1}^{N} P(\text{conv}_i \mid \text{channels} \in S)$$

여기서 "channels $\in S$"란 유저 $i$의 여정에서 $S$에 포함되지 않는 채널의 터치포인트를 제거하고, 남은 터치포인트만으로 전환 확률을 재계산하는 것을 의미한다.

$$\phi_k = \sum_{S \subseteq N \setminus \{k\}} \frac{|S|! \cdot (n - |S| - 1)!}{n!} \cdot \left[v(S \cup \{k\}) - v(S)\right]$$

- 7개 채널 → $2^7 = 128$ coalition → 정확 계산(exact computation)이 가능
- 계산 효율을 위해 5,000명 서브샘플 사용

> **구현**: `part1_simulation/evaluation/ground_truth.py` — `compute_ground_truth_shapley()` (line 204)

### 6.3 Ground Truth 비교

| 채널 | GT-A (강도 분해) | GT-B (반사실적 Shapley) | 차이 |
|------|----------------|---------------------|------|
| Paid Search | **0.311** | **0.289** | -0.022 |
| Email | 0.226 | 0.213 | -0.013 |
| Direct | 0.154 | 0.130 | -0.024 |
| Organic Search | 0.132 | 0.154 | +0.022 |
| Referral | 0.080 | 0.078 | -0.002 |
| Display | 0.060 | 0.086 | +0.026 |
| Social | 0.037 | 0.049 | +0.012 |

**주요 관찰:**

1. **랭킹 일치**: 두 GT 모두 Paid Search > Email을 1, 2위로 산출. 상위 2개 채널에 대한 합의는 강하다.
2. **GT-B가 upper-funnel에 더 많은 크레딧 부여**: Display(0.060 → 0.086), Organic(0.132 → 0.154). 이는 Shapley의 "한계 기여(marginal contribution)" 개념이 시너지 소스 채널의 간접 효과를 더 잘 포착하기 때문이다.
3. **GT-A가 lower-funnel에 더 많은 크레딧 부여**: Paid Search(0.311 → 0.289), Direct(0.154 → 0.130). 강도 분해는 $\beta$ 값이 높은 채널에 직접적으로 더 많은 크레딧을 배분한다.

**실험에서의 사용**: GT-A를 primary benchmark로, GT-B를 Shapley 계열 방법론의 보조 benchmark로 사용한다.

---

## 7. 데이터 검증

생성된 데이터가 현실적이고 내부적으로 일관성 있는지 검증한다.

### 7.1 기본 통계

| 지표 | 값 | 검증 기준 |
|------|---|----------|
| 총 유저 수 | 100,000 | 설정값 |
| 전환 유저 수 | 2,305 | — |
| 전환율 | 2.305% | 목표 범위 [2%, 3%] 내 |
| 평균 여정 길이 | 5.18 | — |
| 중앙값 여정 길이 | 4.0 | — |
| 여정 길이 왜도 | 1.78 | > 0.5 (right-skewed) |
| 타임스탬프 위반 | 0건 | 단조 증가 보장 |

### 7.2 채널 빈도 분포

| 채널 | 빈도 비율 |
|------|----------|
| Organic Search | 19.1% |
| Direct | 15.5% |
| Paid Search | 15.2% |
| Email | 14.5% |
| Social | 13.9% |
| Display | 11.9% |
| Referral | 9.8% |

Organic Search가 최빈 채널인 것은 mid-stage 전이 행렬에서 허브 역할을 하기 때문이다. Display가 가장 낮은 것은 early-stage 이후 빠르게 감소하는 전이 패턴을 반영한다.

### 7.3 세그먼트별 전환율

| 세그먼트 | 전환율 | 해석 |
|---------|-------|------|
| Exploratory | 3.08% | 가장 긴 여정(~7 터치포인트) → 채널 효과 최대 누적 |
| Loyal | 2.49% | 높은 $\eta$(+0.5)이나 짧은 여정(~3)이 상쇄 |
| New | 1.76% | 낮은 $\eta$(-0.3) + 중간 여정(~5) |

---

## 8. 출력 데이터 형식

### 8.1 Journey DataFrame (Long Format)

한 행이 한 유저의 한 터치포인트를 나타내는 long format이다:

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `user_id` | int64 | 유저 고유 식별자 |
| `segment` | category | New / Exploratory / Loyal |
| `touchpoint_idx` | int64 | 여정 내 터치포인트 순서 (0-indexed) |
| `channel` | category | 7개 채널 중 하나 |
| `timestamp` | float64 | 여정 시작 시점으로부터 경과 시간 (시간 단위) |
| `is_last_touchpoint` | bool | 마지막 터치포인트 여부 |
| `converted` | bool | 전환 여부 (유저 레벨, 모든 행 동일) |
| `journey_length` | int64 | 유저의 총 터치포인트 수 |
| `conversion_intensity` | float64 | $\log(\lambda_i(t))$ 값 |

### 8.2 Ground Truth JSON

```json
{
  "ground_truth_A": {
    "method": "intensity_decomposition",
    "channel_credits": {"Paid Search": 0.311, "Email": 0.226, ...},
    "channel_ranking": ["Paid Search", "Email", "Direct", ...]
  },
  "ground_truth_B": {
    "method": "counterfactual_shapley",
    "channel_credits": {"Paid Search": 0.289, "Email": 0.213, ...},
    "channel_ranking": ["Paid Search", "Email", "Organic Search", ...]
  },
  "dgp_parameters": { ... },
  "data_statistics": { ... }
}
```

> **출력 파일**: `data/simulation/journeys.parquet`, `data/simulation/ground_truth.json`, `data/simulation/summary_stats.json`

---

## 9. DGP 확장 가능성

본 DGP는 다음과 같은 실험적 변형을 지원한다 (실험 04, 05, 06에서 활용):

| 변형 | 방법 | 실험 목적 |
|------|------|----------|
| 교차 영향 제거 | 모든 $\delta = 0$ | 시너지 효과 감지 능력 평가 |
| 시간 감쇠 제거 | 모든 $\tau = 1000$일 | 시간 감쇠 무시 시 방법론 성능 변화 |
| 유저 이질성 제거 | 모든 $\eta = 0$ | confounding 제거 시 causal 방법론 이점 소멸 여부 |
| Confounding 강도 조절 | $\eta$ spread $\{0.2, 0.8, 2.0\}$ | correlational vs causal 격차 확대 |
| 기저 전환율 조절 | $\alpha_0 \in \{-3.5, -2.5, -1.8\}$ | incremental vs total attribution 괴리 |
