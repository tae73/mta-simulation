# Part 2: Attribution 방법론 상세 (18개 방법론)

---
**방법론 문서 시리즈**
- [Part 1: DGP 설계](./Methodology_01_DGP_Design.md)
- **Part 2: Attribution 방법론** (현재 문서)
- [Part 3: 실험 설계 및 결과](./Methodology_03_Experimental_Design.md)
- [프로젝트 계획서](./Marketing_Attribution_Project_Plan.md)
---

## 1. 방법론 분류 체계

본 프로젝트에서 구현한 18개 attribution 방법론은 6개 카테고리, 5단계 인과성 수준으로 분류된다 (5-tier 학술 정합성 반영, Methodology_05 § 4.2 참조).

| 카테고리 | 방법론 | 인과성 수준 | 핵심 질문 |
|---------|--------|-----------|----------|
| Rule-based | Last/First/Linear/Time Decay/Position-Based | 없음 (휴리스틱) — pure correlational | 크레딧 배분 규칙은? |
| Statistical | Markov Chain (1st, 2nd order) | 약함 — correlational + structural assumption | 채널 전이 구조의 중요도는? |
| Game-theoretic | Shapley Value (exact, model-based) | 없음 (heuristic, no baseline) — pure correlational | 공정한 기여도 배분은? |
| Deep Learning | LSTM+Attention, Transformer | 없음 — pure correlational (predictive only) | 예측에 중요한 터치포인트는? |
| **Causal (outcome model)** | Survival/Poisson (BackElim/AICPE/**Shapley**), Incremental Shapley, CAMTA | 중간 — regression adjustment, outcome model only (propensity 미보정) | 모델 baseline 차감한 incremental? |
| **Causal (debiased)** | IPW, Doubly Robust, DML | 높음 — propensity-based ATE estimator | 교란 보정 후 채널의 인과 효과? |

> **분류 정정 (5-tier, Methodology_05 § 4.2 참조)**: 이전 "Causal (incremental)" 라벨을 **"Causal (outcome model)"** 로 정정. Survival/Poisson 3종 + Incremental Shapley + CAMTA 는 baseline 차감 ($\hat\lambda(N) - \hat\lambda(\emptyset)$) 을 통해 "incremental" 의도를 가지지만, **technical class 로는 regression adjustment / outcome model 단독 — propensity model 추정 없음**. Shender 본인이 *"observational data → correlational, experimental data → causal"* 로 명시 (논문 Section 1). Strict debiased causal estimator 는 IPW/DR/DML 만 — propensity model 명시 추정.

**모든 방법론의 출력 형식**: 채널별 기여도 dict (`channel_name → float`, 합 = 1.0)

---

## 2. Rule-Based Attribution (5종)

가장 단순한 휴리스틱 방법론이다. 도메인 지식이나 모델 학습 없이, 여정 내 터치포인트의 위치만으로 크레딧을 배분한다.

> **구현**: `part1_simulation/models/rule_based.py`

### 2.1 Last Click

**수식**: 유저 $i$의 마지막 터치포인트 채널에 100% 크레딧을 부여한다.

$$\text{credit}(c, i) = \mathbb{1}[c = \text{channel}(J_i)]$$

여기서 $J_i$는 유저 $i$의 마지막 터치포인트 인덱스이다.

**특성**: Google Analytics의 기본 설정이며, 실무에서 가장 널리 사용된다. lower-funnel 채널(Paid Search, Direct)을 체계적으로 과대평가하고, upper-funnel 채널(Display, Social)을 과소평가한다.

**결과**: MAE = 0.038, Kendall's $\tau$ = 0.81

### 2.2 First Click

**수식**: 유저 $i$의 첫 터치포인트 채널에 100% 크레딧을 부여한다.

$$\text{credit}(c, i) = \mathbb{1}[c = \text{channel}(1)]$$

**특성**: 인지 단계 채널의 중요성을 강조하는 관점. 세그먼트의 시작 채널에 편향되므로, 우리 DGP에서는 New 유저의 Display/Social과 Exploratory의 Organic Search에 과도한 크레딧이 배분된다.

**결과**: MAE = 0.158 (최악), Kendall's $\tau$ = -0.29 (역상관). **Ground truth 랭킹을 거의 반전시킨다.**

### 2.3 Linear

**수식**: 유저 $i$의 모든 터치포인트에 동일한 크레딧을 배분한다.

$$\text{credit}(c, i) = \frac{\text{count}(c \text{ in journey}_i)}{|\text{journey}_i|}$$

**특성**: 채널 간 차별을 두지 않으므로 "가장 안전한" 휴리스틱이지만, 채널별 효과 차이($\beta$)를 포착하지 못한다. 빈도가 높은 Organic Search에 과도한 크레딧이 배분되는 경향.

**결과**: MAE = 0.056, Kendall's $\tau$ = 0.43

### 2.4 Time Decay

**수식**: 전환 시점에 가까운 터치포인트에 지수적으로 높은 가중치를 부여한다.

$$w_j = 2^{-\Delta t_j / \tau_{\text{decay}}}$$

여기서 $\Delta t_j = t_{\text{last}} - t_j$는 마지막 터치포인트로부터의 경과 시간이고, $\tau_{\text{decay}} = 7$일이다.

유저 내에서 가중치를 정규화한 후, 채널별로 집계한다:

$$\text{credit}(c, i) = \frac{\sum_{j: \text{ch}(j)=c} w_j}{\sum_{j} w_j}$$

**특성**: DGP의 시간 감쇠 구조와 유사한 직관을 가지므로 rule-based 중에서 준수한 성능을 보인다. 다만, 단일 반감기(7일)를 모든 채널에 동일하게 적용하므로 채널별 감쇠 차이를 반영하지 못한다.

**결과**: MAE = 0.044, Kendall's $\tau$ = 0.81

### 2.5 Position-Based (40/20/40)

**수식**: 첫 터치포인트에 40%, 마지막 터치포인트에 40%, 나머지에 20%를 균등 배분한다.

$$\text{credit}(c, j) = \begin{cases} 0.4 & j = 1 \text{ (first)} \\ 0.4 & j = J \text{ (last)} \\ 0.2 / (J-2) & \text{otherwise} \end{cases}$$

특수 케이스: 터치포인트 1개 → 100%, 2개 → 50%/50%.

**특성**: First Click과 Last Click의 절충안. 인지 채널과 전환 채널 모두에 크레딧을 부여하지만, 40%의 비율은 임의적이다.

**결과**: MAE = 0.065, Kendall's $\tau$ = 0.33

### 2.6 Rule-Based 종합

| 방법론 | MAE | Kendall's $\tau$ | Top-3 정확도 | 주요 편향 |
|--------|-----|-----------------|-------------|----------|
| Last Click | 0.038 | 0.81 | 100% | Paid Search +0.047, Display -0.028 |
| First Click | 0.158 | -0.29 | 0% | Organic +0.273, Paid Search -0.311 |
| Linear | 0.056 | 0.43 | 67% | Paid Search -0.139, Organic +0.056 |
| Time Decay | 0.044 | 0.81 | 100% | Paid Search -0.108, Referral +0.014 |
| Position-Based | 0.065 | 0.33 | 67% | Paid Search -0.133, Organic +0.093 |

**핵심 발견**: Last Click과 Time Decay가 가장 높은 $\tau$ (0.81)를 달성한다. 이는 우리 DGP에서 lower-funnel 채널의 $\beta$가 실제로 높기 때문이다. 그러나 Last Click은 lower-funnel을 과대평가하고 upper-funnel을 과소평가하는 **체계적 편향**이 존재한다.

---

## 3. Markov Chain Attribution

채널 전이 구조의 중요도를 기반으로 기여도를 산출한다.

> **구현**: `part1_simulation/models/markov.py`

### 3.1 모델 구조

유저 여정을 채널 상태 간의 전이 시퀀스로 모델링한다.

**1차 Markov Chain (order=1)**

상태 공간: $\mathcal{S} = \{\text{Start}, \text{7 channels}, \text{Conversion}, \text{Null}\}$ (10개 상태)

관측된 여정으로부터 전이 확률 행렬 $P$를 추정한다:

$$P(s_j \mid s_{j-1}) = \frac{\text{count}(s_{j-1} \to s_j)}{\text{count}(s_{j-1} \to \cdot)}$$

**2차 Markov Chain (order=2)**

상태 공간: Start + 7 channels + 49 channel pairs + Conversion + Null

$$P(s_j \mid s_{j-2}, s_{j-1}) = \frac{\text{count}(s_{j-2}, s_{j-1} \to s_j)}{\text{count}(s_{j-2}, s_{j-1} \to \cdot)}$$

2차에서는 희소한 쌍에 대해 Laplace smoothing ($\alpha = 0.001$)을 적용한다.

### 3.2 Removal Effect

Markov attribution의 핵심 개념은 **Removal Effect**이다. 채널 $c$를 제거했을 때 전체 전환 확률이 얼마나 감소하는지를 측정한다.

$$\text{RE}(c) = P(\text{Conversion} \mid \text{full chain}) - P(\text{Conversion} \mid \text{chain without } c)$$

채널 $c$를 제거한다는 것은, 전이 행렬에서 $c$로의 모든 전이를 Null(흡수 상태)로 리다이렉트하는 것이다.

**흡수 Markov Chain 풀이:**

전이 행렬을 transient 상태($Q$)와 흡수 상태($R$)로 분해한다:

$$P = \begin{pmatrix} Q & R \\ 0 & I \end{pmatrix}$$

기본 행렬(fundamental matrix): $N = (I - Q)^{-1}$

흡수 확률: $B = N \times R$

$B$의 Start 행, Conversion 열 원소가 전체 전환 확률이다.

### 3.3 기여도 계산

Removal Effect를 정규화하여 기여도를 산출한다:

$$\text{credit}(c) = \frac{\text{RE}(c)}{\sum_{k} \text{RE}(k)}$$

### 3.4 결과

| 방법론 | MAE | Kendall's $\tau$ | 특성 |
|--------|-----|-----------------|------|
| Markov (1st order) | 0.066 | 0.33 | 기본 전이 구조만 반영 |
| Markov (2nd order) | 0.061 | 0.52 | 쌍별 전이까지 포착, 약간 개선 |

**한계**: Markov는 채널 빈도와 전이 패턴에 의존하므로, Paid Search(-0.139~-0.146 bias)를 체계적으로 과소평가한다. Paid Search의 높은 $\beta$를 전이 확률만으로는 포착할 수 없기 때문이다.

---

## 4. Shapley Value Attribution

협력 게임 이론의 Shapley value를 MTA에 적용한다. 7개 채널이므로 $2^7 = 128$개 coalition에 대해 정확 계산이 가능하다.

> **구현**: `part1_simulation/models/shapley.py`

### 4.1 수학적 기초

Shapley value는 다음 네 가지 공리를 만족하는 유일한 배분 방식이다:

1. **효율성(Efficiency)**: $\sum_k \phi_k = v(N)$ (전체 가치를 모두 배분)
2. **대칭성(Symmetry)**: 동일한 기여를 하는 채널은 동일한 크레딧
3. **영 플레이어(Null Player)**: 기여가 없는 채널은 크레딧 0
4. **선형성(Linearity)**: 가치 함수의 합에 대한 Shapley = Shapley의 합

$$\phi_k = \sum_{S \subseteq N \setminus \{k\}} \frac{|S|! \cdot (n - |S| - 1)!}{n!} \cdot \left[v(S \cup \{k\}) - v(S)\right]$$

여기서 $v(S)$는 coalition $S$의 가치 함수이며, $n = 7$이다.

### 4.2 Value Function 정의

**Version A — Conversion Rate Value Function:**

$$v_A(S) = \frac{\text{전환 유저 중 모든 채널이 } S\text{에 포함되는 유저 수}}{\text{모든 채널이 } S\text{에 포함되는 유저 수}}$$

이 정의는 "coalition $S$의 채널만으로 구성된 여정의 전환율"을 의미한다. 문제: 특정 coalition은 관측 데이터가 매우 적어 추정이 불안정하다.

**Version B — Model-Based Value Function (채택):**

1. 유저별 7차원 binary 채널 존재 벡터를 feature로 사용
2. Logistic regression을 학습: $P(\text{conv} \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$
3. Coalition $S$의 가치: $S$에 포함되지 않은 채널의 feature를 0으로 마스킹한 예측값의 평균

$$v_B(S) = \frac{1}{N} \sum_{i=1}^{N} \hat{P}(\text{conv}_i \mid x_{i,k} = 0 \text{ for } k \notin S)$$

Model-based 방식이 더 안정적이며, 본 프로젝트의 primary Shapley 결과로 채택하였다.

### 4.3 결과

| Shapley 변형 | MAE | Kendall's $\tau$ | Top-3 정확도 |
|-------------|-----|-----------------|-------------|
| Model-based (Version B) | 0.035 | 0.90 | 67% |

Shapley (model-based)는 **비인과적 방법론 중 최고의 순위 일치도($\tau = 0.90$)**를 달성한다. 그러나 일부 채널(Display, Paid Search)에서 수 퍼센트포인트의 bias가 존재한다.

---

## 5. Deep Learning Sequence Models

여정을 시퀀스 데이터로 처리하여 전환을 예측하고, 예측에 대한 각 터치포인트의 기여도를 추출한다.

### 5.1 LSTM + Attention

> **구현**: `part1_simulation/models/lstm_attention.py`

#### 아키텍처

```
Input: [channel_one_hot(7), time_delta(1), position(1)] = 9-dim per step
  ↓
LSTM(hidden_size=64, batch_first=True)
  ↓
Dot-product Attention: α_j = softmax(h_j^T · W_a · h_T)
  ↓
Context vector: c = Σ α_j · h_j
  ↓
Dense(64 → 1) + Sigmoid → P(conversion)
```

**입력 표현**: 각 터치포인트를 9차원 벡터로 인코딩한다.
- 채널 one-hot (7차원)
- 시간 간격 ($\Delta t / 24$, 일 단위 정규화)
- 여정 내 위치 비율 ($\text{idx} / \text{length}$)

**Attention 메커니즘**: 마지막 hidden state $h_T$와 각 step의 hidden state $h_j$ 간 dot-product attention을 계산한다.

$$\alpha_j = \frac{\exp(h_j^T W_a h_T)}{\sum_{l} \exp(h_l^T W_a h_T)}$$

$$c = \sum_j \alpha_j \cdot h_j$$

**학습 설정:**
- Loss: Binary Cross-Entropy with class weighting (비전환 유저가 97.7%)
- Optimizer: Adam
- Early stopping: patience = 7, validation AUC 기준
- 패딩: 최대 길이 20으로 zero-padding

#### Attribution 추출 (3가지 방법)

**방법 1 — Attention Weights:**

각 터치포인트의 attention 가중치 $\alpha_j$를 추출하여 채널별로 집계한다.

$$\text{credit}(c) = \frac{\sum_{i \in \text{converted}} \sum_{j: \text{ch}(j)=c} \alpha_{ij}}{\sum_{i \in \text{converted}} \sum_{j} \alpha_{ij}}$$

결과: MAE = 0.036, $\tau$ = 0.71

**방법 2 — Leave-One-Out (LOO):**

각 터치포인트를 하나씩 마스킹하고 예측 변화를 측정한다.

$$\text{LOO}(j, i) = \hat{P}(\text{conv}_i \mid \text{full}) - \hat{P}(\text{conv}_i \mid \text{mask } j)$$

채널별로 집계: $\text{credit}(c) \propto \sum_{i, j: \text{ch}(j)=c} \text{LOO}(j, i)$

결과: MAE = 0.034, $\tau$ = 0.71, **Top-3 정확도 100%**

**방법 3 — SHAP DeepExplainer** (optional):

SHAP 라이브러리의 DeepExplainer를 사용하여 input feature 수준의 Shapley value를 계산한다.

#### LSTM 결과 요약

| Attribution 방법 | MAE | Kendall's $\tau$ | Top-3 정확도 |
|-----------------|-----|-----------------|-------------|
| Attention weights | 0.036 | 0.71 | 100% |
| LOO | 0.034 | 0.71 | 100% |

### 5.2 Transformer

> **구현**: `part1_simulation/models/transformer.py`

#### 아키텍처

```
Input: 9-dim per step → Linear(9, 64) → 64-dim
  ↓
Prepend learned [CLS] token + Positional Encoding (learned)
  ↓
TransformerEncoder(num_layers=2, nhead=2, d_model=64, d_ff=128)
  ↓
[CLS] token output → Dense(64 → 1) + Sigmoid → P(conversion)
```

**Positional Encoding**: 학습 가능한(learned) positional embedding을 사용한다. 최대 시퀀스 길이 21 (CLS + 20 터치포인트).

**Multi-Head Self-Attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

2개 head, $d_k = d_v = 32$ (64 / 2).

#### Attribution 추출

CLS 토큰과 각 터치포인트 간의 multi-head attention weight를 추출하여 채널별로 집계한다.

#### 결과

| 지표 | 값 |
|------|---|
| MAE | 0.119 |
| Kendall's $\tau$ | -0.33 |
| Top-3 정확도 | 0% |

**Transformer가 LSTM보다 크게 열세인 이유:**
1. **짧은 시퀀스**: 평균 여정 길이 5.18은 Transformer의 self-attention이 패턴을 학습하기에 짧음
2. **데이터 효율성**: 2,305명의 전환 유저만 학습에 사용 가능하여, 2-layer Transformer의 파라미터 수 대비 데이터 부족
3. **Inductive bias**: LSTM의 순차적 처리 bias가 마케팅 여정의 시간순서 특성과 자연스럽게 부합

---

## 6. Causal Inference 기반 MTA (6종)

Correlational 방법론(위 1~5)과 달리, confounding(selection bias)을 명시적으로 보정하여 채널의 **순증 전환 효과(incremental effect)**를 추정한다.

### 6.1 Incremental Shapley (Du et al. 2019)

> **구현**: `part1_simulation/models/causal/incremental_shapley.py`
> **논문**: "Causally Driven Incremental Multi Touch Attribution Using a Recurrent Neural Network" (ArXiv 1902.00215, AdKDD 2019)

#### 핵심 아이디어

전통적 Shapley는 **모든 전환**을 채널에 배분한다. 그러나 일부 전환은 광고 없이도 발생했을 **기저 전환(base conversion)**이다. Incremental Shapley는 **광고로 인해 추가된 전환(incremental conversion)**만 배분한다.

Du et al. 원 논문의 2-step 파이프라인을 따른다:
1. **Response Modeling**: 관측 데이터로 response model을 학습하여 $P(\text{conv} \mid \text{exposures})$ 추정
2. **Credit Allocation**: 학습된 모델로 incremental value function을 계산하고 Shapley Value로 배분

본 프로젝트에서는 원 논문의 RNN 대신 2-layer MLP를 response model로 사용한다. 원 논문의 핵심 기여인 **incremental conversion 분리 후 Shapley 배분** 프레임워크는 동일하게 적용한다.

#### 수학적 정의

**Step 1 — Response model 학습 (from data):**

관측 여정 데이터에서 유저별 feature를 추출하고 (채널 존재 여부, 터치 횟수, recency, 세그먼트, 여정 통계), 2-layer MLP를 학습한다:

$$\hat{P}(\text{conv}_i \mid \mathbf{x}_i) = \sigma(\text{MLP}(\mathbf{x}_i; \hat{\Omega}))$$

**Step 2 — 기저 전환율 추정 (learned):**

모든 채널 feature를 0으로 마스킹하여 "광고 노출 없음" 상태의 예측값을 계산한다:

$$\hat{P}_{\text{base}} = \frac{1}{N} \sum_{i=1}^{N} \hat{P}(\text{conv}_i \mid \mathbf{x}_i^{\emptyset})$$

여기서 $\mathbf{x}_i^{\emptyset}$는 모든 채널 관련 feature가 0인 벡터이다.

**Step 3 — Incremental value function:**

Coalition $S$에 대해, $S$에 포함되지 않는 채널의 feature를 0으로 마스킹한 예측값에서 기저 전환율을 뺀다:

$$v_{\text{inc}}(S) = \frac{1}{N} \sum_{i=1}^{N} \left[\hat{P}(\text{conv}_i \mid \mathbf{x}_i^{S}) - \hat{P}_{\text{base}}\right]$$

**Step 4 — Shapley on incremental values:**

$$\phi_k^{\text{inc}} = \sum_{S \subseteq N \setminus \{k\}} \frac{|S|! \cdot (n - |S| - 1)!}{n!} \cdot \left[v_{\text{inc}}(S \cup \{k\}) - v_{\text{inc}}(S)\right]$$

정규화: $\text{credit}(k) = \phi_k^{\text{inc}} / \sum_l \phi_l^{\text{inc}}$

#### 결과

| 지표 | 값 |
|------|---|
| MAE | 0.028 |
| RMSE | 0.030 |
| Kendall's $\tau$ | 0.90 |
| Top-3 정확도 | 67% |

> **Note**: 이전 구현(v1)은 DGP의 `compute_log_intensity` 함수를 직접 호출하여 coalition value를 계산하였으므로, ground truth를 오라클로 사용한 순환논증이었다 (MAE 0.017). 현재 구현(v2)은 관측 데이터로 학습한 response model만 사용하므로 legitimate한 결과를 산출한다.

### 6.2 Survival/Poisson Attribution (Shender et al. 2023, TEDDA)

> **구현**: `part1_simulation/models/causal/survival_attribution.py`
> **논문**: "A Time To Event Framework For Multi-touch Attribution" (ArXiv 2009.08432, JDS 2023)
> **Coverage**: 논문 Section 4 방법론 전수 반영 (4.1.1~4.1.7 + 4.2.1~4.2.3).

#### 모델 구조 (Section 4.1)

각 유저의 path 를 piecewise-constant intensity 인 **interval 단위**로 분할한 뒤, 각 interval 을 1 Poisson 관측치로 보고 회귀한다 (논문 Section 4.1.7).

**Eq 5 — Step-function decay (Section 4.1.1):** 5 시간 구간 (0-24h / 24-72h / 72-168h / 168-336h / 336h+) 으로 채널별 감쇠를 데이터로부터 학습.

**Eq 7 — Multiple ads (Section 4.1.3):** 각 interval 에서 채널 c × bin b 별 active touchpoint count.

**Eq 10 — User features (Section 4.1.4):** segment dummy 가 α₀ shift 로 작용.

**Eq 11 — Experimental data (Section 4.1.5):** `query_events` 인자가 주어지면 ad event 와 query event 를 분리한 모델 적합.

**옵션 hook**: `include_position` (Eq 8), `include_cross_channel` (Eq 9), `include_seasonality` (4.1.6a), `include_self_excitation` (4.1.6c), `extra_ad_features` (Eq 6 gₖ). 모두 default off.

**Eq 12 — Estimation:** Poisson GLM with **offset = log(interval length)**. 우측 절단 (Requirement #1) 은 비전환 유저의 마지막 interval `[t_n, τ]` 을 자동 추가하여 처리.

$$\log(\hat{\lambda}_{ij}) + \log(\Delta t_{ij}) = \hat{\alpha}_0 + \sum_{c,b} \hat{\beta}_{cb} \cdot x_{ijcb} + \sum_s \gamma_s \cdot \mathbb{1}[\text{seg}_i = s] + \cdots$$

#### Credit Assignment (Section 4.2)

**(1) Backwards Elimination — Eq 13 (Section 4.2.1, 기본값):**

$$\text{RawCredit}(j) = \hat{\lambda}(t^*, A^{(j)}) - \hat{\lambda}(t^*, A^{(j-1)})$$

Telescoping: $\sum_j \text{RawCredit}(j) = \hat{\lambda}(A^{(n)}) - \hat{\lambda}(\emptyset)$. 시너지 크레딧이 후반 터치포인트에 집중.

**(2) Incremental Attribution — Eq 19, 20 (Section 4.2.2):** `query_events` 와 함께 `credit_method="incremental"` 로 호출 시, ad-only ablation: query effect 는 유지한 채 ads 만 제거하여 incremental conversions 산출.

**(3) Synergy & Shapley 비교 — Eq 21, 24 (Section 4.2.3):** `compute_synergy_report()` 가

$$S(A^{(j-1)}, A_j) = m(A^{(j)}) - m(A^{(j-1)}) - m(\{A_j\}), \quad m(A) = \hat{\lambda}(A) - \hat{\lambda}(\emptyset)$$

를 계산하여 channel-pair × gap 별 mean synergy 보고. BE vs Shapley 분배 차이 = ½·S 검증.

**Normalization options:** `sum_to_one` (default), `eq17` ($\hat\lambda(A^{(n)})$ 으로 나눔), `eq18` ($\hat\lambda(A^{(n)}) - \hat\lambda(\emptyset)$ 으로 나눔).

**(Non-paper) AICPE:** 독자 확장. 채널의 모든 tb_* feature 를 동시에 0 으로 강제 후 평균 prediction drop. `credit_method="aicpe"` 로 호출.

#### 논문 Primary vs Optional — 모듈식 Ablation 구조

논문에서 명시적으로 "그들의 제안 (their proposal)" 으로 분류한 핵심 구성과 "확장 / future work / experimental data 한정" 으로 분류한 옵션을 분리해 구현했다. **default 호출 (`compute_survival_attribution(journeys)`) 이 논문의 primary configuration 과 1:1 일치**하며, 나머지는 ablation 스타일 hook 으로 켜고 끌 수 있다.

**Always ON — 논문 primary (Eq 7 + Eq 10 + Eq 12 + Eq 13 + Requirement #1)**

| 논문 | 위상 | 구현 |
|---|---|---|
| Eq 7 multiple ads | primary 핵심 | 항상 ON |
| Eq 10 user feature α₀ shift | primary 핵심 | 항상 ON (segment dummy) |
| Eq 5 step-function decay | implementation primary (논문 Section 4.1.7 piecewise-constant 권고) | 항상 ON (5-bin) |
| Eq 12 interval Poisson + log Δt offset | primary estimation | 항상 ON |
| Eq 13 Backwards Elimination | **primary credit ("their proposal")** | default `credit_method="backelim"` |
| 우측 절단 (Requirement #1) | primary 요구사항 | 항상 ON |

> 논문은 decay 함수 f 의 3가지 옵션 (exp mixture Eq 3, spline Eq 4, step Eq 5) 중 하나를 명시적으로 추천하지 않는다 (*"We do not make a specific recommendation here"*, p.7). 다만 Section 4.1.7 의 추정 편의성 (interval Poisson 환원) 으로 인해 **step function (Eq 5) 이 사실상 implementation primary** 로 채택된다.

**Default OFF — 논문이 "확장 / 옵션" 으로 분류한 hook**

| 논문 | 위상 | 인자 | default |
|---|---|---|---|
| Eq 6 ad feature gₖ (Section 4.1.2) | primary 모델에 포함되나 추가 ad-level 피처가 있을 때 | `extra_ad_features=[...]` | `None` |
| Eq 8 position-dependent fⱼ | "For example, ..." 확장 예시 (Section 4.1.3) | `include_position=True` | `False` |
| Eq 9 cross-ad interactions | "This allows us to add..." 확장 예시 (Section 4.1.3) | `include_cross_channel=True` | `False` |
| Eq 11 query/ad 분리 | experimental data 한정 (Section 4.1.5) | `query_events=DataFrame` | `None` |
| 4.1.6 (a) 계절성 | "further refinements" | `include_seasonality=True` | `False` |
| 4.1.6 (c) self-excitation | "further refinements" | `include_self_excitation=True` | `False` |

**Credit 방식 — 셋 중 하나 선택 (mutually exclusive)**

| 값 | 논문 | 비고 |
|---|---|---|
| `"backelim"` (default) | Eq 13, Section 4.2.1 | **paper primary** |
| `"incremental"` | Eq 20, Section 4.2.2 | experimental data 한정 (`query_events` 와 함께) |
| `"aicpe"` | — | 비논문 독자 확장 |

**별도 분석 함수**

- `compute_synergy_report(journeys, model, meta)` — Eq 21/24 분석 도구. 적합된 모델에서 channel-pair × gap 별 synergy 통계만 별도 추출. BE vs Shapley 분배 차이 (= ½·S) 검증용 (Section 4.2.3).

**사용 패턴**

```python
# (1) 논문 primary — default 호출
r = compute_survival_attribution(journeys)

# (2) Section 4.1.3 확장 풀 모델 — position + cross-ad interaction
r = compute_survival_attribution(
    journeys,
    include_position=True,
    include_cross_channel=True,
)

# (3) Section 4.1.5 + 4.2.2 — 실험 데이터 + incremental
r = compute_survival_attribution(
    journeys,
    query_events=query_df,
    credit_method="incremental",
)

# (4) Section 4.2.3 synergy 분석 (별도 함수)
idf, cols, meta = _build_interval_features(journeys)
model = _fit_poisson_model(idf, cols)
synergy_df = compute_synergy_report(journeys, model, meta)
```

**Ablation 실험 설계**

각 hook 의 marginal contribution 을 ground truth 대비 측정 가능 (예: `experiments/04_dgp_sensitivity.py` 패턴):

```python
configs = [
    {"name": "paper_primary",  "include_position": False, "include_cross_channel": False},
    {"name": "+ position",     "include_position": True,  "include_cross_channel": False},
    {"name": "+ cross_channel","include_position": False, "include_cross_channel": True},
    {"name": "+ both",         "include_position": True,  "include_cross_channel": True},
]
```

#### 결과 (v3, full 100K 시뮬레이션)

| Credit 방식 | MAE | $\tau$ | Top-3 | Bootstrap CV |
|------------|-----|--------|-------|---|
| **BackElim (논문 primary, Eq 13)** | 0.046 | **1.000** | **100%** | **0.096** (3rd most stable) |
| AICPE (non-paper extension) | 0.026 | 0.905 | 67% | — |

**핵심 검증 지표** (v3 full 100K):
- **Spearman(learned β[0], GT β) = 0.955** (p=0.001) — 채널별 학습 effect 가 GT β 와 거의 일치
- **Telescoping invariant**: $\sum_j \text{RawCredit}(j) = \hat\lambda(A^{(n)}) - \hat\lambda(\emptyset)$ 단위테스트 1e-9 정확도
- 13/13 unit tests 통과 (Section 4.1.1 ~ 4.2.3 의 핵심 수식 모두 검증)

**학습된 decay 곡선 vs GT 매핑** (BackElim, full 100K):

| Channel | GT β | GT half-life (d) | β[0] (0-1d) | β[1] | β[2] | β[3] | β[4] | 형태 |
|---|---|---|---|---|---|---|---|---|
| Display | 0.30 | 14.0 | 0.26 | 0.27 | 0.28 | 0.17 | 0.18 | 평탄 (long half-life) |
| Social | 0.40 | 3.0 | 0.21 | 0.29 | 0.08 | 0.01 | 0.01 | 빠른 감쇠 |
| Organic Search | 0.50 | 7.0 | 0.37 | 0.32 | 0.27 | 0.10 | 0.11 | 중간 감쇠 |
| **Paid Search** | **1.20** | **1.0** | **0.98** | **0.25** | **0.05** | 0.02 | 0.10 | **급격 감쇠** (가장 큰 β[0] + 가장 짧은 half-life) |
| Email | 0.80 | 5.0 | 0.67 | 0.49 | 0.31 | 0.10 | -0.08 | 중간-빠른 감쇠 |
| Referral | 0.50 | 7.0 | 0.33 | 0.59 | 0.34 | 0.14 | -0.03 | 중간 감쇠 |
| Direct | 0.70 | 2.0 | 0.54 | 0.30 | 0.11 | -0.06 | 0.04 | 중간-빠른 감쇠 |

> **Note (v3, paper-faithful):** v3 는 논문 Section 4 (4.1.1~4.1.7 + 4.2.1~4.2.3) 모든 방법론을 반영 — interval Poisson + log Δt offset (Eq 12), 우측 절단, BE (Eq 13), incremental (Eq 20), synergy (Eq 21), 옵셔널 hooks (Eq 6/8/9, 4.1.6). v2 의 per-user binary outcome aggregation 은 paper-faithful 이 아니므로 deprecated. v2 BackElim MAE=0.041 → v3 0.046 (+0.005); v2 AICPE MAE=0.028 → v3 0.026 (개선). v3 의 핵심 이득은 학습된 β decay 의 GT 정합성 (Spearman 0.955) + bootstrap stability 4.6× 향상 (CV 0.44 → 0.096).

### 6.3 IPW (Inverse Propensity Weighting)

> **구현**: `part1_simulation/models/causal/propensity.py`

#### 수학적 정의

각 채널 $c$를 binary treatment로 취급한다:

$$T_c = \mathbb{1}[\text{유저가 채널 } c\text{를 경험}]$$

**Propensity Score:**

$$e_c(X_i) = P(T_c = 1 \mid X_i)$$

$X_i$는 교란 변수(confounders): 세그먼트 더미 + 다른 채널 노출 여부

**ATE 추정 (Horvitz-Thompson):**

$$\hat{\tau}_c^{\text{IPW}} = \frac{1}{N} \sum_{i=1}^{N} \left[\frac{Y_i \cdot T_{ic}}{e_c(X_i)} - \frac{Y_i \cdot (1 - T_{ic})}{1 - e_c(X_i)}\right]$$

**Clipping**: $e_c(X_i) \in [0.01, 0.99]$로 극단 가중치를 방지한다.

#### 결과

| 지표 | 값 |
|------|---|
| MAE | 0.074 |
| Kendall's $\tau$ | 0.33 |
| Top-3 정확도 | 33% |

**IPW의 한계**: 채널 노출을 binary(있음/없음)로 처리하므로, 빈도와 시간 정보를 모두 손실한다. 이는 우리 DGP에서 exposure 빈도와 타이밍이 중요한 상황에서 큰 정보 손실이다.

### 6.4 Doubly Robust (DR)

> **구현**: `part1_simulation/models/causal/propensity.py`

#### 수학적 정의

IPW와 outcome model을 결합하여, 둘 중 하나만 올바르면 일관된 추정을 보장한다.

$$\hat{\tau}_c^{\text{DR}} = \frac{1}{N} \sum_{i=1}^{N} \left[\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{T_{ic}(Y_i - \hat{\mu}_1(X_i))}{e_c(X_i)} - \frac{(1 - T_{ic})(Y_i - \hat{\mu}_0(X_i))}{1 - e_c(X_i)}\right]$$

여기서:
- $\hat{\mu}_1(X_i) = E[Y \mid X_i, T_c = 1]$ (outcome model, treatment)
- $\hat{\mu}_0(X_i) = E[Y \mid X_i, T_c = 0]$ (outcome model, control)
- $e_c(X_i) = P(T_c = 1 \mid X_i)$ (propensity model)

#### 결과

| 지표 | 값 |
|------|---|
| MAE | 0.078 |
| Kendall's $\tau$ | 0.72 |
| Top-3 정확도 | 100% |

DR이 IPW보다 $\tau$에서 크게 개선(0.33 → 0.72)되었으나, MAE는 여전히 높다. Binary treatment의 근본적 한계가 두 방법 모두에 적용된다.

### 6.5 Double Machine Learning (DML)

> **구현**: `part1_simulation/models/causal/dml.py`

#### 수학적 정의 (Chernozhukov et al. 2018)

Partialling-out 접근으로 교란 변수의 효과를 제거한다.

**Cross-Fitting (5-fold):**

각 fold $l$에 대해:
1. 학습 폴드에서 outcome model 학습: $\hat{E}[Y \mid W]_{-l}$
2. 학습 폴드에서 treatment model 학습: $\hat{E}[T_c \mid W]_{-l}$
3. 검증 폴드에서 잔차 계산:
   - $\tilde{Y}_i = Y_i - \hat{E}[Y_i \mid W_i]_{-l}$
   - $\tilde{T}_i = T_{ic} - \hat{E}[T_{ic} \mid W_i]_{-l}$

**ATE 추정 (Frisch-Waugh-Lovell):**

$$\hat{\tau}_c^{\text{DML}} = \frac{\sum_i \tilde{Y}_i \cdot \tilde{T}_i}{\sum_i \tilde{T}_i^2}$$

교란 변수 $W$: 세그먼트 더미 + 다른 채널 binary 노출

모델: Logistic Regression (outcome, treatment 모두)

#### 결과

| 지표 | 값 |
|------|---|
| MAE | 0.050 |
| Kendall's $\tau$ | 0.52 |
| Top-3 정확도 | 67% |

**DML의 장단점**: IPW/DR보다 개선된 MAE(0.050)를 보이나, 여전히 binary treatment 한계와 logistic regression의 표현력 제약이 존재한다. 데이터 규모에 민감하여 50K+ 유저가 필요하다 (실험 03 결과).

### 6.6 CAMTA (Causal Attention MTA)

> **구현**: `part1_simulation/models/causal/camta.py`

#### 핵심 아이디어

LSTM+Attention 모델에 **인과적 정규화(causal regularization)** 손실을 추가하여, attention weights가 인과적 기여도에 가까워지도록 유도한다.

#### 학습 목적 함수

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda_{\text{causal}} \cdot \mathcal{L}_{\text{causal}}$$

여기서:

$$\mathcal{L}_{\text{causal}} = \text{MSE}\left(\boldsymbol{\alpha}, \hat{\boldsymbol{g}}\right)$$

- $\boldsymbol{\alpha}$: attention weights
- $\hat{\boldsymbol{g}}$: LOO targets (정규화된 leave-one-out effects)

#### Two-Phase Training

1. **Phase 1 (Warmup, 5 epochs)**: $\mathcal{L}_{\text{BCE}}$만 사용하여 기본 예측 성능 확보
2. **Phase 2 (Causal, 35 epochs)**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{causal}}$

**LOO Target 계산:**

각 전환 유저의 각 터치포인트에 대해:

$$g_j = \hat{P}(\text{conv} \mid \text{full}) - \hat{P}(\text{conv} \mid \text{mask } j)$$

$$\hat{g}_j = \frac{\max(0, g_j)}{\sum_l \max(0, g_l)}$$

#### 결과

| 지표 | 값 |
|------|---|
| MAE | 0.043 |
| Kendall's $\tau$ | 0.71 |
| Top-3 정확도 | 100% |

CAMTA는 vanilla LSTM+Attention(LOO) 대비 MAE가 약간 개선(0.034 → 0.043)되지 않았다. 이는 LOO target 자체가 인과적이지 않은 예측 모델에서 계산되므로, causal regularization의 효과가 제한적이기 때문이다.

---

## 7. 방법론 종합 비교

### 7.1 전체 결과 테이블 (Ground Truth A 기준)

MAE 순으로 정렬:

| 순위 | 방법론 | 카테고리 | MAE | RMSE | $\tau$ | Top-3 |
|------|--------|---------|-----|------|--------|-------|
| 1 | Incremental Shapley | Causal | **0.028** | 0.030 | 0.90 | 67% |
| 1 | Survival/Poisson (AICPE) | Causal | **0.028** | 0.032 | 0.90 | 67% |
| 3 | LSTM+Attention (LOO) | DL | 0.034 | 0.040 | 0.71 | 100% |
| 4 | Shapley (model) | Game-theoretic | 0.035 | 0.036 | 0.90 | 67% |
| 5 | LSTM+Attention (attn) | DL | 0.036 | 0.043 | 0.71 | 100% |
| 6 | Last Click | Rule-based | 0.038 | 0.045 | 0.81 | 100% |
| 7 | CAMTA | Causal DL | 0.043 | 0.047 | 0.71 | 100% |
| 8 | Time Decay | Rule-based | 0.044 | 0.053 | 0.81 | 100% |
| 9 | DML | Causal | 0.050 | 0.060 | 0.52 | 67% |
| 10 | Linear | Rule-based | 0.056 | 0.070 | 0.43 | 67% |
| 11 | Markov (2nd) | Statistical | 0.061 | 0.074 | 0.52 | 67% |
| 12 | Position-Based | Rule-based | 0.065 | 0.076 | 0.33 | 67% |
| 13 | Markov (1st) | Statistical | 0.066 | 0.080 | 0.33 | 67% |
| 14 | IPW | Causal | 0.074 | 0.081 | 0.33 | 33% |
| 15 | Doubly Robust | Causal | 0.078 | 0.092 | 0.72 | 100% |
| 16 | Transformer | DL | 0.119 | 0.136 | -0.33 | 0% |
| 17 | First Click | Rule-based | 0.158 | 0.182 | -0.29 | 0% |

> Incremental Shapley v2: 학습 기반 response model. Survival/Poisson v2: time-bin features로 decay 학습. 두 방법 모두 DGP 오라클 없이 데이터만으로 학습한 legitimate 결과. 이전 v1(Incr.Shapley MAE 0.017, Surv/Poisson MAE 0.025)은 DGP 파라미터를 직접 사용한 체리피킹으로 폐기.

### 7.2 카테고리별 분석

**Causal 방법론 (상위 2위):**
- Survival/Poisson과 Incremental Shapley가 MAE 기준 1, 2위
- 단, IPW(14위)와 DR(15위)은 오히려 rule-based보다 열세
- **핵심 차이**: binary treatment vs continuous exposure. Survival/Poisson과 Incremental Shapley는 exposure의 빈도와 시간 정보를 활용하지만, IPW/DR/DML은 binary 처리

**Game-theoretic (안정적 중상위):**
- Shapley (model-based)는 MAE 4위, $\tau$ 공동 1위(0.90)
- 비인과적 방법론 중 가장 균형 잡힌 성능

**Deep Learning (Top-3 정확도 강점):**
- LSTM 계열은 Top-3 정확도 100%로 상위 3개 채널을 정확히 식별
- 그러나 세부 크레딧 값(MAE)에서는 중간 수준

**Rule-based (분산 큰 성능):**
- Last Click(MAE 0.038)과 First Click(MAE 0.158) 사이 4배 차이
- Time Decay가 가장 robust한 rule-based 방법론

### 7.3 채널별 Bias 패턴

**체계적 과대평가 경향:**
- **Direct**: Last Click, CAMTA, LSTM이 과대평가 (마지막 터치포인트 빈도 높음)
- **Organic Search**: First Click, Linear, Position-Based가 과대평가 (빈도 최고 + 초기 여정 비중)

**체계적 과소평가 경향:**
- **Paid Search**: Markov, Linear, Position-Based, Time Decay 모두 과소평가 (높은 $\beta$ 미포착)
- **Email**: LSTM, Time Decay가 과소평가

**편향이 적은 방법론:**
- Incremental Shapley (v2): 최대 채널 bias 0.038 (Display). v1(≤0.028) 대비 다소 증가하나, 학습 기반으로 legitimate
- Survival/Poisson (AICPE, v2): 최대 채널 bias 0.059 (Paid Search). 이전 v1(-0.067)과 유사 수준

### 7.4 실무적 방법론 선택 가이드

| 조건 | 추천 방법론 | 근거 |
|------|-----------|------|
| Incremental 효과 분리 필요 | Incremental Shapley | Base conversion 분리, 학습 기반 |
| 충분한 데이터 (50K+ 유저) | Survival/Poisson | $\tau$ 최고, 5K부터 안정 |
| 데이터 부족 (< 5K) | Time Decay 또는 Last Click | 규모 불변, 추가 학습 불필요 |
| 상위 채널 식별만 필요 | LSTM+Attention (LOO) | Top-3 100%, 빠른 추론 |
| 공정한 크레딧 배분 필요 | Shapley (model-based) | 공리적 기반, $\tau$ 0.90 |
| 빠른 구현 필요 | Last Click | 코드 1줄, MAE 0.038 |
