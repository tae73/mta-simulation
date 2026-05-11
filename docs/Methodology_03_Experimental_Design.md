# Part 3: 실험 설계 및 분석 결과

---
**방법론 문서 시리즈**
- [Part 1: DGP 설계](./Methodology_01_DGP_Design.md)
- [Part 2: Attribution 방법론](./Methodology_02_Attribution_Methods.md)
- **Part 3: 실험 설계 및 결과** (현재 문서)
- [프로젝트 계획서](./Marketing_Attribution_Project_Plan.md)
---

## 1. 평가 프레임워크

### 1.1 평가 메트릭

모든 방법론은 동일한 메트릭으로 평가된다. 입력은 정규화된 채널별 기여도 벡터 $\mathbf{p} = (p_1, ..., p_7)$이고, ground truth는 $\mathbf{g} = (g_1, ..., g_7)$이다.

> **구현**: `part1_simulation/evaluation/metrics.py`

**MAE (Mean Absolute Error):**

$$\text{MAE} = \frac{1}{7} \sum_{k=1}^{7} |p_k - g_k|$$

값의 크기 차이를 측정한다. 모든 실험의 primary metric이다.

**RMSE (Root Mean Squared Error):**

$$\text{RMSE} = \sqrt{\frac{1}{7} \sum_{k=1}^{7} (p_k - g_k)^2}$$

큰 오차에 더 민감하다. 특정 채널에서 큰 편향이 있는 방법론을 페널티한다.

**Kendall's Tau ($\tau$):**

$$\tau = \frac{C - D}{\binom{7}{2}}$$

여기서 $C$는 concordant pairs, $D$는 discordant pairs 수이다. 범위: $[-1, 1]$.

채널 **순위의 일치도**를 측정한다. 절대 값보다 상대적 중요도 순서가 중요한 실무에서 핵심 지표이다.

**Top-K 정확도 (K=3):**

$$\text{Top-3 Acc} = \frac{|\text{pred\_top3} \cap \text{truth\_top3}|}{3}$$

상위 3개 채널을 올바르게 식별하는지 측정한다.

**채널별 Bias:**

$$\text{Bias}(k) = p_k - g_k$$

양수 = 과대평가, 음수 = 과소평가. 방법론의 **체계적 편향 방향**을 진단한다.

### 1.2 Ground Truth 기준

- **Primary benchmark**: Ground Truth A (Intensity Decomposition)
- **보조 benchmark**: Ground Truth B (Counterfactual Shapley) — Shapley 계열 방법론 검증용
- **재현성**: Hydra config + random seed 42

---

## 2. 실험 01: 방법론 정확도 비교 (Main Experiment)

> **구현**: `part1_simulation/experiments/01_method_accuracy.py`
> **결과**: `results/part1/01_method_accuracy.csv`, `results/part1/01_mae_comparison.png`, `results/part1/01_mae_vs_tau.png`

### 가설

$H_1$: Causal inference 기반 방법론이 confounding이 존재하는 DGP에서 correlational 방법론보다 ground truth에 더 가까운 attribution을 산출한다.

$H_2$: 전환 모델의 구조(log-linear intensity)를 정확히 반영하는 방법론이 가장 높은 정확도를 달성한다.

### 실험 설정

- **데이터**: 100,000 유저, 7 채널, 전환율 2.305% (2,305 converters)
- **대상 방법론**: 18종 (Rule-based 5 + Markov 2 + Shapley 2 + DL 3 + Causal 6)
- **비교 기준**: Ground Truth A (intensity decomposition)

### 결과 (v3 Survival 반영, full 100K)

| 순위 | 방법론 | 카테고리 | MAE | $\tau$ | Top-3 |
|------|--------|---------|-----|--------|-------|
| 1 | Incremental Shapley | Causal (outcome model) | **0.028** | 0.90 | 67% |
| 2 | Survival/Poisson v3 AICPE (non-paper) | Causal (outcome model) | **0.026** | 0.91 | 67% |
| 3 | LSTM+Attention (LOO) | DL | 0.034 | 0.71 | 100% |
| 4 | Shapley (model) | Game-theoretic | 0.035 | 0.90 | 67% |
| 5 | LSTM+Attention (attn) | DL | 0.036 | 0.71 | 100% |
| 6 | Last Click | Rule-based | 0.038 | 0.81 | 100% |
| 7 | CAMTA | Causal (outcome model) | 0.043 | 0.71 | 100% |
| 8 | Time Decay | Rule-based | 0.044 | 0.81 | 100% |
| 9 | **Survival/Poisson v3 BackElim (paper primary)** | Causal (outcome model) | **0.046** | **1.000** | **100%** |
| ... | | | | | |
| 16 | Transformer | DL | 0.119 | -0.33 | 0% |
| 17 | First Click | Rule-based | 0.158 | -0.29 | 0% |

> v3 Survival/Poisson 은 Shender 2023 Section 4 전수 반영 (interval Poisson + log Δt offset, BE Eq 13). **BackElim 은 τ=1.0 완벽 순위** + Bootstrap CV 0.096 (3rd most stable across all 17 methods); AICPE 는 MAE 1위. Incremental Shapley 도 v2 (DGP 파라미터 미사용 학습) 기반.

### 분석

**$H_1$ 부분 지지**: 상위 2개 방법론(Incremental Shapley 0.028, Survival/Poisson v3 AICPE 0.026)이 모두 causal이나, IPW(0.074)와 DR(0.078)은 rule-based Last Click(0.038)보다 열세이다. **"Causal이면 항상 우수하다"는 것은 성립하지 않는다.** Binary treatment 한계가 인과 보정의 이점을 상쇄한다.

**$H_2$ 강한 지지**: Survival/Poisson v3 와 Incremental Shapley 모두 DGP 파라미터 없이 학습. v3 AICPE 가 MAE 0.026 으로 단독 1위, BackElim 은 τ=1.0 으로 ranking 정확도 1위. Survival/Poisson의 interval-level Poisson GLM + log Δt offset (Eq 12) 이 DGP의 log-linear intensity 구조와 정합 — 학습된 β decay vs GT β Spearman 0.955 (p=0.001) 입증.

**의외의 발견**: Last Click(0.038)이 많은 sophisticated 방법론보다 우수하다. 이는 우리 DGP에서 lower-funnel 채널($\beta$가 높은 Paid Search, Email, Direct)이 실제로 전환에 가장 크게 기여하고, Last Click이 이 채널들을 자연스럽게 포착하기 때문이다.

---

## 3. 실험 02: 상호작용 효과 포착 (Interaction Effects)

> **구현**: `part1_simulation/experiments/02_interaction_effects.py`
> **결과**: `results/part1/02_interaction_effects.csv`, `results/part1/02_interaction_effects.png`

### 가설

$H$: 교차 채널 영향($\delta_{st}$)이 존재할 때, 이를 감지할 수 있는 방법론과 그렇지 못한 방법론 사이에 attribution 결과의 유의미한 차이가 발생한다.

### 실험 설정

두 가지 조건을 비교한다:

| 조건 | $\delta$ 값 | 설명 |
|------|-----------|------|
| With interactions | Display→Paid(0.4), Social→Email(0.3), Organic→Direct(0.2) | DGP 기본 설정 |
| Without interactions | 모든 $\delta = 0$ | 시너지 제거 |

각 조건에서 방법론을 실행하고, 시너지 쌍(소스→타깃)의 기여도 변화를 측정한다.

**Synergy Detection Score:**

$$\Delta_{\text{synergy}}(s \to t) = \text{credit}(s \mid \delta > 0) - \text{credit}(s \mid \delta = 0)$$

양수 = 시너지가 존재할 때 소스 채널의 기여도가 증가 (시너지 감지)

### 결과

**Display → Paid Search ($\delta = 0.4$, 가장 강한 시너지):**

| 방법론 | $\Delta_{\text{synergy}}$ | 해석 |
|--------|--------------------------|------|
| Shapley (model) | **+0.075** | 가장 민감하게 감지 |
| Survival/Poisson | +0.040 | 시너지를 포착하나 보수적 |
| Last Click | +0.020 | 간접적으로 반영 |
| Linear | +0.017 | 미약한 감지 |
| Time Decay | +0.017 | 미약한 감지 |
| Markov (2nd) | +0.006 | 거의 감지 불가 |
| Markov (1st) | +0.003 | 감지 불가 |

**Social → Email ($\delta = 0.3$):**

대부분의 방법론에서 $\Delta < 0$이 관측되었다. 이는 Social→Email 시너지가 Email의 기여도를 높이면서 상대적으로 다른 채널(Social 포함)의 정규화된 크레딧이 감소하는 역설적 효과이다.

**Organic → Direct ($\delta = 0.2$, 가장 약한 시너지):**

모든 방법론에서 $|\Delta| < 0.01$로, 약한 시너지는 어떤 방법론도 유의미하게 감지하지 못한다.

### 분석

1. **Shapley가 시너지 감지에 가장 민감**: coalition 기반 가치 함수가 채널 조합의 효과를 직접 측정하므로, 시너지가 coalition value 차이에 반영된다.
2. **Markov는 시너지에 무감응**: Markov의 Removal Effect는 채널을 제거했을 때의 전환 확률 변화만 측정하며, 채널 간 조합 효과를 포착하는 구조가 없다.
3. **Survival/Poisson의 중간 감도**: cross-influence가 없어도 decayed exposure의 상관관계를 통해 간접적으로 시너지를 포착한다.
4. **약한 시너지($\delta \leq 0.2$)는 현재 규모(100K)에서 감지 한계** 아래이다.

---

## 4. 실험 03: 데이터 규모 민감도 (Learning Curve)

> **구현**: `part1_simulation/experiments/03_data_scale.py`
> **결과**: `results/part1/03_data_scale.csv`, `results/part1/03_data_scale.png`

### 가설

$H_1$: 모델 기반 방법론(DL, Causal)은 데이터 규모 증가에 따라 정확도가 향상되고, rule-based는 규모에 불변이다.

$H_2$: 방법론마다 "충분한 정확도"를 달성하는 최소 데이터 규모가 다르며, 복잡한 방법론일수록 더 많은 데이터가 필요하다.

### 실험 설정

5개 규모 수준에서 동일한 DGP로 데이터를 생성하고, 대표 방법론 8종을 평가한다:

| $N$ 유저 | 전환자 수 (≈2.3%) |
|---------|-----------------|
| 1,000 | ~32 |
| 5,000 | ~125 |
| 10,000 | ~258 |
| 50,000 | ~1,278 |
| 100,000 | ~2,739 |

### 결과: MAE by Scale

| 방법론 | 1K | 5K | 10K | 50K | 100K | 수렴 시점 |
|--------|-----|-----|------|------|-------|----------|
| Survival/Poisson | 0.057 | **0.020** | 0.023 | 0.016 | **0.017** | **5K** |
| Shapley (model) | 0.118 | 0.056 | **0.023** | 0.026 | 0.030 | 10K |
| Last Click | 0.034 | 0.037 | 0.049 | 0.044 | 0.049 | 규모 불변 |
| Time Decay | 0.030 | 0.034 | 0.033 | 0.035 | 0.035 | 규모 불변 |
| Markov (1st) | 0.069 | 0.061 | 0.058 | 0.059 | 0.060 | 규모 불변 |
| DML | 0.159 | 0.074 | 0.049 | 0.030 | 0.027 | **50K** |
| LSTM+Attention | — | 0.042 | 0.034 | 0.052 | 0.063 | 5K (but noisy) |
| Linear | 0.051 | 0.048 | 0.048 | 0.049 | 0.050 | 규모 불변 |

### 결과: Kendall's $\tau$ by Scale

| 방법론 | 1K | 5K | 10K | 50K | 100K |
|--------|-----|-----|------|------|-------|
| Survival/Poisson | 0.49 | **1.00** | 0.90 | 0.90 | **1.00** |
| Shapley (model) | 0.14 | 0.98 | **1.00** | 0.90 | 0.90 |
| DML | 0.10 | 0.62 | 0.62 | 0.71 | **0.81** |
| LSTM+Attention | — | 0.71 | 0.71 | 0.71 | 0.90 |
| Last Click | 0.62 | 0.81 | 0.81 | 0.71 | 0.71 |

### 분석

**$H_1$ 지지**: Rule-based 방법론(Last Click, Time Decay, Linear)은 데이터 규모에 거의 영향을 받지 않는다. 모델 기반 방법론은 뚜렷한 학습 곡선을 보인다.

**$H_2$ 강력 지지**: 최소 데이터 요구량이 명확히 구분된다.

| 수렴 수준 | 방법론 | 최소 데이터 |
|----------|--------|-----------|
| Very Low | Last Click, Time Decay, Linear | 1K (규모 불변) |
| Low | Survival/Poisson v3 | **10K** (1K=0.079 → 5K=0.053 → 10K=0.040 → 100K=0.049) |
| Medium | Shapley (model), LSTM+Attention | 10K |
| High | DML | **50K** |

**핵심 발견:**

1. **Survival/Poisson v3의 데이터 효율성**: 10K 유저(~200 converters)에서 MAE 0.040 으로 최저점, 5K 에서도 0.053 으로 안정. interval-level Poisson 의 파라미터 수가 적어(35 channel-bin β + intercept + segment dummy ≈ 37) 적은 데이터에서도 수렴. v2 (5K MAE 0.020) 대비 paper-faithful interval split 으로 noise floor 가 약간 높아진 trade-off.

2. **DML의 높은 데이터 요구**: 1K에서 MAE 0.159 (최악 수준)이나, 50K에서 0.030으로 급감. Cross-fitting의 각 fold에 충분한 샘플이 필요하기 때문이다.

3. **LSTM의 비단조적 성능**: 100K에서 MAE가 오히려 증가(0.063). 이는 과적합이 아닌, 클래스 불균형(전환율 2.3%)이 규모 증가와 함께 학습 난이도를 변화시키기 때문으로 추정된다.

---

## 5. 실험 04: DGP 가정 민감도 (Sensitivity Analysis)

> **구현**: `part1_simulation/experiments/04_dgp_sensitivity.py`
> **결과**: `results/part1/04_dgp_sensitivity.csv`, `results/part1/04_dgp_sensitivity.png`

### 가설

$H$: DGP의 핵심 구성 요소(교차 영향, 시간 감쇠, 유저 이질성)를 제거하면, 해당 구성 요소를 활용하는 방법론의 상대적 우위가 변화한다.

### 실험 설정

4가지 DGP 변형(variant)에서 대표 방법론 8종을 실행한다:

| Variant | 변경 내용 | 목적 |
|---------|----------|------|
| Full (baseline) | 기본 DGP | 비교 기준 |
| No interactions | 모든 $\delta = 0$ | 교차 영향 없이 성능 변화? |
| No decay | 모든 $\tau = 1000$일 | 시간 감쇠 무시 시 영향? |
| No heterogeneity | 모든 $\eta = 0$ | confounding 제거 시 causal 이점 변화? |

### 결과: MAE by Variant

| 방법론 | Full | No interact. | No decay | No hetero. |
|--------|------|-------------|----------|-----------|
| Last Click | 0.034 | 0.035 | 0.047 | 0.038 |
| Time Decay | 0.045 | 0.044 | **0.028** | 0.043 |
| Markov (2nd) | 0.066 | 0.067 | 0.052 | 0.061 |
| Shapley (model) | 0.061 | 0.065 | **0.020** | 0.045 |
| Survival/Poisson | **0.024** | **0.016** | 0.023 | **0.021** |
| IPW | 0.071 | 0.063 | 0.051 | 0.066 |
| DML | 0.060 | 0.077 | 0.045 | 0.044 |

### 분석

**Survival/Poisson의 강건성:**
- 4개 variant 모두에서 MAE 0.016~0.024 범위. 가장 robust한 방법론이다.
- No interactions에서 오히려 개선(0.024 → 0.016): 교차 영향이 없으면 Poisson GLM의 단순한 선형 구조가 DGP와 완벽히 일치하므로.

**No decay variant의 효과:**
- 시간 감쇠가 없으면 모든 터치포인트가 동등한 효과를 가지므로, 빈도 기반 방법론이 유리해진다.
- **Shapley 급개선**: 0.061 → 0.020. 시간 감쇠 없이는 binary 채널 존재 여부만으로 전환이 결정되므로, Shapley의 coalition-based value function이 이를 정확히 포착한다.
- **Time Decay 개선**: 0.045 → 0.028. 역설적으로, 시간 감쇠가 없는 상황에서도 Time Decay의 가중치 체계가 우연히 좋은 배분을 만든다.

**No heterogeneity variant의 효과:**
- segment heterogeneity 제거 시 causal 방법론의 이점이 **완전히 소멸하지는 않는다**.
- Survival/Poisson(0.024 → 0.021)과 DML(0.060 → 0.044) 모두 개선되나, 차이가 작다.
- 이는 causal 방법론의 이점이 user feature regression adjustment (Survival/Poisson Eq 10) 또는 propensity-based debiasing (DML) **뿐만 아니라** 모델 구조 (exposure 빈도, 시간 정보 활용) 에서도 기인함을 시사한다. Methodology_05 § 4.2 의 5-tier 분류 (outcome model only vs debiased) 와 일치.

**핵심 결론:**
- "올바른 모델 구조 > 인과 보정"이 정확도의 주된 동인
- DGP가 단순해질수록(구성 요소 제거) 단순한 방법론의 상대적 성능이 개선

---

## 6. 실험 05: Correlational vs Causal 방법론 비교

> **구현**: `part1_simulation/experiments/05_correlational_vs_causal.py`
> **결과**: `results/part1/05_correlational_vs_causal.csv`, `results/part1/05_correlational_vs_causal.png`

### 가설

$H$: Confounding 강도가 증가할수록 correlational 방법론의 성능이 악화되는 반면, causal 방법론은 안정적인 성능을 유지한다.

### 실험 설정

$\eta$ spread를 조절하여 confounding 강도를 변화시킨다:

| 수준 | $\eta$ 분포 | 해석 |
|------|-----------|------|
| Weak | $\eta \in \{-0.1, 0.0, +0.1\}$ (spread = 0.2) | 세그먼트 간 전환 성향 차이 미약 |
| Medium (default) | $\eta \in \{-0.3, 0.0, +0.5\}$ (spread = 0.8) | 기본 설정 |
| Strong | $\eta \in \{-0.8, 0.0, +1.2\}$ (spread = 2.0) | 극단적 confounding |

### 결과

**Correlational 방법론 (평균 MAE):**

| 수준 | Last Click | Time Decay | Markov (2nd) | Shapley |
|------|-----------|-----------|-------------|---------|
| Weak | 0.037 | 0.044 | 0.062 | 0.043 |
| Medium | 0.034 | 0.045 | 0.066 | 0.061 |
| Strong | 0.030 | 0.043 | 0.075 | 0.058 |

**Causal 방법론 (평균 MAE):**

| 수준 | Survival/Poisson | IPW | DML |
|------|-----------------|-----|-----|
| Weak | **0.022** | 0.070 | 0.046 |
| Medium | **0.024** | 0.071 | 0.060 |
| Strong | **0.022** | 0.077 | 0.064 |

### 분석

**$H$ 부분 지지**: Confounding이 강해질수록 Markov의 MAE가 증가(0.062 → 0.075)하지만, Last Click은 오히려 감소(0.037 → 0.030). 이는 강한 confounding에서 Loyal 유저(높은 $\eta$)가 Direct/Paid Search를 더 많이 방문하여, Last Click이 자연스럽게 이 채널을 더 많이 포착하기 때문이다.

**Survival/Poisson의 일관된 우위:**
- 3개 수준 모두에서 MAE 0.022~0.024로 거의 변동 없음
- 세그먼트 더미 변수가 confounding을 효과적으로 보정
- **confounding 강도에 관계없이 안정적**

**격차의 규모:**
- Weak confounding에서의 최고 correlational(Last Click 0.037) vs 최고 causal(Survival 0.022): 차이 0.015
- Strong confounding에서: Last Click 0.030 vs Survival 0.022: 차이 0.008
- **격차가 예상보다 modest**하다. 이는 (1) 세그먼트가 3개로 단순하고, (2) Survival이 이미 세그먼트 더미로 대부분의 confounding을 보정하며, (3) Last Click이 우연히 confounding 방향과 DGP의 $\beta$ 순위가 일치하기 때문이다.

**IPW/DML의 실망스러운 성능:**
- Causal 방법론임에도 correlational 방법론보다 열세
- Binary treatment + logistic regression의 표현력 한계가 인과 보정의 이점을 상쇄

---

## 7. 실험 06: Incremental vs Total Attribution

> **구현**: `part1_simulation/experiments/06_incremental_vs_total.py`
> **결과**: `results/part1/06_incremental_vs_total.csv`, `results/part1/06_incremental_vs_total.png`

### 가설

$H$: 기저 전환율(base conversion rate)이 높을수록 Total Shapley와 Incremental Shapley의 채널별 기여도 괴리가 확대되며, Total Shapley는 lower-funnel 채널을 과대평가한다.

### 실험 설정

$\alpha_0$를 조절하여 기저 전환율을 변화시킨다:

| 수준 | $\alpha_0$ | 기저 전환율 | 전체 전환율 |
|------|-----------|-----------|-----------|
| Low base | -3.5 | ~3.0% | 16.5% |
| Medium base | -2.5 | ~7.9% | 36.1% |
| High base | -1.8 | ~15.1% | 55.6% |

각 수준에서 Total Shapley와 Incremental Shapley를 계산하고, 채널별 크레딧 차이를 분석한다.

### 결과

**채널별 Total vs Incremental 차이 ($\text{Total} - \text{Incremental}$):**

| 채널 | Low base | Medium base | High base | 방향 |
|------|---------|------------|----------|------|
| Paid Search | +0.097 | +0.101 | **+0.111** | Total 과대 ↑ |
| Email | +0.046 | +0.065 | **+0.079** | Total 과대 ↑ |
| Direct | +0.039 | +0.026 | +0.024 | Total 과대 (안정) |
| Organic Search | -0.037 | -0.045 | **-0.051** | Incremental 과대 ↑ |
| Display | -0.073 | -0.082 | **-0.080** | Incremental 과대 ↑ |
| Social | -0.050 | -0.054 | **-0.059** | Incremental 과대 ↑ |
| Referral | -0.021 | -0.010 | -0.025 | Incremental 과대 (안정) |

### 분석

**$H$ 강력 지지**: base rate가 높아질수록 Total과 Incremental의 괴리가 확대된다.

**괴리의 메커니즘:**

1. **Total Shapley**: 전체 전환($v(S)$ = 전환 확률)을 채널에 배분. 기저 전환(광고 없이도 발생)도 채널 크레딧에 포함된다.

2. **Incremental Shapley**: $v_{\text{inc}}(S) = v(S) - v(\emptyset)$로 기저 전환을 차감. 광고로 **추가된** 전환만 배분.

3. 기저 전환율이 높을수록 Total Shapley에서 lower-funnel 채널(Paid Search, Email)이 "기저 전환까지 흡수"하여 과대평가. 반면, Incremental은 upper-funnel(Display, Social)의 **시너지 효과와 인지 기여**를 상대적으로 더 인정.

**비즈니스 시사점:**

| 상황 | 적합한 방법론 | 이유 |
|------|-------------|------|
| 성숙 시장 (높은 기저 전환) | Incremental Shapley | Total은 lower-funnel을 과대평가 → 잘못된 예산 배분 |
| 신규 시장 (낮은 기저 전환) | Total Shapley도 유사 | 기저 전환이 작아 괴리가 작음 |
| Display/Social 예산 정당화 | Incremental Shapley | upper-funnel의 간접 기여를 정당하게 반영 |
| Last-click 이관 검토 | Total → Incremental 비교 | 두 방법의 괴리가 Last Click bias의 규모를 정량화 |

---

## 8. 종합 분석

### 8.1 핵심 발견 요약

| 발견 | 근거 실험 | 시사점 |
|------|----------|--------|
| DGP 구조를 반영하는 모델이 최고 정확도 | 실험 01, 04 | "올바른 모델 구조 > 인과 보정" |
| Binary treatment 기반 causal은 한계 | 실험 01, 05 | IPW/DR < Last Click, exposure 활용이 핵심 |
| 데이터 규모 요구는 방법론마다 크게 상이 | 실험 03 | Survival 5K, DML 50K+, 사전 규모 검토 필수 |
| 교차 영향은 Shapley가 최고 민감도 | 실험 02 | 시너지 분석에는 game-theoretic 접근이 적합 |
| Confounding 격차는 예상보다 modest | 실험 05 | 실무에서 causal의 이점은 모델 구조에서 더 많이 기인 |
| 기저 전환율이 높을수록 Total ≠ Incremental | 실험 06 | 성숙 시장에서는 Incremental 필수 |

### 8.2 방법론 선택 의사결정 프레임워크

```
데이터 규모는?
├─ < 5K 유저
│   └─ Time Decay (MAE ~0.035, 규모 불변, $\tau$ 0.81)
├─ 5K ~ 50K 유저
│   ├─ Exposure 시간 정보 있음? → Survival/Poisson v3 (MAE 0.040 @10K, $\tau$ 0.81~1.0, paper-faithful Eq 13)
│   └─ Binary 채널 정보만? → Shapley model-based (MAE 0.023, $\tau$ 1.00)
└─ > 50K 유저
    ├─ Confounding 우려 강함? → DML (MAE 0.027, 교란 보정)
    ├─ Incremental 효과 필요? → Incremental Shapley (MAE 0.028)
    └─ 안정성 우선 (production)? → Survival/Poisson v3 (CV 0.096, τ 1.0)
```

### 8.3 방법론별 데이터 요구 / 정확도 / 복잡도 매트릭스

| 방법론 | 최소 데이터 | MAE@100K | Bootstrap CV | 구현 복잡도 | 인과적? |
|--------|-----------|----------|---|-----------|--------|
| Last Click | 1K | 0.038 | 0.13 | 매우 낮음 | No |
| Time Decay | 1K | 0.044 | 0.13 | 낮음 | No |
| Markov (2nd) | 5K | 0.061 | 0.05 | 중간 | No |
| Shapley (model) | 10K | 0.035 | 0.99 | 중간 | No |
| LSTM+Attention | 5K | 0.034 | 0.10 | 높음 | No |
| **Survival/Poisson v3 BackElim** | **10K** | **0.046** | **0.096** | 중간 | **Yes** |
| Survival/Poisson v3 AICPE | 10K | 0.026 | 0.096 | 중간 | Yes (non-paper) |
| DML | 50K | 0.050 | 0.93 | 높음 | Yes |
| Incremental Shapley | 10K | 0.028 | — | 높음 | **Yes** |

> †Survival/Poisson v3: Shender 2023 Section 4 전수 반영 (interval Poisson + log Δt offset, BE Eq 13). Incremental Shapley v2: 학습 기반 response model (DGP 오라클 불필요). v1 (MAE 0.017) 은 DGP 직접 호출로 폐기.

### 8.4 Part 2, Part 3으로의 연결

본 실험 결과는 시뮬레이션 데이터에서의 발견이다. 다음 단계에서:

- **Part 2 (Criteo)**: 실 데이터(16.5M events)에서 LSTM+Attention과 Transformer의 행동 패턴이 시뮬레이션과 일치하는지 검증한다. Ground truth가 없으므로 방법론 간 **일관성(consistency)**과 **scale-up 행동**을 분석한다.

- **Part 3 (MMM)**: 집계 수준 Bayesian MMM의 채널 기여도가 유저 수준 MTA 결과와 어떻게 대응되는지 **Triangulation** 분석을 수행한다.

---

## 9. Real-World Validation 실험 (08–11)

### 9.1 동기

실험 01~07은 모두 **simulation ground truth가 알려진 상태에서**의 정확도를 측정했다 (MAE/Tau against $\beta$, $f$, $\delta$). 그러나 실무 배포 환경에서는 GT가 없다. 따라서 다음 4개의 GT-free 운영 지표를 별도로 평가하여, 시뮬 벤치마크가 실제 의사결정에 얼마나 transferable한지를 검증한다.

| 실험 | 질문 | 메트릭 | 구현 |
|------|------|--------|------|
| 08 | OOS 일반화? | AUC, PR-AUC, Brier | `experiments/08_predictive_validation.py` |
| 09 | 의사결정 임팩트? | Revenue Lift % | `experiments/09_decision_impact.py` |
| 10 | 안정성? | Bootstrap CV, 95% CI | `experiments/10_bootstrap_stability.py` |
| 11 | 합의도? | Pairwise Tau, consensus rank | `experiments/11_convergent_validity.py` |

### 9.2 실험 08: Out-of-Sample Predictive Validation

**가설**: GT-MAE가 낮은 방법은 held-out 유저 전환 예측에서도 우월하다 (sim 벤치마크 ↔ deployment outcome).

**설정**
- 100K 유저를 **random 80/20 user-level split** (seed=42). DGP timestamp는 유저별로 0에서 시작하므로 calendar time 기반 split 불가.
- 각 방법론을 train slice (80K)로 재학습 → channel weights $w_k$
- Test journey score $s(j) = \sum_{TP \in j} w_{c(TP)}$ where $c(TP)$ is the channel of the touchpoint
- 메트릭: AUC, PR-AUC, Brier (min-max scaled)

**해석축**
- **GT-MAE × OOS-AUC** 산점도의 Pearson r — 음수일수록 sim 벤치마크가 실무 신호 proxy로 작동
- 모든 방법이 random(0.5) 상회하면 channel weight 자체가 generalizable signal

### 9.3 실험 09: Decision Impact (Expected Revenue Lift)

**가설**: Allocation MAE는 추상적 거리 지표. **Expected revenue lift**가 stakeholder가 직접 묻는 운영 지표이며, 두 지표는 음의 상관을 보여야 한다.

**설정 (Approach A 닫힌해)**

Linear Response 가정 하에서 expected paid conversions의 closed form:

$$\mathbb{E}[\text{conversions}] = \sum_k \text{spend}_k \times \text{eff}_k$$

where $\text{eff}_k = \beta_k \cdot \mathbb{E}[f_k] / c_k$ (DGP 알려진 효율).

- **baseline**: 현재 관측된 paid spend share (Paid Search 99% 집중)
- **GT-optimal**: `optimal_allocation_fraction` (proportional-to-efficiency 규칙) — 본 프로젝트의 GT 기준
- **Linear LP ceiling**: all-on-best-channel (Email 100%) — 순수 LP 최적
- 메트릭: $\text{lift}_{\%} = \text{rev}_{\text{method}} / \text{rev}_{\text{baseline}} - 1$

**Note**: GT-optimal은 *concave saturation 가정 하의* 최적이며, 순수 linear에서는 일부 방법이 GT-optimal을 초과 가능 (Email 과집중). 이는 capping하지 않고 그대로 보고하여 saturation-free 환경에서의 "공격적 재배분" 행동을 투명하게 드러낸다.

### 9.4 실험 10: Bootstrap Stability

**가설**: 평균 정확도가 비슷해도 finite-sample 변동성은 의사결정 위험을 좌우. 인과/구조적 방법(Survival/Poisson, Shapley)이 DL 방법(LSTM, Transformer)보다 안정적 — DGP 구조를 명시적으로 활용하기 때문.

**설정 (3-tier bootstrap N)**
- 5K 유저 bootstrap resample (with replacement, with user-id reassignment)
- **Tier-1 (light)**: rule-based 5종, Markov 2종, Shapley → N=100
- **Tier-2 (medium)**: LSTM-attention, Transformer, Inc. Shapley, IPW, DR → N=20
- **Tier-3 (heavy)**: DML, CAMTA, Survival/Poisson → N=5

**메트릭**
- per (method, channel): bootstrap mean, std, CV = std/mean, 95% CI width = q97.5 - q2.5
- per method: 평균 CV across channels (전체 안정성 요약)

**해석축**
- mean CV vs GT-MAE plot에서 두 축 모두 낮은 방법 = **robust winner**
- GT-MAE 낮지만 CV 높은 방법 = **fragile winner** (시뮬 통과, 실무 risk 큼)

### 9.5 실험 11: Convergent Validity

**가설**: GT가 없는 환경에서, 이질적 방법론들의 **합의 채널 순위(consensus rank)** 는 단일 best method 만큼 신뢰할 수 있다.

**설정**
- 17개 방법의 attribution을 채널 순위로 변환 (`01_method_accuracy.csv`의 bias 컬럼에서 `credit = max(0, gt_a + bias)` 후 정규화)
- pairwise Kendall's Tau matrix (17×17) — hierarchical clustering으로 시각화
- 채널별 disagreement = std(rank across methods) — 어느 채널이 contested인지 식별
- consensus rank = 평균 rank across methods → GT-A와의 Tau 비교

**해석축**
- **consensus tau vs best individual tau**: consensus가 best에 근접 → triangulation이 GT-free 환경에서 유효한 fallback
- **클러스터 구조**: 같은 카테고리 내부 방법론이 클러스터화하는지 (within-category 합의 > between-category 합의)
- **contested channels**: rank std 큰 채널은 단일 방법 의존 금지, 추가 검증 필요

### 9.6 종합 매트릭스 (notebooks/part1/05 §13)

GT-aware 정확도(Exp 01) + 4개 GT-free 지표(Exp 08~11)를 통합 rank 매트릭스로 결합:

- **Robust Winner**: 모든 5축 상위 — Survival/Poisson, Shapley (model-based), Incremental Shapley
- **Fragile Winner**: GT-aware 좋으나 GT-free 약함 — DL 계열 (LSTM-LOO, CAMTA), DML
- **Stable but Inaccurate**: CV 낮으나 정확도 떨어짐 — Last Click, Linear

**실무 권장사항**
- 데이터 충분 (>5K 전환): Survival/Poisson 또는 Shapley (model-based) 1차, Incremental Shapley로 cross-validate
- 데이터 부족 (<2K 전환): Time Decay baseline + bootstrap CI 보고
- 의사결정 압박 큰 상황: 2~3개 이질적 방법론의 consensus rank 우선
- DL 방법은 단독 의사결정 근거 사용 금지 — Shapley/Survival와 cross-check 필수
