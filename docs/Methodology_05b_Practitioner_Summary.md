# Methodology 05b — Causal Attribution: 실무진용 요약

> **목적**: 사내 DS↔마케팅 핸드오프용 단일 요약 문서. 본 프로젝트가 채택한 attribution 방법론과 추가로 연구·개발한 항목, 의사결정 매트릭스를 한 문서로 모았다.
>
> **독자**: 마케팅 분석가 / PM / DS-마케팅 협업 담당. 수식은 최소, 코드 없음, 의사결정·해석 중심.
>
> **선행/심화 문서**:
> - 학술 풀버전: `docs/Methodology_05_Causal_Attribution_Frameworks.md`
> - 마케팅 한 장 핸드아웃 (Conditional vs Marginal): `docs/Marketing_Handout_Conditional_vs_Marginal.md`
> - 실무 분석 노트북: `notebooks/part1/02_main_survival_incremental_shapley.ipynb`
> - 18종 방법론 분류: `docs/Methodology_02_Attribution_Methods.md`

---

## 0. 한 페이지 요약 (TL;DR)

- **본 프로젝트의 중심 method**: Survival/Poisson 모델 × Shapley credit (채널 단위) + Path-level Δ (캠페인 시나리오 단위). 같은 fitted 모델 위에서 두 단위 보고가 자동 일관.
- **두 가지 view 분리**: Conditional Shapley (사후 정산 = 전환자 모집단) vs Marginal G-computation Shapley (미래 예산 = 전 모집단). 본 시뮬에서 채널 ranking 완전 일치 (ρ=1.000) → robust.
- **본 프로젝트가 추가 연구·개발한 것**: ① Survival 위에 Shapley credit 통합 (Du + Shender 의 수학적 동치 활용) ② 4-cell ablation matrix (long-path bias 해소 1) ③ Channel↔Path duality (long-path bias 해소 2) ④ Conditional vs Marginal 분리 (selection/collider bias 해소) ⑤ 5-tier 정직한 causal label ⑥ Multivariate user feature 일반화.
- **핵심 수치** (100K 시뮬, 4 method):

  | Method | MAE ↓ | Kendall τ ↑ | Alloc MAE ↓ | Bootstrap CV ↓ |
  |---|---|---|---|---|
  | **Survival/Poisson Shapley** ⭐ | **0.016** | 0.905 | **0.019** | 0.126 |
  | Survival/Poisson BackElim | 0.046 | **1.000** | 0.083 | **0.096** |
  | Survival/Poisson AICPE | 0.026 | 0.905 | — | 0.096 |
  | Incremental Shapley (LR) | 0.029 | 0.905 | 0.013 | 0.600 |

- **권장 default**: 채널 예산 결정에는 Survival/Poisson Shapley + Marginal G-comp. 캠페인 시나리오 분석에는 Path-level Δ. 사후 정산 보고에는 Conditional Shapley.

---

## 1. Attribution 이 답해야 하는 질문 두 개 — Conditional vs Marginal

### 1.1 두 질문은 다른 estimand 다

| 질문 | View | 답하는 것 |
|---|---|---|
| "이번 분기 전환 크레딧을 어떻게 채널에 나눌까?" | **Conditional** | 전환자 모집단 안에서 채널 mix 분배 |
| "Email 예산 -10% 시 모집단 전환 몇 % 줄까?" | **Marginal G-comp** | 전 모집단의 counterfactual lift |

**시간 방향이 핵심**: 과거 분석 → Conditional, 미래 결정 → Marginal.

### 1.2 왜 같은 데이터로 두 view 가 다른 답을 내는가 — 응급실 비유

> "응급실 환자만 분석해서 '구급차 탄 사람이 더 자주 죽는다' 고 결론내면 안 됨. 구급차가 사망 원인이 아니라, 위중한 사람들이 구급차를 탔던 것."

광고에 자주 노출된 전환자 = "구급차 탄 사람". **광고 효과 ≠ 노출 빈도와 전환의 raw correlation**. 전환 (post-treatment outcome) 에 conditioning 하면 Pearl/Hernán 의 standard 결과로 collider bias 가 도입된다.

### 1.3 의사결정 → view 매핑

| 의사결정 | View | 비고 |
|---|---|---|
| 사후 attribution audit (분기 정산 보고) | Conditional | 전환 발생분의 채널 배분 |
| 광고비 ROI 정산 (retrospective) | Conditional | 이번 분기 spend ↔ credit |
| **채널 예산 재배분 (forward)** | **Marginal G-comp** | A/B test estimand 와 정렬 |
| **A/B test 사전 effect size** | **Marginal G-comp** | Channel holdout 효과 예측 |
| 캠페인/journey 시나리오 디자인 | 둘 다 | Path-level 보고와 결합 |

본 시뮬에서 두 view 의 ranking 은 완전 일치 (ρ=1.000), magnitude 만 일부 차이 (Paid Search Cond 0.318 → Marg 0.267, -5%p). **Ranking 일치 → 권고 robust**, **magnitude 차이 → 예산 의사결정 시 Marginal 권장**.

---

## 2. 비교군 방법론 — 한 단락씩

본 프로젝트의 중심은 §3 의 Survival/Poisson 통합 framework. 본 섹션은 비교 baseline 으로 다른 5 계열을 짧게 요약한다.

### 2.1 Rule-based heuristics (Last / First / Linear / Time Decay / Position-Based)

고정 규칙으로 즉시 credit 배분. journey 받으면 추가 학습 없이 즉시 산출 가능. **장점**: 빠름, 설명 단순, baseline benchmark. **한계**: 시너지·시간 감쇠·baseline 모두 무시, causal claim 불가능. 의사결정 보조보다는 "다른 방법론이 얼마나 다른가" 의 비교 기준으로 활용.

### 2.2 Markov Chain (1st/2nd order + Removal Effect)

채널 간 전이확률을 추정해 graph 를 만들고, 특정 채널을 "제거" 했을 때의 전환율 감소량을 그 채널의 credit 으로 사용 (Removal Effect). **장점**: 채널 dependency 일부 포착, structural model 이라 약한 causal claim 가능. **한계**: order 가 짧으면 long-range 시너지 못 잡음, 시간 감쇠 미반영. 본 프로젝트 분류로는 "Correlational + structural" tier.

### 2.3 Total Shapley (model-based, 128 coalitions)

게임이론 가치배분을 attribution 에 적용. response model 의 conversion 확률을 value function 으로 두고, 7 채널 = 128 coalition 의 평균 marginal contribution 을 계산. **장점**: efficiency 공리 (합 = 전체 value) 로 "fair" 한 시너지 분배. **한계**: baseline 미차감 → "pure correlational" tier. "incremental" 효과를 잡지 못함.

### 2.4 Du Incremental Shapley (KDD 2019)

별도 response model (원 권고 LSTM, 본 프로젝트는 단순 LR) 로 conversion 확률 학습, **baseline (광고 안 봐도 일어났을 전환) 을 명시적으로 차감** 한 incremental value $v(S) = \hat P(Y\mid S) - \hat P(Y\mid \emptyset)$ 에 exact Shapley 적용. **장점**: incremental 의미 명시. **한계**: 시간 감쇠 explicit 처리 안 됨, 우측 절단 미처리.

본 프로젝트와의 관계: §3.2 에서 Survival 위에 Du-style Shapley 를 통합해 **두 paper 가 단일 framework 의 두 instance** 임을 보였다.

### 2.5 Causal Debiased (IPW / Doubly Robust / DML)

Propensity score $e(W)$ 를 명시적으로 추정해 weighting/matching/DR/DML 로 채널별 ATE 산출. **장점**: strict causal estimator (DR/DML 은 outcome 또는 propensity 중 하나만 정확해도 consistent). **한계**: 채널별 ATE 만 산출 — credit 합이 1 이 아니라 budget 결정 직접 활용이 어렵다.

본 프로젝트 위치: Methodology 05 §8.1 의 "Survival × IPW Hybrid (doubly robust survival)" 가 후속 개발 1순위.

---

## 3. 본 프로젝트의 중심 — Survival/Poisson 통합 framework

§3 가 본 문서의 본론이다. §3.1 은 paper 기반 base 채택의 이유, §3.2~3.7 은 본 프로젝트가 추가로 연구·개발한 6개 항목이다.

### 3.1 Base — Shender 2023 TEDDA 의 채택과 이유

**무엇 (모델)** — inhomogeneous Poisson process 기반의 log-linear intensity 모델.

$$\log \lambda(t) = \alpha_0 + \sum_j f(t - t_j) + (\text{user feature shift})$$

- $\lambda(t)$ = 시점 $t$ 에서의 conversion intensity (시간 의존)
- $f(\cdot)$ = 광고 노출 시점부터의 시간 감쇠 함수, 채널별로 다른 형태 가능
- log-additive ⇒ rate-multiplicative: 광고 두 개를 본 효과는 rate 의 곱

**어떻게 (알고리즘 흐름)**:

1. **Interval split** — 각 user 의 journey 를 intensity 가 piecewise constant 인 시간 구간으로 분할. 각 interval = 하나의 Poisson regression observation.
2. **Poisson GLM 적합** — 각 interval 의 conversion count 를 response, 길이 $\Delta t$ 를 offset (log $\Delta t$) 으로 두고 채널별 step-function bin 더미 + user feature 더미를 regressor 로 fit (statsmodels GLM, log link).
3. **Intensity 평가** — fitted GLM 으로 임의 광고 부분집합 $A$ 의 $\hat\lambda(t^*, A)$ 산출.
4. **Credit allocation** — §3.2 에서 다룰 BackElim / Shapley / AICPE 중 선택해 채널별 credit 분배.

**왜 이 방식인가** (다른 모델 대비 우위):

| 요소 | TEDDA Poisson | 단순 LR | Rule-based |
|---|---|---|---|
| 우측 절단 (8h 미관측 ≠ 0) | ✅ Requirement #1 명시 처리 | ❌ | ❌ |
| 채널별 시간 감쇠 차별화 | ✅ Requirement #2, $f(t-t_j)$ explicit | ❌ | ❌ |
| User feature regression adjustment | ✅ Eq 10 더미 | 가능 | ❌ |
| Interval Poisson 으로 추정 단순 | ✅ Eq 12 환원 | ✅ | n/a |

본 프로젝트 채택 spec: **5-bin step function** (0-24h / 24-72h / 72-168h / 168-336h / 336h+). AIC 검증에서 단순 단일 step 대비 충분한 개선 + position hook 까지 포함 시 best AIC (notebook 02 (Main) §2).

### 3.2 추가 ① — Survival × Shapley 통합 (Du 와 Shender 의 수학적 동치)

**무엇을 추가했나** — Shender paper 본문은 BackElim 이 primary credit (§4.2.1), Shapley 는 비교용으로만 등장한다 (§4.2.3). 본 프로젝트는 **동일 fitted GLM 위에 Shapley credit 을 1급 시민으로 통합**해, paper 두 편 (Shender TEDDA + Du Incremental Shapley) 을 단일 framework 의 두 instance 로 통일했다.

**이론적 근거 — Shapley constant-invariance**:

$$\phi_i(v) = \phi_i(v + c) \quad \text{for any constant } c$$

⇒ $v(A) = \hat\lambda(A)$ 와 $v(A) = \hat\lambda(A) - \hat\lambda(\emptyset)$ 가 동일 channel credit 을 산출. Shender 의 intensity backbone 위에 Du 의 "incremental" framework 가 자연스럽게 올라간다.

**알고리즘 흐름**:

1. 같은 fitted GLM 의 intensity predictor 호출
2. 7 채널 = 128 coalition 의 $\hat\lambda$ 평가
3. exact Shapley 평균 marginal:

$$\phi_j = \sum_{S \subseteq N\setminus\{j\}} \frac{|S|!(n-|S|-1)!}{n!}\,[v(S\cup\{j\}) - v(S)]$$

**실증 근거** (본 시뮬 100K 측정):

| Method | MAE ↓ | Allocation MAE ↓ |
|---|---|---|
| Survival/Poisson **Shapley** | **0.016** | **0.019** |
| Survival/Poisson **BackElim** | 0.046 | 0.083 |

같은 GLM 위 credit 분배 방식만 바꿔도 **MAE 2.9× / Allocation 4.4× 개선**.

**운영 해석**:
- **BackElim**: 시너지를 마지막 광고에 집중 → bidding 시점 의사결정 (현재 광고 marginal 만 고려) 에 정렬
- **Shapley**: 시너지를 광고들에 균등 분배 → magnitude 정확도 + budget allocation 정확도 우위 → **사후 정산·예산 배분의 1차 권고**

### 3.3 추가 ② — 4-Cell Ablation Matrix (Long-Path Credit Bias 의 credit-allocation 측면 해결)

**풀려는 bias 문제 — Long-Path Credit Concentration**:

- 본 시뮬의 16-20 step long path 가 1-2 step path 대비 **3.8× 높은 conversion rate** (within New segment).
- 그러나 16-20 step 유저의 **96.3% 가 전환 안 함** → long path 는 conversion 의 *correlation* (selection effect) 일 뿐 *causation* 이 아님.
- **Credit allocation 단계에서의 증폭**: BackElim 류는 시너지를 *마지막 광고* 에 100% 집중. 20-step path 에서는 ad #20 이 누적 시너지를 통째로 흡수 → **long path 의 last touch 채널 (보통 lower-funnel: Direct, Email) 이 over-credit** 된다.
- 결과: BackElim 채널 credit 이 lower-funnel 로 inflate → 예산 의사결정에 bias 도입.

**무엇으로 풀었나 — 2D ablation matrix**:

같은 attribution 문제를 **response model 축** × **credit allocation 축** 의 2D matrix 로 펼쳐 bias 의 두 출처를 분리한다.

| Response \\ Credit | BackElim | **Shapley** ⭐ | AICPE |
|---|---|---|---|
| **Poisson interval (Shender)** | 현 BE | **현 Shapley** (신규) | 현 AICPE |
| **LR (sklearn)** | (의미 약함) | **Du IncShap (LR)** | (의미 약함) |
| **LSTM (Du 원 권고)** | (미구현) | 미구현 | (미구현) |

분리되는 bias 출처:
- **(a) response model**: long-path 응답의 시간 감쇠·user feature adjustment — Poisson 이 LR 보다 우월 (시간 감쇠 explicit)
- **(b) credit allocation**: 분배 방식 — Shapley 가 시너지를 광고들에 균등 분배 → BE 의 last-ad 집중 문제 직접 해소

**실증 결과 — 어느 축이 더 결정적인가**:

| 비교 | MAE 변화 | 의미 |
|---|---|---|
| 같은 Poisson, BE → Shapley | 0.046 → **0.016** (2.9× ↑) | credit allocation 축 |
| 같은 Shapley, LR → Poisson | 0.029 → 0.016 (1.8× ↑) | response model 축 |
| 같은 BE, Allocation MAE | 0.083 → 0.019 (4.4× ↑) | credit allocation 축 |

→ **credit allocation 선택이 backbone 선택보다 long-path bias 해소에 더 결정적**. 본 프로젝트가 Shapley credit 을 1차 권고하는 정량 근거.

### 3.4 추가 ③ — Channel ↔ Path Duality (Long-Path Bias 의 aggregation 측면 해결)

**풀려는 bias 문제 — Naive Path Credit 의 Conversion-Count Inflation**:

마케팅 보고에 자주 등장하는 "어떤 path template 이 가장 conversion 을 많이 가져왔는가" 류 분석은 보통 **observed conversion count 기반** (path 별 전환자 수, path 별 평균 credit) 으로 산출된다. 그러나 long path 가 conversion 과 correlate 하므로 이 집계 방식은 자동으로 long path 를 inflate 시킨다.

- 예: "20-step path 가 1-step path 보다 raw count 로 더 많은 conversion 을 가져왔다" → "긴 캠페인이 효과적" 으로 결론
- 실제: high-intent 유저가 long path 를 자기-선택 → selection bias
- Collider bias: 전환에 conditioning 한 path 비교는 표준 인과 추론에서 spurious link 도입

**무엇으로 풀었나 — 모델 기반 Path Credit + Efficiency Axiom**:

Path-level credit 을 **observed conversion count 가 아닌**, fitted GLM 으로부터 계산:

$$\Delta_\text{path} = \hat\lambda(\text{path}) - \hat\lambda(\emptyset)$$

GLM 이 자동 처리하는 bias 완화 요소:
- (a) 채널별 시간 감쇠 (step function decay) — long path 후반부 광고의 marginal contribution 자동 감쇠
- (b) User feature adjustment — segment intercept shift 로 selection 일부 보정
- (c) Interval Poisson 의 saturation — long-path 의 marginal 증가가 bounded

⇒ 20-step path 가 1-step path 의 20× credit 을 받지 않는다. GLM 이 산출하는 incremental intensity 차이로 bounded.

**수학적 일관성 — Efficiency Axiom 의 역할**:

| Aggregation 단위 | Quantity | 용도 |
|---|---|---|
| Channel-level | $\phi_c$ (per-channel Shapley) | 예산 결정 |
| Path-level | $\Delta_\text{path} = \hat\lambda(\text{path}) - \hat\lambda(\emptyset)$ | 캠페인 시나리오 |

Shapley efficiency 공리에 의해 path 내 channel credit 의 합 = $\Delta_\text{path}$. 즉 **두 view 가 별도 method 가 아니라 같은 incremental Shapley 의 다른 aggregation**. 한 fitted GLM 으로 두 보고를 산출하면 magnitude 가 자동 일관된다.

**왜 path-level 까지 명시 framework 화 했나**:

- 마케팅 의사결정 단위가 두 가지 — *채널 예산* vs *캠페인/journey 시나리오 디자인*
- 분리해서 풀면 두 보고의 magnitude 가 어긋남 (sum 불일치) → 의사결정자 혼란
- 본 프로젝트의 통합: 단일 GLM + efficiency 보증 → 어느 level 에서 보고해도 long-path bias 회피 + magnitude 일관

> **주의**: §3.4 의 path-level Δ 는 *aggregation 단계* 의 inflation 을 회피하지만, *selection bias 자체* (high-intent 유저가 long path 를 선택) 는 §3.5 의 Marginal G-computation 이 모집단 평균으로 추가 해소한다. 두 항목은 **상호보완**.

### 3.5 추가 ④ — Conditional vs Marginal G-computation 분리

**무엇을 추가했나** — 같은 channel credit 을 두 estimand 로 산출하는 framework. §1 에서 개념 도입했고, 여기서는 본 프로젝트의 구현 의의를 정리한다.

| View | 모집단 | 답하는 질문 |
|---|---|---|
| Conditional Shapley | 전환자만 | 분기 정산: 전환 발생분을 어떻게 분배? |
| Marginal G-comp Shapley | 전 모집단 (전환자 + 비전환자) | 미래 결정: 모집단에서 채널 lift 는? |

**왜 분리가 필요했나** (collider bias 핵심):

- 전환은 **post-treatment outcome**. 여기에 conditioning 하는 Conditional view 는 Pearl/Hernán 의 standard 결과로 ATE 추정에 spurious link 를 도입
- 마케팅 예산 의사결정 (Email +30% → 모집단 효과) 의 estimand 와 **Marginal view 가 정확히 일치** → A/B test 가 측정하는 quantity 와 동등
- 시뮬레이션에서 DGP-known counterfactual Shapley (전 모집단 기반) 와 비교 시 **Marginal 이 MAE 0.020 으로 가장 가까움** (Conditional 0.012 는 sample 내 truth 와 일치하나, 모집단 causal truth 와는 거리 있음)

**알고리즘 흐름**:

1. 같은 fitted GLM
2. **Conditional**: converted users 만 group-by → 채널별 Shapley 평균
3. **Marginal**: 전 모집단 g-computation — 각 user 의 counterfactual $\hat\lambda(\text{full}_u) - \hat\lambda(\text{full}_u \setminus c)$ 계산 후 평균

**본 시뮬 결과**:

| 항목 | 값 |
|---|---|
| 채널 ranking 일치도 | Spearman ρ = **1.000** |
| 최대 magnitude 차이 | Paid Search Cond 0.318 → Marg 0.267 (-5%p) |
| 나머지 6 채널 \|Δ\| | < 0.025 |

→ **Ranking 일치 → 권고 robust**. **Magnitude 차이 → 예산 결정 시 Marginal 권장**, 사후 정산에는 Conditional 그대로.

### 3.6 추가 ⑤ — "Causal" 라벨의 정직한 5-tier

**무엇을 추가했나** — 18종 attribution method 를 **causal 강도 5단계** 로 재분류해 "incremental = 자동 causal" 오해를 정정했다.

| 강도 | Method (본 프로젝트) | 정당성 |
|---|---|---|
| Truly causal (data-based) | Shender §4.2.2 (Eq 19/20) + query events | RCT 식별, 실험 데이터 필수 |
| Truly causal (debiased) | IPW / DR / DML | propensity-based ATE |
| **Causal — outcome model only** | Survival/Poisson 3종, Du IncShap, CAMTA | regression adjustment, **propensity 미보정** |
| Correlational + structural | Markov 1st/2nd | 약한 structural causal claim |
| Pure correlational | Last/First/Linear/Time Decay/Position/Total Shapley/LSTM/Transformer | heuristic / association |

**왜 추가했나** — 본 프로젝트의 Survival 계열 (BE/Shapley/AICPE) 은 baseline 차감 ($\sum \phi = \hat\lambda(N) - \hat\lambda(\emptyset)$) 을 수행하나, 이는 **outcome model 정확성 + no-unobserved-confounders 가정의 결과** 이지 propensity 보정 (strict debiasing) 이 아니다. 마케팅 의사결정에서 "incremental" 라벨이 "정확한 causal lift" 로 오해되지 않도록 본 프로젝트는 outcome model only 와 debiased 를 **두 sub-category 로 명시 분리**했다.

**핵심 차이**:
- **Outcome model only**: outcome model 이 잘못되면 bias. propensity 모델링 없음.
- **Debiased (DR/DML)**: outcome 또는 propensity 둘 중 하나만 맞아도 consistent.

### 3.7 추가 ⑥ — Multivariate User Feature 일반화

**무엇을 추가했나** — Shender Eq 10 의 user feature 는 paper 에서 단일 age bucket 예시만 제시. 본 프로젝트는 **임의 multivariate user feature** (segment + device + country + registration cohort 등) 를 generic 으로 지원하도록 API 일반화.

**흐름** — feature 목록을 받으면 alphabetical 첫 level 을 reference 로 두고 level-1 개 더미 컬럼을 자동 추가, Poisson GLM 의 regressor 에 합류.

**왜 추가했나** — 실 운영 데이터에서는 단일 confounder 가정 불가. device, geography, registration cohort 등 다중 user feature 가 필수. paper 의 single-feature framing 을 실 운영 가능 형태로 generic 화했다.

**DAG-based 가이드라인** (실무 적용 시 주의):

1. **Pre-treatment only**: 모든 user feature 는 광고 노출 *이전* 시점 측정.
2. **Mediator 회피**: 광고가 영향을 줄 수 있는 변수 (이 세션의 page view 수 등) 는 제외.
3. **Collider 회피**: conversion 의 후속 변수, 또는 conversion 과 confounder 의 공동 자손은 제외.
4. **DAG-based selection**: domain knowledge + DAG 로 backdoor 차단 confounder 만 선택. 모든 가용 column 을 무작정 던지지 말 것.
5. **현 framework 한계**: outcome model only — propensity 미보정. Strict debiasing 은 §3.8 Future.

### 3.8 미구현 / 다음 단계

| 우선순위 | 항목 | 의의 |
|---|---|---|
| 1 | **Survival × IPW Hybrid** (doubly robust survival) | propensity 보정 추가 → "Causal (outcome model)" 에서 "Causal (debiased)" 로 승급 |
| 2 | LSTM response model (Du 원 권고) | 4-cell matrix 의 (LSTM, Shapley) cell 완성 |
| 3 | Experimental data 시뮬 (query_events 분리) | Shender §4.2.2 의 strict causal incremental 평가 |
| 4 | Sensitivity analysis (E-value, Rosenbaum bounds) | unobserved confounding robustness 정량화 |

상세는 Methodology 05 §8.

---

## 4. 본 시뮬레이션 결과 — 어떤 method 가 무엇에 강한가

100K 시뮬, GT-A (intensity-based ground truth) 대비.

### 4.1 정확도 (MAE, Kendall τ, Top-3 hit rate)

| Method | MAE ↓ | Kendall τ ↑ | Top-3 ↑ |
|---|---|---|---|
| **Survival/Poisson Shapley** ⭐ | **0.016** | 0.905 | 67% |
| Survival/Poisson AICPE | 0.026 | 0.905 | 67% |
| Incremental Shapley (LR) | 0.029 | 0.905 | 67% |
| Survival/Poisson BackElim | 0.046 | **1.000** | 100% |

→ **Shapley credit on Survival backbone 이 magnitude 정확도 1위**. BackElim 은 ranking 측면 1위.

### 4.2 안정성 (Bootstrap CV)

| Method | mean CV ↓ |
|---|---|
| Survival/Poisson BE / AICPE | 0.096 |
| **Survival/Poisson Shapley** | 0.126 |
| Incremental Shapley (LR) | 0.600 |

→ BE 와 Shapley 의 stability 거의 동등. Du LR-IncShap 은 4.8× 불안정.

### 4.3 예산 배분 정확도 (Allocation MAE)

| Method | Allocation MAE ↓ |
|---|---|
| Incremental Shapley (LR) | 0.013 |
| **Survival/Poisson Shapley** | **0.019** |
| Survival/Poisson BackElim | 0.083 |

→ Shapley credit on Survival 이 BE 대비 4.4× 개선, Du LR-IncShap 에 근접.

### 4.4 Conditional vs Marginal (본 시뮬)

| 항목 | 값 |
|---|---|
| 채널 ranking | ρ = 1.000 (완전 일치) |
| Paid Search magnitude | Cond 0.318 → Marg 0.267 (-5%p) |
| 나머지 6 채널 \|Δ\| | < 0.025 |

### 4.5 데이터 스케일 민감도

| n_users | BackElim MAE | **Shapley MAE** |
|---|---|---|
| 1K | 0.079 | 0.063 |
| 10K | 0.040 | 0.028 |
| 50K | 0.048 | 0.016 |
| 100K | 0.049 | **0.012** |

→ Shapley credit 의 sample efficiency 가 BE 보다 우월. 큰 데이터에서 격차가 더 커진다.

---

## 5. 목적별 method 선택 가이드 (의사결정 매트릭스)

| 목적 | 1차 권장 | 2차 권장 | 비고 |
|---|---|---|---|
| **사후 정산 audit (분기 보고)** | Conditional Shapley (Survival × Shapley, channel) | Conditional BackElim | Cond view, 전환자 모집단 |
| **채널 예산 재배분 (forward)** | **Marginal G-comp Shapley** ⭐ | Survival × Shapley (Cond) | Marg view, A/B test estimand 정렬 |
| **캠페인 / journey 시나리오 디자인** | **Path-level Δ_path** ⭐ | (BE path total — telescoping 으로 동일 quantity) | §3.4 aggregation |
| **Channel ranking 만 필요** | Survival/Poisson BackElim | Survival × Shapley | τ=1.000 vs 0.905 |
| **Credit magnitude 정확** | **Survival × Shapley** ⭐ | Survival/Poisson AICPE | MAE 0.016 |
| **균형잡힌 production 1차** | **Survival × Shapley** ⭐ | (Survival BE 보조) | MAE + Alloc + CV 모두 상위 |
| **강한 causal claim — debiased** | DML | DR / IPW | unconfoundedness 가정 |
| **강한 causal claim — data-based** | Survival × incremental (+ `query_events`) | (RCT 데이터 필요) | experimental data 필수 |
| **빠른 baseline (개념 설명용)** | Last Click / Time Decay | Linear | 해석 단순 |
| **Paper-faithful TEDDA** | Survival × BackElim | — | Shender §4.2.1 primary |
| **Paper-faithful Du** | Survival × Shapley | Incremental Shapley (LR) | 본 프로젝트 통합 |

---

## 6. 한계와 다음 단계

### 6.1 본 framework 의 한계

- **Outcome model 의존**: 본 프로젝트의 Survival 계열은 "Causal — outcome model only" tier. outcome model 정확성 + no-unobserved-confounders 가정에 의존. propensity 미보정.
- **Confounder set W 의존**: user feature 더미가 잡지 못하는 unobserved confounder 가 있으면 bias 잔존.
- **Observational data**: 본 시뮬과 실 데이터 모두 RCT 가 아님. Shender §4.2.2 의 strict causal claim 은 query_events 분리 데이터가 필요.
- **분석은 의사결정 *지원* 도구이지 *증명* 도구가 아님**. Ultimate 검증은 A/B test.

### 6.2 다음 단계 (우선순위)

1. **Survival × IPW Hybrid** (Methodology 05 §8.1): propensity model 추가 → doubly robust survival. outcome model 또는 propensity 둘 중 하나만 정확해도 consistent.
2. **A/B test holdout 검증**: ultimate ground truth. 채널 holdout 실험으로 Marginal G-comp 의 estimand 와 직접 비교.
3. **추가 W (device, country, registration cohort, history)**: 실 데이터 적용 시 multivariate user feature 활용 (§3.7).
4. **LSTM response model**: 4-cell matrix 의 (LSTM, Shapley) cell 완성 (Du 원 권고).

---

## 부록 — 두 소스 문서와의 매핑

| 본 문서 | Methodology 05 (학술) | Marketing Handout (한 장) |
|---|---|---|
| §0 TL;DR | §0 Overview + §7 Decision | TL;DR |
| §1 Conditional vs Marginal | §3.4 + §3.5 (4-layer) | 본문 전체 |
| §2 비교군 5계열 | §4 Causal 5-tier 의 카테고리 | (없음) |
| §3.1 Survival base | §1.3 (Shender Eq 2-10) | (없음) |
| §3.2 Shapley 통합 | §3.1 (constant invariance) + §3.3 (BE vs Shapley) | (없음) |
| §3.3 4-cell ablation | §3.2 | (없음) |
| §3.4 Channel↔Path | §3.4 | 본 시뮬 결과 |
| §3.5 Cond vs Marg | §3.5 | 본문 핵심 |
| §3.6 5-tier causal | §4.2-4.3 | (없음) |
| §3.7 Multivariate user feature | §1.3 (Eq 10) + 단위테스트 18-19 | (없음) |
| §4 시뮬 결과 | §6 | 본 시뮬 적용 |
| §5 의사결정 매트릭스 | §7 | 의사결정 매트릭스 |
| §6 한계 / Future | §8 | 한계 + Next Steps |
