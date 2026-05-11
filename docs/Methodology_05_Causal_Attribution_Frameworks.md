# Methodology 05 — Causal Attribution Frameworks: Shender TEDDA + Du Incremental Shapley 통합

> **목적**: Survival/Poisson Attribution (Shender et al. 2023) 과 Incremental Shapley (Du et al. 2019) 의 수학적 통합 framework 제시 + "causal" 표기의 정직한 구분 + 본 codebase 구현 안내.
>
> **선행 문서**: `Methodology_02_Attribution_Methods.md` Section 6.2 (Survival 단독), Section 6.6 (Incremental Shapley 단독). 본 문서는 두 방법론의 **공통 backbone** 과 **credit allocation 의 다양성** 을 다루며, Methodology_02 의 보강이자 학술 정합성 정리.

---

## 0. Overview — 왜 통합 framework 가 필요한가

본 codebase 의 attribution 방법론은 18종 (Methodology_02 참조). 그중 **Survival/Poisson** 과 **Incremental Shapley** 두 방법론은 표면적으로 다른 framework 처럼 보이나, 실제로는:

1. **공통 구조**: 둘 다 "response model + credit allocation" 의 2 단계 파이프라인
2. **차이점**: response model 의 형태 (Poisson intensity vs. Logistic regression) 와 credit 분배 방식 (BackElim vs. Shapley)
3. **수학적 동치**: Shapley credit on Survival/Poisson backbone = Du-style Incremental Shapley with Poisson response

이 문서는:
- Shender 2023 TEDDA 의 Section 4 전수 정리
- Du 2019 Incremental Shapley 의 framework 정리
- 두 방법의 **통합 ablation matrix** (response × credit)
- "causal" 라벨의 정확한 의미 (Shender 본인 진술 인용)

---

## 1. Shender 2023 TEDDA 전수 정리

### 1.1 논문 메타

- **제목**: A Time To Event Framework For Multi-touch Attribution
- **저자**: Shender, Amini, Bao, Dikmen, Richardson, Wang (Google)
- **출처**: Journal of Data Science (2023), ArXiv 2009.08432
- **약칭**: TEDDA (Time to Event Data Driven Attribution)

### 1.2 두 가지 핵심 요구사항

논문 Section 3 에서 명시:

**Requirement #1 — 우측 절단 (censoring)**:
> 실시간 데이터는 본질적으로 incomplete. 광고 노출 후 8시간 동안 전환이 없다 ≠ 결과 0. 향후 1일/1주 의 결과를 모르므로 right-censored 로 처리해야 함.

**Requirement #2 — 시간 의존 효과 (time-varying)**:
> 광고의 전환 효과는 시간에 따라 다름 (보통 노출 직후 강하고 점차 감소). 단순 lookback window 가 아닌 연속 감쇠 함수로 모델링해야 함.

### 1.3 Section 4.1 Modeling User Conversion Behavior — 점진적 일반화

#### 4.1.1 단일 광고 모델 (Eq 2-5)

가장 단순한 형태:

$$\log \lambda(t) = \alpha_0 + f(t - t_1) \quad \text{[Eq 2]}$$

- $\alpha_0$: baseline log-intensity (광고 없을 때)
- $f(\cdot)$: 시간 감쇠 함수, $f(x) = 0$ for $x \le 0$ (광고 전에는 효과 없음)

논문은 $f$ 의 3가지 parameterization 을 제시:

**옵션 1 (Eq 3) — exponential mixture**:
$$\log \lambda(t) = \alpha_0 + \sum_{l=1}^{L} \beta_l \exp(-\theta_l (t - t_1))$$

**옵션 2 (Eq 4) — spline basis**:
$$\log \lambda(t) = \alpha_0 + \sum_{l=1}^{L} \beta_l b_l(t - t_1)$$

**옵션 3 (Eq 5) — step function** (← 본 codebase 채택):
$$\log \lambda(t) = \alpha_0 + \beta_1 \mathbb{1}\{t-t_1 \le 24\} + \beta_2 \mathbb{1}\{24 < t-t_1 \le 48\} + \beta_3 \mathbb{1}\{48 < t-t_1 \le 720\}$$

> *"We do not make a specific recommendation here; rather, this is an area for future study."* (논문 p.7)

본 구현은 **5-bin step function** 사용: 0-24h / 24-72h / 72-168h / 168-336h / 336h+. Section 4.1.7 의 추정 편의성 (interval Poisson 환원) 으로 step function 이 사실상 implementation primary.

#### 4.1.2 광고 feature (Eq 6)

광고에 부수 feature (포맷, device 등) 추가:

$$\log \lambda(t) = \alpha_0 + f(t-t_1) + \sum_{k=1}^{K} g_k(t-t_1, x_{1k}) \quad \text{[Eq 6]}$$

- $g_k$: 각 feature 의 시간 감쇠 함수 (보통 $f$ 보다 단순한 형태)
- 본 codebase: `extra_ad_features` hook (default off)

#### 4.1.3 다중 광고 (Eq 7) + 위치 효과 (Eq 8) + cross-광고 상호작용 (Eq 9)

**Eq 7** — 다중 광고 효과는 log-scale 에서 additive (multiplicative on rate):

$$\log \lambda(t) = \alpha_0 + \sum_j f(t-t_j) + \sum_{j,k} g_k(t-t_j, x_{jk}) \quad \text{[Eq 7]}$$

→ "두 광고를 보면 conversion rate 가 곱해짐" — 직관적이지 않은 부분. 50개 광고 시 2^50 배 곱해지는 비현실 → 광고 간 상호작용으로 보정 (Eq 9).

**Eq 8** — 광고 위치 효과 (1번째 vs 2번째 광고가 다른 효과):

$$\log \lambda(t) = \alpha_0 + \sum_j f_j(t-t_j) \quad \text{[Eq 8]}$$

→ $f_j$ = j번째 광고의 효과 함수. 본 codebase: `include_position` hook (first/last/middle 더미).

**Eq 9** — cross-광고 시너지 (이전 광고가 다음 광고 효과에 영향):

$$\log \lambda(t) = \alpha_0 + \sum_j f(t-t_j) + \sum_j g_1(t-t_j) \cdot \mathbb{1}\{t_j - t_{j'} < \Delta \text{ for some } j' < j\} \quad \text{[Eq 9]}$$

→ 본 codebase: `include_cross_channel` hook (24h 윈도우 내 인접 광고 페어).

#### 4.1.4 유저 feature (Eq 10)

유저 특성 (age bucket — 논문 예시; 또는 country, device, segment 등) 이 baseline 에만 영향 시:

$$\log \lambda(t) = \alpha_0 + \alpha_{\text{age bucket}} + \sum_j f(t-t_j) + \sum_{j,k} g_k(t-t_j, x_{jk}) \quad \text{[Eq 10]}$$

논문은 generic **"user feature"** 로 표현 — age bucket 을 예시로 들며, 어떤 user-level 변수든 가능. **"confounding variable" 이라는 용어 부재** — 단순히 baseline α₀ shift 의 regression adjustment.

→ 본 codebase: `compute_survival_attribution(..., user_features=("segment", ...))` 인자로 임의 multivariate user feature 지원. Default 는 backward-compat 유지 위해 `("segment",)`. 각 feature 는 alphabetical 순 first level 을 reference 로 두고 $\text{level} - 1$ 개 더미 컬럼 (`u_{feature}_{level}`) 을 자동 추가. 실제 운영 데이터 (device, country, age bucket 등) 에서는 `user_features=("segment", "device", "country")` 등으로 generic 호출.

> **중요한 framing 구분** — 논문의 user feature ≠ 자동으로 causal confounder
> - **논문 Eq 10**: predictive baseline shift (어떤 user feature 든 OK, mediator/collider 도 포함 가능). Section 1 명시: observational data → correlational.
> - **Causal interpretation**: 해당 user feature 가 DAG 상 backdoor confounder 일 때만 causal claim. Mediator/collider 면 오히려 bias 유발.
> - 본 시뮬의 segment 는 **공교롭게 둘 다 만족** (segment → channel 선호 + segment → conversion baseline 의 common cause). 그러나 이는 우리 DGP 설계의 결과지 논문이 segment 를 confounder 로 명시했기 때문이 아님.
> - **현 구현의 정확한 위치**: Eq 10 의 user feature 더미 추가는 **regression adjustment / outcome model 의 일부** (causal inference 의 1차 접근 중 하나) — propensity-based debiasing 이 아님. Strict causal claim 을 위해서는 § 8 Future Work 의 Survival × IPW hybrid 가 필요.

#### 4.1.5 Experimental data (Eq 11)

실험 데이터 (광고 ablation 가능) 가 있을 때, **query event** 와 **ad event** 분리 모델:

$$\log \lambda(t) = \alpha_0 + \alpha_{\text{user}} + \sum_j f(t-t_j) \mathbb{1}\{\text{ad j shown}\} + \sum_{j,k} g_k(t-t_j, x_{jk}) \mathbb{1}\{\text{ad shown}\} + \sum_j m(t-t_j) + \sum_{j,k} n_k(t-t_j, x_{jk}) \quad \text{[Eq 11]}$$

여기서:
- $m, n_k$: query effect (광고 의도 자체의 효과 — 광고가 ablated 되어도 발생)
- $f, g_k$: ad effect (광고가 실제로 보여졌을 때의 추가 효과)
- $\mathbb{1}\{\text{ad j shown}\}$: ablation 마스크

→ **이 모델은 query/ad 분리 데이터 (experimental data) 가 있어야만 의미 있음.** 본 codebase: `query_events: pd.DataFrame` 인자 (default None — observational mode).

#### 4.1.6 추가 정제 (refinements)

**(a) 계절성**: time-of-day, day-of-week, holidays → indicator 또는 spline.
- 본 codebase: `include_seasonality` hook.

**(b) 반복 전환**: 한 유저가 여러 번 전환 가능 시 (예: 의류 재구매). 단일 전환만 가능한 경우는 single-occurrence survival model 로 환원.
- 본 codebase: `interval_df["conversion_count"]` 가 integer 로 자동 처리.

**(c) Mutually exciting Poisson process**: 이전 전환이 미래 conversion intensity 에 영향 (예: 옷 한 번 사면 또 살 가능성 ↑).
- 본 codebase: `include_self_excitation` hook (DGP 가 1-conversion-per-user 라 default off).

#### 4.1.7 Estimation (Eq 12) — 추정 방법

논문의 핵심 estimation 권고:

> *"If $f, g_k$ are either piecewise constant or can be approximated as such, then by **breaking each user's path into intervals where the intensity $\lambda(t)$ is constant**, we can treat each interval as an observation in a Poisson regression problem. The number of conversions in the interval is then the response, and the **length of the interval is the offset**."* (논문 Section 4.1.7)

inhomogeneous Poisson process 의 log-likelihood (Eq 12):

$$\sum_{i=1}^{N} \left[ -\int_0^\tau \lambda_i(t)\,dt + \sum_{j=1}^{C_i} \log \lambda_i(T_{ij}) \right] \quad \text{[Eq 12]}$$

→ piecewise-constant 환경에서 **interval-level Poisson regression with offset = log(Δt)** 으로 환원. 본 codebase v3 의 핵심 추정 방식.

### 1.4 Section 4.2 Credit Assignment

#### 4.2.1 Backwards Elimination (Eq 13) — paper primary

> *"The method we propose, which we call backwards elimination..."* (논문 p.12)

**Eq 13** — 정의:

$$\text{RawCredit}(j) = \hat\lambda(t^*, A^{(j)}) - \hat\lambda(t^*, A^{(j-1)}) \quad \text{[Eq 13]}$$

여기서 $A^{(j)}$ = 첫 j 개 광고 집합, $A^{(0)} = \emptyset$.

**알고리즘**:
1. 전환 시점 $t^*$ 에서 $\hat\lambda(t^*, A^{(n)})$ (모든 광고 활성) 부터 시작
2. 마지막 광고 부터 역순으로 제거하며 intensity 감소분을 해당 광고 credit 으로 누적
3. Telescoping: $\sum_{j=1}^n \text{RawCredit}(j) = \hat\lambda(A^{(n)}) - \hat\lambda(\emptyset)$

**특징**: **시너지를 마지막 광고에 몰아줌** (super-additive 시 +, sub-additive 시 −). 광고 입찰 (ad bidding) 응용에 적합 — bidding 시점에 미래 광고를 알 수 없으므로 현재 광고의 marginal + 과거 광고와의 시너지 만 고려하는 것이 합리적.

**Normalization 옵션** (Eq 17, 18):

$$\text{NormalizedCredit}(j) = \frac{\hat\lambda(t^*, A^{(j)}) - \hat\lambda(t^*, A^{(j-1)})}{\hat\lambda(t^*, A^{(n)})} \quad \text{[Eq 17, 분모 포함 baseline]}$$

$$\text{NonBaselineNormalizedCredit}(j) = \frac{\hat\lambda(t^*, A^{(j)}) - \hat\lambda(t^*, A^{(j-1)})}{\hat\lambda(t^*, A^{(n)}) - \hat\lambda(t^*, \emptyset)} \quad \text{[Eq 18, baseline 제외]}$$

본 codebase: `normalize="sum_to_one"` (default), `"eq17"`, `"eq18"` 옵션.

#### 4.2.2 Incremental Attribution (Eq 19, 20) — experimental data 한정

실험 데이터 (Eq 11 모델) 가 있을 때, ad-only ablation:

$$\log \hat\lambda(t^*, A^{(l)}) = \hat\alpha_0 + \hat\alpha_{\text{user}} + \sum_{j=1}^{l} \hat f(t^*-t_j) \mathbb{1}\{\text{ad shown}\} + \sum_{j=1}^{l}\sum_k \hat g_k \mathbb{1}\{\text{ad shown}\} + \sum_{j=1}^{n} \hat m(t^*-t_j) + \sum_{j=1}^{n}\sum_k \hat n_k \quad \text{[Eq 20]}$$

→ ad effect 만 ablate ($f, g_k$), query effect ($m, n_k$) 는 모든 query 유지. 결과적으로 **incremental conversions** (광고 인한 추가 전환) 만 광고들에 배분.

→ 본 codebase: `credit_method="incremental"` + `query_events` 인자. **observational data 환경 (본 시뮬레이션 포함) 에서는 이 method 의 strict causal claim 은 무효.**

#### 4.2.3 Synergy & Shapley 비교 (Eq 21, 24, 25)

**Eq 21** — 시너지 정의:

$$S(A^{(j-1)}, A_j) = m(A^{(j)}) - m(A^{(j-1)}) - m(\{A_j\})$$

여기서 $m(A) = \hat\lambda(A) - \hat\lambda(\emptyset)$ (marginal credit).

**Eq 24** — 광고 간 시간 간격이 멀수록 시너지 감소:

$$S(\{A_1\}, A_2) = (\exp(f_1(t^* - t_1)) - 1)(\exp(f_2(t^* - t_2)) - 1)$$

**Eq 25** — Shapley value:

$$\phi_j(v) = \sum_{O \subseteq \Omega \setminus \{j\}} \frac{|O|! (N-|O|-1)!}{N!} [v(O \cup \{j\}) - v(O)]$$

→ 논문은 광고 attribution 의 value function 으로 $v(\cdot) = \hat\lambda(t^*, \cdot)$ 을 명시 (또는 동등하게 $v(A) = \hat\lambda(A) - \hat\lambda(\emptyset)$):

> *"In the case of ad attribution, the value function is $v(\cdot) = \hat\lambda(t^*, \cdot)$ (or equivalently, $v(A) = \hat\lambda(t^*, A) - \hat\lambda(t^*, \emptyset)$ for any set A)"* (논문 p.17)

**핵심 정리** — BE vs Shapley 분배 차이 (Eq 22-27, 2-광고 toy):

$$\text{BE}(A_2) - \text{Shapley}(A_2) = \tfrac{1}{2} S(\{A_1\}, A_2)$$

→ BE 는 시너지 100% 를 마지막 광고에, Shapley 는 50:50 으로 분배.

→ 본 codebase: 단위테스트 13번 (`test_be_minus_shapley_equals_half_synergy`) 이 이 관계를 1e-9 정확도로 검증.

---

## 2. Du et al. 2019 Incremental Shapley 정리

### 2.1 논문 메타

- **제목**: Causal Inference for Recommendation: A Real-World Multi-touch Attribution Application
- **저자**: Du, Lee, Ghaffarzadegan, Wang (JD.com / Stanford GSB)
- **출처**: KDD 2019

### 2.2 Two-Step Pipeline

**Step 1 — Response Modeling**:

관측 데이터로 conversion 확률 모델 학습:

$$\hat P(Y=1 \mid \text{exposure features}) = \text{model}(X)$$

원 논문은 **RNN/LSTM** 권고 (시퀀스 처리). 본 codebase 는 **LogisticRegression** 으로 단순화 (sample 효율 + 빠른 학습 우선).

**Step 2 — Incremental Value Function & Shapley**:

$$v(S) = \hat E[Y \mid S \text{ exposed}] - \hat E[Y \mid \emptyset] \quad \text{[Du Eq, baseline subtracted]}$$

→ **"광고 인한 incremental conversion"** 만 분배. baseline conversion (광고 안 봐도 일어났을 전환) 은 어떤 광고에도 배분 안 됨.

Shapley credit:

$$\phi_j = \sum_{S \subseteq N\setminus\{j\}} \frac{|S|!(n-|S|-1)!}{n!} [v(S \cup \{j\}) - v(S)]$$

### 2.3 Du vs. Shender 의 framework 비교

| 항목 | Du et al. 2019 | Shender et al. 2023 |
|---|---|---|
| 출처 | KDD '19 | JDS '23 |
| Response model | RNN/LSTM (원 권고) — 본 codebase LR | Inhomogeneous Poisson process (intensity model) |
| Value function unit | Conversion **probability** ∈ [0,1] | Conversion **rate** (intensity, events / unit time) |
| Credit allocation | Shapley (Section 2.2 Step 2) | BackElim (paper primary) 또는 Shapley (Section 4.2.3) |
| "Incremental" 정의 | $v(S) - v(\emptyset)$ — baseline 차감 | Shender 4.2.2 (data-based, experimental) 또는 model-based (BE telescoping = $\hat\lambda(N) - \hat\lambda(\emptyset)$) |
| 시간 정보 활용 | RNN 의 sequence representation | $f(t-t_j)$ 의 explicit decay |
| 우측 절단 | 명시적 미처리 | **명시적 처리** (Requirement #1) |

---

## 3. 통합 Framework — Same Backbone, Different Credit

### 3.1 핵심 통찰: Shapley Constant-Invariance

Shapley 의 수학적 성질:

$$\phi_i(v) = \phi_i(v + c) \quad \text{for any constant } c$$

증명: $v'(A) = v(A) + c$ 이면, marginal $v'(S\cup\{i\}) - v'(S) = v(S\cup\{i\}) - v(S)$ 로 상수가 상쇄. 따라서 Shapley credits 동일.

이로부터:

$$\phi_i(v(A) = \hat\lambda(A)) = \phi_i(v(A) = \hat\lambda(A) - \hat\lambda(\emptyset))$$

→ Shender 의 4.2.3 Shapley 변형 ($v = \hat\lambda$) **= Du 의 incremental Shapley framework with Poisson response** ($v = \hat\lambda - \hat\lambda(\emptyset)$). **수학적으로 동치**.

### 3.2 4-Cell Ablation Matrix

response model × credit allocation 의 2D matrix:

| Response \ Credit | BackElim (Shender 4.2.1) | Shapley (Shender 4.2.3 / Du) | AICPE (non-paper) |
|---|---|---|---|
| **Poisson interval (Shender)** | **현 v3 BackElim** ⭐ paper primary | **신규 통합** ⭐ Section 4.2.3 | 현 v3 AICPE |
| **LR (sklearn)** | (의미 약함 — temporal info 손실) | **현 `incremental_shapley.py`** | (의미 약함) |
| **LSTM (Du 원 논문)** | (가능, 미구현) | **Du 원 논문 정확 재현** (미구현) | (가능, 미구현) |

본 codebase 의 위치:
- ✅ Poisson + BackElim (`compute_survival_attribution(credit_method="backelim")`)
- ✅ Poisson + Shapley (`compute_survival_attribution(credit_method="shapley")`) — **신규 추가**
- ✅ Poisson + AICPE (`compute_survival_attribution(credit_method="aicpe")`)
- ✅ LR + Shapley (`compute_incremental_shapley(...)`)
- ⏳ LSTM + Shapley (미구현, future work)

### 3.3 BE vs Shapley 의 정량적 차이

같은 $\hat\lambda$ backbone 에서 BE 와 Shapley 의 channel credit 차이는 정확히:

$$\text{BE}(A_j) - \text{Shapley}(A_j) = \tfrac{1}{2} \sum_{j'\le j-1} S(A^{(j-1)}, A_j) - \tfrac{1}{2} \sum_{j'\ge j+1} S(A^{(j)}, A_{j'}) + \cdots$$

(2-광고 단순화 시 정확히 ½·S, 3+ 광고 시 multi-way synergy 항 포함)

**해석**:
- 시너지 (super-additive 광고 조합) 가 클수록 BE 와 Shapley 차이 ↑
- BE 는 시너지를 후반 광고에 집중 → 후반 광고가 "biased upward"
- Shapley 는 시너지를 전 광고에 균등 분배 → "fair" 하나 magnitude 정확도 ↓ 가능

**우리 100K 시뮬레이션 결과** (v3 측정, 2026-05-08):

| Method | MAE ↓ | Kendall τ ↑ | Top-3 ↑ | Bootstrap CV ↓ | Allocation MAE ↓ |
|---|---|---|---|---|---|
| Survival/Poisson **BackElim** (paper primary) | 0.046 | **1.000** | 100% | **0.096** | 0.083 |
| **Survival/Poisson Shapley** ⭐ (통합) | **0.016** | 0.905 | 67% | 0.126 | **0.019** |
| Survival/Poisson **AICPE** | 0.026 | 0.905 | 67% | 0.096 | (재측정 필요) |
| Incremental Shapley (LR) | 0.029 | 0.905 | 67% | 0.600 | 0.013 |

**핵심 발견** (Phase 2 patch 결과):
- **Shapley credit 의 정확도 (MAE 0.016) 가 BackElim (0.046) 보다 2.9× 우수** — 시너지를 광고들에 균등 분배하는 것이 magnitude 정확도 측면에서 유리
- **Allocation MAE 도 0.019** (BE 0.083 대비 4.4× 개선) — Shapley credit 이 budget allocation 에 더 적합
- Bootstrap CV 0.126 (BE 0.096 와 비슷) — stability 거의 동등
- **Kendall τ 만 BE 의 1.000 보다 0.905 로 약간 낮음** — magnitude 정확하나 ranking 1-2 위 swap 발생 가능

---

## 4. "Causal" 라벨의 정직한 구분

### 4.1 Shender 본인 진술 (논문 Section 1)

> *"Our model can be used with either observational data or data from randomized experiments... **Observational data can be interpreted in terms of the correlation between showing ads and conversions, while data from randomized experiments allows for a causal estimate** of the number of conversions caused by ads."*

→ **TEDDA 자체로는 자동 causal 이 아님**. observational data → correlational, experimental data (Eq 11 + 4.2.2) → causal.

### 4.2 Causal claim 의 5 단계 (정정 — 학술 정합성)

이전 3 단계 분류를 5 단계로 세분화. 핵심 정정은 Survival/Poisson 계열 (BE/Shapley/AICPE) 과 Du IncShap, CAMTA 를 **"Causal — outcome model only"** 로 재분류 — 이들은 strict 의미의 *debiasing* (propensity 보정) 이 아닌 **regression adjustment / outcome model** 접근이며, causal inference 의 한 갈래이지만 propensity-based debiased estimator 와 구별해야 함.

| 강도 | Method | 정당성 |
|---|---|---|
| **Truly causal (data-based)** | Shender 4.2.2 (with `query_events`) | 실험 데이터에서 ad-shown vs ad-withheld 직접 ablation. Eq 19/20. RCT 식별. |
| **Truly causal (debiased)** | IPW, Doubly Robust, DML | propensity-based ATE estimator. unconfoundedness 가정 + propensity model. 잘못된 outcome model 에도 (DR/DML) consistent. |
| **Causal — outcome model only** ⚠️ | Survival/Poisson (BE/Shapley/AICPE), Du Incremental Shapley (LR/LSTM/Poisson), CAMTA | **regression adjustment**. user feature 더미 (Eq 10) + outcome model fit 만으로 causal 구조 가정. **Outcome model 정확성 + no unobserved confounders 양쪽 가정 의존**. propensity 모델링 없음 → strict debiasing 아님. |
| **Correlational with structural assumption** | Markov Chain (1st/2nd order, Removal Effect) | structural Markov 모델은 transition 구조 + removal effect 를 통해 약한 causal interpretation 시도. 그러나 outcome model 도 propensity 도 아님. |
| **Pure correlational** | Last/First Click, Linear, Time Decay, Position-Based, Total Shapley (model-based) | heuristic 또는 mode-free association. baseline 차감 없음. causal claim 불가능. |

**정정 포인트**:

1. **이전 "Model-based incremental"** → **"Causal — outcome model only"**. "incremental" 라벨이 자동으로 causal 을 함의하는 것처럼 오해될 수 있어 더 정확한 framework 표기로 변경. baseline 차감 ≠ propensity 보정.
2. **Markov Chain** 은 기존 "Correlational" 단독 분류였으나, structural model 의 약한 causal claim 을 인정해 별도 tier 로 분리.
3. **Strict debiasing 의 정의**: propensity model $e(W) = P(\text{exposure} \mid W)$ 를 명시적으로 추정하고 weighting / matching / DR / DML 으로 causal estimator 를 구성하는 것. Outcome model 단독은 이 조건 미충족.

### 4.3 본 codebase 의 분류 (5-tier 적용)

`METHOD_CATEGORIES` dict (모든 experiments + docs 일관 적용):

```python
METHOD_CATEGORIES = {
    # Rule-based heuristics — pure correlational
    "Last Click":   "Rule-based",
    "First Click":  "Rule-based",
    "Linear":       "Rule-based",
    "Time Decay":   "Rule-based",
    "Position-Based": "Rule-based",

    # Statistical — correlational with structural assumption
    "Markov (1st)": "Statistical",
    "Markov (2nd)": "Statistical",

    # Game-theoretic — pure correlational (no baseline)
    "Shapley (model-based)": "Game-theoretic",

    # Deep Learning — pure correlational (predictive only)
    "LSTM+Attention (LOO)":  "Deep Learning",
    "LSTM+Attention (attn)": "Deep Learning",
    "Transformer (2L/2H)":   "Deep Learning",

    # Causal — outcome model only (regression adjustment)
    # 모델 가정 (no unobserved confounders + correct outcome spec) 의존, propensity 미보정
    "Survival/Poisson (BackElim)": "Causal (outcome model)",
    "Survival/Poisson (AICPE)":    "Causal (outcome model)",
    "Survival/Poisson (Shapley)":  "Causal (outcome model)",
    "Incremental Shapley":         "Causal (outcome model)",
    "CAMTA (Causal Attention)":    "Causal (outcome model)",  # causal regularization

    # Causal — debiased (propensity-based ATE)
    "IPW":           "Causal (debiased)",
    "Doubly Robust": "Causal (debiased)",
    "DML":           "Causal (debiased)",
}
```

→ 이전 "Causal (incremental)" 라벨 폐기. 두 sub-category 로 명확히 구분:
- **"Causal (outcome model)"**: Survival/Poisson 3종 + Incremental Shapley + CAMTA — **regression adjustment** 기반. user feature 더미 (Eq 10) 으로 baseline 차감하나 propensity 미보정. Outcome model 정확성 가정 단독 의존.
- **"Causal (debiased)"**: IPW / DR / DML — propensity model 명시적 추정 기반 ATE estimator. Outcome model 부정확해도 (DR/DML) propensity 정확하면 consistent.

**왜 "incremental" 라벨을 버리는가**:
- "incremental" 은 "baseline 차감 → 추가 전환 분배" 의 *의도* 를 표현한 라벨이지만, 그 자체로 자동 causal 함의가 없음.
- Survival/Poisson Shapley 는 $\sum \phi_i = \hat\lambda(N) - \hat\lambda(\emptyset)$ 의 incremental sum 을 분배하지만 — 이 "incremental" 이 strict causal incremental 이 되려면 $\hat\lambda(\emptyset)$ 이 정확한 counterfactual baseline 이어야 함 (no unobserved confounders 가정). 가정 충족 여부와 무관하게 method 자체는 **outcome model 의 prediction subtraction** 이지 propensity-based estimator 가 아님.
- 따라서 method 의 *technical class* 를 정확히 표기하는 "outcome model" 이 더 적절.

### 4.4 함의 — Exp 05 재해석

기존 Exp 05 결과: "Causal MAE 0.045 vs Correlational MAE 0.045 (거의 동일)" → 가설 약하게 지지.

5-tier 분류 적용 후 재해석:

| 카테고리 | 평균 MAE (medium confounding) |
|---|---|
| Pure correlational (Last/Time Decay/Position/Total Shapley) | (Phase 2 후 갱신) |
| Correlational + structural (Markov 1st/2nd) | (Phase 2 후 갱신) |
| Causal — outcome model (Survival 3종 + Inc Shapley + CAMTA) | (Phase 2 후 갱신) |
| Causal — debiased (IPW/DR/DML) | (Phase 2 후 갱신) |

→ "Causal (debiased)" 가 confounding 보정 효과를 보이는지 **단독** 확인 가능. 기존 통합 비교는 outcome-model method 들이 평균을 끌어내려 propensity-based debiasing 의 효과가 묻혀짐.

---

## 5. 본 codebase 구현 안내

### 5.1 파일 구조

```
part1_simulation/models/causal/
├── survival_attribution.py     # Shender TEDDA + 통합 Shapley
│   ├── _build_interval_features  # Eq 12 interval split
│   ├── _fit_poisson_model        # Poisson GLM + log Δt offset
│   ├── _predict_intensity_at     # λ̂(t*, A) 평가
│   ├── _backwards_elimination_credits  # Eq 13 BE
│   ├── _shapley_credits          # Eq 25 Shapley (신규, 통합 핵심)
│   ├── _aicpe_credits            # non-paper extension
│   ├── _incremental_credits      # Eq 19/20 (experimental data)
│   ├── _compute_synergy_for_path # Eq 21
│   ├── compute_synergy_report    # 분석 도구
│   └── compute_survival_attribution  # 외부 API (4 credit modes)
└── incremental_shapley.py      # Du 2019 (LR response)
    ├── _build_user_features
    ├── _train_response_model     # LogisticRegression
    ├── _compute_exact_shapley
    └── compute_incremental_shapley
```

### 5.2 사용 예시

#### 통합 framework — 동일 backbone, 4가지 credit:

```python
from part1_simulation.models.causal.survival_attribution import compute_survival_attribution

# 논문 primary — BackElim (Eq 13)
r_be = compute_survival_attribution(journeys, credit_method="backelim")

# Section 4.2.3 Shapley 변형 — 통합 framework 핵심
r_sh = compute_survival_attribution(journeys, credit_method="shapley")

# Non-paper AICPE
r_aic = compute_survival_attribution(journeys, credit_method="aicpe")

# 실험 데이터 (Eq 19/20)
r_inc = compute_survival_attribution(
    journeys, credit_method="incremental", query_events=query_df
)
```

#### Du 의 LR 기반 IncShap (별도 함수):

```python
from part1_simulation.models.causal.incremental_shapley import compute_incremental_shapley

r_du = compute_incremental_shapley(journeys, sample_users=3000)
```

#### 모든 옵션 hook (Eq 6/8/9/10, 4.1.6):

```python
r = compute_survival_attribution(
    journeys,
    credit_method="shapley",
    user_features=("segment", "device", "country"),  # Eq 10 multivariate
    include_position=True,        # Eq 8
    include_cross_channel=True,   # Eq 9
    include_seasonality=True,     # 4.1.6 (a)
    include_self_excitation=True, # 4.1.6 (c)
    extra_ad_features=["format"], # Eq 6 gₖ
    cross_channel_window_hours=24.0,
)
```

#### Multivariate user feature 사용 시 권장 가이드라인

Eq 10 의 user feature 는 **predictive baseline shift** — 어떤 user-level 변수든 모델에 추가 가능하나, causal interpretation 시에는 다음 주의:

1. **Pre-treatment only**: 모든 user feature 는 광고 노출 *이전* 시점에 측정된 변수여야 함. 노출 후 변수는 mediator/collider 가능성 → bias 유발.
2. **DAG-based selection**: domain knowledge + DAG 로 backdoor 경로를 막는 confounder 후보를 식별 (예: device, geography, registration cohort). 모든 가용 column 을 던져넣지 말 것.
3. **Mediator 회피**: 광고 노출이 영향을 줄 수 있는 변수 (예: "이 세션의 페이지뷰 수") 는 mediator → 제외.
4. **Collider 회피**: outcome (conversion) 의 후속 변수 또는 conversion 과 confounder 의 공동 자손은 conditioning 시 새로운 bias 경로 개방 → 제외.
5. **현 framework 의 한계**: 이 가이드라인을 따라도 *outcome model* 만 보정 — propensity 미보정. Strict debiasing 은 § 8 Future Work 참조.

### 5.3 단위테스트 (19개, Section 4.1.1 ~ 4.2.3 전수 + Eq 10 multivariate)

`part1_simulation/tests/test_survival_attribution.py`:

| # | 테스트 | 검증 대상 |
|---|---|---|
| 1 | `test_interval_construction_single_user` | 4.1.7 interval split |
| 2 | `test_offset_present_in_design` | Eq 12 log Δt offset |
| 3 | `test_paper_eq5_recovery_single_channel` | Eq 5 step-function decay 회복 |
| 4 | `test_eq7_multiplicative_combination` | Eq 7 log-additive |
| 5 | `test_eq10_segment_intercept_shift` | Eq 10 user feature shift (단일) |
| 6 | `test_eq11_query_ad_split` | Eq 11 query/ad 분리 |
| 7 | `test_repeat_conversion_integer_response` | 4.1.6 (b) 반복 전환 |
| 8 | `test_telescoping_invariant_unclamped` | Eq 13 telescoping |
| 9 | `test_normalization_eq17_eq18_run` | Eq 17/18 normalization |
| 10 | `test_eq20_incremental_keeps_query_effect` | Eq 20 incremental query 유지 |
| 11 | `test_eq21_synergy_paper_example` | Eq 21 시너지 (paper Example: S=2) |
| 12 | `test_eq24_synergy_decreases_with_gap` | Eq 24 단조 감소 |
| 13 | `test_be_minus_shapley_equals_half_synergy` | Section 4.2.3 BE−Shapley=½S |
| 14 | `test_shapley_constant_invariance` | Shapley 상수 불변성 |
| 15 | `test_shapley_efficiency_paper_example` | Σφ = λ̂(N)−λ̂(∅) |
| 16 | `test_shapley_2_player_paper_example` | paper Eq 26-27 (φ_D=2, φ_E=3) |
| 17 | `test_shapley_du_incremental_equivalence` | v=λ̂ vs v=λ̂−λ̂(∅) 동치 |
| **18** | **`test_multivariate_user_features`** | **Eq 10 multivariate (segment + device): u_segment_* + u_device_* 동시 추가, log-additive 검증** |
| **19** | **`test_user_features_default_backward_compat`** | **default `("segment",)` = explicit `["segment"]` 1e-9 동치** |

**실행**:
```bash
PYTHONPATH=. python -m pytest part1_simulation/tests/test_survival_attribution.py -v
# 19/19 passed
```

---

## 6. 실험 결과 — 통합 framework 의 4-method 비교

(Phase 2 patch 결과 자동 반영. 100K 시뮬, GT-A 대비)

### 6.1 정확도 비교 (Exp 01, v3 측정 100K)

| Method | MAE ↓ | Kendall τ ↑ | Top-3 ↑ |
|---|---|---|---|
| **Survival/Poisson Shapley** ⭐ | **0.016** | 0.905 | 67% |
| Survival/Poisson AICPE | 0.026 | 0.905 | 67% |
| Incremental Shapley (LR) | 0.029 | 0.905 | 67% |
| Survival/Poisson BackElim | 0.046 | **1.000** | 100% |

→ **Shapley credit on Survival backbone 이 정확도 1위**. Du LR-IncShap (0.029) 보다 1.8× 우수, BE (0.046) 보다 2.9× 우수. paper Section 4.2.3 Shapley 변형 + log Δt offset interval Poisson 의 결합 효과.

### 6.2 안정성 비교 (Exp 10, Bootstrap CV)

| Method | mean CV ↓ |
|---|---|
| Survival/Poisson BackElim/AICPE | 0.096 |
| **Survival/Poisson Shapley** | 0.126 |
| Incremental Shapley (LR) | 0.600 |

→ Shapley credit 의 stability 가 BE/AICPE 와 거의 동등 (0.126 vs 0.096). Du LR (0.600) 대비 4.8× 개선.

### 6.3 Budget Allocation (Exp 07, v3 측정)

| Method | Allocation MAE ↓ |
|---|---|
| Incremental Shapley (LR) | 0.013 |
| **Survival/Poisson Shapley** | **0.019** |
| Survival/Poisson BackElim | 0.083 |

→ Shapley credit 이 BE 대비 **4.4× 개선** (0.083 → 0.019). Incremental Shapley LR (0.013) 에 근접. **시너지 균등 분배 특성이 budget allocation 에서 정확** (BE 의 시너지 집중은 lower-funnel 채널 over-allocation 을 유발).

### 6.4 Data scale curve (Exp 03, v3 측정)

| n_users | BackElim MAE | **Shapley MAE** | LR-IncShap MAE |
|---|---|---|---|
| 1K | 0.079 | 0.063 | (n/a sample) |
| 5K | 0.053 | 0.032 | (다른 표) |
| 10K | 0.040 | 0.028 | |
| 50K | 0.048 | 0.016 | |
| 100K | 0.049 | **0.012** | |

→ **Shapley credit 의 sample efficiency 가 BackElim 보다 우월** — 50K-100K 에서 0.016→0.012 로 감소 (BE 는 0.048→0.049 정체). 큰 데이터에서 paper-faithful Shapley 가 가장 정확.

### 6.4 Channel Credit 패턴 차이

(Phase 2 후 시각화 추가)

---

## 7. 권장 사항 — 목적별 method 선택

| 목적 | 1차 권장 | 2차 권장 | 비고 |
|---|---|---|---|
| **Channel ranking 만 필요** | Survival/Poisson **BackElim** | Survival/Poisson **Shapley** | BE: τ=1.0 (perfect), Shapley: τ=0.91 |
| **Credit magnitude 정확** | **Survival/Poisson Shapley** ⭐ | Survival/Poisson AICPE | Shapley MAE=0.016 (단연 1위) |
| **Budget allocation** | Incremental Shapley (LR) | **Survival/Poisson Shapley** | Alloc MAE 0.013 vs 0.019 (근접) |
| **균형잡힌 production 1차** | **Survival/Poisson Shapley** ⭐ | (Survival BE 보조) | MAE + Alloc + CV 모두 상위 |
| **Causal claim 강함 (debiased)** | DML | DR / IPW | unconfoundedness 가정 |
| **Causal claim 강함 (data-based)** | Survival/Poisson **incremental** + `query_events` | (필요 시 RCT 데이터) | experimental data 필수 |
| **Stability 우선** | Survival/Poisson (any credit) | Markov | CV 0.096 |
| **Paper-faithful TEDDA** | Survival/Poisson **BackElim** | — | Section 4.2.1 primary |
| **Paper-faithful Du** | Survival/Poisson **Shapley** | Incremental Shapley (LR) | 통합 framework |

---

## 8. 향후 작업 (Future Work)

### 8.1 Survival × IPW Hybrid (Debiased Survival/Poisson) — 우선순위 1

**문제 의식**: 현 Survival/Poisson 계열 (BE/Shapley/AICPE) 은 § 4.2 의 분류상 **"Causal — outcome model only"** 즉 regression adjustment. propensity model 을 추정하지 않으므로 strict 의미의 debiased causal estimator 가 아님. 본 hybrid 는 outcome model + propensity model 두 가지를 결합하여 **doubly robust** 성질 (둘 중 하나만 정확해도 consistent) 을 갖는 Survival 변형을 구축.

**Pipeline 설계**:

1. **User feature W 식별** (DAG 기반): 광고 노출과 conversion 의 backdoor 경로를 막는 pre-treatment user feature 집합 $W$ 선정. 본 시뮬의 경우 segment ($W = \text{segment}$); 실 데이터의 경우 device, country, registration cohort 등.

2. **Per-channel propensity 추정**: 각 채널 $c$ 별로
   $$e_c(W_i) = P(\text{user } i \text{ ever exposed to channel } c \mid W_i)$$
   를 user 수준 logistic regression 으로 추정. multinomial / per-channel 두 옵션 모두 가능.

3. **Stabilized IPW weights**:
   $$w_{i,c} = \frac{P(\text{exposed to } c)}{e_c(W_i)} \quad \text{(stabilized)}$$
   각 interval row 에 해당 channel 의 weight 부여. multi-channel exposure 시 weight 의 곱 (assumption: per-channel exposure 의 conditional independence) 또는 propensity score *vector* 의 별도 처리.

4. **Weighted Poisson GLM**:
   ```python
   sm.GLM(y, X, family=Poisson(link=Log()), offset=log_dt, freq_weights=ipw_weights)
   ```
   IPW weighting 으로 propensity-corrected outcome model 추정.

5. **BackElim/Shapley credit on weighted model**: 기존 § 4.2 알고리즘을 가중 모델에 적용. credit 의 의미는 "propensity-adjusted incremental intensity contribution".

**Doubly robust 성질**:
- Outcome model (Eq 12 Poisson + user feature 더미) 정확 → consistent
- Propensity model ($e_c(W)$ logistic) 정확 → consistent
- 둘 다 정확 → semi-parametric 효율적
- 둘 다 잘못 → bias

**Status**: 미구현, 후속 task. 구현 시 신규 파일 `survival_propensity.py` 또는 `survival_attribution.py` 끝부분에 stub 함수 (현재 NotImplementedError).

**관련 문헌**:
- Robins, Rotnitzky & Zhao (1994) — IPW/DR foundations
- Bang & Robins (2005) — doubly robust outcome+propensity
- Chernozhukov et al. (2018) — DML (cross-fitted DR with ML nuisance models)

### 8.2 DAG-based Confounder Selection Helper

**미구현**. `select_confounders(dag_edges, treatment, outcome)` 헬퍼 함수로 backdoor 경로를 막는 minimal sufficient adjustment set 을 자동 식별. networkx 기반 backdoor 알고리즘 + d-separation 검증.

### 8.3 Sensitivity Analysis (E-value, Rosenbaum bounds)

**미구현**. 현 outcome-model 기반 attribution 의 unobserved confounding 에 대한 robustness 정량화. E-value (VanderWeele & Ding 2017) 또는 Rosenbaum bounds.

### 8.4 Time-varying Confounding 처리

**미구현, 우선순위 낮음** (현 DGP 가 정적 segment 만 가지므로). 실 운영 데이터 적용 시 고려.

### 8.5 LSTM Response Model

**부분 구현** (`incremental_shapley_lstm.py`). 후속: 4 alternative DGPs 에서 LSTM 의 robustness 검증, hyperparameter sweep.

### 8.6 Experimental Data Strict Causal Evaluation

**미구현**. DGP 확장: query/ad split 시뮬 → Shender 4.2.2 (Eq 19/20) 의 strict causal claim 평가. 현 시뮬에는 query_events 가 없어 incremental mode 가 BE 로 fallback.

### 8.7 Triangulation Framework

**미구현**. Survival 3-credit + IncShap-LR/LSTM + DML 의 cross-validation framework. 다 method 가 동일 (또는 가까운) 채널 ranking 산출 시 신뢰도 ↑, 큰 disagreement 시 method-specific assumption violation 진단.

---

## 부록 A — 수학적 derivation: Shapley constant-invariance

Shapley value:

$$\phi_i(v) = \sum_{S \subseteq N\setminus\{i\}} \binom{n-1}{|S|}^{-1} \frac{1}{n} [v(S\cup\{i\}) - v(S)]$$

(상수 weight 표기 단순화).

Constant shift: $v'(A) = v(A) + c$. 그러면:
$$v'(S\cup\{i\}) - v'(S) = (v(S\cup\{i\}) + c) - (v(S) + c) = v(S\cup\{i\}) - v(S)$$

→ 모든 marginal 항이 동일 → Shapley credits 동일 (각 항이 marginal 의 가중합이므로). $\square$

이로부터 $v(A) = \hat\lambda(A)$ 와 $v(A) = \hat\lambda(A) - \hat\lambda(\emptyset)$ 가 같은 Shapley credit 산출. 단 **합계는 다름**:

$$\sum_i \phi_i(v(A) = \hat\lambda(A)) = \hat\lambda(N) - \hat\lambda(\emptyset)$$

(Shapley efficiency). 이 합계는 Du 의 "incremental conversions 총량" 과 일치 — 따라서 **Survival 위에 Shapley 를 올린 통합 method 는 자동으로 Du-style incremental** 이 됨.

---

## 부록 B — Section 4.2.3 paper Example 수치 검증

**Setup** (논문 p.16-17):
- $\hat\lambda(\emptyset) = 1$
- $\hat\lambda(\{A_1\}) = 2$
- $\hat\lambda(\{A_2\}) = 3$
- $\hat\lambda(\{A_1, A_2\}) = 6$ (super-additive: $2 \times 3 = 6 > 2 + 3 - 1 = 4$)

**Marginal credits**:
- $m(A_1) = 2 - 1 = 1$
- $m(A_2) = 3 - 1 = 2$
- $m(\{A_1, A_2\}) = 6 - 1 = 5$

**Synergy (Eq 21)**:
$$S(\{A_1\}, A_2) = m(\{A_1, A_2\}) - m(\{A_1\}) - m(\{A_2\}) = 5 - 1 - 2 = 2$$

**BackElim (Eq 13, 역순 제거)**:
- BE($A_2$) = $\hat\lambda(\{A_1, A_2\}) - \hat\lambda(\{A_1\}) = 6 - 2 = 4$
- BE($A_1$) = $\hat\lambda(\{A_1\}) - \hat\lambda(\emptyset) = 2 - 1 = 1$
- 합 = 5 ✓ (= $\hat\lambda(N) - \hat\lambda(\emptyset)$)

**Shapley (Eq 25, 2-player)**:
- $\phi(A_1) = \tfrac{1}{2}[(\hat\lambda(\{A_1\}) - \hat\lambda(\emptyset)) + (\hat\lambda(\{A_1, A_2\}) - \hat\lambda(\{A_2\}))] = \tfrac{1}{2}[1 + 3] = 2$
- $\phi(A_2) = \tfrac{1}{2}[(\hat\lambda(\{A_2\}) - \hat\lambda(\emptyset)) + (\hat\lambda(\{A_1, A_2\}) - \hat\lambda(\{A_1\}))] = \tfrac{1}{2}[2 + 4] = 3$
- 합 = 5 ✓

**BE − Shapley**:
- BE($A_1$) − Shapley($A_1$) = 1 − 2 = **−1** = −½·S
- BE($A_2$) − Shapley($A_2$) = 4 − 3 = **+1** = +½·S

→ BE 는 시너지(2) 100% 를 마지막 광고($A_2$) 에 부여. Shapley 는 시너지 50:50 으로 분배. 단위테스트 13 (`test_be_minus_shapley_equals_half_synergy`) 이 이 정확한 관계를 1e-9 정확도로 검증. $\square$
