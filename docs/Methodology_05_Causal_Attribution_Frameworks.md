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
- **Conditional Shapley (전환자 모집단) vs Marginal G-computation Shapley (전 모집단)** 의 estimand 분리 — collider bias 회피 + A/B test estimand 정렬 (§3.5)
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

> **Estimand 사전 공지** — Shender §4.2 의 BackElim/Shapley/Incremental 모두 paper default 로 **converted users 집계 (conditional estimand)** 를 가정한다. 즉 $v(S)$ 의 평균 모집단이 전환자만으로 한정된 conditional view. 모집단 수준 causal lift 로의 확장 (regression standardization / g-formula on ALL users) 은 본 프로젝트가 추가한 **Marginal G-computation 계열** — §3.5 참조. 본 §1.4 의 모든 수식은 paper-faithful conditional 형태로 기술됨.

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

> **세 axis 의 직교성** — §3.2 의 matrix 는 (response model × credit allocation) 의 2D. 본 문서는 두 축을 추가로 명시한다:
> - **§3.4 — Aggregation level** (channel-level $\phi_c$ ↔ path-level $\Delta_\text{path}$): 같은 fitted GLM 으로 두 보고 단위 자동 일관 (efficiency axiom)
> - **§3.5 — Estimand population** (Conditional: converted users only ↔ Marginal G-comp: ALL users): 같은 Shapley 알고리즘에서 value function $v(S)$ 의 평균 모집단 만 교체. collider bias 와 decision-alignment 가 핵심 이슈.
>
> 세 axis 모두 **단일 fitted Poisson GLM 위에서 호환** → 한 모델 fit 으로 (credit method × aggregation × estimand) 의 모든 조합 보고 가능. 본 codebase 의 통합 framework 가 fragmented method 모음 대비 갖는 핵심 이점.

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

### 3.4 Aggregation Level — Channel ↔ Path Duality

§3.1-3.3 의 통합 framework 는 **credit allocation method × response model** 의 2D matrix 였다. 실제 운영에서는 third axis 인 **aggregation level (channel vs path)** 도 marketing 의사결정 단위에 직결된다. 채널별 credit 은 *예산* 결정 도구, path-template 별 credit 은 *캠페인 시나리오* 결정 도구.

#### Two-level decomposition

같은 fitted GLM ($\hat\lambda$) 위에서 동일 game value $G = \hat\lambda(\text{full}_\text{population}) - \hat\lambda(\emptyset)$ 를 두 단위로 분해 가능:

| Aggregation | 단위 | 수식 |
|---|---|---|
| **Channel-level** (§3.1-3.3) | 7 channels | $G = \sum_c \phi_c$ — per-channel Shapley credit |
| **Path-level** (§3.4 신규) | path template (수천) | $G = \sum_\text{path} \text{count}_\text{path} \cdot \bar\Delta_\text{path}$ |

여기서 path 별:

$$\Delta_\text{path} \;\equiv\; \hat\lambda(\text{path}) - \hat\lambda(\emptyset) \;=\; \sum_{c\,\in\,\text{path}} \phi_c^{(\text{path})}$$

Shapley **efficiency axiom** 에 의해 path 내 channel Shapley credit 의 sum 은 $\hat\lambda(\text{path}) - \hat\lambda(\emptyset)$ 와 정확히 동일. **별도 method 가 아니라 같은 Incremental Shapley framework 의 다른 aggregation**.

#### Aggregation matrix (§3.2 4-cell 의 확장 view)

| Level \ Method | BackElim | Shapley (Incremental) ⭐ | AICPE |
|---|---|---|---|
| **Channel** ($\phi_c$) | per-channel ablation credit (Shender 4.2.1) | per-channel Shapley credit (Shender 4.2.3 / Du) | per-channel AICPE |
| **Path** ($\Delta_\text{path}$) | path total via telescoping | **path-level Incremental Shapley** (efficiency axiom) | — |

→ Shapley column 은 channel ↔ path 두 level 모두 일관 (efficiency 공리 보장). codebase standard 는 Shapley (마케팅 magnitude 정확도 우수) 이므로 path-level 도 Shapley framing 으로 통일.

#### 마케팅 의사결정 단위 매핑

| 의사결정 | Aggregation | 1차 권장 |
|---|---|---|
| **채널 예산 배분** | Channel-level | Survival/Poisson Shapley (§3.2 ⭐) |
| **캠페인/journey 시나리오 디자인** | Path-level | Path-level Incremental Shapley (§3.4 ⭐) |
| **예산 + 시나리오 동시 권고** | 두 view 병행 | 같은 fitted GLM, 다른 unit. 통일성은 efficiency axiom 으로 보장 |

#### 구현 노트

- `compute_survival_attribution(credit_method="shapley")` → channel-level (per-channel $\phi_c$)
- Path-level: 같은 fitted GLM 의 `_predict_intensity_at` 으로 $\hat\lambda(\text{path}) - \hat\lambda(\emptyset)$ 직접 계산 후 template 별 aggregate. **per-user explicit Shapley 호출 불필요** (efficiency axiom).
- Population 통일: §4 와 동일하게 converted users 만 사용 → exact identity ($\sum_u \Delta_u = \sum_c \phi_c \cdot n_\text{converted}$)
- 노트북 예제: `notebooks/part1/02_main_survival_incremental_shapley.ipynb` §7.5 — total identity check (rel err < 0.1%) 로 framework unity 실증

#### §3.1 constant-invariance 와의 관계

- §3.1: per-channel Shapley credits ($\phi_c$) 는 baseline 차감 여부에 invariant
- §3.4: 그러나 **합** (path total = $\hat\lambda(\text{path}) - \hat\lambda(\emptyset)$) 은 baseline 차감 필수 — incremental 의미를 보존
- 두 성질이 함께 path-level aggregation 의 well-definedness 보장 (부록 A derivation 참고)

### 3.5 Conditional vs Marginal G-Computation Estimand — 모집단 분리

§3.1-3.4 까지의 모든 credit (BackElim / Conditional Shapley / AICPE) 은 paper-faithful default 인 **converted users 모집단** 위에서 정의된다 (§1.4 사전 공지). 본 절은 같은 fitted GLM 위에서 **estimand 모집단** 만 교체하는 직교 axis — Marginal G-computation — 을 1급 시민으로 추가한다. **사용자 요청의 핵심 framing 인 "BackElim vs Incremental Shapley (Conditional) vs Incremental Shapley (Marginal G-comp)" 의 3-way 비교가 본 절의 출발점**.

#### 3.5.1 두 estimand 의 정의

| Estimand | 평균 모집단 | 답하는 질문 | 마케팅 의사결정 매핑 |
|---|---|---|---|
| **Conditional** (paper-faithful default) | converted users 만 ($U_\text{conv}$) | "이번 분기 *전환자* 의 채널 mix 를 어떻게 분배?" | 사후 attribution audit, ROI 정산 (retrospective) |
| **Marginal G-computation** (본 프로젝트 추가) | 전 모집단 ($U$ = converters + non-converters) | "Email 예산 -10% 시 *모집단* 전환 몇 % 줄까?" | Forward 채널 예산 재배분, A/B test 사전 effect size |

#### 3.5.2 수식 — value function 의 모집단만 교체

같은 fitted GLM 의 intensity predictor $\hat\lambda_u(t_u^*, A)$ 위에서:

$$v_\text{cond}(S) = \frac{1}{|U_\text{conv}|} \sum_{u \in U_\text{conv}} \hat\lambda_u(t_u^*, A_u \cap S)$$

$$v_\text{marg}(S) = \frac{1}{|U|} \sum_{u \in U} \hat\lambda_u(t_u^*, A_u \cap S)$$

이후 Shapley 분배는 **완전히 동일**:

$$\phi_c^\bullet = \sum_{S \subseteq N\setminus\{c\}} \frac{|S|!(n-|S|-1)!}{n!} [v_\bullet(S \cup \{c\}) - v_\bullet(S)] \quad (\bullet \in \{\text{cond}, \text{marg}\})$$

알고리즘적으로 **유일한 차이는 모집단 loop** (`user_source = journeys[converted]` vs `journeys`) — 코드 ref `_survival_credits.py` `_shapley_credits()` 의 `user_source` 분기 및 `_backwards_elimination_credits()` 의 `user_iter_source` 분기 (둘 다 `subpopulation` 파라미터로 제어).

#### 3.5.3 왜 분리가 필요한가 — collider bias + decision-alignment

**(a) Collider bias** — 전환은 **post-treatment outcome**. 광고 노출 → 전환 의 인과 그래프에서 전환에 conditioning 하는 Conditional view 는 Pearl/Hernán 의 standard 결과 (e.g., *Causal Inference: What If*, Ch. 8) 로 채널 효과 추정에 spurious link 를 도입한다. 직관적으로 **광고에 자주 노출된 전환자** 는 high-intent self-selection 의 결과지 광고 효과 그 자체가 아니다.

> **응급실 비유** (`Marketing_Handout_Conditional_vs_Marginal.md` 인용): *"응급실 환자만 분석해서 '구급차 탄 사람이 더 자주 죽는다' 고 결론내면 안 됨. 구급차가 사망 원인이 아니라 위중한 사람들이 구급차를 탔던 것."* — 광고에 자주 노출된 전환자 = "구급차 탄 사람".

**(b) Decision-alignment** — 마케팅 채널 예산 의사결정 (Email +30% → 모집단 전체 노출 변화) 의 estimand 는 정확히 모집단 평균 lift:

$$\text{ATE}_c = E_u\big[\hat\lambda(\text{full}_u) - \hat\lambda(\text{full}_u \setminus c)\big]$$

이는 **A/B test 가 측정하는 quantity** (channel holdout vs full exposure 의 모집단 평균 차이) 와 동등. Marginal G-comp 의 efficiency 합 $\sum_c \phi_c^\text{marg} = E_u[\hat\lambda(\text{full}_u) - \hat\lambda(\emptyset)]$ 가 정확히 이 estimand 의 채널 분해.

**(c) "G-computation" 명명 근거** — Pearl 의 g-formula / Robins 의 regression standardization: fitted outcome model 을 **각 user 의 actual covariate 위에서 evaluate 후 모집단 평균**. backdoor 조건이 충족되면 (no unobserved confounders, $W$ 가 backdoor 차단) 이 평균이 ATE 와 일치.

#### 3.5.4 §3.1 Constant-invariance 와의 직교성

§3.1 의 Shapley invariance ($\phi_i(v) = \phi_i(v + c)$) 는 **baseline 차감 여부** ($v = \hat\lambda$ vs $v = \hat\lambda - \hat\lambda(\emptyset)$) 에 대한 성질이지, **모집단 선택** (converters vs all) 에 대한 성질이 *아니다*. 두 axis 는 직교:

| Axis | 영향 | 결과 |
|---|---|---|
| Baseline 차감 (§3.1) | $v$ 에 상수 더하기 | **Shapley credit 동일** (invariant), 합만 차이 |
| 모집단 (§3.5) | 평균 대상 user 집합 변경 | **Shapley credit 변경** (각 user 의 covariate 가 다름) |

따라서 모집단 변경은 진짜로 다른 estimand 를 산출. 단, fitted GLM 이 monotone 하고 두 모집단의 covariate 분포가 충분히 겹치면 ranking 은 일치할 수 있다 (본 시뮬에서 ρ=1.000).

#### 3.5.5 3-Way 비교 — BackElim vs Cond Shapley vs Marg G-comp Shapley

| | **BackElim** | **Incremental Shapley (Conditional)** | **Incremental Shapley (Marginal G-comp)** |
|---|---|---|---|
| 출처 | Shender 4.2.1 (Eq 13, paper primary) | Shender 4.2.3 + Du 2019 통합 (§3.1-§3.3) | **본 프로젝트 추가** — Pearl regression standardization |
| Credit 알고리즘 | Sequential telescoping (역순 제거) | Exact Shapley (128 coalitions, §3.1) | Exact Shapley (128 coalitions) — 알고리즘 동일 |
| Value function 모집단 | converted users (default) | converted users (default) | **ALL users** (전 모집단) |
| Estimand | conditional intensity drop | conditional channel credit (전환자 평균) | $E_{u \in U}[\hat\lambda(\text{full}_u) - \hat\lambda(\text{full}_u \setminus c)]$ — 모집단 ATE 분해 |
| 시너지 분배 | last-ad 집중 (super-additive 시 마지막 광고 over-credit) | 광고들 균등 분배 (½ each in 2-player) | 광고들 균등 분배 (모집단 평균 위에서) |
| Selection / collider bias | HIGH (last-ad 집중 + outcome conditioning) | MEDIUM (outcome conditioning 잔존) | **LOW** (모집단 평균으로 collider 회피) |
| 답하는 질문 | "현재 광고 marginal + 과거 시너지" (bidding) | "전환자의 채널 mix 분배" (사후 정산) | "모집단에서 채널 lift 는?" (forward 예산, A/B estimand) |
| `credit_method` 인자 | `"backelim"` | `"shapley"` (default) | `"shapley"` + `subpopulation="all"` |
| 편의 호출 | `compute_survival_attribution(...)` | `compute_survival_attribution(...)` | `compute_survival_gcomp_attribution(...)` |
| Causal tier (§4.2) | Outcome model only | Outcome model only | Outcome model only (**propensity 미보정 — 여전히 strict debiasing 아님**) |

> **중요**: Marginal G-comp 도 §4.2 의 "Causal — outcome model only" tier. 모집단 estimand 로 collider 를 회피하나 **propensity 모델링이 없으므로 여전히 outcome model 정확성 + no-unobserved-confounders 가정에 의존**. Strict debiasing (DR/DML) 은 §8.1 Survival × IPW Hybrid 가 future work.

#### 3.5.6 시뮬 결과 (notebook 02 (Main) §10)

100K 시뮬, GT-A (intensity-based, 전환자 truth) + GT-B (counterfactual Shapley, 모집단 truth) 두 ground truth 동시 비교:

| 항목 | 값 |
|---|---|
| Channel ranking Spearman ρ (Cond vs Marg) | **1.000** (완전 일치) |
| Paid Search magnitude | Cond 0.318 → Marg 0.267 (**-5%p**) |
| 나머지 6 채널 \|Δ\| | < 0.025 |
| GT-A (sample 내 전환자 truth) 와의 MAE | Conditional **0.012** (sample-faithful) |
| GT-B (모집단 counterfactual truth) 와의 MAE | **Marginal 0.020** (causal-faithful) |

**해석**:
- Ranking 완전 일치 → 채널 우선순위 권고는 두 view 어느 쪽으로 보고해도 robust
- Magnitude 차이 (Paid Search -5%p) → 전환자 모집단에서 Paid Search 가 over-credit 되어 있음 (high-intent 유저가 Paid Search 를 더 많이 봄). **예산 결정 시 Paid Search 비중 ~5%p down-weight** 권장
- 두 ground truth 와의 MAE 분리 → "어떤 estimand 를 답하는지" 가 정확도 평가 자체에 영향. Conditional 은 sample 내 GT-A 와, Marginal 은 모집단 GT-B 와 일치 — 각각의 estimand 를 정확히 추정

#### 3.5.7 구현 — `subpopulation` 파라미터

```python
from part1_simulation.models.causal.survival_attribution import (
    compute_survival_attribution,
    compute_survival_gcomp_attribution,
)

# Conditional (default, paper-faithful — 사후 정산용)
r_cond = compute_survival_attribution(journeys, credit_method="shapley")
# subpopulation="converters" 가 default

# Marginal G-computation (전 모집단 — forward 예산 결정용)
r_marg = compute_survival_attribution(
    journeys, credit_method="shapley", subpopulation="all"
)

# 편의 wrapper (subpopulation="all" 고정)
r_marg_short = compute_survival_gcomp_attribution(journeys, credit_method="shapley")

# BackElim 도 동일하게 marginal estimand 지원
r_be_marg = compute_survival_attribution(
    journeys, credit_method="backelim", subpopulation="all"
)
```

`AttributionResult.method` naming: marginal mode 시 `"Survival/Poisson (Shapley) [G-comp marginal]"` 류 suffix 가 자동 부여됨 (코드 ref `survival_attribution.py` `compute_survival_attribution()` 의 `subpop_suffix` + credit-method dispatch 블록). 보고서/플롯의 method label 만 보고도 어느 estimand 인지 식별 가능.

> **AICPE / Incremental (`query_events`) 는 미지원**: AICPE 는 interval-level 독립 제거 방식이라 모집단 평균 분기점이 없고, `incremental` mode 는 experimental data 의 ad-vs-query split 자체가 모집단 가정을 다르게 가져가므로 현 구현에서 `subpopulation` 파라미터는 BackElim/Shapley 에만 적용 (코드 ref `survival_attribution.py` `compute_survival_attribution()` 의 `aicpe` 분기 — "AICPE is interval-level … subpopulation moot" 주석).

#### 3.5.8 권장 매트릭스 (§7 의 forward link)

| 의사결정 시점 | 권장 view | credit_method 조합 |
|---|---|---|
| 사후 attribution audit / 분기 정산 보고 | **Conditional Shapley** | `credit_method="shapley"` (default) |
| 광고비 ROI 정산 (retrospective) | **Conditional Shapley** | 위와 동일 |
| **채널 예산 재배분 (forward)** | **Marginal G-comp Shapley** | `credit_method="shapley", subpopulation="all"` |
| **A/B test 사전 effect size 추정** | **Marginal G-comp Shapley** | 위와 동일 |
| 캠페인 / journey 시나리오 디자인 | **둘 다 + Path-level** $\Delta_\text{path}$ (§3.4) | (Conditional channel + Marginal channel + Path Δ 3-view) |

세부 의사결정 매트릭스는 §7 Recommendations 참조.

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
| **Causal — outcome model only** ⚠️ | Survival/Poisson (BE/Shapley/AICPE) — **Conditional 및 Marginal G-comp 모두 동일 tier**, Du Incremental Shapley (LR/LSTM/Poisson), CAMTA | **regression adjustment**. user feature 더미 (Eq 10) + outcome model fit 만으로 causal 구조 가정. **Outcome model 정확성 + no unobserved confounders 양쪽 가정 의존**. propensity 모델링 없음 → strict debiasing 아님. <br> **Marginal G-computation (§3.5)** 은 같은 outcome model 위에서 *모집단 estimand* 로 계산되어 collider bias 를 **회피** 하지만, propensity 모델링이 없으므로 여전히 outcome-model only tier — Conditional 대비 estimand 가 마케팅 의사결정과 정렬될 뿐 causal 강도 자체는 동일 가정 하. Strict debiasing (DR/DML) 은 §8.1 Survival × IPW Hybrid Future Work. |
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

#### Conditional vs Marginal G-Computation (§3.5) — `subpopulation` 파라미터:

```python
from part1_simulation.models.causal.survival_attribution import (
    compute_survival_attribution,
    compute_survival_gcomp_attribution,
)

# Conditional (default, paper-faithful — 사후 정산용)
# subpopulation="converters" 가 default, 명시 호출도 동일
r_cond = compute_survival_attribution(journeys, credit_method="shapley")

# Marginal G-computation (전 모집단 — forward 예산 결정용)
r_marg = compute_survival_attribution(
    journeys, credit_method="shapley", subpopulation="all"
)

# 편의 wrapper (subpopulation="all" 고정)
r_marg_short = compute_survival_gcomp_attribution(journeys, credit_method="shapley")

# BackElim 도 동일하게 marginal estimand 지원
r_be_marg = compute_survival_attribution(
    journeys, credit_method="backelim", subpopulation="all"
)
```

`AttributionResult.method` naming: marginal mode 시 자동으로 `" [G-comp marginal]"` suffix 부여 (e.g., `"Survival/Poisson (Shapley) [G-comp marginal]"`). 보고서/플롯의 method label 만으로 estimand 식별 가능 (코드 ref `survival_attribution.py` `compute_survival_attribution()` 의 `subpop_suffix` + credit-method dispatch 블록). **`aicpe` / `incremental` mode 는 `subpopulation` 미지원** — BackElim/Shapley 에만 적용.

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
# 21/21 passed
```

> **Note (§3.5 테스트 커버리지)**: 현 21개 테스트. `subpopulation="all"` (Marginal G-computation) 분기는 두 sanity check 로 명시적으로 검증된다 — `test_path_incrementality_efficiency_axiom_vs_shapley` ($\Sigma_u \Delta_{\text{path}} = \Sigma_c \phi_c^{\text{marg}} \times n_{\text{users}}$, efficiency axiom) 와 `test_path_incrementality_telescoping_matches_be_raw_unclamped` (telescoping = BE raw unclamped total). 두 테스트는 path-level helper `compute_path_incrementality` (notebook 02 (Main) §7.5/§10-C/§10-D 가 공유, 단일 fit 재사용·refit 없음) 가 conditional·marginal 양 모집단에서 credit total identity 를 만족함을 보장한다.

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

### 6.5 Channel Credit 패턴 차이

BackElim vs Shapley 의 채널별 credit 분포 차이는 notebook 02 (Main) §4 (paired bar + Spearman ρ) 에서 시각화된다. GT-free 추정 vs ground truth 의 채널별 패턴 비교는 notebook 02 (Main) 부록 (estimand × GT matched bar + MAE 교차표) 에 수록 — Conditional Shapley 는 GT-A (전환자 truth) 에서 채널별 최소 오차 (MAE 0.012), Marginal G-comp 는 GT-B (모집단 counterfactual truth) 에서 최소 오차 (MAE 0.020) 로, estimand 정합성이 채널 패턴 수준에서 확인된다 (§6.6 참조).

### 6.6 Conditional vs Marginal G-Computation (§3.5, notebook 02 (Main) §10)

같은 fitted GLM 위에서 `subpopulation="converters"` (Conditional) vs `subpopulation="all"` (Marginal G-comp) 비교. 100K 시뮬, GT-A (intensity-based, 전환자 truth) + GT-B (counterfactual Shapley, 모집단 truth) 두 ground truth 동시 평가:

| 항목 | 값 | 해석 |
|---|---|---|
| Channel ranking Spearman ρ (Cond vs Marg) | **1.000** | 완전 일치 — 우선순위 권고 robust |
| Paid Search magnitude | Cond 0.318 → Marg 0.267 | **-5%p** — 전환자 모집단에서 over-credit (high-intent self-selection) |
| 나머지 6 채널 \|Δ\| | < 0.025 | 무시할 만한 magnitude 차이 |
| GT-A (sample 내 전환자 truth) MAE | Conditional **0.012** | sample-faithful estimand |
| GT-B (모집단 counterfactual truth) MAE | **Marginal 0.020** | causal-faithful estimand |

→ **Ranking robust** + **magnitude 5%p 차이**. 사후 정산 보고에는 Conditional 그대로, **forward 예산 결정 시 Paid Search 비중 ~5%p down-weight 권장**. 두 ground truth 와의 MAE 분리는 "어떤 estimand 를 답하는지" 가 정확도 평가 자체를 결정함을 보여준다.

---

## 7. 권장 사항 — 목적별 method 선택

| 목적 | 1차 권장 | 2차 권장 | 비고 |
|---|---|---|---|
| **Channel ranking 만 필요** | Survival/Poisson **BackElim** | Survival/Poisson **Shapley** | BE: τ=1.0 (perfect), Shapley: τ=0.91 |
| **Credit magnitude 정확** | **Survival/Poisson Shapley** ⭐ | Survival/Poisson AICPE | Shapley MAE=0.016 (단연 1위) |
| **Budget allocation** | Incremental Shapley (LR) | **Survival/Poisson Shapley** | Alloc MAE 0.013 vs 0.019 (근접) |
| **사후 attribution audit / 분기 정산 보고** ⭐ | **Survival × Shapley (Conditional)** | Survival × BackElim (Conditional) | §3.5 Conditional view (default), 전환자 모집단 estimand |
| **Forward 채널 예산 재배분** ⭐ | **Survival × Shapley (Marginal G-comp)** ⭐ | Survival × Shapley (Conditional) — 비교용 | §3.5 Marginal view, A/B test estimand 정렬, `subpopulation="all"` |
| **A/B test 사전 effect size 추정** | **Survival × Shapley (Marginal G-comp)** | (실제 A/B 실험) | A/B 가 측정하는 estimand 와 동등 |
| **균형잡힌 production 1차** | **Survival/Poisson Shapley (Conditional)** ⭐ | (Marginal G-comp 병행 보고) | default Conditional, forward 결정 보고서엔 Marginal 추가 권장 |
| **Causal claim 강함 (debiased)** | DML | DR / IPW | unconfoundedness 가정 |
| **Causal claim 강함 (data-based)** | Survival/Poisson **incremental** + `query_events` | (필요 시 RCT 데이터) | experimental data 필수 |
| **Stability 우선** | Survival/Poisson (any credit) | Markov | CV 0.096 |
| **Paper-faithful TEDDA** | Survival/Poisson **BackElim** | — | Section 4.2.1 primary |
| **Paper-faithful Du** | Survival/Poisson **Shapley (Conditional)** | Incremental Shapley (LR) | 통합 framework, Du default 도 conditional |
| **Campaign / journey 시나리오 디자인** | **Path-level Incremental Shapley** ($\Delta_\text{path}$) + Cond/Marg 채널 view 병행 | (BackElim path total — telescoping 으로 동일 quantity) | §3.4 aggregation × §3.5 estimand 두 axis 동시 활용. 예: 노트북 02 (Main) §7.5 / §10 |

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

### 8.8 Path-Level Bootstrap CI

**부분 구현** (channel-level bootstrap 은 노트북 02 (Main) §5 에 있음). Path-level 확장: path template 별 $\Delta_\text{path}$ 의 90% CI 를 user-resample bootstrap 으로 추정. 어떤 path 의 ranking 이 robust 한지 (CI 가 0 으로부터 멀고 좁은지) marketing 보고에 활용. Path template 수가 많으므로 Top-K (e.g., K=50) 만 bootstrap 하는 selective 전략 권장.

### 8.9 IPW-Weighted Path Incrementality

**미구현, 우선순위 2** (§8.1 Survival × IPW Hybrid 의 path-level 확장). Path 자체가 endogenous (유저의 path 선택은 광고 schedule + 행동의 결과) — selection bias 보정을 위해 path-propensity model:

$$e_\text{path}(W_i) = P(\text{user } i \text{ exposed to path } p \mid W_i)$$

쥬닉이 만족스러운 propensity model 추정 후 IPW-reweighted Δ_path 로 path-level causal estimate 강화. Path 수가 많아 per-path propensity 추정이 어려우므로, **transition propensity** (`channel_prev → channel_next` 단위) 의 product 로 분해 권장. Doubly robust (path-outcome model + path-propensity model) 변형도 가능.

### 8.10 Converted vs Non-Converted Path 비교

**미구현, 우선순위 3**. 같은 path template 의 converted vs non-converted 유저 비교 → unobserved confounding 단서. 만약 같은 path 인데 conversion rate 가 segment 별로 크게 다르면, segment 변수 외에 추가 confounder 존재 시그널.

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
