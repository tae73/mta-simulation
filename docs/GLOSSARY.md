# Glossary / 용어집 (KO ↔ EN)

> Part 1 포트폴리오([`part1_simulation/README.md`](../part1_simulation/README.md) · [EN](../part1_simulation/README.en.md))의
> 용어 단일기준(single source of terminology). 한국어 본문은 영어 전문용어를 병기하며, 두 언어 문서의 용어·정의는 이 표를 따른다.

## 방법론 핵심 / Core method

| 한국어 | English | 정의 / Definition |
|---|---|---|
| Poisson Survival 백본 | Poisson Survival backbone | 터치포인트를 시간 구간으로 쪼갠 interval-split Poisson GLM. 전환 intensity $\lambda(t)$를 모델링하며 우측 절단을 자연 처리. The interval-split Poisson GLM modeling conversion intensity $\lambda(t)$; handles right-censoring natively. |
| 단일 fitted GLM | single fitted GLM | 한 번만 적합한 GLM. 이후 모든 credit·path·population 분해가 같은 모델의 질의. One GLM fit once; all downstream decompositions are queries of it. |
| 채널 기여도 | channel credit | 적합된 intensity를 채널별로 분해한 값 $\phi_c$. Per-channel decomposition $\phi_c$ of the fitted intensity. |
| BackElim (Backwards Elimination) | BackElim | 광고를 나중→처음 순서로 제거하며 $\hat\lambda$ 하락분을 크레딧으로(순서 의존, Eq. 13). Remove ads last→first, credit the $\hat\lambda$ drop (order-dependent, Eq. 13). |
| Shapley | Shapley | 128개 coalition에 대한 한계기여 평균(순서 무관, coalition-fair, Eq. 25). Mean marginal contribution over 128 coalitions (order-free, coalition-fair, Eq. 25). |
| Incremental Shapley | Incremental Shapley | baseline 전환을 뺀 *순증* lift만 배분(Du et al. 2019). Allocates only the *incremental* lift after subtracting baseline conversion (Du et al. 2019). |
| Total Shapley (model-based) | Total Shapley (model-based) | baseline까지 포함한 크레딧. lower-funnel 과대평가, high base에서 붕괴. Credit that includes the baseline; over-values lower-funnel, collapses at high base rate. |
| Multi-path / path-level Incremental Shapley | Multi-path / path-level Incremental Shapley | 채널이 아닌 여정 템플릿 단위로 $\Delta_{\text{path}}$를 합산. Aggregates $\Delta_{\text{path}}$ by journey template, not by channel. |
| 효율성 항등식 | efficiency identity | $\sum_{\text{paths}}\Delta_{\text{path}} = \sum_c \phi_c = \mathbb{E}_u[\Delta_u]$ (상대오차 0.00%). The identity unifying path/channel/population views (0.00% rel. error). |
| AICPE | AICPE | interval 단위 독립 제거로 산출한 survival credit 변형. A survival-credit variant using interval-level independent removal. |

## 인과 estimand / Causal estimands

| 한국어 | English | 정의 / Definition |
|---|---|---|
| Conditional Shapley | Conditional Shapley | 전환자(converters)만 집계. 사후 attribution audit용. GT_A 정합. Aggregated over converters only; for retrospective audit; matches GT_A. |
| Marginal G-computation | Marginal G-computation | 전체 유저 집계. 전향적 예산 결정·A/B 정렬. GT_B 정합. Aggregated over all users; for forward budget decisions, A/B-aligned; matches GT_B. |
| Collider bias | collider bias | 전환(post-treatment outcome)에 조건부로 집계할 때 유입되는 편향(Pearl/Hernán). Bias introduced by conditioning on conversion (a post-treatment outcome). |
| 교란 / Confounding | confounding | 유저 의도·세그먼트(η)가 채널 노출과 전환에 동시 영향. Latent user intent/segment (η) affecting both exposure and conversion. |
| GT_A (sample intensity) | GT_A (sample intensity) | per-user $\hat\lambda(\text{full})-\hat\lambda(\varnothing)$의 채널 분해(sample truth). Per-user intensity decomposition (sample truth). |
| GT_B (counterfactual) | GT_B (counterfactual) | 채널 제거의 모집단 기대효과(population ATE). Population expectation of channel removal (population ATE). |

## 평가 지표 / Evaluation metrics

| 한국어 | English | 정의 / Definition |
|---|---|---|
| MAE (채널 기여도 오차) | MAE (channel-credit error) | ground truth 대비 채널 크레딧 평균절대오차(↓ 좋음). Mean absolute error of channel credit vs ground truth (↓ better). |
| Kendall τ (순위 일치도) | Kendall τ (ranking agreement) | 추정 순위와 GT 순위의 일치도(↑ 좋음, 음수=역전). Rank correlation with GT (↑ better, negative = inverted). |
| 예산 배분 MAE | budget allocation MAE | 추정 최적 예산 배분과 GT 최적 배분의 오차. Error of estimated optimal allocation vs GT optimum. |
| 부트스트랩 CV | bootstrap CV | 부트스트랩 표본의 변동계수(std/mean, ↓=안정). Coefficient of variation across bootstrap resamples (↓ = stable). |
| OOS AUC | OOS AUC | hold-out 예측 판별력(reasonableness 게이트). Out-of-sample predictive discrimination (a reasonableness gate). |

## 산업 활용 / Industry

| 한국어 | English | 정의 / Definition |
|---|---|---|
| Effect ≠ Efficiency | Effect ≠ Efficiency | 효과(β)가 큰 채널이 돈당 효율은 낮을 수 있음(예: Paid Search). A high-effect channel can be inefficient per dollar (e.g. Paid Search). |
| 효율 (전환/$) | efficiency (conv/$) | 비용 구조를 결합한 채널의 단위비용당 기대 전환. Expected conversions per unit cost after combining the cost structure. |
| robust 여정 템플릿 | robust journey template | `count ≥ 5`로 거른 재현 가능한 여정(35개). Reproducible journey forms after a `count ≥ 5` filter (35 of them). |
| 보고 신뢰도 게이트 | reporting-confidence gate | 부트스트랩 CI 폭으로 보고 가능 여부를 판단. Use bootstrap CI width to decide whether a channel is reportable. |

## 데이터·세팅 / Data & setup

| 한국어 | English | 정의 / Definition |
|---|---|---|
| DGP (Data Generation Process) | DGP | Du(2019)+Shender(2023)+CDA(2025)를 통합한 시뮬레이션 생성기. The simulation generator integrating the three frameworks. |
| 7 채널 | 7 channels | Display, Social, Organic Search, Paid Search, Email, Referral, Direct. |
| 세그먼트 (η) | segment (η) | New(−0.3) / Exploratory(0.0) / Loyal(+0.5)의 baseline 전환 이질성. Baseline-conversion heterogeneity across user segments. |
| 교차 영향 (δ) | cross-influence (δ) | Display→Paid Search 0.4, Social→Email 0.3, Organic→Direct 0.2. |
| ground truth | ground truth | 알려진 DGP 파라미터(β, f_channel, δ, η)로부터 도출한 정답 기여도. True credit derived from the known DGP parameters. |
