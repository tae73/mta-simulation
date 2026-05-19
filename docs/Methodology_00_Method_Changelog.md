# Methodology 00 — Method Version Changelog (canonical)

본 프로젝트의 causal 방법론은 정합성 개선을 거치며 여러 버전을 폐기했다. 노트북·문서
곳곳의 "v1/v2/v3" 인라인 주석의 **단일 출처(canonical)** 가 본 문서다. 다른 곳은 본
문서를 참조만 한다.

> **핵심 원칙:** ground truth(DGP 오라클)를 attribution 계산에 직접 사용한 버전은
> 모두 폐기한다 — 순환논증(circular reasoning)이기 때문이다. 채택 버전은 *관측 데이터만*
> 사용한다.

## Incremental Shapley (Du et al. 2019)

| 버전 | 내용 | 상태 | 측정 |
|------|------|------|------|
| v1 | coalition value 를 DGP `compute_log_intensity` 직접 호출로 계산 | **폐기** — DGP 오라클 순환논증 (cherry-pick) | MAE 0.017 (무효) |
| v2 | 관측 데이터로 학습한 LogisticRegression response model 의 $\hat P(Y\mid S)-\hat P(Y\mid\emptyset)$ 에 exact Shapley | **채택** | MAE 0.028, τ 0.90 |

## Survival/Poisson (Shender et al. 2023 TEDDA)

| 버전 | 내용 | 상태 | 측정 |
|------|------|------|------|
| v1 | DGP 오라클 기반 AICPE | **폐기** — 순환논증 | MAE 0.025 (무효) |
| v2 | per-user binary outcome aggregation | **deprecated** — paper-faithful 아님 (Shender §4.2 는 interval Poisson) | MAE 0.041 |
| v3 | interval split + Poisson GLM (log Δt offset, Eq 12) + BackElim(Eq 13) | **채택 (paper-faithful)** | BackElim MAE 0.046, τ 1.000 |
| v3+ | 동일 backbone 위 Shapley credit(§4.2.3, Du 통합) | **채택 (Main 1차 권고)** | Shapley MAE 0.016 |

## 분류(라벨) 정정 이력

- **METHOD_CATEGORIES 5-tier 정정**: 구 "Causal (incremental)" 라벨 폐기. Survival 3종 +
  Inc Shapley + CAMTA = **"Causal (outcome model)"** (regression adjustment, propensity
  미보정), IPW/DR/DML = **"Causal (debiased)"**. Shender 본인 진술(observational →
  correlational) + 학술 정합성에 근거. 상세: `Methodology_05 §4`.

## Conditional vs Marginal G-computation (신규 통합)

- `compute_survival_attribution(..., subpopulation=)` 파라미터로 동일 fitted GLM 위에서
  Conditional(전환자, paper-faithful default) vs Marginal G-comp(전 모집단) 두 estimand
  제공. fitted β 불변, credit aggregation 모집단만 교체. 상세: `Methodology_05 §3.5`,
  실무 요약 `Methodology_05b`, 1-page `Marketing_Handout_Conditional_vs_Marginal.md`,
  노트북 `notebooks/part1/02_main_survival_incremental_shapley.ipynb` §10.

## 폐기 버전을 문서에 남기는 이유

순환논증으로 얻은 비현실적 고성능(IncShap v1 MAE 0.017, Survival v1 MAE 0.025)을
명시적으로 기록해, 후속 작업이 같은 함정을 반복하지 않도록 한다. 노트북 본문의 v-note 는
요약일 뿐이며 권위 있는 기록은 본 문서다.
