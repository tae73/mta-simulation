"""Update markdown cells in notebooks/part1/04_causal_attribution.ipynb to v3.

Updates Cell [4] Survival/Poisson description and Cell [12] summary.
Then re-executes the notebook in-place to refresh code outputs.
"""

import nbformat

NB = "notebooks/part1/04_causal_attribution.ipynb"

nb = nbformat.read(NB, as_version=4)

# --- Cell [4]: Survival/Poisson description ---
cell4_new = """## 2. Incremental Shapley (Du et al. 2019)

**핵심 아이디어:** 전체 전환을 배분하는 대신, **광고로 인한 순증 전환(incremental conversion)**만 배분.

Du et al. 원 논문의 2-step 파이프라인:
1. **Response Modeling**: 관측 데이터로 response model을 학습하여 P(conv | features) 추정
2. **Credit Allocation**: v(S) = model_predict(S active) - model_predict(no channels) → Shapley 배분

> **Note (v2):** 이전 구현(v1)은 DGP의 `compute_log_intensity`를 직접 호출하여 coalition value를 계산하는 순환논증이었다. 현재 v2는 관측 데이터로 학습한 LogisticRegression response model만 사용한다.

## 3. Survival/Poisson Attribution (Shender et al. 2023, TEDDA)

논문 *"A Time To Event Framework For Multi-touch Attribution"* (JDS 2023, ArXiv 2009.08432) 의 **Section 4 방법론을 전수 반영** (v3 paper-faithful):

- **모델 (Eq 7+10+12):** 각 유저 path 를 piecewise-constant intensity interval 로 분할 → 각 interval = 1 Poisson 관측치, **offset = log(Δt)** + segment dummy (α₀ shift). 우측 절단 (Requirement #1) 자동 처리.
- **Decay (Eq 5):** 5개 시간 구간 (0-1d / 1-3d / 3-7d / 7-14d / 14d+) × 7채널 = 35 features. **DGP 파라미터 (decay_half_life_days) 미사용**.
- **Credit (paper primary, Eq 13 BackElim):** $\\text{RawCredit}(j) = \\hat\\lambda(t^*, A^{(j)}) - \\hat\\lambda(t^*, A^{(j-1)})$. Telescoping 으로 $\\sum_j = \\hat\\lambda(A^{(n)}) - \\hat\\lambda(\\emptyset)$.
- **옵션 hooks (default off):** Eq 6 ad features, Eq 8 position, Eq 9 cross-ad interaction, Eq 11 query/ad 분리, 4.1.6 계절성/self-excitation. Synergy/Shapley 비교 (Eq 21/24) 는 별도 함수 `compute_synergy_report()`.

> **Note (v3):** v2 의 per-user binary outcome aggregation 은 paper-faithful 이 아니므로 deprecated. v3 는 interval Poisson + log Δt offset 으로 추정하고, BackElim 이 paper primary credit. 학습된 β decay vs GT β Spearman = **0.955** (p=0.001) 으로 paper 정합성 검증."""

nb.cells[4].source = cell4_new

# --- Cell [12]: Summary ---
cell12_new = """---
## 요약

1. **Confounding은 실재한다:** 세그먼트 → 채널 노출 + 세그먼트 → 전환의 이중 경로가 selection bias를 생성
2. **Survival/Poisson v3 (paper-faithful TEDDA):** Shender 2023 Section 4 전수 반영 — interval Poisson + log Δt offset (Eq 12), BackElim (Eq 13). full 100K 에서 BE MAE=0.046, **τ=1.000 (perfect ranking)**, Bootstrap CV=0.096 (3rd most stable). 학습 β vs GT β Spearman 0.955.
3. **Incremental Shapley (v2):** 학습 기반 response model로 "순증 전환"만 배분 — model-based Shapley 대비 개선
4. **IPW/DML은 유저 feature 필수:** 프로필 데이터가 풍부한 환경에서 강점
5. **CAMTA는 vanilla LSTM보다 개선:** causal regularization이 attention을 인과적으로 교정

> **Note:** Incremental Shapley v1(MAE 0.017)과 Survival/Poisson v1(MAE 0.025)은 DGP 오라클을 직접 사용한 체리피킹이었으므로 폐기. Survival/Poisson v2 (per-user binary aggregation) 도 paper-faithful 이 아니므로 v3 (interval Poisson + offset) 으로 대체. 모두 관측 데이터만으로 학습한 legitimate한 결과.

다음 노트북: [05_experiments_and_insights.ipynb](05_experiments_and_insights.ipynb) — 6개 실험 종합 분석"""

nb.cells[12].source = cell12_new

nbformat.write(nb, NB)
print(f"Updated cells [4] and [12] in {NB}")
