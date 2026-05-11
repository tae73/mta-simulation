"""Update markdown cell in notebooks/part1/06_cost_and_budget_optimization.ipynb to v3."""

import nbformat

NB = "notebooks/part1/06_cost_and_budget_optimization.ipynb"
nb = nbformat.read(NB, as_version=4)

# Cell [15] — Exp 07 conclusion
nb.cells[15].source = """## 5. Experiment 07 결론 (v3 Survival 반영)

**가설 검증 결과: 부분 지지**

1. **Top 3 (v3 패치 후)**: Incremental Shapley (Alloc MAE=0.013), LSTM+Attention (0.033), CAMTA (0.036)
   - true β를 정확히 복원하는 방법론이 비용 효율성 ranking도 정확히 복원
2. **Survival/Poisson v3 BackElim 의 Alloc MAE=0.083** — paper-faithful BE 의 Paid Search 시너지 집중이 budget 배분에서는 손해 (v2 AICPE 0.006 → v3 BE 0.083 swing 은 BackElim 의 정의된 동작; **AICPE 모드 호출 시 v2 수준 회복 가능**: `compute_survival_attribution(j, credit_method="aicpe")`)
3. **Causal 카테고리 평균 MAE(0.070) < Rule-based(0.080)**: 가설 방향 일치
4. **17개 중 16개가 Allocation Tau=1.0**: Linear Response에서는 efficiency ranking이 trivial
   - 대부분의 방법론이 "Email > Display > Social > Paid Search" 순서는 맞춤
   - **차이는 magnitude(배분 비율)에서 발생** — 일부 Causal이 GT에 더 근접
5. **Shapley (model-based) 는 하위권** (MAE=0.117): Attribution 정확도(Exp 01) 에서는 상위권이지만 paid channel credit 분배 패턴이 budget 최적화에 불리

**실무적 시사점:**
- Attribution 정확도(MAE vs GT-A)와 Budget 최적화 성능은 **높은 상관이지만 완전 일치는 아님**. Survival/Poisson v3 BackElim 의 사례 (τ=1.0 perfect ranking 이지만 Alloc MAE 큼) 가 단적 예
- 예산 의사결정이 목적이라면 attribution 정확도뿐 아니라 **비용 효율성 관점의 평가가 별도로 필요** — credit_method 선택 (BackElim vs AICPE) 도 운영 목표에 따라 다름
- Linear Response의 한계: 포화 없어 ranking만 중요하고 magnitude 차이의 실무적 의미가 제한적
  → Approach C (DGP-derived concave response) 확장 시 더 의미 있는 결과 기대"""

nbformat.write(nb, NB)
print(f"Updated cell [15] in {NB}")
