"""Update markdown cells in notebooks/part1/05_experiments_and_insights.ipynb to v3."""

import nbformat

NB = "notebooks/part1/05_experiments_and_insights.ipynb"
nb = nbformat.read(NB, as_version=4)

# Cell [5] — Exp 01 conclusion
nb.cells[5].source = """**Experiment 01 결론 (v3 Survival 반영, full 100K):**
- **Top Causal**: Incremental Shapley (MAE=0.028) 와 Survival/Poisson v3 AICPE (MAE=0.026) 가 정확도 1-2 위. **Survival/Poisson v3 BackElim** 은 paper-faithful primary credit (Eq 13) 으로 **Kendall τ=1.000** 완벽 순위 + Bootstrap CV=0.096 (3rd most stable).
- **Shapley (model-based)** 가 Game-theoretic 중 최고 (MAE=0.035) — 로지스틱 회귀의 효과적 smoothing
- **LSTM LOO가 Top-3 100%** — 반사실적 추출 방법이 상위 3개 채널을 정확히 복원
- **First Click은 anti-correlated** (Tau=-0.29) — 인지 채널에 100% 크레딧 배분의 치명적 오류
- **Transformer는 LSTM보다 열등** (MAE 0.119 vs 0.034) — 평균 5 터치포인트에서 self-attention 이점 없음

> **Note:** Incremental Shapley v1(MAE 0.017)은 DGP 오라클 직접 호출로 폐기, v2 는 학습 response model. Survival/Poisson v2 (per-user binary) → v3 (interval Poisson + log Δt offset, paper-faithful Section 4 전수). 학습된 β decay vs GT β Spearman = **0.955** (p=0.001).

---
## 2. Experiment 02 — 채널 간 시너지 탐지 능력

> **가설:** 시퀀스 모델(Markov, LSTM)과 인과 모델이 Display→Paid Search 시너지(δ=0.4)를 flat 모델보다 잘 포착한다.
>
> **설정:** 동일 DGP에서 cross-influence를 켠 경우(δ>0)와 끈 경우(δ=0)를 비교. Synergy detection score = 채널 쌍 크레딧 변화량."""

# Cell [7] — Exp 02 conclusion
nb.cells[7].source = """**Experiment 02 결론 (v3 Survival 반영):**
- **Survival/Poisson v3** 와 **Shapley (model-based)** 가 Display→PaidSearch 시너지 탐지 — interval-level Poisson 의 channel-bin count term 이 cross-influence 영향을 흡수
- **Markov Chain은 pairwise 상호작용에 둔감** (+0.006) — Removal Effect는 개별 채널 제거만 측정하므로 쌍별 시너지를 직접 포착하지 못함
- Rule-based 방법들도 위치 효과(positional effect)를 통해 간접적으로 시너지를 일부 포착
- **교차 영향이 강할수록** 이를 명시적으로 모델링하는 방법론이 유리. Survival/Poisson v3 의 옵션 hook `include_cross_channel=True` (Eq 9) 으로 명시적 cross-channel feature 추가 가능 (default off).

---
## 3. Experiment 03 — 데이터 규모 민감도 (Learning Curve)

> **가설:** DL 방법론은 통계적 방법론보다 더 많은 데이터를 필요로 한다. Markov/Shapley는 5K에서 안정, LSTM은 10K+ 필요.
>
> **설정:** n_users = {1K, 5K, 10K, 50K, 100K}에서 각 방법론의 MAE 추적. DL은 5K 미만에서 N/A 처리."""

# Cell [9] — Exp 03 conclusion
nb.cells[9].source = """**Experiment 03 결론 (v3 Survival 반영):**
- **Survival/Poisson v3**: 1K MAE=0.079, 5K=0.053, **10K=0.040 (best)**, 50K=0.048, 100K=0.049 — 10K 부터 안정. v2 (5K MAE 0.020) 대비 paper-faithful interval split 으로 noise floor 가 약간 높아진 trade-off (학습 β decay 가 GT β 와 더 정합)
- **DML**: 1K에서 MAE=0.159로 불안정 → 50K에서 0.028로 수렴. Cross-fitting에 충분한 데이터 필요
- **Last Click / Time Decay**: 1K에서도 안정 (heuristic은 학습이 없으므로 데이터 무관)
- **LSTM**: 5K 미만에서 학습 불가 (전환 유저 ~125명으로 불충분)
- **실무 가이드**: 전환 유저 5K 이상이면 Survival/Poisson v3 추천 (paper-faithful + perfect ranking τ=1.0), 50K 이상이면 DML 추가 고려

---
## 4. Experiment 04 — DGP 가정 민감도

> **가설:** 상호작용 제거 시 시퀀스 모델의 이점 소멸. 이질성 제거 시 causal 보정의 이점 소멸.
>
> **설정:** 4 DGP 변형 (Full / No interactions / No decay / No heterogeneity) × 8 methods. 각 20K 유저."""

# Cell [11] — Exp 04 conclusion
nb.cells[11].source = """**Experiment 04 결론 (v3 Survival 반영):**
- **Survival/Poisson v3 는 모든 DGP 변형에서 robust** (MAE 0.043~0.047, τ=0.91 일관) — interval Poisson 이 가정 위반에 둔감
- **No decay**: Shapley 크게 개선 (0.041→0.019). 시간 감쇠가 없으면 단순 채널 조합만 보면 되므로 Shapley의 coalition value가 정확해짐
- **No interactions (δ=0)**: 대부분 성능 하락. cross-influence 없이도 여전히 비선형 구조(decay, heterogeneity) 존재
- **No heterogeneity (η=0)**: DML이 개선 (confounding 소실). IPW는 미세 변화
- **데이터의 어떤 특성이 방법론 선택에 영향을 미치는가?** → 시간 감쇠가 강하면 temporal 모델 필수, 상호작용이 강하면 시퀀스/인과 모델 유리

---
## 5. Experiment 05 — Correlational vs Causal Attribution

> **가설:** Confounding 강도 증가 → correlational 방법론 악화, causal 방법론은 안정.
>
> **설정:** 세그먼트 η 값 spread를 조절하여 confounding 강도를 3단계로 변화 (0.2 / 0.8 / 2.0).
> η가 클수록 세그먼트별 전환율 차이가 크고, 세그먼트별 채널 선호가 강해져 confounding이 심화."""

# Cell [31] — Exp 10 conclusion
nb.cells[31].source = """**Experiment 10 결론 (v3 Survival 반영):**
- **가장 안정적인 방법론** (mean CV 최저)은 Markov order=1 (0.039) — transition matrix 의 large counts 평균이 sample-to-sample 변동에 둔감.
- **Survival/Poisson v3 의 stability 대폭 개선**: v2 AICPE CV=0.44 → v3 (BE 호출, AICPE 라벨 보존) **CV=0.096 (3rd most stable)**. interval Poisson + log Δt offset 의 분산 감소 효과 — Markov 다음으로 안정적인 causal method
- **DL 방법론(LSTM, Transformer)**은 학습 noise + 작은 sample size가 결합되어 CV가 큰 편
- **Shapley (model-based)** 는 fragile winner 의 전형 — 정확도(Exp 01) 상위지만 bootstrap CV 16/16 으로 단일 운영에는 위험
- 95% CI width heatmap에서 Email/Paid Search가 가장 넓은 (= 가장 contested) 채널인지 확인
- Violin plot에서 GT-A를 가로지르는 분포 (median이 GT-A 근처 + 좁은 IQR)가 이상적"""

# Cell [40] — Final synthesis
nb.cells[40].source = """**종합 결론 (Real-World Validation, v3 Survival 반영):**

1. **Robust Winner**: 모든 5개 지표에서 상위 (GT-aware 정확도 + OOS 예측 + 의사결정 임팩트 + 안정성 + 합의도) — 시뮬과 실무 모두에서 신뢰 가능.
   - 후보: **Survival/Poisson v3 BackElim** (τ=1.0 + CV=0.096 + paper-faithful), Shapley (model-based), Incremental Shapley
2. **Fragile Winner**: GT-aware는 좋지만 GT-free 지표 일부가 약함 — 시뮬 벤치마크는 통과하지만 실무 risk 큼.
   - 예시 후보: DML (정확도는 OK이나 작은 sample bootstrap CV 큼), LSTM-LOO (정확도 좋으나 학습 noise)
3. **Stable but Inaccurate**: Bootstrap CV 낮지만 정확도 떨어짐.
   - 예시: Last Click, Linear (안정적으로 틀림)

**실무 권장사항**
- 데이터 충분 (>5K 전환): **Survival/Poisson v3 BackElim** (paper-faithful TEDDA, perfect ranking τ=1.0, stability CV 0.096) 1차 선택, **Incremental Shapley**로 cross-validate. AICPE 모드는 budget allocation 최적 (Alloc MAE 더 낮음)
- 데이터 부족 (<2K 전환): **Time Decay** baseline + **bootstrap CI** 보고
- 의사결정 압박 큰 상황: **2~3개 이질적 방법론의 consensus rank** 우선 제시 (Exp 11 τ_consensus=0.81 vs τ_best=1.0)
- DL 방법은 단독 의사결정 근거로 사용 금지 — Shapley/Survival 결과와 cross-check 필수

**다음 단계 (Part 2/3)**
- Part 2 Criteo (16.5M)에서 동일 패턴 재현 여부 확인 → Exp 11 consensus 유효성 외적 타당성
- Part 3 MMM (집계 수준)과의 triangulation으로 user-level vs aggregate-level 일치도 점검"""

nbformat.write(nb, NB)
print(f"Updated cells [5,7,9,11,31,40] in {NB}")
