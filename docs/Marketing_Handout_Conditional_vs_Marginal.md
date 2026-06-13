# Channel Attribution — Conditional vs Marginal 의사결정 가이드

> **1-page handout** (data scientist → marketing team). 본 분석의 두 가지 channel attribution view (Conditional, Marginal G-computation) 의 차이, 언제 어느 것을 사용할지, 그리고 그 결정이 fair 한 이유.

---

## TL;DR

- **Conditional Shapley (§4)**: 전환자만 본 채널 credit 분배 — **사후 attribution audit** 용
- **Marginal G-comp Shapley (§10)**: 모집단 평균 causal lift — **미래 budget 의사결정** 용
- **본 분석 결과**: 두 view 의 채널 ranking 완전 일치 (Spearman ρ = 1.000) → **robust 권고**

---

## 핵심 비유

> **"응급실 환자만 분석해서 '구급차 탄 사람이 더 자주 죽는다' 고 결론내면 안 됨. 구급차가 사망 원인이 아니라, 위중한 사람들이 구급차를 탔던 것."**

광고에 자주 노출된 전환자 = "구급차 탄 사람". **광고 효과 ≠ 노출 빈도와 전환의 raw correlation**.

---

## 의사결정 매트릭스

| 의사결정 | 사용할 view | 답하는 질문 |
|---|---|---|
| 사후 attribution audit | **Conditional** | "전환한 유저의 channel mix 분배" |
| 광고비 ROI 정산 (retrospective) | **Conditional** | "이번 분기 전환 credit ↔ 채널 spend" |
| **채널 예산 재배분 (forward)** | **Marginal G-comp** | "Email -10% 시 모집단 전환 감소?" |
| **A/B test 사전 effect estimate** | **Marginal G-comp** | "Channel holdout 효과 크기" |
| 캠페인 시나리오 디자인 | **둘 다** | 관찰 vs 모집단 시나리오 |

→ **시간 방향이 핵심**: 과거 분석 → Conditional, 미래 결정 → Marginal

---

## "Bias 가 맞다" 증거 3가지

1. **Non-converters 도 long path 많음** — 16-20 step 유저의 **96.3% 가 전환 안 함**. Long path 가 conversion 을 *correlate* 할 뿐 *cause* 하지 않음.

2. **Within-segment 잔존** — New segment 안에서도 1-2 step 0.81% vs 16-20 step 3.04% (**3.8배**). Segment 통제로도 selection effect 못 잡음.

3. **Collider bias** — 전환 (post-treatment outcome) 에 conditioning 하면 Pearl/Hernán 의 standard 결과로 인과 효과 추정에 spurious link 도입.

---

## "Marginal 이 Fair 한 이유" 2가지

1. **Decision-aligned estimand**: 마케팅 의사결정은 모집단 전체 노출을 바꿈 (Email 예산 +30% → 모든 유저 풀에 영향). Marginal 이 정확히 이 질문에 답: $E_{u}[\hat\lambda(\text{full}) - \hat\lambda(\text{full}\setminus c)]$. **A/B test 가 측정하는 estimand 와 일치**.

2. **시뮬 데이터 입증**: DGP-known ground truth 두 가지 중
   - GT_A (intensity, converters): Conditional 이 MAE 0.012 로 일치 (sample 내 truth)
   - GT_B (counterfactual Shapley, ALL users): **Marginal 이 MAE 0.020 으로 가장 가까움** (population causal truth)

---

## 5-Step 설득 시나리오

1. **직관적 비유**: 구급차 (위 참조)
2. **Conditional 의 자기참조 지적**: *"전환자의 합은 미래 예측이 아님 — 사후 정산 도구"*
3. **시뮬레이션 GT 비교**: Marginal 이 counterfactual Shapley (GT_B) 와 가장 가까움 → 인과 ATE 와 정렬
4. **Conservative framing**: 둘 다 보고서 포함, ranking 일치 (본 시뮬 ρ=1.0) 시 robust 권고
5. **Future trigger**: A/B test (ultimate), DR Survival/Poisson (Methodology 05 §8.1, propensity-adjusted) 가 다음 단계

---

## 본 시뮬 결과 실무 적용

- ρ = 1.000 → **Conditional 과 Marginal 어느 쪽이든 채널 ranking 동일**
- 최대 magnitude 차이: **Paid Search Cond 0.318 → Marg 0.267 (-5%p)** — converters 에 over-credited
  - **Budget 결정**: Marginal 권장 → Paid Search 비중 ~5%p down-weight
  - **사후 정산**: Conditional 그대로 OK
- 나머지 6개 채널: 두 view 거의 동일 (|Δ| < 0.025)

---

## 한계 + Next Steps

- 두 view 모두 outcome model + $W$ (confounder set) 충분 가정 의존 (Methodology 05 §4.2)
- **Ultimate 검증은 A/B test** — observational 분석은 의사결정 *지원* 도구이지 *증명* 아님
- 실 데이터 적용 시: 추가 $W$ features (device, country, history) + Tier 2 (DR/IPW Survival/Poisson, Methodology 05 §8.1)

---

**참고 자료**:
- 분석 코드: `notebooks/part1/02_main_survival_incremental_shapley.ipynb` §10
- Methodology 문서: `docs/Methodology_05_Causal_Attribution_Frameworks.md` §3.5 (Conditional vs Marginal G-computation), §4.2 (Causal tier); "4-layer framing" 은 notebook 02 (Main) §10
