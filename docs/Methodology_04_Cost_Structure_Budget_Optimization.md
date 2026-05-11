# Methodology 04: Cost Structure & Budget Optimization

## 1. 개요

Part 1 시뮬레이션의 18개 attribution 방법론 비교에 **비용 레이어**를 추가하여, "어떤 attribution 방법론이 더 나은 예산 의사결정으로 이어지는가"를 ground truth 기반으로 평가한다.

**핵심 워크플로우**: Attribution → ROI/ROAS → Budget Allocation → Ground Truth 대비 평가

---

## 2. Cost Model 설계 원칙

### 2.1 Observation Layer

Cost는 **관측 레이어(observation layer)**로서, DGP의 전환 확률 계산에 **일체 관여하지 않는다**.

```
[DGP Pipeline]
  assign_segments → channel_sequences → timestamps → compute_conversions
                                                          ↓
                                                   [Cost Layer]  ← 여기서 비용만 부착
                                                   assign_touchpoint_costs
```

- `conversion_model.py`는 변경 없음
- 동일 seed → 동일 journeys + conversions 생성 보장
- Cost는 전환 이후에 metadata로 append

### 2.2 현실 마케팅 비용 구조 반영

| 비용 유형 | 설명 | 해당 채널 |
|----------|------|----------|
| CPM (Cost Per Mille) | 1,000 노출당 과금, 인지도 중심 | Display, Social |
| CPC (Cost Per Click) | 클릭당 과금, 성과 중심 | Paid Search |
| Fixed (per-send) | 건당 고정 비용 | Email |
| Zero | 미디어비 없음 (고정비만) | Organic Search, Referral, Direct |

### 2.3 세그먼트별 비용 차등

실무에서 유저 타겟팅 비용은 세그먼트에 따라 다르다:
- **New (Prospecting)**: 넓은 오디언스 타겟팅 → 비용 높음 (multiplier > 1.0)
- **Exploratory**: 기본 비용 (multiplier = 1.0)
- **Loyal (Retargeting)**: 좁은 오디언스, 높은 CTR → 비용 낮음 (multiplier < 1.0)

Zero-cost 채널은 세그먼트 무관하게 비용 0.

---

## 3. 채널별 비용 파라미터

| Channel | Cost Type | Base Cost/TP ($) | New | Exploratory | Loyal | 근거 |
|---------|-----------|-----------------|-----|-------------|-------|------|
| Display | CPM | 0.005 | 1.2 | 1.0 | 0.8 | ~$5 CPM, 프로스펙팅 비용 높음 |
| Social | CPM | 0.008 | 1.3 | 1.0 | 0.7 | ~$8 CPM, lookalike 타겟팅 비용 |
| Organic Search | zero | 0.0 | — | — | — | SEO는 고정비, 미디어비 없음 |
| Paid Search | CPC | 2.50 | 1.1 | 1.0 | 0.9 | ~$2.50 avg CPC, 비브랜드 키워드 |
| Email | fixed | 0.003 | 1.0 | 1.0 | 1.0 | ESP 발송 단가 |
| Referral | zero | 0.0 | — | — | — | 추천인 비용 없음 |
| Direct | zero | 0.0 | — | — | — | 직접 방문, 미디어비 없음 |

### 3.1 비용 노이즈 모델

개별 터치포인트 비용에 log-normal 노이즈를 추가하여 결정론적이지 않게 만든다:

```
actual_cost = base_cost × segment_multiplier × exp(ε),  ε ~ N(0, σ²)
```

- `σ = 0.1` (약 10% 변동)
- Zero-cost 채널은 노이즈 없이 항상 0.0

### 3.2 실제 비용 분포 (100K 유저 실행 결과)

100K 유저, 517,893 터치포인트, 2,305 converters (2.31%) 기준:

| Channel | Touchpoints | Avg Cost/TP | Total Spend | Spend Share |
|---------|------------|-------------|-------------|-------------|
| Paid Search | 78,606 | $2.603 | $204,645 | 99.39% |
| Social | 72,017 | $0.009 | $683 | 0.33% |
| Display | 61,491 | $0.006 | $348 | 0.17% |
| Email | 75,213 | $0.003 | $227 | 0.11% |
| Organic Search | 99,154 | $0.000 | $0 | 0% |
| Direct | 80,495 | $0.000 | $0 | 0% |
| Referral | 50,917 | $0.000 | $0 | 0% |

- **Total spend: $205,903** | **CPA: $89.33**
- Avg cost/TP가 base cost와 약간 다른 것은 segment multiplier + log-normal noise(σ=0.1) 때문

Paid Search의 비용 집중(99.4%)은 **의도적이며 현실적**이다. 실무 performance marketing에서 검색 광고가 전체 예산의 70-90%를 차지하는 것은 일반적이다. 이 극단적 집중이 budget optimization의 핵심 질문을 만든다: "Paid Search에 대한 과투자인가, 적정 투자인가?"

---

## 4. Budget Optimization 방법론

### 4.1 Approach A: Linear Response (현재 채택)

**가정**: 예산과 전환이 비례한다.

```
Budget_k → n_k = B_k / c_k           (터치포인트 수)
Conversions_k ∝ n_k × effect_k       (전환 비례)
Marginal Conv/$ = effect_k / c_k     (상수, 채널별 고정)
```

**Attribution-based 최적 배분**:

각 attribution 방법론의 `channel_credits`를 `effect_k` 추정치로 사용:

```
estimated_efficiency_k = channel_credits_k / c_k   (유료 채널만)
B_k = B × (efficiency_k / Σ_j efficiency_j)        (효율성 비례 배분)
```

### 4.2 Ground Truth 최적 배분 (Linear)

알려진 DGP 파라미터로부터 true channel effect 계산:

```
marginal_effect_k = β_k × E[f_k(Δt)]
```

여기서 `E[f_k(Δt)]`는 채널 k의 평균 시간 감쇠 (journeys 데이터에서 추출).

유료 채널의 true efficiency:

```
true_efficiency_k = marginal_effect_k / c_k
```

Linear response에서 최적해는 **가장 효율적인 채널에 예산 집중**이다.

이는 trivial하지만, 18개 방법론이 이 **efficiency ranking을 얼마나 정확히 복원하는지**가 핵심 비교 포인트이다.

### 4.2.1 실제 Ground Truth Budget 결과 (100K 유저)

**Marginal Effects** (β_k × avg_decay):

| Channel | β | Avg Decay | Marginal Effect | Rank |
|---------|---|-----------|----------------|------|
| Paid Search | 1.2 | 0.390 | 0.468 | 1 |
| Email | 0.8 | 0.573 | 0.458 | 2 |
| Direct | 0.7 | 0.466 | 0.326 | 3 |
| Referral | 0.5 | 0.537 | 0.268 | 4 |
| Organic Search | 0.5 | 0.464 | 0.232 | 5 |
| Display | 0.3 | 0.638 | 0.192 | 6 |
| Social | 0.4 | 0.321 | 0.129 | 7 |

**핵심 인사이트: Effect ≠ Efficiency**

| Paid Channel | Marginal Effect | Cost/TP | Efficiency (effect/$) | Optimal Alloc |
|-------------|----------------|---------|----------------------|---------------|
| Email | 0.458 | $0.003 | **152.67** | **73.7%** ($147,345) |
| Display | 0.192 | $0.005 | 38.30 | 18.5% ($36,969) |
| Social | 0.129 | $0.008 | 16.07 | 7.8% ($15,505) |
| Paid Search | 0.468 | $2.50 | **0.19** | **0.09%** ($181) |

- Paid Search는 **effect 1위**이지만 비용 대비 **efficiency 최하위**
- Email은 effect 2위이나 발송 비용이 극히 낮아 **efficiency 압도적 1위**
- 이 역전 현상이 "어떤 attribution이 더 나은 예산 결정을 만드는가"의 핵심 테스트

### 4.3 한계 및 향후 확장

Linear response의 한계:
- 포화(saturation) 없음 → 단일 채널 집중이 항상 "최적"
- 실무 마케터는 diminishing returns를 당연히 가정

**Approach C (DGP-Derived Concave Response)** — 향후 확장 후보:
- DGP의 Poisson 모델에 자연스러운 포화가 내장: `∂P/∂n_k ∝ β_k · f̄_k · λ · exp(-λ)`
- `λ·exp(-λ)` 항이 diminishing returns 제공 (λ=1에서 최대, 이후 감소)
- 임의 파라미터 없이 ground truth response curve 도출 가능
- Approach A 결과 확인 후 확장 여부 결정

---

## 5. 평가 메트릭

### 5.1 Budget Allocation 메트릭

| 메트릭 | 수식 | 설명 |
|--------|------|------|
| Allocation MAE | `(1/K) × Σ|B̂_k - B*_k|` | 예측 배분 비율 vs GT 최적 배분의 MAE |
| Rank Correlation | `Kendall's Tau(rank(B̂), rank(B*))` | 예산 배분 순위 일치도 |
| Efficiency Ratio | `Conv(B̂) / Conv(B*)` | 예측 배분의 전환 / 최적 전환 (0~1) |

### 5.2 ROI/ROAS 메트릭 (채널별)

| 메트릭 | 수식 | 설명 |
|--------|------|------|
| ROAS | `(credit_k × total_conv × rev_per_conv) / cost_k` | 채널별 광고 수익률 |
| CPA | `cost_k / (credit_k × total_conv)` | 채널별 전환당 비용 |

무료 채널(Organic, Referral, Direct)은 cost=0이므로 ROAS=∞, CPA=0 → 유료 채널만 대상으로 계산.

---

## 6. 실험 07: Budget Optimization 평가

### 6.1 가설

> Ground truth에 가까운 채널 기여도를 추정하는 attribution 방법론(Survival/Poisson, Incremental Shapley)이
> 최적에 가까운 예산 배분을 도출하고, rule-based 방법론(Last Click 등)은 lower-funnel 편향으로
> Paid Search에 과투자하는 비효율적 배분을 만든다.

### 6.2 실험 설계

18개 attribution 방법론 각각에 대해:

1. **Attribution 실행** → `channel_credits` (기존 결과 재활용)
2. **ROI 계산** → `ROAS_k = (credit_k × conversions × revenue) / cost_k`
3. **Budget 배분** → `B_k = B × (efficiency_k / Σ efficiency_j)` (유료 채널만)
4. **GT 대비 평가** → Allocation MAE, Rank Tau, Efficiency Ratio

### 6.3 예상 결과

| 방법론 그룹 | 예상 Allocation MAE | 근거 |
|------------|--------------------|----- |
| Causal (Survival, Inc. Shapley) | 낮음 | True β 복원 → true efficiency ranking |
| Game-theoretic (Shapley) | 중간 | β ranking 유사하지만 magnitude 차이 |
| DL (LSTM, Transformer) | 중간 | 예측력은 높지만 causal effect ≠ attribution |
| Rule-based (Last Click 등) | 높음 | Lower-funnel 편향 → Paid Search 과추정 |

---

## 7. 구현 아키텍처

### 7.1 새로운 Type 정의

```python
class CostDef(NamedTuple):
    channel_name: str
    cost_type: str                      # "cpm", "cpc", "fixed", "zero"
    base_cost_per_touchpoint: float
    segment_multipliers: Dict[str, float]

class BudgetConfig(NamedTuple):
    total_budget: float = 200_000.0
    revenue_per_conversion: float = 100.0
    cost_noise_sigma: float = 0.1
    cost_defs: Tuple[CostDef, ...] = ()
```

### 7.2 Config 로딩

기존 `load_dgp_config()` 변경 없음. 별도 `load_budget_config()` 함수 추가.
기존 13곳의 호출 코드 수정 불필요.

### 7.3 파일 구조

| 파일 | 역할 |
|------|------|
| `part1_simulation/dgp/cost_model.py` (신규) | 터치포인트별 비용 할당 + 집계 |
| `part1_simulation/evaluation/budget_ground_truth.py` (신규) | GT 최적 배분 도출 |
| `part1_simulation/optimization/budget_optimizer.py` (신규) | Attribution → Budget 배분 |
