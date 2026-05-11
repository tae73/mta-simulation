# Methodology 07 — Multivariate User-Feature Recovery 실험

> **목적**: `compute_survival_attribution()` 의 신규 `user_features: Tuple[str, ...]` API (Methodology_05 § 1.3 / 5.2 참조) 가 multivariate 환경에서 실제로 attribution 정확도를 개선하는지 정량 검증.
>
> **관련 문서**: Methodology_05 § 4.2 (5-tier causal taxonomy), § 8.1 (Survival × IPW future work). 본 실험은 outcome model 단독 (regression adjustment) 의 한계를 드러내는 evidence 도 함께 제공.

---

## 1. Hypothesis

**H_recovery**: DGP 에 두 user feature (segment + device) 가 모두 baseline 을 shift 시킬 때, multivariate Survival/Poisson Shapley (`user_features=("segment","device")`) 가 univariate (`user_features=("segment",)`) 보다 ground truth 채널 기여도 대비 MAE 가 낮다.

**Secondary**:
- omit-one (segment 만 / device 만) 의 MAE 가 둘 다 포함한 multivariate 보다 높다 (omitted variable 의 baseline shift 정보 손실)
- 둘 다 omit (`user_features=()`) 의 MAE 가 가장 높다

---

## 2. DGP — `multivariate_dgp.py`

**설계**: canonical Poisson 백본 (`compute_log_intensity`) 에 device 추가 — 7 channels 의 β 와 segment η 는 그대로, device η 만 추가:

$$\log \lambda_i(t) = \alpha_0 + \sum_{j} \beta_k \cdot \text{decay}(t - t_j) + \sum \delta_{ij} \cdot \text{cross} + \eta_{\text{segment}_i} + \eta_{\text{device}_i}$$

| 파라미터 | 값 |
|---|---|
| n_users | 20,000 |
| target conversion rate | 2.5% |
| device levels | desktop / mobile / tablet |
| device_etas (log scale) | desktop 0.0 (ref), mobile −0.4, tablet +0.3 |
| device_proportions | desktop 0.45, mobile 0.40, tablet 0.15 |
| segment_etas (canonical) | New −0.3, Exploratory 0.0, Loyal +0.5 |
| α₀ (calibrated, mean) | ≈ −5.44 |
| seeds | 42, 1, 7, 13, 100 |

**Ground truth**: `_decompose_user_intensity_multivariate` — `evaluation/ground_truth.py` 의 Ground Truth A 로직과 동일하나 user-level η 가 segment.eta + device_eta 의 *합* 으로 채널에 비례 분배. GT sum = 1.0 정규화.

**검증**:
- conversion rate: 2.49% ± 0.07% (5 seeds)
- user-level device constancy: OK (한 user = 한 device)
- GT sum-to-one: 1.0000

---

## 3. 실험 설정

5 seed × 4 user_feature config × 2 credit method = **40 runs**.

| user_features | 의미 |
|---|---|
| `("segment", "device")` | **multivariate** (모든 true user feature 포함) |
| `("segment",)` | univariate (default) — device omit |
| `("device",)` | device only — segment omit |
| `()` | 둘 다 omit (baseline α₀ 만) |

Credit methods: `shapley` (Methodology_05 통합 framework primary), `backelim` (Shender 2023 paper primary).

스크립트: `scripts/run_multivariate_recovery.py`. 결과: `results/part1/multivariate_recovery.csv` (40 rows).

---

## 4. 결과 — 5-seed mean ± std

### 4.1 Shapley credit

| user_features | MAE | RMSE | Kendall τ | Top-3 |
|---|---|---|---|---|
| **multivariate (segment+device)** ⭐ | **0.0160 ± 0.0055** | 0.0212 ± 0.0066 | 0.867 ± 0.085 | 93% |
| univariate (segment) | 0.0161 ± 0.0053 | 0.0213 ± 0.0061 | 0.886 ± 0.080 | 93% |
| device only | 0.0207 ± 0.0029 | 0.0270 ± 0.0035 | 0.943 ± 0.052 | 100% |
| no user feature | 0.0215 ± 0.0034 | 0.0277 ± 0.0040 | 0.924 ± 0.043 | 100% |

### 4.2 BackElim credit

| user_features | MAE | RMSE | Kendall τ | Top-3 |
|---|---|---|---|---|
| multivariate (segment+device) | 0.0536 ± 0.0053 | 0.0623 ± 0.0061 | 0.924 ± 0.080 | 100% |
| univariate (segment) | 0.0538 ± 0.0075 | 0.0623 ± 0.0080 | 0.924 ± 0.080 | 100% |
| device only | 0.0559 ± 0.0034 | 0.0639 ± 0.0042 | 0.943 ± 0.052 | 100% |
| no user feature | 0.0561 ± 0.0056 | 0.0641 ± 0.0061 | 0.943 ± 0.052 | 100% |

### 4.3 H_recovery 검증 — per-seed 비교

| credit | seeds where multi < uni | mean Δ MAE (multi − uni) | conclusion |
|---|---|---|---|
| **shapley** | **3/5** (seeds 1, 42, 100) | **−0.0001** | **약하게 지지** (추세는 맞으나 magnitude 가 noise floor 미만) |
| backelim | 1/5 (seed 42) | −0.0002 | 미지지 — 4 seed 에서 multivariate 가 미세하게 더 나쁨 |

**Per-seed Δ MAE (multi − uni)**:
- shapley: {1: −0.0014, 7: +0.0016, 13: +0.0009, 42: −0.0013, 100: −0.0004}
- backelim: {1: +0.0020, 7: +0.0008, 13: +0.0013, 42: −0.0053, 100: +0.0005}

---

## 5. 분석 — 왜 multivariate 의 이점이 작은가

### 5.1 Sample noise floor

20K user × 2.5% conv rate ≈ 500 converters. MAE 의 sample noise floor 는 약 **0.005-0.01** 수준 (5-seed std 가 0.005-0.008). 관측된 Δ MAE = −0.0001 ~ −0.0002 는 **이 noise floor 보다 1-2 orders of magnitude 작음** → 통계적 유의성 도달 불가.

### 5.2 Segment-channel correlation 가 multivariate 효과를 마스킹

본 DGP 의 segment 는 *start_channel* 에도 영향 (Methodology_01 § segment definition: `start_channels` per segment). 즉:
- segment → channel 분포 (channel 빈도가 segment-dependent)
- segment → conversion baseline (η_segment)

이는 segment 가 backdoor confounder 임을 시사하나, segment effect 의 일부는 이미 *채널 β estimate 에 흡수* 됨 (각 segment 가 다른 채널 mix 를 보므로 β estimate 가 segment-conditional 평균에 가까움). 따라서 segment dummy 추가 시 marginal information 이 작음.

반면 **device 는 채널 분포와 완전히 orthogonal** (모든 device 가 동일한 channel sequence 분포 사용) — device dummy 가 추가하는 정보는 "pure baseline shift" 로, segment 보다 더 깨끗하게 식별 가능. 그럼에도 magnitude 가 noise 에 묻힘.

### 5.3 Kendall τ 와 MAE 의 trade-off (역설)

Shapley 결과에서 흥미로운 패턴:
- **MAE 1위**: multivariate (0.0160)
- **τ 1위**: device only (0.943) — multivariate (0.867) 보다 높음

해석: segment dummy 를 빼면 (device only / no feature) magnitude estimate 는 약간 부정확해지나 ranking 안정성이 올라감. segment dummy 가 작은 채널 ranking swap 을 유발 (특히 small effect 채널 간) — magnitude 정확도와 ranking 정확도가 다른 winner 를 가지는 익숙한 현상 (Methodology_06 § 3 의 BackElim τ=1.0 vs Shapley τ=0.91 와 동일 패턴).

### 5.4 Outcome-model only 의 한계 (Methodology_05 § 4.2 와 일치)

본 실험은 **regression adjustment 단독 (outcome model only)** 의 한계 evidence:
- segment + device 둘 다 true confounder 이지만 multivariate Survival 의 개선은 noise 수준
- Strict debiasing (propensity-based) 가 필요할 가능성 — Methodology_05 § 8.1 의 Survival × IPW hybrid 가 향후 검증 대상
- 단순히 "user feature 더 추가 = 더 정확" 가설은 이 DGP/sample size 에서 *실증적으로 약하게 지지*

---

## 6. 결론

### 6.1 H_recovery 판정

- **shapley 에서 약한 지지** (3/5 seeds, Δ MAE = −0.0001 — noise floor 미만)
- **backelim 에서 미지지** (1/5 seeds)
- magnitude 는 작으나 **방향성 (multivariate ≤ univariate) 은 평균적으로 일치**
- 결정적 지지에는 더 큰 샘플 (50K-100K converter) 또는 더 강한 device effect 필요

### 6.2 정직한 함의

1. **Multivariate API 는 정상 작동** — 코드 검증 완료 (Methodology_05 § 5.3 단위테스트 18, 19; 본 실험에서 large-scale 호출 정상)
2. **이론적 정당성**: Eq 10 generic user feature 의 multivariate 확장은 학술적으로 명확
3. **실증적 효과는 작음**: 본 DGP 설정에서 multivariate 의 marginal MAE 개선 < 0.001 (noise floor 미만)
4. **운영 권장**: multivariate user feature 사용은
   - (i) **사전 정당화** (DAG 기반 backdoor confounder 식별, mediator/collider 회피 — Methodology_05 § 5.2)
   - (ii) **충분한 converter 수** (≥ 1000 권장)
   - (iii) **strict causal claim 시 propensity-based hybrid 고려** (§ 8.1 future work)
   - 의 조건 하에서만 의미 있음

### 6.3 다음 단계

1. **Sample size 확대 실험**: 50K, 100K user 에서 multivariate 의 효과 크기 변화 측정 (sample efficiency curve)
2. **Device effect magnitude sweep**: device_etas magnitude 를 0.1 / 0.4 / 0.8 로 변화시켜 detectability boundary 식별
3. **Survival × IPW Hybrid 구현** (Methodology_05 § 8.1): propensity-based debiasing 이 multivariate outcome model 단독보다 효과적인지 검증

---

## 부록 A — 재현 명령

```bash
# Single seed smoke test (5K users, ~30s)
PYTHONPATH=. python scripts/run_multivariate_recovery.py --seeds 42 --n-users 5000

# Full 5-seed × 20K (~90s)
PYTHONPATH=. python scripts/run_multivariate_recovery.py
# → results/part1/multivariate_recovery.csv (40 rows)
```

## 부록 B — DGP 파일

- `part1_simulation/dgp/alternatives/multivariate_dgp.py` — `generate_dgp_multivariate()` (160줄, alternative DGP interface 준수)
- `scripts/run_multivariate_recovery.py` — 실험 runner + summary + H_recovery test
- `results/part1/multivariate_recovery.csv` — raw results

## 부록 C — α₀ calibration 안정성 (5 seeds)

| seed | conv_rate | α₀ |
|---|---|---|
| 42 | 0.02465 | −5.438 |
| 1 | 0.02475 | (≈ −5.44) |
| 7 | 0.02485 | (≈ −5.44) |
| 13 | 0.02540 | (≈ −5.44) |
| 100 | 0.02490 | −5.438 |

→ Binary search converges within ±0.5%p of target across all seeds. α₀ stability 확인.
