# Methodology 06 — DGP Robustness Validation + Du LSTM IncShap

> **목적**: Survival/Poisson Shapley 의 우월성이 **DGP-method matching advantage** 인지 검증. 4가지 다른 probability process 기반 DGP 에서 모든 method 성능 측정.
>
> **결론 미리보기**: Survival/Poisson Shapley 는 4 DGPs 중 3개 (logistic, cox, hawkes) 에서 우수하나 **Markov DGP 에서는 11위** 로 추락. Mean MAE 기준 **1위는 Time Decay (0.034)**, Survival Shapley 는 6위 (0.055). **순위(τ) 기준으로는 Survival BackElim 이 가장 robust** (mean τ=0.786). 결과적으로 "Survival/Poisson 이 universally best" 는 사실이 아님 — DGP 구조에 따라 다름.

---

## 1. 4가지 alternative DGP 설계

### 1.1 Logistic response — `logistic_dgp.py`

$$P(\text{conv} \mid \text{user}) = \sigma\!\left(\alpha_0 + \sum_k w_k \cdot n_k(\text{user}) + \eta_{\text{seg}}\right)$$

- $n_k$ = 유저의 채널 k 누적 touchpoint 수
- LR-friendly form — Du LR-based IncShap **자연 우대 가설**

### 1.2 Discrete Markov chain — `markov_dgp.py`

9-state Markov: 7 channels + Conv (absorbing) + Drop (absorbing).
$$P(s_{t+1} = j \mid s_t = i) \text{ — 채널별 } (P_\text{conv}, P_\text{drop}, P_\text{cont})$$

- GT: removal effect (채널 k 제거 시 reach Conv 확률 변화)
- **Markov method 자연 우대 가설**

### 1.3 Cox PH with Weibull baseline — `cox_dgp.py`

$$\lambda(t) = h_0(t) \cdot \exp(\beta^T x(t)), \quad h_0(t) = \tfrac{k}{\theta}\!\left(\tfrac{t}{\theta}\right)^{k-1}$$

- Weibull shape $k=1.5$ → 비-log-linear baseline
- Survival/Poisson 의 step-function approximation 한계 테스트
- GT: $(\exp(\beta_k) - 1)$ 정규화

### 1.4 Multivariate self-exciting Hawkes — `hawkes_dgp.py`

$$\lambda(t) = \mu + \sum_{t_j < t} \alpha_{c(t_j)} \exp(-\beta(t-t_j))$$

- 각 광고가 미래 conversion intensity 를 self-excite
- Ogata thinning 으로 시뮬레이션
- Survival 의 self-excitation hook 검증

---

## 2. Cross-DGP MAE 매트릭스 (full result)

| Method | Cox | Hawkes | Logistic | Markov | **Mean** |
|---|---|---|---|---|---|
| **Time Decay (7.0d)** | 0.047 | 0.044 | 0.024 | 0.021 | **0.034 ⭐ #1** |
| Linear | 0.058 | 0.055 | 0.034 | 0.017 | 0.041 |
| Last Click | 0.038 | 0.036 | 0.048 | 0.053 | 0.044 |
| Markov (order=2) | 0.067 | 0.066 | 0.045 | **0.005** | 0.046 |
| Markov (order=1) | 0.074 | 0.070 | 0.056 | **0.004** | 0.051 |
| **Survival/Poisson (Shapley)** | **0.035** | **0.025** | **0.014** | 0.147 | **0.055 #6** |
| Survival/Poisson (BackElim) | **0.014** | **0.016** | 0.066 | 0.125 | 0.055 |
| Position-Based | 0.076 | 0.069 | 0.069 | 0.019 | 0.058 |
| Survival/Poisson (AICPE) | 0.037 | 0.035 | 0.021 | 0.166 | 0.065 |
| Doubly Robust | 0.076 | 0.105 | 0.071 | 0.037 | 0.072 |
| Incremental Shapley (LR) | 0.038 | 0.065 | 0.059 | 0.149 | 0.078 |
| **Incremental Shapley (LSTM)** | 0.079 | 0.075 | 0.133 | **0.028** | 0.079 |
| Shapley (model-based) | 0.058 | 0.076 | 0.027 | 0.165 | 0.082 |
| DML | 0.033 | 0.061 | 0.093 | 0.177 | 0.091 |
| IPW | 0.043 | 0.083 | 0.065 | 0.174 | 0.091 |
| First Click | 0.154 | 0.144 | 0.202 | 0.038 | 0.135 |

**굵게 표기**: 해당 column 의 best (DGP-specific winner).

---

## 3. Cross-DGP Kendall τ 매트릭스 (ranking accuracy)

| Method | Cox | Hawkes | Logistic | Markov | **Mean τ** |
|---|---|---|---|---|---|
| **Survival/Poisson (BackElim)** | **1.00** | **0.91** | 0.81 | 0.43 | **0.79 ⭐ #1** |
| Last Click | 0.71 | 0.71 | 0.81 | 0.81 | 0.76 |
| **Survival/Poisson (Shapley)** | **1.00** | 0.91 | **0.91** | 0.20 | 0.75 |
| Survival/Poisson (AICPE) | **1.00** | 0.68 | **0.91** | 0.41 | 0.75 |
| Time Decay | 0.62 | 0.52 | 0.81 | 0.81 | 0.69 |
| DML | 0.91 | 0.43 | 0.82 | 0.62 | 0.69 |
| Shapley (model-based) | 0.78 | 0.41 | 0.81 | 0.62 | 0.66 |
| Incremental Shapley (LR) | 0.81 | 0.51 | 0.62 | 0.51 | 0.61 |
| IPW | 0.71 | 0.43 | 0.43 | 0.62 | 0.55 |
| Doubly Robust | 0.51 | 0.51 | 0.82 | 0.00 | 0.46 |
| Incremental Shapley (LSTM) | 0.43 | 0.43 | 0.28 | 0.71 | 0.46 |
| Markov (order=2) | 0.24 | 0.24 | 0.43 | 0.81 | 0.43 |
| Markov (order=1) | 0.24 | 0.24 | 0.24 | **0.91** | 0.41 |
| Linear | 0.14 | 0.14 | 0.52 | 0.81 | 0.41 |
| Position-Based | 0.14 | 0.24 | 0.24 | **0.91** | 0.38 |
| First Click | -0.29 | -0.29 | -0.39 | 0.33 | -0.16 |

---

## 4. 핵심 분석 질문 답변

### Q1: Survival Shapley 가 4 DGPs 모두에서 top-3 인가?

**No.** 4 DGPs 의 method ranking 에서 Survival Shapley 위치:

| DGP | Survival Shapley rank (MAE) | 이유 |
|---|---|---|
| Logistic | **#1** (0.014) | Sigmoid 구조에도 channel-bin Poisson 이 잘 맞음 |
| Cox | **#2** (0.035, BackElim 0.014 다음) | Weibull baseline vs step-function approximation 손실 |
| Hawkes | **#2** (0.025, BackElim 0.016 다음) | 자기-가산 구조도 interval Poisson 이 흡수 |
| **Markov** | **#11** (0.147) | 9-state absorbing Markov 가 Poisson process 와 근본적으로 다름 |

→ **3/4 DGPs 에서 top-2** (excellent), **but Markov DGP 에서 catastrophic failure**.

### Q2: DGP-method matching 효과는 얼마나 큰가?

| DGP | "자연 우대" 예측 method | 실제 winner | 매칭 정도 |
|---|---|---|---|
| Logistic | Du LR-IncShap | Survival Shapley (0.014) | LR-IncShap 은 0.059 — 가설 **반증**! Survival 이 logistic 도 더 잘 fit |
| Markov | Markov method | Markov (1st) (0.004) | **완벽 매칭** — Markov 가 30× 우수 |
| Cox | Survival/Poisson | Survival BackElim (0.014) | **매칭** — 그러나 Cox 의 Weibull baseline 으로 Survival 이 다른 DGP 보다는 정확도 낮음 |
| Hawkes | Survival self-excitation | Survival BackElim (0.016) | **매칭** — 자기-가산이 step-function 으로 흡수됨 |

→ **Markov DGP 에서만 명확한 matching 효과** (Markov 가 30× 차이로 압도). 다른 DGPs 에서는 Survival 이 매칭 외에서도 잘 작동.

### Q3: Du LR vs LSTM IncShap — LSTM 이 일관되게 우월?

**No.** Mixed results:

| DGP | Du LR-IncShap MAE | Du LSTM-IncShap MAE | LSTM 우월? |
|---|---|---|---|
| Logistic | 0.059 | **0.133** | ❌ LSTM 2.3× 악화 (overfit on simple structure) |
| Markov | 0.149 | **0.028** | ✅ LSTM 5.3× 우월 (sequence 구조 포착) |
| Cox | **0.038** | 0.079 | ❌ LSTM 2.1× 악화 |
| Hawkes | **0.065** | 0.075 | ≈ 비슷 |

→ **DGP 구조에 따라 LSTM 우열 결정**. LSTM 이 universal upgrade 아님. Logistic/Cox 같은 단순 구조에서는 **LSTM 이 overfit**, Markov 같은 sequence-heavy 에서는 LSTM 이 우월. **Du 원 논문의 LSTM 권고가 항상 맞지는 않음** (sample size + DGP 구조 의존).

### Q4: Survival Shapley vs Du LSTM-IncShap — 같은 Shapley credit, 다른 response model

| DGP | Survival Shapley (Poisson 응답) | Du LSTM-IncShap | 승자 |
|---|---|---|---|
| Logistic | 0.014 | 0.133 | **Survival** (9.5×) |
| Markov | 0.147 | 0.028 | **Du LSTM** (5.3×) |
| Cox | 0.035 | 0.079 | **Survival** (2.3×) |
| Hawkes | 0.025 | 0.075 | **Survival** (3.0×) |

→ Survival 이 3/4 에서 우월. **Markov DGP 에서만 LSTM response 가 결정적 advantage**. 

**Response model 의 효과**:
- Poisson interval GLM: 시간 순서 + 채널 빈도 동시 포착, 작은 sample 에서도 안정
- LSTM: sequence representation 우수, but small sample 에서 overfit
- 본 시뮬 (10K-20K user) 에서는 Poisson 이 LSTM 보다 유리한 경우 多

---

## 5. 종합 ranking 재해석

### 5.1 Mean MAE 기준 ranking

```
1. Time Decay (7.0d)         0.034  ← rule-based 가 1위!
2. Linear                    0.041
3. Last Click                0.044
4. Markov (order=2)          0.046
5. Markov (order=1)          0.051
6. Survival/Poisson (Shapley) 0.055
6. Survival/Poisson (BackElim) 0.055
8. Position-Based            0.058
9. Survival/Poisson (AICPE)  0.065
10. Doubly Robust            0.072
11. Incremental Shapley (LR) 0.078
12. Incremental Shapley (LSTM) 0.079
13. Shapley (model-based)    0.082
14. DML / IPW                0.091
16. First Click              0.135
```

→ **Rule-based methods (Time Decay, Linear, Last Click) 이 cross-DGP 평균 1-3위**! 이는 robust = "구조 가정 적은" 방법이 다양한 DGP 에서 평균적으로 잘 작동함을 시사.

### 5.2 Mean Kendall τ 기준 ranking

```
1. Survival/Poisson (BackElim) 0.79  ← 채널 ranking 일관 best
2. Last Click                  0.76
3. Survival/Poisson (Shapley)  0.75
4. Survival/Poisson (AICPE)    0.75
5. Time Decay                  0.69
6. DML                         0.69
...
```

→ **순위 정확도 (τ) 에서는 Survival BackElim 이 일관 1위**. magnitude 정확도와 ranking 정확도가 다른 method 에 유리한 흥미로운 발견.

### 5.3 핵심 trade-off

| 지표 | 1위 | 의미 |
|---|---|---|
| **Mean MAE (magnitude)** | Time Decay (0.034) | Rule-based 가 cross-DGP 에서 robust |
| **Mean Kendall τ (ranking)** | Survival BackElim (0.79) | TEDDA 의 paper-faithful 구조가 ranking 에 강함 |
| **Single-DGP best** | (DGP-dependent) | DGP-method matching 효과 강함 |

---

## 6. 논문화 시사점

이전 결론 "Survival/Poisson Shapley = 18-method 종합 best" 는 **단일 DGP (Shender-flavored) 에서의 결과**. Cross-DGP 결과는 다음을 시사:

1. **Survival/Poisson 의 우월성은 DGP-conditional**. Poisson process 와 비슷한 구조의 DGP 에서 강함, Markov-style 에서 약함.

2. **Rule-based Time Decay 의 robustness 가 의외로 강함** — paper 작성 시 baseline 으로 신중히 다뤄야 함.

3. **Du LSTM IncShap 은 universal upgrade 아님** — sample size + DGP 구조 의존. Du 원 논문의 강한 LSTM 권고는 본 시뮬 규모에서는 부분 지지.

4. **Magnitude vs Ranking trade-off**:
   - Magnitude 정확도: Rule-based / Markov 가 robust
   - Ranking 정확도: Survival/Poisson 이 robust
   - "최적 method" 가 evaluation metric 에 따라 다름

### 논문화 시 권장 표현

- **NOT**: *"Survival/Poisson Shapley is the best method"*
- **YES**: *"Survival/Poisson Shapley achieves competitive performance on Poisson-like DGPs (matching its structural assumptions) while showing robustness in ranking (Kendall τ)"*
- **Honest limitation**: *"The method's MAE performance degrades on discrete-state DGPs (Markov chain) where the underlying probability structure differs fundamentally from Poisson"*

### 향후 연구 방향

1. **Hybrid response model**: Markov-aware Survival (transition-augmented intensity) — Markov DGP 에서도 robust 한 unified method
2. **Real-world validation (Criteo)**: 실 데이터의 underlying structure 추정 (Cox? Logistic? Hybrid?) → 어느 method 가 deployment 에 적합한지
3. **Larger LSTM evaluation**: 100K+ users 에서 Du LSTM 의 진정한 잠재력 평가 (현재 sample size 가 LSTM 학습에 부족할 수 있음)
4. **Confidence intervals**: Bootstrap CV 를 cross-DGP 에 적용 → method × DGP 의 stability 정량화

---

## 7. 기술 세부 — 구현 + 재현

### 코드 위치
```
part1_simulation/dgp/alternatives/
├── __init__.py
├── logistic_dgp.py       # generate_dgp_logistic()
├── markov_dgp.py         # generate_dgp_markov()
├── cox_dgp.py            # generate_dgp_cox()
└── hawkes_dgp.py         # generate_dgp_hawkes()

part1_simulation/models/causal/
└── incremental_shapley_lstm.py  # Du LSTM + Attention IncShap

scripts/
├── run_dgp_robustness.py        # 4 DGPs × non-LSTM methods
└── run_du_lstm_on_dgps.py       # 4 DGPs × Du LSTM IncShap
```

### 재현 명령
```bash
# 1. Cross-DGP non-LSTM evaluation (~10분)
PYTHONPATH=. python scripts/run_dgp_robustness.py

# 2. Du LSTM IncShap on 4 DGPs (~5분)
PYTHONPATH=. python scripts/run_du_lstm_on_dgps.py

# 3. 결과 확인
python -c "
import pandas as pd
df = pd.read_csv('results/part1/dgp_robustness.csv')
print(df.pivot(index='method', columns='dgp', values='mae').to_string(float_format='%.4f'))
"
```

### Conversion rate 검증
| DGP | Target | Achieved |
|---|---|---|
| Logistic | 2.5% | 2.66% |
| Markov | 2.5% | 2.34% |
| Cox | 2.5% | 1.57% |
| Hawkes | 2.5% | 2.85% |

→ 모두 1-3% 범위 내. Cox 가 약간 낮음 (Weibull baseline 의 calibration 어려움).
