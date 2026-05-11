# Marketing Attribution: From Simulation to Scale
## 프로젝트 계획서

**작성자:** Senior Marketing Data Scientist  
**작성일:** 2026-03-31  
**프로젝트 유형:** 개인 연구 / 포트폴리오  
**핵심 키워드:** Multi-Touch Attribution, Media Mix Modeling, Causal Inference, Bayesian Statistics, Deep Learning Sequence Models

---

## 1. 프로젝트 개요

### 1.1 배경 및 동기

Marketing Attribution은 "어떤 마케팅 채널이 전환에 얼마나 기여했는가?"라는 질문에 답하는 문제이다. 이 문제는 단순해 보이지만, 실무에서는 데이터의 한계, 방법론 간 결과 불일치, 인과 추론의 어려움, 프라이버시 규제 등 복합적인 도전에 직면한다.

현재 공개된 MTA 데이터셋 환경을 진단하면 다음과 같다. Criteo Attribution Dataset(2018)은 1,600만 건의 대규모 실 데이터이나 모든 feature가 해시/익명화되어 도메인 해석이 불가능하다. GA4 BigQuery Public Dataset은 해석 가능한 필드가 존재하나, traffic_source가 유저 레벨 first-touch로 고정되어 세션별 채널 시퀀스를 신뢰성 있게 재구성할 수 없으며, obfuscation으로 내부 일관성이 제한적이다. CriteoPrivateAd(2025)는 uuid가 일 단위로 리셋되어 멀티데이 여정 구성이 원천적으로 불가능하므로 MTA에 사용할 수 없다. Hillstrom, Criteo Uplift 등 인과추론 데이터셋은 treatment가 단일(binary)이어서 멀티채널 attribution 시나리오에 직접 적용이 어렵다.

즉, "해석 가능한 feature를 가진 대규모 공개 MTA 데이터셋"은 사실상 존재하지 않는다. 이 구조적 공백을 인식한 위에서, 본 프로젝트는 시뮬레이션 데이터(해석 가능, ground truth 존재)와 실 데이터(Criteo, 대규모)를 결합하는 하이브리드 접근을 채택한다.

### 1.2 프로젝트 목표

본 프로젝트의 목표는 세 가지이다.

첫째, Marketing Attribution 방법론의 체계적 비교 및 정량 평가. Rule-based 휴리스틱부터 Shapley Value, Markov Chain, LSTM/Transformer 기반 시퀀스 모델까지, 동일한 데이터에 복수의 방법론을 적용하고 ground truth 대비 정확도를 정량적으로 비교한다.

둘째, User-level MTA와 Aggregate-level MMM의 Triangulation. 유저 레벨 시퀀스 분석(MTA)과 집계 수준 시계열 분석(MMM)의 결과를 교차 검증하여, 두 접근의 일치/불일치를 분석하고 실무적 시사점을 도출한다.

셋째, 데이터 특성에 따른 방법론 선택 기준 제시. 데이터 규모, 시퀀스 길이, 채널 수, 해석 필요성 등 조건에 따라 어떤 방법론이 최적인지를 실험적 근거와 함께 정리한다.

### 1.3 프로젝트 구성 요약

| Part | 데이터 | 핵심 역할 | 방법론 |
|------|--------|----------|--------|
| Part 1 | 직접 설계한 시뮬레이션 데이터 | 방법론 정확도 검증 (ground truth 비교) | Rule-based, Markov, Shapley, 시퀀스 모델 |
| Part 2 | Criteo Attribution Dataset (2018) | 대규모 실 데이터에서의 스케일 검증 | LSTM + Attention, Transformer, SHAP |
| Part 3 | Meta Robyn / PyMC-Marketing 데이터 | Bayesian MMM 및 예산 최적화 | Bayesian Regression, Adstock, Saturation |
| 통합 | Part 1~3 결과 종합 | Triangulation 및 방법론 선택 프레임워크 | 비교 분석, sensitivity analysis |

---

## 2. Part 1 — 시뮬레이션 기반 MTA 방법론 비교

### 2.1 목적

해석 가능한 채널 구조와 알려진 ground truth를 가진 합성 데이터를 직접 설계하여, 각 MTA 방법론이 "실제 기여도"를 얼마나 정확하게 복원하는지를 정량적으로 평가한다.

### 2.2 시뮬레이션 DGP 학술적 레퍼런스

MTA 시뮬레이션의 DGP를 독자적으로 설계하는 대신, 학술적으로 검증된 기존 시뮬레이션 프레임워크를 기반으로 구축한다. 조사 결과, MTA 시뮬레이션 DGP를 명시적으로 제시한 논문은 세 편이 확인되었으며, 이들을 본 프로젝트의 DGP 설계 기반으로 채택한다.

#### 2.2.1 [Primary] Du et al. (2019) — JD.com / Stanford GSB

**논문:** "Causally Driven Incremental Multi Touch Attribution Using a Recurrent Neural Network"
**저자:** Ruihuan Du, Yu Zhong, Harikesh Nair, Bo Cui, Ruyang Shou
**게재:** ArXiv 1902.00215, AdKDD 2019 (Stanford Graduate School of Business Working Paper No. 3761)
**코드:** `github.com/jd-ads-data/jd-mta` (TensorFlow 기반, 시뮬레이션 + 모델 코드 공개)

**DGP 핵심 구조:**

이 논문은 JD.com의 실제 프로덕션 MTA 시스템을 위해 개발된 시뮬레이션 엔진을 제공한다. DGP는 유저 i, 브랜드 b, 일자 t에 대해 다음 변수를 생성한다.

- x_{i,t,b}: 유저 i가 브랜드 b에 대해 일자 t에 각 광고 포지션에서 받은 노출 벡터
- r_{b,t}: 브랜드 b의 일자 t 가격 지수 (경쟁/외부 변수)
- d_i: 유저의 시간 불변 특성 (user heterogeneity)
- h_{i,0,b}: 유저 i의 브랜드 b에 대한 초기 조건 (이전 구매 이력 등)
- y_{i,t,b}: 유저 i의 브랜드 b 일자 t 전환 여부 (binary)

시뮬레이션 파라미터는 브랜드 수(B), 시퀀스 일수(T), 광고 포지션 수, 유저당 일일 평균 노출 수, 총 유저 수(N)를 포함한다. 전환 확률은 광고 노출의 sequential dependence를 반영하여, 광고 강도(intensity), 타이밍(timing), 경쟁(competition), 유저 이질성(user heterogeneity)을 포착하는 구조로 설계되어 있다.

**모델 파이프라인:** 시뮬레이션 데이터 생성 → RNN 기반 전환 예측(response modeling) → Shapley Value 기반 credit allocation의 2단계 구조이다. Shapley Value는 공리적 기반(axiomatic foundations)과 공정성(fairness) 조건을 충족하도록 설계되었으며, 전체 incremental conversion을 광고에 배분한다.

**본 프로젝트에서의 활용:** DGP의 기본 골격(유저-브랜드-일자 구조, sequential dependence, user heterogeneity)을 채택한다. GitHub의 `generate_data.py`를 시작점으로 삼되, 채널명을 해석 가능한 마케팅 채널(Display, Search, Email 등)로 재정의하고, 파라미터를 본 프로젝트의 시나리오에 맞게 조정한다.

#### 2.2.2 [Secondary] Shender et al. (2020/2023) — Google Research

**논문:** "A Time To Event Framework For Multi-touch Attribution"
**저자:** Dinah Shender, Ali Nasiri Amini, Xinlong Bao, Mert Dikmen, Amy Richardson, Jing Wang (Google)
**게재:** ArXiv 2009.08432 (2020), Journal of Data Science, Vol.22, pp.56-76 (2023)

**DGP 핵심 구조:**

이 논문은 전환을 비동질 포아송 과정(inhomogeneous Poisson process)의 실현으로 모델링한다. 유저 i의 전환 강도(conversion intensity)를 다음과 같은 log-linear 모델로 정의한다.

```
log(λᵢ(t)) = α₀ + Σⱼ Σₖ βₖ · xⱼₖ · f(t - tⱼ)
```

여기서 λᵢ(t)는 시점 t에서 유저 i의 순간 전환 강도, tⱼ는 j번째 광고 노출 시점, xⱼₖ는 j번째 광고의 k번째 feature 값, f(t - tⱼ)는 광고 효과의 시간 감쇠 함수이다.

시뮬레이션에서 광고 효과의 시간 감쇠를 명시적으로 제어할 수 있다. 예를 들어, 광고 유형 1의 효과를 "1일차 1.5배, 2일차 1.2배, 3일차 이후 효과 없음"으로 설정하고, 광고 유형 2는 "장기적으로 효과가 지속"하도록 설정할 수 있다. 이를 통해 채널별 시간 감쇠 패턴의 ground truth를 정밀하게 제어할 수 있다.

검증은 500회 반복 시뮬레이션으로 수행하며, 각 시나리오에서 ground truth 계수와 추정 계수를 비교하고 2.5%/97.5% 분위수 기반 신뢰구간을 보고한다. 3개 시나리오(단일 광고 유형, 두 광고 유형, 효과 없는 광고 유형 포함)를 통해 모델의 강건성을 확인한다.

**Attribution 알고리즘:** 경로에서 마지막 광고를 반복적으로 제거하며 AICPE(Average Incremental Conversion Probability per Exposure)를 계산하는 방식을 사용한다.

**본 프로젝트에서의 활용:** Du et al.의 DGP에 Shender et al.의 **시간 감쇠 모델링**을 보강한다. 구체적으로, 채널별 광고 효과의 temporal decay function f(t - tⱼ)를 채널 유형에 따라 차별화하여 설정한다(예: Display는 장기 인지 효과로 느린 감쇠, Paid Search는 즉시 효과로 빠른 감쇠). 또한 Poisson process 기반 전환 모델은 Survival Analysis 기반 attribution(Part 1의 방법론 중 하나)과 자연스럽게 연결되므로, 시뮬레이션과 분석 방법론의 일관성을 확보할 수 있다.

#### 2.2.3 [Complementary] CDA Framework (2025) — 최신 레퍼런스

**논문:** "Causal-driven attribution (CDA): Estimating channel influence without user-level data"
**게재:** ArXiv 2512.21211 (December 2025)

**DGP 핵심 구조:**

이 논문은 가장 최근(2025년 12월)에 발표된 MTA 시뮬레이션 레퍼런스이다. 연구팀은 "거의 2년간의 탐색 끝에도 적절한 공개 데이터셋을 찾을 수 없었으며, 기존 출판물들은 두 개의 공개 데이터셋에만 의존하고 있으나 어느 것도 요구사항을 충족하지 못한다"고 밝히며, 합성 데이터 생성의 필요성을 명시적으로 논증한다. 이는 본 프로젝트의 시뮬레이션 접근과 동일한 문제 인식이다.

DGP는 집계 수준(aggregate-level)에서 설계되며, 다음 요소를 포함한다.

- 각 채널의 기본 노출 수(baseline impressions)
- 채널별 성장률(growth rate)
- 채널별 전환율(conversion rate)
- 변동성을 포착하기 위한 랜덤 노이즈
- 마케팅 채널 간 교차 영향(cross-channel influence) — 마지막 단계로 도입

실제 마케팅 데이터를 완벽하게 시뮬레이션할 수 없다는 점(고객 인지와 행동은 심리적 요인의 영향을 받음)을 인정하면서도, 단순화 가정 하에 현실적 시나리오를 구성한다.

검증은 PCMCI(temporal causal discovery) + Structural Causal Model을 사용하며, true causal graph가 주어진 경우 평균 relative RMSE 9.50%, predicted graph 사용 시 24.23%를 달성한다.

**본 프로젝트에서의 활용:** 이 논문의 핵심 기여인 **채널 간 교차 영향(cross-channel influence)** 모델링을 Du et al.의 DGP에 통합한다. 구체적으로, 특정 채널의 노출이 다른 채널의 후속 노출 확률이나 전환 기여에 영향을 미치는 구조(예: Display 노출 → Paid Search 클릭 확률 증가)를 DGP에 반영한다. 또한, 이 논문이 집계 수준 데이터만으로 attribution을 수행하는 프레임워크를 제시하므로, Part 3(MMM)과의 연결점으로도 활용할 수 있다.

#### 2.2.4 통합 DGP 설계 전략

본 프로젝트의 DGP는 위 세 논문의 강점을 다음과 같이 통합한다.

| DGP 요소 | 채택 소스 | 구체적 활용 |
|----------|----------|-----------|
| 기본 골격 (유저-채널-일자, sequential dependence) | Du et al. (2019) | `jd-mta` 코드의 `generate_data.py`를 시작점으로 활용 |
| 유저 이질성 (user heterogeneity) | Du et al. (2019) | 유저 특성 d_i를 세그먼트(신규/탐색/충성)로 구현 |
| 광고 강도 및 경쟁 효과 | Du et al. (2019) | 브랜드/캠페인 간 경쟁 구조 반영 |
| 시간 감쇠 함수 (temporal decay) | Shender et al. (2023) | 채널별 차별화된 f(t - tⱼ) 설정 |
| Poisson 기반 전환 모델 | Shender et al. (2023) | log-linear intensity model로 전환 확률 생성 |
| 채널 간 교차 영향 (cross-influence) | CDA (2025) | 채널 시너지/대체 효과를 전이 확률과 전환 확률에 반영 |
| 해석 가능한 채널명 | 본 프로젝트 독자 설계 | 7개 현업 채널명으로 재정의 |
| Ground truth 검증 방법론 | Shender et al. (반복 시뮬레이션) + Du et al. (5가지 방법론 비교) | 다수 반복 + 다수 방법론 비교를 결합 |

### 2.3 데이터 생성 프로세스 (DGP) 상세 설계

#### 2.3.1 채널 구조

7개의 명시적 마케팅 채널을 정의한다. 각 채널은 현업에서 관찰되는 역할과 특성을 반영한다.

| 채널 | 퍼널 역할 | 전환 직접 기여 | 인지/어시스트 기여 | 설계 의도 |
|------|-----------|-------------|-----------------|----------|
| Display Ads | Upper Funnel | 낮음 | 높음 | 인지 채널, 여정 초반에 등장, 낮은 클릭률 |
| Social Media (Paid) | Upper~Mid Funnel | 낮음~중간 | 중간 | 관심 유발, Display보다 약간 높은 engagement |
| Organic Search | Mid Funnel | 중간 | 중간 | 정보 탐색 단계, 자연스러운 유입 |
| Paid Search | Mid~Lower Funnel | 높음 | 낮음 | 구매 의도가 있는 유저의 검색, 전환 직전 |
| Email | Mid~Lower Funnel | 중간~높음 | 중간 | 기존 유저 재방문 유도, 리텐션 채널 |
| Referral | Mid Funnel | 중간 | 중간 | 외부 사이트 추천, 신뢰 기반 |
| Direct | Lower Funnel | 높음 | 낮음 | 브랜드 인지가 높은 유저의 직접 방문 |

#### 2.3.2 유저 여정 생성 로직 (Du et al. DGP 기반 확장)

**Step 1 — 유저 세그먼트 정의:**
3개 세그먼트를 정의한다. 신규 유저(전체의 50%, Display/Social에서 시작하는 경향), 탐색 유저(30%, Organic Search 중심의 긴 여정), 충성 유저(20%, Email/Direct 중심의 짧은 여정).

**Step 2 — 여정 길이 결정:**
세그먼트별로 여정 길이(터치포인트 수)의 분포를 다르게 설정한다. 신규 유저는 Geometric(p=0.25) + 1로 평균 약 5개, 탐색 유저는 Geometric(p=0.2) + 2로 평균 약 7개, 충성 유저는 Geometric(p=0.5) + 1로 평균 약 3개이다.

**Step 3 — 채널 시퀀스 생성 (위치 의존적 전이 확률):**
각 터치포인트의 채널은 이전 채널과 현재 위치(여정 내 순서)에 의존하는 전이 확률로 결정한다. 이는 1차 Markov + Position Effect를 결합한 구조이다.

예시 전이 확률 (여정 초반, position ≤ 0.3):
Display → Display 0.1, Display → Social 0.25, Display → Organic Search 0.35, Display → Paid Search 0.1, Display → Email 0.05, Display → Referral 0.1, Display → Direct 0.05 등으로, 초반에는 인지 채널에서 탐색 채널로의 전이가 우세하다.

예시 전이 확률 (여정 후반, position > 0.7):
Organic Search → Paid Search 0.3, Organic Search → Direct 0.25, Organic Search → Email 0.2 등으로, 후반에는 전환 채널로의 전이가 우세하다.

**Step 4 — 전환 확률 결정 (핵심 — Ground Truth):**
전환 확률은 채널 시퀀스의 함수로 정의한다. 이것이 곧 각 채널의 "실제 기여도"이며, 모든 attribution 모델의 평가 기준이 된다.

전환 확률 모델은 Du et al. (2019)의 sequential dependence 구조에 Shender et al. (2023)의 log-linear intensity model을 결합하여 다음과 같이 설계한다.

```
log(λᵢ(t)) = α₀ + Σⱼ Σₖ βₖ · xⱼₖ · f_channel(t - tⱼ)
                 + Σᵢⱼ δᵢⱼ · cross_influence(channelᵢ, channelⱼ)
                 + d_i · η
```

여기서 λᵢ(t)는 유저 i의 시점 t에서의 전환 강도(Shender et al. 기반), f_channel(t - tⱼ)는 채널별 시간 감쇠 함수, δᵢⱼ는 채널 간 교차 영향(CDA 2025 기반), d_i · η는 유저 이질성 효과(Du et al. 기반)이다. 최종 전환 여부는 이 intensity에 기반한 포아송 과정의 실현으로 결정된다.

**Ground Truth 파라미터 설정 예시:**

*채널 효과 계수 β (Du et al. 구조):*

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| β_display | 0.3 | Display 존재 시 전환 강도 소폭 증가 (인지 효과) |
| β_paid_search | 1.2 | Paid Search 존재 시 전환 강도 대폭 증가 (직접 전환) |
| β_email | 0.8 | Email 존재 시 중간 수준 증가 |

*채널별 시간 감쇠 함수 f_channel (Shender et al. 구조):*

| 채널 | 감쇠 패턴 | 구체적 설정 | 근거 |
|------|----------|-----------|------|
| Display Ads | 느린 감쇠 (장기 인지) | f(Δt) = exp(-Δt/14일) | 인지 효과는 2주에 걸쳐 서서히 감쇠 |
| Paid Search | 빠른 감쇠 (즉시 효과) | f(Δt) = exp(-Δt/1일) | 검색 의도는 당일~1일 내 소멸 |
| Email | 중간 감쇠 | f(Δt) = exp(-Δt/5일) | 이메일 열람 후 5일 정도 효과 지속 |
| Social Media | 중간~빠른 감쇠 | f(Δt) = exp(-Δt/3일) | 소셜 노출 효과 3일 정도 |
| Organic Search | 중간 감쇠 | f(Δt) = exp(-Δt/7일) | 정보 탐색 효과 1주일 정도 |
| Referral | 중간 감쇠 | f(Δt) = exp(-Δt/7일) | 추천 신뢰 효과 1주일 |
| Direct | 빠른 감쇠 | f(Δt) = exp(-Δt/2일) | 직접 방문은 즉시 의도 반영 |

*채널 간 교차 영향 δ (CDA 2025 구조):*

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| δ_display→paid_search | 0.4 | Display 노출 후 Paid Search 시 시너지 (인지→검색 전환) |
| δ_social→email | 0.3 | Social 노출 후 Email 시 시너지 (관심→리텐션 전환) |
| δ_organic→direct | 0.2 | Organic Search 후 Direct 시 시너지 (정보 탐색→브랜드 기억) |

이 파라미터들이 ground truth이며, 각 attribution 모델이 이를 얼마나 정확하게 복원하는지가 핵심 평가 지표이다. 검증은 Shender et al.의 방법론을 따라 다수 반복 시뮬레이션(100~500회)으로 추정의 안정성을 확인하고, Du et al.의 방법론을 따라 5가지 이상의 attribution 접근법을 동시에 비교한다.

**Step 5 — 시간 정보 생성:**
각 터치포인트에 timestamp를 부여한다. 터치포인트 간 간격은 Exponential(λ=1/48시간)로 생성하여, 평균 2일 간격의 현실적 패턴을 만든다.

#### 2.3.3 데이터 규모

| 지표 | 목표값 | 근거 |
|------|--------|------|
| 총 유저 수 | 100,000 | Markov/Shapley에 충분하고, LSTM도 학습 가능한 규모 |
| 전환율 | 2~3% | 현실적 이커머스 전환율 |
| 전환 유저 수 | 2,000~3,000 | 딥러닝 학습 최소 기준 충족 |
| 평균 터치포인트 수 | 4~5 | 시퀀스 모델의 의미가 있는 길이 |
| 최대 터치포인트 수 | 15~20 | 긴 여정에서의 모델 성능 테스트 |
| 고유 채널 수 | 7 | 정확 Shapley 계산 가능 (2⁷ = 128 coalition) |

#### 2.3.4 검증: DGP가 현실적인지 확인

데이터 생성 후 다음을 확인한다. 전환율이 2~3% 범위인지, 채널별 빈도 분포가 현실적인지(Direct와 Organic Search가 가장 많고, Display가 중간, Referral이 가장 적은 등), 여정 길이 분포가 right-skewed인지(대부분 2~5, 소수가 10+), 전환 여정과 비전환 여정의 채널 구성이 통계적으로 유의하게 다른지를 확인한다.

### 2.4 적용 방법론

#### 2.4.1 Rule-Based Attribution (5종)

Last Click, First Click, Linear, Time Decay(반감기 7일), Position-based(40/20/40)를 적용한다. 각 모델에서 산출한 채널별 기여도 비율을 ground truth(β, γ, δ 파라미터에서 도출한 Shapley value)와 비교한다.

**평가 지표:** 채널별 기여도의 Mean Absolute Error(MAE), 채널 랭킹의 Kendall's Tau(순위 일치도), 특정 채널의 과대/과소평가 패턴 분석.

#### 2.4.2 Markov Chain Attribution

**1차 Markov:** 채널 간 전이 확률 행렬 추정 → Removal Effect 계산 → 기여도 산출. 전이 확률 행렬 히트맵, 채널 네트워크 그래프(Sankey diagram) 시각화.

**2차 Markov:** 2-step 전이 확률 추정. 상태 공간 확장에 따른 희소성 문제 관찰. 1차와 2차의 기여도 차이 분석.

**Higher-order Markov (3차 이상):** 데이터 충분성 여부에 따라 시도. Laplace smoothing 적용.

**평가:** Ground truth 대비 MAE, 특히 상호작용 효과(δ)가 있는 채널 쌍에서 Markov가 이를 포착하는지 확인.

#### 2.4.3 Shapley Value Attribution

**가치 함수 v(S) — 두 가지 버전:**

Version A (전환율 기반): v(S) = 채널 집합 S만 거친 여정의 전환율. 직관적이나, 희귀 채널 조합에서 불안정.

Version B (모델 기반): 로지스틱 회귀를 학습한 뒤, 채널 feature를 마스킹하여 v(S) 산출. 더 안정적이나, 모델 오차가 전파.

채널 수가 7개이므로 정확 계산이 가능하다(2⁷ = 128 coalition). 두 버전의 결과 차이와, ground truth 대비 정확도를 비교한다.

**Shapley 공리 검증:** 효율성(모든 채널 기여도 합 = 전체 전환), 대칭성, Null Player 속성을 실제 결과에서 확인한다.

#### 2.4.4 LSTM + Attention

**모델 아키텍처:**
입력: 패딩된 채널 시퀀스 (max_length × feature_dim). 각 터치포인트의 feature: [channel_one_hot(7), time_since_previous, position_in_journey]. Embedding → LSTM(hidden_dim=64) → Attention → Dense → Sigmoid.

**기여도 추출 (3가지 방법 비교):**
Attention Weight: 각 time step의 attention score를 기여도로 사용. SHAP (DeepExplainer): 학습된 모델에 SHAP을 적용. Leave-One-Out: 각 터치포인트를 제거하고 예측 확률 변화 측정.

**평가:** 세 가지 기여도 추출 방법의 상관관계, ground truth 대비 MAE, Rule-based 및 Markov 결과와의 비교.

#### 2.4.5 Transformer (조건부)

LSTM과 동일한 입력에 대해, Encoder-only Transformer(1~2 layer, 2 heads)를 적용한다. 시퀀스 길이별 LSTM vs Transformer 성능 비교(length 2-3, 4-5, 6-10, 10+)를 수행하여, "어떤 시퀀스 길이에서 Transformer가 LSTM을 능가하는가?"를 실험적으로 확인한다.

#### 2.4.6 Causal Inference 기반 MTA

기존 MTA 방법론(Rule-based, Markov, Shapley, 딥러닝)은 "전환을 예측하는 데 어떤 채널이 중요한가?"라는 상관(correlational) 질문에 답한다. 반면, Causal MTA는 "이 채널의 광고가 없었다면 전환이 일어나지 않았을 것인가?"라는 반사실적(counterfactual) 질문에 답한다. 이 차이는 실무에서 매우 중요하다. 예를 들어, Paid Search가 Last Click 기준으로 기여도가 높더라도, 해당 유저가 Paid Search 없이도 어차피 전환했을 가능성이 있다면 Paid Search의 인과적 기여는 상관적 기여보다 낮다.

본 프로젝트에서는 시뮬레이션 데이터의 DGP를 알고 있으므로, 반사실적 전환 확률을 직접 계산하여 causal ground truth를 구할 수 있다는 고유한 장점이 있다. 이를 활용하여 다음 방법론을 적용한다.

**A. Incremental Shapley Value (Du et al. 2019 기반)**

Du et al.의 핵심 기여는 전통적 Shapley Value가 전체 전환 크레딧을 배분하는 것과 달리, incremental conversion(광고로 인한 순증 전환)만을 배분하는 Incremental Shapley를 제안한 점이다.

구현 방식: (1) Response model(RNN 또는 로지스틱 회귀)로 P(conversion | exposure=x)를 추정한다. (2) Counterfactual P(conversion | exposure=0), 즉 광고 노출이 전혀 없었을 경우의 전환 확률을 추정한다. (3) Incremental conversion = P(conversion | exposure=x) - P(conversion | exposure=0)을 계산한다. (4) 이 incremental 부분만을 Shapley Value로 각 채널에 배분한다.

이 접근의 장점은 "기저 전환(base conversion)"과 "광고로 인한 추가 전환"을 분리한다는 것이다. 기저 전환은 광고 없이도 발생했을 전환이므로, 어떤 채널에도 크레딧을 부여하지 않는 것이 인과적으로 올바르다.

**B. Survival Analysis 기반 Causal Attribution (Shender et al. 2023 기반)**

Shender et al.의 Poisson process 모델은 본질적으로 인과적 해석이 가능하다. 전환 강도 λ(t)를 광고 노출의 함수로 모델링하므로, 특정 광고를 경로에서 제거했을 때 λ(t)가 얼마나 감소하는지를 직접 계산할 수 있다.

구현 방식: (1) Inhomogeneous Poisson process 기반 전환 모델을 학습한다: log(λᵢ(t)) = α₀ + Σⱼ βₖ · f(t - tⱼ). (2) 각 광고 j에 대해 counterfactual intensity λ_{-j}(t)를 계산한다 (광고 j의 효과를 제거). (3) AICPE(Average Incremental Conversion Probability per Exposure) = [λ(t) - λ_{-j}(t)] / λ(t)로 기여도를 산출한다.

이 접근의 장점은 시간 축을 명시적으로 모델링하므로, "이 채널은 전환을 며칠 앞당겼는가?"라는 질문에도 답할 수 있다는 것이다. Cox Proportional Hazards model의 time-varying covariate 확장도 비교 대상으로 포함한다.

**C. Propensity Score 기반 Causal Attribution**

시뮬레이션 데이터에서 유저 세그먼트에 따라 채널 노출 확률이 다르게 설계되어 있으므로(예: 충성 유저는 Email을 더 많이 받음), selection bias가 존재한다. 이를 보정하지 않으면 Email의 효과가 과대추정될 수 있다.

구현 방식: (1) Propensity Score 추정: 각 채널 c에 대해, P(channel_c 노출 | user_features)를 로지스틱 회귀로 추정한다. (2) Inverse Propensity Weighting (IPW): 각 터치포인트의 기여도를 1/propensity로 가중하여 selection bias를 보정한다. (3) Doubly Robust Estimator: outcome model(전환 예측)과 propensity model을 결합하여, 둘 중 하나만 올바르면 일관된 추정을 보장한다.

이 접근의 목적은 관찰 데이터에서 발생하는 confounding을 명시적으로 다루는 것이며, DGP의 유저 세그먼트 구조가 confounding의 역할을 한다.

**D. Double Machine Learning (DML) for MTA**

Chernozhukov et al. (2018)의 DML 프레임워크를 MTA에 적용한다. DML은 nuisance parameter(교란 변수의 영향)를 ML 모델로 추정하고 제거한 뒤 treatment effect를 추정하는 방법으로, 고차원 confounding에 강건하다.

구현 방식: (1) 각 채널 c를 treatment로, 전환 여부를 outcome으로, 나머지 채널 노출과 유저 특성을 confounders로 설정한다. (2) Cross-fitting을 통해 과적합 편향을 방지한다. (3) 각 채널의 ATE(Average Treatment Effect)를 추정하여 인과적 기여도로 사용한다.

EconML 라이브러리의 `DML` 또는 `LinearDML` 클래스를 활용한다. 멀티채널 환경에서는 각 채널을 순차적으로 treatment로 설정하여 7개 채널의 ATE를 개별 추정한다.

**E. Causal Deep Learning Attribution (CAMTA / CausalMTA 계열)**

최근 학술 연구에서 인과적 관점을 딥러닝 시퀀스 모델에 통합하는 접근이 활발히 제안되고 있다.

CAMTA (Kumar et al., ICDM Workshop 2020): Causal Attention Model for Multi-Touch Attribution. RNN 기반으로 time-varying confounders를 통제하면서 채널별 causal effect를 추정한다. Attention mechanism이 인과적 기여도를 직접 학습하도록 설계되었다.

CausalMTA (Yao et al., 2022): 정적 유저 속성과 동적 행동 feature로부터 발생하는 confounding bias를 counterfactual prediction으로 제거하는 모델이다.

DCRMTA (Tang et al., 2024): Deep Causal Representation for MTA. 유저 행동과 전환 사이의 causal feature를 추출하면서 confounding variable의 영향을 분리하는 end-to-end 접근이다.

본 프로젝트에서는 CAMTA의 아이디어를 기반으로, Part 1의 LSTM + Attention 모델에 causal regularization을 추가하는 변형을 구현한다. 구체적으로, attention weight가 단순 예측 기여가 아닌 counterfactual 기여(해당 터치포인트 제거 시 예측 변화)와 정렬되도록 auxiliary loss를 추가한다.

**F. 방법론 카테고리 정리 (업데이트)**

| 카테고리 | 방법론 | 핵심 질문 | 인과성 수준 |
|---------|--------|----------|-----------|
| Rule-based | Last/First/Linear/Time Decay/Position | "크레딧을 어떤 규칙으로 나눌 것인가?" | 없음 (휴리스틱) |
| Statistical | Markov Chain | "채널 전이 구조에서 어떤 채널이 중요한가?" | 약함 (Removal Effect ≈ 반사실적) |
| Game-theoretic | Shapley Value | "공정한 기여도 배분은?" | 약함 (공리 기반, 인과 아님) |
| Predictive DL | LSTM/Transformer + Attention | "전환 예측에 어떤 터치포인트가 중요한가?" | 약함 (상관 기반) |
| **Incremental Causal** | **Incremental Shapley (Du et al.)** | **"광고로 인한 순증 전환은 얼마인가?"** | **중간 (모델 기반 반사실)** |
| **Time-to-Event Causal** | **Survival/Poisson Attribution (Shender et al.)** | **"이 광고가 전환을 얼마나 앞당겼는가?"** | **중간 (intensity 기반 반사실)** |
| **Debiased Causal** | **IPW / Doubly Robust / DML** | **"Selection bias를 보정한 채널 효과는?"** | **높음 (교란 보정)** |
| **Causal DL** | **CAMTA 변형 (Causal Attention)** | **"딥러닝 기여도가 인과적으로 타당한가?"** | **중간~높음 (모델 의존)** |

### 2.5 핵심 실험 및 분석

**실험 1 — 방법론 정확도 비교 (메인 실험):**
7개 방법론(Rule-based 5 + Markov + Shapley)과 2~3개 딥러닝 변형, 그리고 4개 causal 방법론의 채널별 기여도를 ground truth와 비교한다. 결과를 하나의 요약 테이블과 레이더 차트로 시각화한다.

**실험 2 — 상호작용 효과 포착 능력:**
DGP에 설정한 채널 간 시너지(δ_display→paid_search 등)를 각 모델이 얼마나 포착하는지를 분석한다. 상호작용을 명시적으로 모델링하지 않는 Rule-based 모델 vs Markov(시퀀스 반영) vs 딥러닝(비선형 상호작용 학습) vs Causal 방법론(교란 보정 후 상호작용)의 차이를 보여준다.

**실험 3 — 데이터 규모 민감도 (Learning Curve):**
유저 수를 1,000 → 5,000 → 10,000 → 50,000 → 100,000으로 변화시키며, 각 모델의 ground truth 대비 MAE 변화를 추적한다. "이 방법론은 최소 N명의 데이터가 있어야 신뢰할 수 있다"는 실무 가이드를 도출한다.

**실험 4 — DGP 가정 변화 민감도:**
상호작용 효과를 제거한 DGP, 시간 감쇠를 제거한 DGP, 유저 세그먼트 이질성을 제거한 DGP 등에서 각 모델의 성능 변화를 관찰한다. "데이터의 어떤 특성이 방법론 선택에 영향을 미치는가?"를 분석한다.

**실험 5 — Correlational vs Causal Attribution 비교 (신규):**
동일 데이터에 대해 상관적 방법론(Shapley, LSTM Attention)과 인과적 방법론(Incremental Shapley, IPW, DML)의 결과를 직접 비교한다. 시뮬레이션 DGP에서 유저 세그먼트별 채널 노출 확률(confounding 강도)을 조절하며, confounding이 강할수록 두 접근의 결과 차이가 커지는지를 확인한다. 이 실험은 "언제 causal 보정이 필요한가?"라는 실무적 판단 기준을 제공한다.

**실험 6 — Incremental vs Total Attribution:**
Du et al.의 Incremental Shapley와 전통적 Shapley의 결과를 비교한다. DGP의 base conversion rate(광고 없는 자연 전환율)를 0%, 5%, 10%, 20%로 변화시키며, base conversion이 높을수록 Incremental Shapley와 전통 Shapley의 괴리가 커지는지를 확인한다. 이는 "자연 전환이 많은 비즈니스에서 전통 MTA가 얼마나 오도할 수 있는가?"를 보여준다.

### 2.6 기대 산출물

- 시뮬레이션 DGP 코드 (재현 가능, 파라미터 조절 가능, Du et al./Shender et al./CDA 통합)
- 10+ 방법론의 기여도 비교 테이블 및 시각화 (Rule-based 5 + Markov + Shapley + DL 2~3 + Causal 4)
- Ground truth 대비 정확도 랭킹 (correlational vs causal 방법론 분리 비교)
- Correlational vs Causal attribution 괴리 분석 (confounding 강도별)
- Incremental vs Total attribution 비교 (base conversion rate별)
- 방법론 선택 의사결정 프레임워크 (조건 → 추천 방법론, 인과성 수준 포함)

---

## 3. Part 2 — Criteo Attribution 기반 스케일 검증 및 딥러닝 Attribution

### 3.1 목적

Part 1의 시뮬레이션에서 검증한 방법론을, 대규모 실 데이터(Criteo Attribution Dataset, 1,600만 이벤트, 260만 여정)에서 적용하여 스케일러빌리티를 확인한다. 특히, 시뮬레이션에서는 충분한 데이터가 보장되었으나 실 데이터에서는 클래스 불균형, 노이즈, 비정형 패턴 등이 존재하므로, 방법론의 실 환경 강건성(robustness)을 테스트한다.

### 3.2 데이터셋: Criteo Attribution Dataset (2018)

| 항목 | 상세 |
|------|------|
| 규모 | 약 16.5M impression/click 이벤트, ~2.6M 고유 여정 |
| 유저 식별 | uid (전체 기간 일관) |
| 터치포인트 Feature | campaign(해시), cat1~cat9(해시, 의미 불명), click(0/1), cost |
| 전환 | conversion(0/1), conversion_timestamp |
| 시간 | timestamp (변환됨, 순서 보존) |
| 핵심 강점 | View-through vs Click-through 구분, cost 포함, 대규모 |
| 핵심 한계 | 모든 feature 해시/익명화, 도메인 해석 불가 |

### 3.3 해석 가능성 한계에 대한 입장

Criteo 데이터의 cat1~cat9은 해시되어 "이 채널이 Display인지 Search인지"를 알 수 없다. 본 프로젝트에서는 이 한계를 명시적으로 인정하며, Criteo 데이터의 역할을 다음과 같이 한정한다.

**Criteo 데이터에서 하는 것:** 방법론이 대규모 실 데이터에서 기술적으로 작동하는지 검증. 딥러닝 시퀀스 모델의 학습 및 attention/SHAP 기반 기여도 추출. 모델 간 기여도 순위의 일관성(agreement) 분석. View-through vs Click-through의 기여도 차이 분석. Cost 기반 ROI attribution.

**Criteo 데이터에서 하지 않는 것:** "채널 X의 마케팅 예산을 늘려야 한다"는 실무적 해석. cat1~cat9의 도메인 의미 추측(클러스터링 기반 사후 해석은 부록에서만 탐색적으로 다루며, 본론의 결론에는 포함하지 않음).

### 3.4 분석 파이프라인

#### Phase 1: 데이터 전처리 및 여정 재구성

uid로 그룹핑하고 timestamp로 정렬하여 유저별 터치포인트 시퀀스를 구성한다. Attribution window(7일, 14일, 30일)를 적용하여 전환 이전 터치포인트만 포함한다. 전환/비전환 여정을 분리하고, 기초 통계를 산출한다(여정 길이 분포, 전환율, 채널별 빈도 등).

#### Phase 2: Baseline 모델

Part 1과 동일한 Rule-based 5종, Markov Chain, Shapley Value를 적용한다. campaign ID를 채널 proxy로 사용하되, 고유 campaign 수가 많으면 상위 N개 또는 cat1 기반 그룹핑을 적용한다. Part 1 시뮬레이션 결과와의 패턴 유사성/차이를 비교한다.

#### Phase 3: 딥러닝 시퀀스 모델

**3-1. LSTM + Attention:**
입력: [campaign_embedding, cat1~cat9_embedding, click, cost, time_delta]. Embedding dimension: campaign과 각 cat에 대해 8~16차원 학습. LSTM(hidden=128) → Multi-head Attention(2 heads) → Dense → Sigmoid. Loss: Binary Cross-Entropy (class weight로 불균형 보정). 평가: AUC-ROC, AUC-PR, Calibration.

**3-2. Transformer:**
Encoder-only Transformer (2 layer, 4 heads, d_model=64). Temporal Encoding: 절대적 positional encoding 대신, 터치포인트 간 시간 간격을 인코딩. CLS token → classification head → Sigmoid.

**3-3. 기여도 추출 비교:**
동일한 전환 여정에 대해 Attention weight, SHAP(DeepExplainer), Leave-One-Out 세 가지 방법으로 터치포인트별 기여도를 추출한다. 세 방법의 순위 일치도(Kendall's Tau)를 계산한다. Part 1의 시뮬레이션에서의 결과와 비교하여, "실 데이터에서도 세 방법의 일치도가 유사한가?"를 확인한다.

#### Phase 4: View-Through vs Click-Through 분석

Criteo 데이터의 고유 강점인 click(0/1) 필드를 활용한다. View-through(노출만, click=0) 터치포인트와 Click-through(click=1) 터치포인트의 기여도를 별도로 분석한다. "노출만으로도 전환에 기여하는가?"라는 실무적 질문에 데이터 기반으로 답한다.

#### Phase 5: Cost-Efficiency Attribution

터치포인트별 cost 정보를 활용하여, 채널(campaign 그룹)별 Cost Per Attributed Conversion을 계산한다. attribution 모델에 따라 ROAS(Return on Ad Spend) 추정치가 어떻게 달라지는지를 보여준다.

### 3.5 핵심 실험

**실험 5 — 시뮬레이션 vs 실 데이터 방법론 일관성:**
Part 1(시뮬레이션)과 Part 2(Criteo)에서 공통 적용한 방법론(Markov, Shapley)의 행동 패턴을 비교한다. 예: "Markov가 Shapley보다 특정 채널을 과대평가하는 패턴이 시뮬레이션과 실 데이터에서 모두 관찰되는가?"

**실험 6 — 시퀀스 길이별 딥러닝 성능:**
여정 길이를 구간별(2-3, 4-5, 6-10, 10+)로 나누어 LSTM과 Transformer의 예측 성능(AUC)을 비교한다. "Transformer가 LSTM을 능가하는 시퀀스 길이의 임계점"을 식별한다.

**실험 7 — Attention ≠ Attribution 검증:**
Attention weight와 SHAP value의 상관관계를 정량화한다. 불일치가 큰 여정을 case study로 분석하여, "attention이 기여도를 반영하지 않는 경우"의 패턴을 식별한다.

### 3.6 기대 산출물

- 대규모 실 데이터에서의 방법론 기술적 작동 확인
- LSTM vs Transformer 성능 비교 (시퀀스 길이별)
- Attention vs SHAP vs Leave-One-Out 기여도 비교 분석
- View-through attribution 분석 결과
- Cost-efficiency attribution 결과

---

## 4. Part 3 — Bayesian Media Mix Modeling

### 4.1 목적

Part 1~2의 유저 레벨 MTA와 대비되는, 집계 수준(aggregate-level)의 채널 효과 추정을 수행한다. Media Mix Modeling(MMM)은 개인 식별 정보 없이 작동하므로 프라이버시 규제 환경에서 특히 중요하며, 채널별 지출 → 매출의 인과적 관계를 추정하고 최적 예산 배분을 계산한다.

### 4.2 데이터셋

**Primary: Meta Robyn dt_simulated_weekly**

| Feature | 설명 |
|---------|------|
| DATE | 주간 날짜 (208주, 4년) |
| revenue | 주간 매출 (종속변수) |
| tv_S, ooh_S, print_S | 전통 미디어 채널 지출 |
| search_S, facebook_S | 디지털 채널 지출 |
| newsletter | CRM 채널 |
| competitor_sales_B | 경쟁사 매출 (외부 통제 변수) |
| events | 프로모션/이벤트 |

모든 feature가 명시적이고 해석 가능하다. TV, OOH, Print, Search, Facebook, Newsletter의 6개 미디어 채널과 경쟁사 매출, 이벤트 등 통제 변수가 포함되어 있어 현실적 MMM 시나리오를 구성할 수 있다.

**Secondary: PyMC-Marketing delayed_saturated_mmm 시뮬레이터**

Adstock rate와 Saturation curve의 파라미터를 직접 지정하여 데이터를 생성할 수 있다. Ground truth가 알려져 있으므로 모델 검증(parameter recovery)에 활용한다.

### 4.3 분석 파이프라인

#### Phase 1: EDA 및 시계열 특성 분석

채널별 지출 시계열의 추세, 계절성, 자기상관 분석. 채널 간 지출 상관관계 확인 (다중공선성 진단). 매출과 각 채널 지출의 시차 상관(cross-correlation) 분석을 통한 Adstock 초기값 추정.

#### Phase 2: Bayesian MMM 구축 (PyMC-Marketing)

**모델 구조:**

```
revenue_t = intercept 
            + Σ_c β_c × Saturation(Adstock(media_c,t)) 
            + Σ_k γ_k × control_k,t 
            + ε_t
```

**Adstock Transformation:** Geometric Adstock과 Weibull Adstock 두 가지를 비교한다. Geometric: Adstock_t = Media_t + λ × Adstock_{t-1}, λ의 사전분포 Beta(3, 3). Weibull: shape와 scale 파라미터에 대한 사전분포 설정, 지연 효과(delayed peak) 모델링.

**Saturation Transformation:** Hill function: f(x) = x^α / (x^α + K^α). K(반포화점)와 α(곡선 기울기)의 사전분포 설정.

**Prior 설정 전략:** 각 채널의 β에 대한 사전분포를 비정보적(weakly informative)으로 시작하고, Prior Predictive Check를 통해 사전분포가 비현실적인 매출 범위를 생성하지 않는지 확인한다.

#### Phase 3: Bayesian Workflow

**3-1. Prior Predictive Check:**
사전분포에서 샘플링하여 예측 매출 분포를 생성한다. 이 분포가 합리적인 범위(음수 매출 없음, 관측값의 ±2배 이내 등)인지 확인한다.

**3-2. MCMC 샘플링:**
NUTS(No U-Turn Sampler)로 사후분포를 추정한다. 수렴 진단: Rhat(< 1.01), ESS(> 400), trace plot, rank plot. 4 chains × 2,000 samples(1,000 warmup + 1,000 draw).

**3-3. Posterior Predictive Check:**
추정된 모델이 실제 매출 시계열을 얼마나 잘 재현하는지 확인한다. 잔차 분석 (자기상관 잔존 여부).

**3-4. 모델 비교:**
Geometric Adstock vs Weibull Adstock 모델을 LOO-CV(Leave-One-Out Cross-Validation) 또는 WAIC(Widely Applicable Information Criterion)으로 비교한다.

#### Phase 4: 채널 효과 해석

**채널별 기여도 분해 (Waterfall Chart):**
총 매출을 Base(마케팅 없는 기본 매출) + 각 채널 기여 + 통제 변수 효과로 분해한다.

**Adstock 파라미터 해석:**
각 채널의 추정된 반감기(half-life)를 비교한다. 예: TV의 반감기가 3주, Search의 반감기가 0.5주라면, TV는 장기 효과, Search는 즉시 효과라는 해석.

**Saturation 파라미터 해석:**
각 채널의 반포화점(K)과 현재 지출 수준을 비교한다. 현재 지출 > K이면 해당 채널은 이미 수확 체감 구간에 있다.

**ROAS 추정:**
채널별 posterior β에서 ROAS의 사후분포를 도출한다. 점추정이 아닌 불확실성 구간(credible interval)을 포함한 ROAS 보고.

#### Phase 5: Budget Optimization

현재 예산 배분 대비, 동일 총 예산에서 매출을 최대화하는 최적 배분을 계산한다. 추정된 response curve(Adstock + Saturation)를 기반으로 Lagrange multiplier 또는 numerical optimization을 적용한다. 최적 배분과 현재 배분의 차이를 시각화하여, "어떤 채널에서 빼서 어디로 옮겨야 하는가?"를 제시한다.

#### Phase 6: Prior Sensitivity Analysis

채널 β의 사전분포를 변경(더 tight / 더 loose)하며 결과 변화를 관찰한다. "이 결론은 사전분포 선택에 얼마나 민감한가?"를 정량화한다. 현업에서 incrementality 실험 결과를 prior로 반영하는 "calibrated MMM"의 시뮬레이션.

### 4.4 핵심 실험

**실험 8 — Parameter Recovery (PyMC-Marketing 시뮬레이터):**
알려진 ground truth 파라미터(Adstock rate, Saturation K, channel β)로 데이터를 생성하고, Bayesian MMM이 이를 얼마나 정확하게 복원하는지 확인한다. 데이터 기간(52주, 104주, 208주)에 따른 복원 정확도 변화를 관찰한다.

**실험 9 — Adstock 모델 비교:**
Geometric vs Weibull Adstock의 LOO-CV 성능을 비교한다. 각 채널에 대해 "이 채널은 Geometric이 적합한가, Weibull이 적합한가?"를 데이터 기반으로 판단한다.

**실험 10 — Budget Optimization 시나리오:**
현재 예산의 ±20% 변동 시나리오에서의 최적 배분 변화를 분석한다. "예산이 줄어도 유지해야 하는 채널"과 "예산이 늘면 가장 먼저 증액할 채널"을 식별한다.

### 4.5 기대 산출물

- End-to-end Bayesian MMM 파이프라인
- 채널별 기여도 분해 (Waterfall chart)
- Adstock/Saturation 파라미터 해석 보고서
- ROAS 추정 (불확실성 구간 포함)
- Budget Optimization 결과 및 시나리오 분석
- Prior Sensitivity Analysis 결과

---

## 5. 통합 분석 — Triangulation

### 5.1 목적

Part 1~3의 결과를 하나의 프레임워크에서 종합하여, MTA와 MMM의 관계, 방법론 간 trade-off, 실무 의사결정 가이드를 도출한다.

### 5.2 분석 축

**축 1 — MTA vs MMM 결과 비교:**
Part 1의 시뮬레이션 데이터에서 MTA(Shapley)가 추정한 채널별 기여도와, 동일 데이터를 주간 집계하여 MMM으로 추정한 채널 효과를 비교한다. 이를 위해 시뮬레이션 DGP에서 주간 집계 매출과 채널별 주간 터치포인트 수(또는 비용)를 산출하여 MMM 입력 데이터로 변환한다.

두 접근의 결과가 일치하면 강한 근거가 되며, 불일치하면 그 원인(집계 수준의 정보 손실, Simpson's Paradox 등)을 분석한다.

**축 2 — 방법론 복잡도 vs 정확도 Trade-off:**
Part 1의 ground truth 기반 결과에서, 방법론 복잡도(Rule-based < Markov < Shapley < LSTM < Transformer < Causal DL)와 ground truth 대비 MAE의 관계를 정리한다. "추가적인 복잡도가 정당화되는 정확도 향상이 있는가?"를 정량화한다.

**축 3 — Correlational vs Causal Attribution:**
실험 5(Correlational vs Causal 비교)의 결과를 기반으로, confounding 강도에 따른 두 접근의 결과 괴리를 정리한다. "자사 데이터에서 selection bias가 어느 정도인가?"에 따라 causal 보정의 필요성을 판단하는 실무 기준을 제시한다. 예를 들어, 유저 세그먼트별 채널 노출 확률이 크게 다르지 않은 환경(약한 confounding)에서는 Shapley만으로 충분하지만, CRM 기반 타겟팅이 강한 환경(강한 confounding)에서는 IPW 또는 DML 보정이 필수적일 수 있다.

**축 4 — 데이터 요건 매트릭스:**
각 방법론이 요구하는 최소 데이터 조건을 Part 1의 Learning Curve 실험 결과를 기반으로 정리한다.

| 방법론 | 최소 전환 수 | 최소 채널 수 | 시퀀스 길이 요건 | 시간 정보 필요 | 유저 feature 필요 | 인과성 수준 |
|--------|------------|------------|----------------|-------------|-----------------|-----------|
| Last Click | 100+ | 2+ | 1+ | 아니오 | 아니오 | 없음 |
| Markov | 1,000+ | 3+ | 2+ | 아니오 | 아니오 | 약함 |
| Shapley | 500+ | ≤15 (정확), 15+ (근사) | 1+ | 아니오 | 아니오 | 약함 |
| LSTM/Transformer | 5,000~10,000+ | 3+ | 3~5+ | 권장 | 아니오 | 약함 |
| Incremental Shapley | 500+ | ≤15 | 1+ | 아니오 | 아니오 | 중간 |
| Survival/Poisson MTA | 1,000+ | 2+ | 2+ | **필수** | 권장 | 중간 |
| IPW / Doubly Robust | 2,000+ | 2+ | 1+ | 아니오 | **필수** | 높음 |
| DML | 5,000+ | 2+ | 1+ | 아니오 | **필수** | 높음 |
| CAMTA (Causal DL) | 10,000+ | 3+ | 3+ | 권장 | 권장 | 중간~높음 |
| Bayesian MMM | N/A (주간 52+) | 3+ | N/A | 필수 (시계열) | N/A | 중간 |

(위 수치는 Part 1 실험 결과에 따라 업데이트한다.)

**축 5 — 실무 의사결정 Flow Chart:**
실무자가 자신의 데이터 조건에서 어떤 방법론을 선택해야 하는지를 안내하는 의사결정 트리를 제시한다. 핵심 분기점은 다음과 같다: (1) 유저 레벨 데이터가 있는가? → 없으면 MMM. (2) RCT/실험 데이터인가, 관찰 데이터인가? (3) Selection bias가 강한가? → 강하면 Causal 방법론 필수. (4) 데이터 규모가 충분한가? → 적으면 Markov/Shapley, 충분하면 DL 가능. (5) 시간 정보가 정밀한가? → 정밀하면 Survival/Poisson, 아니면 Shapley.

### 5.3 기대 산출물

- MTA vs MMM 비교 분석 보고서
- 방법론 선택 의사결정 프레임워크 (flow chart + 조건표)
- 전체 프로젝트 요약 테이블

---

## 6. 기술 스택

| 영역 | 도구 |
|------|------|
| 데이터 처리 | Python (Pandas, NumPy, Polars) |
| 시각화 | Matplotlib, Seaborn, Plotly |
| 통계 모델 | Statsmodels, SciPy |
| Markov Chain | pychattr 또는 직접 구현 |
| Shapley Value | 직접 구현 (정확 계산) + shap 라이브러리 |
| 딥러닝 | PyTorch |
| Bayesian MMM | PyMC-Marketing (PyMC 기반) |
| 인과추론 | DoWhy, EconML (DML, Doubly Robust), causalml (IPW) |
| 실험 관리 | MLflow 또는 Weights & Biases |
| 버전 관리 | Git / GitHub |
| 문서화 | Jupyter Notebook + Markdown |

---

## 7. 일정 계획

| 주차 | 마일스톤 | 상세 |
|------|---------|------|
| W1-2 | Part 1 — DGP 설계 및 데이터 생성 | 채널 구조 설계, 전환 확률 모델 구현, 데이터 검증 |
| W3-4 | Part 1 — Rule-based + Markov + Shapley | 7개 전통/통계 방법론 적용, ground truth 비교 |
| W5-6 | Part 1 — LSTM + Transformer | 딥러닝 모델 구축, 기여도 추출, 세 가지 해석 방법 비교 |
| W7-8 | Part 1 — Causal MTA 방법론 | Incremental Shapley, Survival/Poisson, IPW/DR, DML, CAMTA 변형 구현. 실험 5~6 (Correlational vs Causal, Incremental vs Total) |
| W9 | Part 1 — 실험 분석 및 정리 | 실험 1~6 정리, 시각화, 중간 보고서 |
| W10-11 | Part 2 — Criteo 데이터 전처리 + Baseline | 여정 재구성, Rule-based + Markov + Shapley 적용 |
| W12-13 | Part 2 — 딥러닝 + 기여도 비교 | LSTM/Transformer 학습, Attention vs SHAP 비교 |
| W14 | Part 2 — View-through + Cost 분석 | 실험 정리 |
| W15-16 | Part 3 — Bayesian MMM | EDA, 모델 구축, Prior/Posterior 검증 |
| W17 | Part 3 — Budget Optimization + Sensitivity | 최적화, 시나리오 분석 |
| W18 | 통합 분석 | MTA vs MMM 비교, Correlational vs Causal 통합, 방법론 선택 프레임워크 정리 |
| W19-20 | 문서화 및 포트폴리오 정리 | README, 블로그 포스트, 코드 정리 |

총 약 20주 (5개월)이며, 파트타임(주 10~15시간) 기준이다. 풀타임 집중 시 10~12주로 단축 가능하다.

---

## 8. 리스크 관리

| 리스크 | 영향 | 확률 | 완화 전략 |
|--------|------|------|----------|
| 시뮬레이션 DGP가 비현실적 | Part 1 결과의 외적 타당성 저하 | 중 | 현업 데이터 특성(전환율, 여정 길이 등)을 문헌/경험에서 참고하여 DGP 교정 |
| Criteo 데이터의 전환 여정 수 부족 (딥러닝) | LSTM/Transformer 과적합 | 낮 | 2.6M 여정은 충분. 다만 k-fold CV, 강한 정규화 적용 |
| Transformer가 LSTM 대비 유의미한 향상 없음 | Part 2 딥러닝 파트의 novelty 저하 | 높 | 이 결과 자체가 의미 있는 finding ("짧은 시퀀스에서는 Transformer 불필요")으로 프레이밍 |
| Bayesian MMM 수렴 실패 | Part 3 결과 불가 | 낮 | Prior 조정, 모델 단순화, 참고 문헌의 사전분포 활용 |
| 전체 프로젝트 범위 과다 | 완성 지연 | 중 | Part 1을 MVP로 확보한 뒤 Part 2, 3을 순차 추가. Part 1만으로도 독립 포트폴리오 가치 있음 |

---

## 9. 포트폴리오 프레젠테이션 전략

### 9.1 GitHub Repository 구조

```
marketing-attribution/
├── README.md                    # 프로젝트 요약 및 핵심 결과
├── part1_simulation/
│   ├── dgp/                     # 데이터 생성 프로세스
│   ├── models/                  # MTA 방법론 구현
│   ├── experiments/             # 실험 노트북
│   └── results/                 # 결과 시각화
├── part2_criteo/
│   ├── preprocessing/           # 여정 재구성
│   ├── models/                  # 딥러닝 모델
│   ├── experiments/             # 실험 노트북
│   └── results/
├── part3_mmm/
│   ├── eda/                     # 탐색적 분석
│   ├── models/                  # Bayesian MMM
│   ├── optimization/            # Budget optimization
│   └── results/
├── integration/                 # 통합 분석
│   ├── triangulation.ipynb
│   └── decision_framework.md
└── docs/
    ├── project_plan.md          # 본 문서
    └── methodology_guide.md     # 방법론 상세 설명
```

### 9.2 면접 대비 핵심 메시지

**메시지 1 — 데이터 진단 역량:** "공개 데이터셋의 한계를 정확히 진단하고(GA4의 세션별 채널 부재, CriteoPrivateAd의 일 단위 uuid 리셋 등), 그 한계에 맞는 분석 전략을 설계했습니다."

**메시지 2 — Ground Truth 기반 평가:** "시뮬레이션으로 ground truth를 설정하고, 각 방법론의 정확도를 정량적으로 비교했습니다. 실 데이터에서는 불가능한 이 평가가, 방법론 선택의 근거를 제공합니다."

**메시지 3 — 복잡도 ≠ 좋음:** "Transformer가 항상 최선이 아닙니다. 시퀀스 길이가 짧은 환경에서는 Markov Chain이면 충분하며, 이를 실험적으로 보여주었습니다."

**메시지 4 — Triangulation:** "MTA와 MMM은 동일 문제를 다른 렌즈로 보는 것입니다. 두 접근의 결과를 교차 검증하는 것이 실무에서의 표준 프로세스이며, 이를 직접 구현했습니다."

**메시지 5 — Bayesian Thinking:** "점추정이 아닌 불확실성 구간을 포함한 의사결정을 수행했습니다. Prior Sensitivity Analysis를 통해 결론의 강건성을 검증했습니다."

**메시지 6 — Correlation ≠ Causation의 실증:** "전통적 MTA는 '전환을 잘 예측하는 채널'을 찾지만, 이는 '광고로 인해 전환이 발생한 채널'과 다를 수 있습니다. 시뮬레이션에서 confounding 강도를 조절하며 두 접근의 결과 괴리를 정량적으로 보여주었고, selection bias가 강한 환경에서 causal 보정이 필수적임을 실험적으로 입증했습니다."

---

## 10. 확장 가능성

본 프로젝트 완료 후, 다음 방향으로 확장이 가능하다.

**확장 1 — Causal Incrementality (Geo-Lift / CausalImpact):** Part 3의 MMM 결과를 Meta GeoLift 또는 Google CausalImpact로 교차 검증하는 지역 실험 시뮬레이션.

**확장 2 — Privacy-Preserving Attribution:** CriteoPrivateAd(2025) 데이터를 활용한 DP 환경에서의 CTR/CVR 예측 성능 저하 분석. MTA와 직접 관련은 없으나, post-cookie 시대의 attribution 트렌드에 해당.

**확장 3 — Heterogeneous Treatment Effect (HTE):** Hillstrom 또는 Taobao 데이터에서 유저 세그먼트별 마케팅 효과 이질성 분석. Causal Forest, X-Learner 등 적용.

**확장 4 — LLM 기반 Attribution 해석:** 학습된 딥러닝 모델의 기여도 결과를 LLM으로 자연어 해석하여 비기술 이해관계자에게 전달하는 파이프라인 구축.

---

*본 계획서는 프로젝트 진행 과정에서 실험 결과에 따라 업데이트될 수 있다.*

---

## 부록 A. 핵심 참고 문헌

### A.1 시뮬레이션 DGP 레퍼런스 (Part 1 기반)

| # | 논문 | 저자 | 게재 | 핵심 기여 | 코드 |
|---|------|------|------|----------|------|
| **[R1]** | Causally Driven Incremental Multi Touch Attribution Using a Recurrent Neural Network | Du, Zhong, Nair, Cui, Shou | ArXiv 1902.00215 / AdKDD 2019 | MTA 시뮬레이션 DGP (sequential dependence, user heterogeneity, competition) + RNN response model + Shapley credit allocation | `github.com/jd-ads-data/jd-mta` |
| **[R2]** | A Time To Event Framework For Multi-touch Attribution | Shender, Nasiri Amini, Bao, Dikmen, Richardson, Wang | Journal of Data Science, 22, pp.56-76 (2023) | Inhomogeneous Poisson process 기반 전환 모델 + 채널별 시간 감쇠 함수 + 500회 반복 시뮬레이션 검증 | 비공개 |
| **[R3]** | Causal-driven attribution (CDA): Estimating channel influence without user-level data | (2025 저자) | ArXiv 2512.21211 (Dec 2025) | 채널 간 cross-influence 시뮬레이션 + 집계 수준 인과 attribution (PCMCI + SCM) + 공개 MTA 데이터 부재 문제 명시 | 비공개 |

### A.2 MTA 방법론 레퍼런스

| # | 논문 | 핵심 내용 | Part |
|---|------|----------|------|
| [R4] | Shao & Li (2011). Data-Driven Multi-Touch Attribution Models. KDD. | Bagged logistic regression 기반 MTA, bivariate metric 제안 | Part 1, 2 |
| [R5] | Anderl et al. (2016). Mapping the Customer Journey. Journal of Marketing Research. | Markov Chain 기반 MTA의 마케팅 저널 논문 | Part 1 |
| [R6] | Ren et al. (2018). Learning Multi-touch Conversion Attribution with Dual-attention Mechanisms. CIKM. | DARNN: Dual-Attention RNN, Criteo Attribution 데이터 벤치마크 | Part 2 |
| [R7] | Singal et al. (2019). Shapley Meets Uniform: An Axiomatic Framework for Attribution. | Shapley Value의 공리적 확장을 MTA에 적용 | Part 1 |
| [R8] | Yang et al. (2020). Interpretable Deep Learning Model for Online Multi-touch Attribution. | DeepMTA: Phased-LSTM + Shapley 결합 해석 가능 모델 | Part 2 |
| [R9] | Shender et al. (2020). A Time To Event Framework For Multi-touch Attribution. | Survival analysis 기반 MTA (Google Research) | Part 1 |

### A.3 Media Mix Modeling 레퍼런스

| # | 논문 / 프레임워크 | 핵심 내용 | Part |
|---|------------------|----------|------|
| [R10] | Jin et al. (2017). Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects. Google Research. | Bayesian MMM 기초, Adstock/Saturation | Part 3 |
| [R11] | Meta (2022). Robyn: Continuous & Semi-Automated MMM. | Ridge + Nevergrad, Pareto-optimal 모델 선택 | Part 3 |
| [R12] | Google (2024). Meridian: An Open-Source MMM Framework. | Bayesian Hierarchical, Geo-level MMM | Part 3 |
| [R13] | PyMC-Marketing Documentation. pymc-marketing.readthedocs.io | Python Bayesian MMM 구현 | Part 3 |

### A.4 Causal Inference 기반 MTA 레퍼런스

| # | 논문 | 핵심 내용 | Part |
|---|------|----------|------|
| [R14] | Dalessandro et al. (2012). Causally Motivated Attribution for Online Advertising. ADKDD. | 인과 추론 관점에서 MTA를 최초로 재정의, Shapley 기반 인과적 credit allocation | Part 1 |
| [R15] | Kumar et al. (2020). CAMTA: Causal Attention Model for Multi-Touch Attribution. ICDM Workshop. | Causal RNN + Attention으로 time-varying confounders 통제, 인과적 기여도 학습 | Part 1, 2 |
| [R16] | Yao et al. (2022). CausalMTA. Alibaba. | 정적/동적 confounding bias를 counterfactual prediction으로 제거하는 MTA 모델 | Part 1 |
| [R17] | Tang et al. (2024). DCRMTA: Deep Causal Representation for MTA. | 인과적 feature와 confounding variable을 분리하는 end-to-end 접근 | Part 1 |

### A.5 Causal Inference 일반 레퍼런스

| # | 논문 | 핵심 내용 | Part |
|---|------|----------|------|
| [R18] | Chernozhukov et al. (2018). Double/Debiased Machine Learning. Econometrics Journal. | DML 프레임워크, cross-fitting을 통한 과적합 방지 | Part 1 |
| [R19] | Wager & Athey (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. JASA. | Causal Forest | 확장 |
| [R20] | Gordon et al. (2019). A Comparison of Approaches to Advertising Measurement. Marketing Science. | RCT vs 관찰적 방법론 비교 (Facebook 대규모 실험) | 통합 |
| [R21] | Lewis & Rao (2015). The Unfavorable Economics of Measuring the Returns to Advertising. QJE. | 광고 효과 측정의 통계적 파워 문제 | 통합 |

### A.6 Attention 해석 논쟁 레퍼런스

| # | 논문 | 핵심 주장 | Part |
|---|------|----------|------|
| [R22] | Jain & Wallace (2019). Attention is not Explanation. NAACL. | Attention weight가 feature importance와 상관 낮을 수 있음 | Part 2 |
| [R23] | Wiegreffe & Pinter (2019). Attention is not not Explanation. EMNLP. | 적절한 조건에서 attention이 설명력을 가질 수 있음 | Part 2 |

### A.7 데이터셋 레퍼런스

| # | 데이터셋 | 출처 | 본 프로젝트에서의 역할 |
|---|---------|------|---------------------|
| [D1] | Criteo Attribution Modeling for Bidding Dataset (2018) | criteo.com / Diemert et al., AdKDD 2017 | Part 2: 대규모 실 데이터 스케일 검증 |
| [D2] | CriteoPrivateAd (2025) | huggingface.co/datasets/criteo/CriteoPrivateAd / Sebbar et al., ArXiv 2502.12103 | 참고 전용 (MTA 부적합 — uuid 일 단위 리셋) |
| [D3] | GA4 BigQuery Public Dataset | bigquery-public-data.ga4_obfuscated_sample_ecommerce | 참고 전용 (MTA 부적합 — 세션별 채널 시퀀스 재구성 불가) |
| [D4] | Meta Robyn dt_simulated_weekly | github.com/facebookexperimental/Robyn | Part 3: Bayesian MMM |
| [D5] | PyMC-Marketing simulated data | github.com/pymc-labs/pymc-marketing | Part 3: Parameter recovery 실험 |
