# Part 1 Notebooks — 읽는 순서와 구조

서사: **Setup → Main → Benchmark → Validation**. 본 프로젝트의 중심 방법론은
**Poisson Survival backbone + Incremental/Shapley channel credit + multi-path
(path-level Incremental Shapley) + Conditional vs Marginal G-computation**이며,
나머지 모든 계열은 그 **성능 benchmark(baseline)** 이다.

## 읽는 순서

| # | Notebook | 블록 | 역할 |
|---|----------|------|------|
| 01 | `01_dgp_design_and_eda.ipynb` | Setup | DGP 설계 + 생성 데이터 EDA |
| 02 | `02_main_survival_incremental_shapley.ipynb` | **Main** | **중심 방법론**: Survival/Poisson, Incremental/Shapley credit, multi-path duality, Conditional vs Marginal G-comp, decay 해석, bootstrap/OOS |
| 03 | `03_benchmark_traditional.ipynb` | Benchmark | rule-based / Markov / Total Shapley (baseline) |
| 04 | `04_benchmark_deep_learning.ipynb` | Benchmark | LSTM+Attention / Transformer (baseline) |
| 05 | `05_benchmark_causal_baselines.ipynb` | Benchmark | IPW / DR / DML / CAMTA (baseline) |
| 06 | `06_benchmark_comparison.ipynb` | Benchmark | Exp 01–06 sweep = Main vs 전 baseline + 방법론 선택 framework |
| 07 | `07_cost_and_budget_optimization.ipynb` | Validation/Applied | Exp 07 비용·예산 최적화 |
| 08 | `08_realworld_validation.ipynb` | Validation | Exp 08–11 GT-free 검증 + Robust/Fragile 종합 |

→ Part 2 (Criteo 대규모 실데이터)

## 실험 ID → Notebook 매핑

실험 ID(01–11)는 `results/part1/NN_*.csv` 파일명 및 docs 수치와 강결합이므로 **불변**.
Notebook 파일 번호와는 별개 식별자다.

| Exp ID | 실험 | Notebook | 분류 |
|--------|------|----------|------|
| 01 | Method Accuracy | 06 | Benchmark sweep |
| 02 | Interaction Effects | 06 | Benchmark sweep |
| 03 | Data Scale | 06 | Benchmark sweep |
| 04 | DGP Sensitivity | 06 | Benchmark sweep |
| 05 | Correlational vs Causal | 06 | Benchmark sweep |
| 06 | Incremental vs Total | 06 | Benchmark sweep |
| 07 | Budget Optimization | 07 | Applied |
| 08 | OOS Predictive Validation | 08 | Validation |
| 09 | Decision Impact (Revenue Lift) | 08 | Validation |
| 10 | Bootstrap Stability | 08 | Validation |
| 11 | Convergent Validity | 08 | Validation |

Main 방법론의 *특성* 분석(decay 해석, BackElim↔Shapley 일관성, Conditional vs
Marginal, multi-path duality)은 02 에 있으며 Exp ID 가 없는 분석 섹션이다.

## 단일 소유(canonical) 주제

중복을 피하기 위해 주제별 canonical owner 를 둔다. 다른 노트북은 요약 후 링크만 한다.

| 주제 | Canonical |
|------|-----------|
| Main 방법론 (Survival/IncShap/multi-path/Cond-Marg) | **02** |
| Confounding / corr≠causation 개념 | **05 §1** |
| 예산 / ROI 메커니즘 | **07** |
| 방법론 선택 framework | **06 §7** |
| Conditional vs Marginal 설득 산문·응급실 비유 | `docs/Marketing_Handout_Conditional_vs_Marginal.md` |

## 관련 문서

- 학술 풀버전: `docs/Methodology_05_Causal_Attribution_Frameworks.md`
- 실무진 요약: `docs/Methodology_05b_Practitioner_Summary.md`
- 실험 설계/결과: `docs/Methodology_03_Experimental_Design.md` (Exp 01–06 본문, Exp 08–11 §9)
- 결과 CSV: `results/part1/NN_*.csv` (실험 재실행 없이 노트북이 로드)

## 재현 / behavior 주의

노트북 셀 출력은 사전 실행 결과(frozen)다. 본 리팩토링은 셀 이동·분할·재배치만
수행했고 수치 결과는 비트 단위로 보존된다. 재실행이 필요하면 conda env
`mta-simulation` 사용 (`python`, not system `python3`). 경로는
`from part1_simulation.notebook_setup import DATA_DIR, RESULTS_DIR, load_exp_csv`
상수를 권장.
