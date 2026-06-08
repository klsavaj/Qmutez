# 05 - Multi-Objective Matching Model

## The Problem
A traditional rideshare algorithm focuses on one objective: getting the rider a car as fast as possible.
The Qmutez P2P model must balance conflicting objectives:
1.  Minimize the detour for the commuter-driver.
2.  Maximize the probability that the match is accepted by both parties.

## The Model Pipeline (`src/models/xgb_matching.py`)

### 1. The Baseline Justification
We implement a `RuleBasedBaseline` strict-threshold system and a simple `LogisticRegressionBaseline`. This scientifically proves that simple distance cutoffs leave money on the table, setting a benchmark for the advanced modeling.

### 2. Regression (Detour Prediction)
An `XGBRegressor` is trained explicitly to forecast the continuous variable: *How far off route will this driver have to travel?*

### 3. Classification (Acceptance Probability)
An `XGBClassifier` models human economic behavior, utilizing the `overlap_ratio` and estimated fare splits to output a probabilistic decimal (e.g., `0.85` likelihood of acceptance).

The classifier trains on **9 features**: `driver_orig_dist`, `overlap_ratio`, `time_diff_mins`, `driver_approx_fare`, `rider_approx_fare`, `fare_ratio`, `pickup_hour`, `pickup_dow`, and `shareability_count`. Note that `detour_distance_km` is intentionally **excluded** from the classifier's inputs — it is the regressor's prediction target, so including it would leak the answer. (The `LogisticRegressionBaseline`, which has no regression objective, does include `detour_distance_km`, giving it 10 features.) To counter class imbalance, the classifier sets `scale_pos_weight` to the train-set negative/positive ratio.

### 4. The Unified Ranking Score
The `compute_match_score()` function combines these non-correlative outputs into a single metric capable of ranking an ordered queue.
*   `Rank Score = Acceptance Probability - (Penalty Weight * Predicted Detour)`

### 5. Bipartite Graph Optimization (Production Inference)
The training pipeline collects ALL feasible pairs with independently assigned stochastic labels, giving the model the complete distribution of matches to learn from. 

At **inference time** (production deployment), once all Candidate Pairs within a batch window receive their ML-predicted `Rank Score`, the system solves the assignment optimally. The candidates are represented as a **Bipartite Graph**:
*   **Nodes:** Drivers ($D$) and Riders ($R$).
*   **Edges ($x_{ij}$):** Feasible matches connecting $D_i$ to $R_j$.
*   **Weights ($w_{ij}$):** The ML-generated `Rank Score` (from the trained XGBoost model).

The system executes a Maximum Weight Bipartite Matching algorithm (e.g., Hungarian Algorithm via `networkx.max_weight_matching`) to find the globally optimal combination that maximizes ride acceptance while minimizing total system detour:
$$ \max \sum_{i} \sum_{j} w_{ij} x_{ij} $$
Subject to the constraints that every Driver and every Rider can only be matched a maximum of once:
*   $ \sum_{j} x_{ij} \le 1 $  (For all $i$)
*   $ \sum_{i} x_{ij} \le 1 $  (For all $j$)

*Note: This separation (training sees all pairs, inference selects optimal assignments) eliminates selection bias and ensures the model learns general acceptance patterns, not just "winning-pair" patterns.*

### 6. Validation & Evaluation
To make the reported numbers trustworthy and interpretable, the training routine performs:
*   **Driver-level train/test split:** Unique `driver_id`s are partitioned 80/20 (`random_state=42`) so that no driver appears in both train and test. Splitting on individual pairs would let the same driver leak across the boundary and inflate test scores.
*   **Hyperparameter tuning:** A 5-fold `GridSearchCV` sweeps the classifier over `n_estimators` (50/100/200), `max_depth` (3/5/7), and `learning_rate` (0.05/0.1/0.2); the regressor adopts the winning configuration.
*   **Cross-validation & metrics:** 5-fold cross-validation accompanies the held-out test metrics — MAE and RMSE for the detour regressor; accuracy, F1, and ROC-AUC for the acceptance classifier.
*   **Feature ablation:** Each feature is removed in turn and the classifier retrained, measuring the ROC-AUC drop, to quantify each feature's contribution (`figures/ablation_study.png`).
*   **Learning curves:** Training vs. cross-validation accuracy across increasing training-set sizes diagnose over/under-fitting (`figures/learning_curve.png`).
*   **System-level matching evaluation:** Beyond per-pair model metrics, the trained scores drive a `networkx.max_weight_matching` over the test set to report system outcomes — match rate, average driver detour, average driver earnings, average rider savings, and CO₂ saved — plus a confidence-threshold trade-off curve (`figures/system_tradeoff_curve.png`). Supporting plots `figures/feature_importance.png` and `figures/roc_curve.png` are also produced.
