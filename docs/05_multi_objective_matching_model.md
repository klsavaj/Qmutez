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
