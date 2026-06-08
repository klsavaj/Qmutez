# Qmutez: End-to-End System Architecture

This document provides a comprehensive, start-to-finish architectural overview of the Qmutez Peer-to-Peer (P2P) commuter carpooling matching engine.

---

## Phase 1: Data Ingestion & Central Storage (`src/data/loader.py`)

**The Challenge:** Real P2P carpooling data is proprietary (Uber, Waze). We must train our models on public single-passenger NYC Taxi Data to simulate a P2P network.
**The Constraint:** In July 2016, the NYC Taxi & Limousine Commission (TLC) stopped providing exact Latitude/Longitude GPS coordinates, switching to generic "Taxi Zones." Because our simulation requires exact route math (Haversine distance), we utilize the **August 2025** dataset and synthesize coordinates programmatically.

### 1. External Fetching
*   The system reads the target dataset (Year, Month) from `config/dev_config.yaml` or `config/prod_config.yaml`.
*   The `DataLoader` class connects to the AWS Cloudfront TLC servers and streams the massive `.parquet` file in 8KB chunks (to prevent local memory overflow).
*   The file is saved as immutable raw data in `data/raw/`.

### 2. Geographic Data Cleaning & Flexible Hub Creation
*   The system loads the raw Parquet file into pandas.
*   **Coordinate Cleaning:** Drops invalid coordinates outside the NYC Tri-State bounding box (Lat: `40.5`-`41.2`, Lon: `-74.3`-`-73.7`).
*   **Zonal Bounding Box Distribution + Point-in-Polygon Refinement:** Extracted Zone IDs (`PULocationID`) are mapped to their true geographic shapefile bounds. Instead of clustering trips at artificial "centroids," pickups are uniformly distributed across the full area of their respective zones. A subsequent polygon validation step uses `shapely.contains()` rejection sampling to ensure every coordinate falls within the actual zone polygon (not in water or across boundaries), preserving the true spatial density and ensuring the model correctly evaluates "edge-case" detours near zone boundaries.
*   **Economic Check:** Strips rows with negative fares, zero distances.
*   The clean dataset is saved to `data/processed/`.

---

## Phase 2: Feature Engineering Core (`src/simulation/feature_eng.py`)

This module houses the pure, stateless mathematical logic required to evaluate the viability of two separate commuters sharing a route.

1.  **`calculate_detour_cost()`:** Uses `geopy.geodesic()` to determine exactly how far out of their way the Driver has to go.
    *   *Formula:* `(Driver Origin -> Rider Origin -> Rider Dest -> Driver Dest) - (Driver Origin -> Driver Dest)`
2.  **`compute_spatial_overlap()`:** A heuristic ratio showing what percentage of the new shared trip overlaps with the driver's original intended route. Clamped to [0, 1] to handle negative detours (rider directly on the way).
3.  **`calculate_time_overlap()`:** Ensures the Rider requested the trip within an acceptable margin (e.g., 15 mins) of the Driver's start time.

---

## Phase 3: The P2P Simulation Engine (`src/simulation/pair_generator.py`)

Because we use single-passenger taxi data, we must programmatically force a simulation of a P2P marketplace. This generates the synthetic training dataset required for the machine learning models.

### 1. Temporal Filtering
Filters millions of trips down to the peak commute windows. Both rush hours are covered via the `time_windows` list in config — the **AM** window (07:00 to 10:00) and the **PM** window (16:00 to 19:00) — since commuter carpooling occurs in both directions of the daily commute. In `dev` mode the filtered trips are then downsampled to 25% for fast iteration.

### 2. Role Assignment
Uses algorithmic splits (with a set random seed for reproducibility) to tag 50% of the commuters as "Drivers" aiming to offset their driving costs, and 50% as "Riders" requesting a ride.

### 3. BallTree Spatial Indexing & Pair Generation
Comparing *every* Driver to *every* Rider creates millions of mathematical calculations, crashing memory instantly.
*   **BallTree Spatial Index:** For each temporal batch, a `sklearn.neighbors.BallTree` is constructed from rider pickup coordinates using the Haversine metric on the unit sphere. Each driver queries the tree for neighbors within a 5km geodesic radius in **O(log N)** time — replacing the old O(N) linear scan per driver.
*   For the remaining feasible candidates, the engine runs all functions from `feature_eng.py`.
*   **Shareability Signal:** The number of riders that fall within the driver's search radius in a batch is recorded as `shareability_count` — a neighbor-density measure (Santi et al., PNAS 2014) indicating how "poolable" a driver's corridor is. (Negative samples, drawn from random infeasible pairs, carry a `shareability_count` of 0.)
*   **Multi-Factor Stochastic Labeling:** Labels are derived from a composite of 4 independent behavioral scores (detour severity, fare profitability, temporal inconvenience, trip length ratio) plus Gumbel noise for human irrationality. This replaces simple threshold-based labeling and forces the model to learn realistic acceptance patterns.
*   The final dataset is saved to `data/simulated/candidate_pairs.parquet`. Each row carries driver/rider IDs, the driver's original distance, detour km, overlap ratio, time difference, both fare estimates, plus the engineered interaction features (`fare_ratio`, `pickup_hour`, `pickup_dow`, `shareability_count`) and the binary acceptance label.

---

## Phase 4: Machine Learning & Multi-Objective Matching (`src/models/`)

With the candidate pairs generated, the system creates models to predict the feasibility and efficiency of matches on unseen data.

### 1. The Baselines (`baselines.py`)
To prove the advanced model is necessary, the architecture provides two benchmarks:
*   **RuleBasedBaseline:** Represents a naive application that simply accepts *any* pairing where the detour is under 3km, operating with zero economic intelligence.
*   **LogisticRegressionBaseline:** A classic statistical model attempting to predict the `1` or `0` label using standardized features. It struggles to capture non-linear interactions (e.g., how tolerable a detour is depending on the length of the original trip).

### 2. The Advanced Multi-Objective Matching Model (`xgb_matching.py`)
The ultimate goal of Qmutez is to rank candidate pairings effectively. It uses **XGBoost** (Gradient Boosted Trees), which excels at tabular, non-linear geographic/economic relationships over continuous metrics (deep learning). 

It divides the matching problem into two dual-objectives:
*   **Objective A (Regression):** An `XGBRegressor` predicts exactly how many kilometers the detour will cost the driver.
*   **Objective B (Classification):** An `XGBClassifier` models human economic behavior, predicting the boolean probability (from `0.0` to `1.0`) that the driver will actually accept the detour given the overlapping route and the estimated fare they will earn.

Both sub-models train on the same engineered feature space. The classifier uses **9 features** — `driver_orig_dist`, `overlap_ratio`, `time_diff_mins`, `driver_approx_fare`, `rider_approx_fare`, `fare_ratio`, `pickup_hour`, `pickup_dow`, and `shareability_count`. `detour_distance_km` is deliberately excluded from the classifier's inputs because it is the regressor's target, and feeding it in would leak the answer.

**Training rigor:**
*   **Driver-level train/test split:** Rather than splitting on individual pairs, the unique `driver_id`s are partitioned 80/20 (`random_state=42`) so that no driver appears in both train and test. This prevents entity leakage that would otherwise inflate test scores.
*   **Class balance:** `scale_pos_weight` (= negatives / positives, measured on the train set) is passed to the classifier so the minority "accept" class is not crushed.
*   **Hyperparameter tuning:** A `GridSearchCV` sweep over `n_estimators`, `max_depth`, and `learning_rate` (5-fold) selects the classifier's parameters; the regressor reuses the winning configuration.
*   **Validation & metrics:** 5-fold cross-validation is reported alongside held-out test metrics — MAE and RMSE for the regressor; accuracy, F1, and ROC-AUC for the classifier.
*   **Interpretability & system evaluation:** The pipeline also emits a feature-importance plot, an ROC curve, a per-feature **ablation study** (ROC-AUC drop when each feature is removed), **learning curves** (train vs. CV accuracy across training-set sizes) to diagnose over/under-fitting, and a **system-level bipartite-matching evaluation** (match rate, average driver detour, driver earnings, rider savings, CO₂ saved, plus a confidence-threshold trade-off curve). Figures are written to `figures/`.

### 3. The Match Score Combiner
The `compute_match_score()` function mathematically fuses these dual outputs into a singular ranking score.
*   `Rank Score = Acceptance Probability - (Predicted Detour In KM * Penalty Weight)`

### 4. Batch Allocation & Bipartite Graph Optimization
The unified `Rank Score` acts as the edge weight in a global graph. The simulation holds incoming requests for a micro-batch temporal window (e.g., 30 to 60 seconds). It maps all waiting Drivers to all waiting Riders simultaneously, executing a **Maximum Weight Bipartite Matching Algorithm** to solve for the absolute highest cumulative `Rank Score` across the entire ecosystem, guaranteeing the global optimum instead of a suboptimal greedy pairing.

---

## Output Phase (`notebooks/03_model_evaluation.ipynb`)

The final piece of the architecture is visual evaluation. This Jupyter Notebook loads the trained Multi-Objective Matcher and renders scatter plot visualizations using `seaborn`. It demonstrates the core premise of the academic project/startup: highlighting the "Trade-off Frontier" between how much distance a driver must sacrifice vs. how likely they are to accept the matching incentive.
