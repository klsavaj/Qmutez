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

1.  **`calculate_haversine()`:** A trigonometric algorithm that calculates the "Great Circle" distance between two GPS coordinates using the radius of the Earth. (Far more accurate than plotting Euclidean geometry on a flat map axis).
2.  **`calculate_detour_cost()`:** Determines exactly how far out of their way the Driver has to go.
    *   *Formula:* `(Driver Origin -> Rider Origin -> Rider Dest -> Driver Dest) - (Driver Origin -> Driver Dest)`
3.  **`compute_spatial_overlap()`:** A heuristic ratio showing what percentage of the new shared trip overlaps with the driver's original intended route.
4.  **`calculate_time_overlap()`:** Ensures the Rider requested the trip within an acceptable margin (e.g., 15 mins) of the Driver's start time.

---

## Phase 3: The P2P Simulation Engine (`src/simulation/pair_generator.py`)

Because we use single-passenger taxi data, we must programmatically force a simulation of a P2P marketplace. This generates the synthetic training dataset required for the machine learning models.

### 1. Temporal Filtering
Filters millions of trips down strictly to the "Morning Commute" window (e.g., 07:00 to 10:00 AM).

### 2. Role Assignment
Uses algorithmic splits (with a set random seed for reproducibility) to tag 50% of the commuters as "Drivers" aiming to offset their driving costs, and 50% as "Riders" requesting a ride.

### 3. The O(N²) Bounding Filter & Pair Generation
Comparing *every* Driver to *every* Rider creates millions of mathematical calculations, crashing memory instantly.
*   **Heuristic Pre-filter:** Before calculating the expensive Haversine distance, the engine checks if a Rider's pickup point is roughly within 5 kilometers (`lat +/- 0.05`) of the Driver's pickup point. If not, the pair is discarded immediately.
*   For the remaining feasible candidates, the engine runs all functions from `feature_eng.py`.
*   **Ground Truth Labeling:** Assigns a `1` (Match Feasible) if the detour is under the maximum threshold (e.g., < 3.0 km) and a `0` (Match Infeasible) if it is over.
*   The final dataset (containing driver IDs, detour km, overlap ratios, economic fare estimates, and the binary label) is saved to `data/simulated/candidate_pairs.parquet`.

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

### 3. The Match Score Combiner
The `compute_match_score()` function mathematically fuses these dual outputs into a singular ranking score.
*   `Rank Score = Acceptance Probability - (Predicted Detour In KM * Penalty Weight)`

### 4. Batch Allocation & Bipartite Graph Optimization
The unified `Rank Score` acts as the edge weight in a global graph. The simulation holds incoming requests for a micro-batch temporal window (e.g., 30 to 60 seconds). It maps all waiting Drivers to all waiting Riders simultaneously, executing a **Maximum Weight Bipartite Matching Algorithm** to solve for the absolute highest cumulative `Rank Score` across the entire ecosystem, guaranteeing the global optimum instead of a suboptimal greedy pairing.

---

## Output Phase (`notebooks/03_model_evaluation.ipynb`)

The final piece of the architecture is visual evaluation. This Jupyter Notebook loads the trained Multi-Objective Matcher and renders scatter plot visualizations using `seaborn`. It demonstrates the core premise of the academic project/startup: highlighting the "Trade-off Frontier" between how much distance a driver must sacrifice vs. how likely they are to accept the matching incentive.
