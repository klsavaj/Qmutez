# 04 - P2P Simulation Logic

## The "Missing Data" Problem
Native peer-to-peer (P2P) commuter matching data is heavily proprietary (locked inside apps like Waze Carpool or Uber). To train a matching framework, we must simulate "Drivers" and "Riders" from a dataset of single-passenger rides.

## Simulation Pipeline (`src/simulation/pair_generator.py`)

### 1. The Temporal Filter
We strictly isolate trips that occur during traditional peak commuting hours (e.g., `07:00` to `10:00`) to represent a realistic commuter scenario.

### 2. Driver/Rider Assignment
We use a random algorithmic split to assign a set percentage (e.g., 50%) of the trips to the "Driver Profile" and the rest to the "Rider Request Profile".

### 3. Spatial Indexing (BallTree)
Attempting a purely combinatorial pair matching between thousands of drivers and thousands of riders results in millions of calculations, instantly crashing local memory limits.
*   **BallTree Spatial Index:** For each temporal batch, a `sklearn.neighbors.BallTree` is constructed from rider pickup coordinates using the Haversine metric on the unit sphere. Each driver queries the tree for neighbors within a 5km geodesic radius in **O(log N)** time — replacing the old O(N) linear scan per driver.

### 4. Temporal Aggregation (Batch Matching)
Real-world systems do not match users instantly (which creates greedy, suboptimal pairings). Qmutez simulates a 5-minute aggregation window where drivers and riders are pooled over a short interval prior to executing the spatial queries. This mirrors how a production system would batch incoming requests.

### 5. Training Data Collection
All feasible pairs within the detour threshold are collected with independently assigned **multi-factor stochastic labels**. The label models acceptance as a composite of 4 behavioral factors (detour severity, fare profitability, temporal inconvenience, trip length ratio) plus Gumbel noise for human irrationality. This gives the model the full distribution of matches — both great and mediocre — rather than only bipartite-optimal winners.

*Note: Bipartite Graph Matching (via `networkx.max_weight_matching`) is reserved for **production inference**, where the trained model's prediction scores serve as edge weights to select globally optimal 1:1 assignments.*

### 6. Mathematical Feature Engineering
*   **Haversine Distance:** Used because it accounts for the actual spherical shape of the Earth, providing an exact shortest-path distance over simple Euclidean grids.
*   **Detour Cost formulation:** The new distance `(Driver Origin -> Rider Pickup -> Rider Dropoff -> Driver Destination)` minus the original `(Driver Origin -> Driver Destination)`.
*   **Overlap Ratio:** `original_dist / (original_dist + detour)` — clamped to [0, 1] to handle edge cases where the rider is directly on the way (negative detour).
