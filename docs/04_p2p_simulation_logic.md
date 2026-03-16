# 04 - P2P Simulation Logic

## The "Missing Data" Problem
Native peer-to-peer (P2P) commuter matching data is heavily proprietary (locked inside apps like Waze Carpool or Uber). To train a matching framework, we must simulate "Drivers" and "Riders" from a dataset of single-passenger rides.

## Simulation Pipeline (`src/simulation/pair_generator.py`)

### 1. The Temporal Filter
We strictly isolate trips that occur during traditional peak commuting hours (e.g., `07:00` to `10:00`) to represent a realistic commuter scenario.

### 2. Driver/Rider Assignment
We use a random algorithmic split to assign a set percentage (e.g., 50%) of the trips to the "Driver Profile" and the rest to the "Rider Request Profile".

### 3. Solving the O(N²) Computation Problem
Attempting a purely combinatorial pair matching between thousands of drivers and thousands of riders results in millions of calculations, instantly crashing local memory limits.
*   **Geographic Pre-filter:** Before calculating exact detours, the script checks a loose bounding box constraint: Is the rider's pickup broadly near the driver's origin point? If not, the pair is instantly bypassed.

### 4. Temporal Aggregation (Batch Matching)
Real-world systems do not match users instantly (which creates greedy, suboptimal pairings). Qmutez simulates an aggregation window where thousands of riders and drivers are pooled over a short interval (e.g., 30-60 seconds) prior to executing the matching algorithm. This enables the calculation of global optimal matches.

### 5. Mathematical Feature Engineering
*   **Haversine Distance:** Used because it accounts for the actual spherical shape of the Earth, providing an exact shortest-path distance over simple Euclidean grids.
*   **Detour Cost formulation:** The new distance `(Driver Origin -> Rider Pickup -> Rider Dropoff -> Driver Destination)` minus the original `(Driver Origin -> Driver Destination)`.
