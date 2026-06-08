# 08 - Nationwide Scaling Architecture

The Qmutez matching engine is trained on single-passenger NYC Taxi Data because New York City offers the densest open-source mobility dataset in existence. However, the ultimate business goal of Qmutez is to deploy a live Peer-to-Peer carpooling application across the entire United States.

A major technical challenge arises: **How does an AI model trained on specific, dense 3km detours in Manhattan scale to the massive, 15km highway detours of Texas or the Midwest without failing?**

This document explains the mathematical architecture—**Feature Abstraction** and **Dynamic Configuration Clustering**—that allows a model trained on *"New York Logic"* to scale universally to America perfectly.

---

## 1. Feature Abstraction (Why the AI Survives)
If we trained our XGBoost Matching Algorithm on absolute geographic coordinates (e.g., teaching it that the route from Times Square to Brooklyn is highly profitable), the AI would immediately crash when deployed in Chicago or Los Angeles, because it mathematically *"overfits"* to the local geography of New York.

To solve this, the Qmutez pipeline abstracts raw geography into **Universal Ratio Features**. 

If you analyze the exact parameters fed into the XGBoost Regressor and Classifier in `src/models/xgb_matching.py`, you will find:
1.  `detour_distance_km`: The absolute metric distance added to the route.
2.  `overlap_ratio`: The calculation (`Original Commute Distance / (Original Commute Distance + Detour)`).

### The Math of Generalization
Because the AI learns the *psychological breaking point* of the `overlap_ratio` rather than the absolute distance, it flawlessly generalizes behavior mathematically across different state densities.

**Scenario A: The New York Commuter**
*   **Driver Commute:** 5 km (Dense traffic)
*   **Rider Detour Request:** 5 km 
*   **Math:** `Overlap Ratio = 5 / (5 + 5) = 50%`. 
*   **AI Logic:** The system correctly assesses that a 50% overlap means the driver is literally doubling their workload for very little financial `fare_amount`. The Acceptance Probability Classifier correctly returns a `0.10` likelihood of success. The match is rejected.

**Scenario B: The Texas / Highway Commuter**
*   **Driver Commute:** 80 km (Interstate highway)
*   **Rider Detour Request:** 10 km (A short exit off the highway)
*   **Math:** `Overlap Ratio = 80 / (80 + 10) = 88%`.
*   **AI Logic:** Despite the Texas detour (10km) being exactly double the New York detour (5km), the `overlap_ratio` is incredibly high (88%). The detour represents a negligible inconvenience over the length of the 80km highway stretch, and the `fare_amount` split will be highly profitable. The Acceptance Probability Classifier correctly returns a `0.92` likelihood of success. The match is accepted via Bipartite formulation.

Therefore, **one single machine learning model** controls the entire country. The fundamental human geoeconomic behavior ("*Was the money worth the extra hassle?*") is mathematically universal across all state lines.

---

## 2. Dynamic Configuration Clustering (Regional Adaptation)
While the AI scales perfectly via abstraction, the `PairGenerator` simulation engine uses a configurable detour threshold (`max_detour_km`) to define the feasibility boundary for each geographic region. This threshold controls the BallTree spatial query radius and the detour acceptability cutoff, adapting the matching behavior to local driving conditions.

In both `config/prod_config.yaml` and `config/dev_config.yaml`, the threshold is currently configured to `max_detour_km: 3.0` for dense NYC (dev and prod share the matching thresholds so training is consistent; they differ only in `sample_ratio`). If Qmutez scales to the Midwest, a `3.0` km limit is violently restrictive and will result in zero matches on a highway system — which is exactly why the threshold is configuration-driven and tiered per region below.

### Solution: Geographic Density Tiers
We decouple the routing logic from the codebase. When the live Qmutez app receives a Rider Request, the backend server pings the user's GPS coordinates and applies a **Density Clustering Heuristic**:

1.  **Tier 1 (Hyper-Dense):** Manhattan, Downtown Chicago, SF.
    *   *Configuration Triggered:* `max_detour_km = 3.0`.
    *   *Logic:* In traffic gridlock, 3.0 km is a brutal 25-minute delay. The shield must be narrow.
2.  **Tier 2 (Suburban/Medium):** Atlanta, Denver, Suburbs.
    *   *Configuration Triggered:* `max_detour_km = 7.0`.
    *   *Logic:* Moderate traffic allows for wider Bipartite Graph candidate generation without destroying the driver's schedule.
3.  **Tier 3 (Highway/Rural):** Maine, Texas, Interstates.
    *   *Configuration Triggered:* `max_detour_km = 20.0`, `max_detour_mins = 15.0`.
    *   *The Speed Limit Logic:* How does a massive 20km detour align with a standard 15-minute wait time tolerance? Because in rural states like Maine, the detour occurs on a 70-mph (112 km/h) highway. Covering 20km on I-95 only taking 10.5 minutes. The algorithmic shield must be open up to 20km precisely *because* the speed limit allows the driver to clear it within the psychological `max_detour_mins` threshold.

### Spatial Integrity: Point-in-Polygon Validation
A third critical scaling mechanism operates at the coordinate synthesis layer. NYC Taxi Zones are compact and roughly rectangular, so scattering pickups across bounding boxes works well. But suburban census tracts and rural county polygons are often highly irregular — following rivers, wrapping around lakes, or forming L-shapes along highways. A simple bounding box scatter would place a significant fraction of synthesized coordinates in water, forests, or neighboring jurisdictions.

To solve this, `loader.py` applies **Point-in-Polygon Refinement** after the initial bounding box scatter. Using vectorized `shapely.contains()`, every coordinate is validated against the actual zone polygon. Points that fall outside are automatically resampled via rejection sampling until they land within the polygon boundary. This ensures that when the synthesis pipeline processes non-NYC geographic data (Census tracts, GTFS zones, custom regions), every synthesized coordinate is geographically valid regardless of polygon shape.

### Conclusion
By mathematically abstracting the dataset to Universal Ratios, dynamically tiering the `max_detour_km` and `max_detour_mins` computational shields based on geographic density and average vehicular speed, and enforcing point-in-polygon spatial validity for arbitrary zone geometries, the Qmutez backend architecture serves precisely optimized P2P carpools nationwide using a single, robust codebase.
