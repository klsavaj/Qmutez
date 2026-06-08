# Project Proposal

## Title

**Incentive-Aware Multi-Objective Matching for Peer-to-Peer Commuter Carpooling**

---

## 1. Problem Statement and Motivation

Urban commuting systems experience congestion, high transportation costs, and underutilized private vehicle capacity. While ride-sharing platforms such as Uber and Lyft offer pooled ride options, their optimization frameworks are primarily designed for professional drivers and real-time demand rather than peer commuters with partially fixed routes.

In commuter-oriented peer-to-peer settings (e.g., university campuses or office corridors), participants are regular commuters with limited tolerance for detours. Effective ride matching must therefore simultaneously:

- Minimize additional detour time for drivers  
- Increase economic benefit for participants  
- Maximize overall match rate  
- Reduce cancellation likelihood  
- Maintain lower rider cost compared to solo alternatives  

Traditional geographic threshold–based matching does not explicitly model trade-offs between detour cost, economic gain, and match probability. In this project, I propose to design and evaluate a machine learning–based multi-objective matching framework that predicts route compatibility and match success likelihood, and optimizes driver–rider pairing under competing objectives.

---

## 2. Dataset Description

### Primary Dataset

NYC Taxi Trip Record Data  
Provided by the New York City Taxi and Limousine Commission (NYC TLC)

Official page:  
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Direct download directory (example – Yellow Taxi):  
https://d37ci6vzurychx.cloudfront.net/trip-data/

The dataset includes:

- Pickup and drop-off timestamps  
- Pickup and drop-off latitude/longitude  
- Trip distance  
- Trip duration  
- Fare amount  
- Passenger count  

### Optional Supplementary Dataset

Chicago Taxi Trips Dataset (City of Chicago Open Data):  
https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew

---

### Data Construction Plan

Because the datasets contain single-passenger trips rather than shared rides, I will construct a simulation framework:

1. Cluster trips into commuting time windows (e.g., 7–10 AM, 4–7 PM).
2. Define candidate driver–rider pairs based on spatial and temporal overlap.
3. Engineer features including:
   - Route overlap ratio  
   - Detour distance  
   - Detour time  
   - Estimated incremental earnings  
   - Time window compatibility  

This synthetic pairing framework enables controlled evaluation of matching strategies.

---

## 3. Baseline Model

I will implement two baseline approaches:

### 1. Rule-Based Matching

- **Objective:** Provide a simple, interpretable benchmark to evaluate more advanced ML methods.  
- **Method:**
  - Compute straight-line (Haversine) distance between driver route and rider pickup/drop points.
  - Accept matches only if detour distance is below a fixed threshold (e.g., 2–3 km).  
  - Time window compatibility: rider pickup must fall within driver commute time ±15 minutes.  
  - Cost sharing: fare split proportional to distance traveled.  
- **Input Features:**  
  - Driver start/end coordinates  
  - Rider pickup/drop coordinates  
  - Estimated detour distance/time  
- **Output:**  
  - Binary match feasibility (1 = feasible, 0 = infeasible)  
- **Rationale:** Provides interpretable baseline and allows quick sanity checks for route compatibility.

### 2. Logistic Regression Baseline

- **Objective:** Introduce a simple predictive model that captures linear relationships between key features and match feasibility.  
- **Method:**
  - Use the same candidate pairs as the rule-based method.  
  - Train a logistic regression classifier to predict match feasibility.  
- **Input Features:**
  - Detour distance  
  - Detour time  
  - Route overlap ratio  
  - Time window overlap  
  - Rider-driver distance from route midpoint  
- **Output:** Probability of match success (acceptance probability)  
- **Rationale:** Provides a quantitative benchmark for comparison with non-linear, multi-objective models.  

---

## 4. Advanced Model

The advanced model focuses on **multi-objective prediction** and ranking of driver–rider pairs.

### Core Model

- **Algorithm:** Gradient Boosted Trees (XGBoost)  
- **Objectives:**  
  1. Predict detour cost (continuous regression)  
  2. Predict match probability (classification)  
  3. Compute a composite score for multi-objective matching  
- **Input Features:** *(as implemented — relative/abstracted features rather than absolute coordinates, so the model generalizes across cities)*
  - `driver_orig_dist` — driver's original solo trip distance
  - `overlap_ratio` — fraction of the shared trip aligned with the driver's intended route
  - `time_diff_mins` — temporal gap between driver and rider pickups
  - `driver_approx_fare` / `rider_approx_fare` — economic baselines
  - `fare_ratio` — rider fare relative to driver fare
  - `pickup_hour` / `pickup_dow` — temporal context (AM/PM, weekday)
  - `shareability_count` — corridor density (riders within the driver's search radius)
  *(The detour-prediction regressor uses these same 9 features; `detour_distance_km` is its target, not an input.)*
- **Output:**  
  - Predicted detour distance (km)  
  - Probability of driver accepting match  
  - Multi-objective match score 

### Rationale

- Captures non-linear relationships between detour, time, and likelihood of acceptance.  
- Enables multi-objective optimization for realistic commuter scenarios.  
- Provides flexible scoring framework that can be extended for future applications.

---

## 5. Evaluation Metrics

### Model-Level

Regression:
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)

Classification:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

### System-Level Simulation

- Match rate (%)  
- Average driver detour  
- Average driver earnings increase  
- Average rider cost reduction  
- Trade-off curves (earnings vs detour)  

Baseline vs advanced model comparison will be performed.

> **Status:** All of the above are implemented. The regressor reports MAE and RMSE; the classifier reports accuracy, precision, recall, F1, and ROC-AUC; and a system-level bipartite-matching evaluation reports match rate, average driver detour, driver earnings, rider savings, CO₂ saved, and a confidence-threshold trade-off curve.

---

## 6. Planned Timeline (Now – End of April)

### Week 1
- Collect and clean NYC/Chicago taxi datasets
- Perform exploratory data analysis (EDA)
- Cluster trips into commuting time windows
- Engineer route similarity features, detour distance, detour time, and time window compatibility 

### Week 2
- Implement rule-based matching system
- Implement logistic regression baseline
- Evaluate baseline models on initial synthetic driver–rider pairs
- Analyze feature importance and baseline performance 

### Week 3
- Implement Gradient Boosted Trees (XGBoost) for multi-objective matching
- Train regression for detour prediction and classification for match probability
- Construct multi-objective scoring function combining detour, earnings, and acceptance probability
- Perform initial hyperparameter tuning and validation

---

## 7. Areas of Uncertainty / Assistance Requested

- Since my match labels are simulated, what strategies would you suggest for validating that these labels are realistic and suitable for ML training? Are there best practices for train/test splitting or sensitivity analysis for synthetic labels?
- How can I ensure that my evaluation methodology is rigorous and scientifically credible, given that the dataset and labels are simulated? For example, how should I document assumptions, benchmark against baselines, and present robustness analysis?