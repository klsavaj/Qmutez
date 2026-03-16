# 07 - Startup Roadmap & Future Work

While the current Qmutez architecture implements a robust, Master's-level Machine Learning pipeline (Feature Abstraction, Driver Acceptance Prediction, and Bipartite Batch Matching), scaling this platform into a live, venture-backed mobility startup requires specialized algorithmic extensions.

This document outlines the theoretical and mathematical roadmap for future enterprise-grade enhancements.

---

## 1. Predictive Commute Matching
Typical mobility platforms (like Uber) act *reactively*: a user opens the app, requests a ride, and the system scrambles to find a driver. 

However, peer-to-peer carpooling is fundamentally based on **stable recurring routines** (e.g., commuting to a university or tech park at 8:30 AM every weekday). Qmutez will eventually transition from a reactive model to a **Predictive Time-Series Matching Model**.

### Mathematical Formulation
The system will log historical trips and utilize a Recurrent Neural Network (RNN) or Gradient Boosting model to detect temporal recurrence. It will calculate the probability that a user will request a specific trip tomorrow:

$$ P(trip \mid user_i, time_t, day_d, origin_z, dest_z) $$

If the probability exceeds a confidence threshold (e.g., $> 0.90$), the system will **pre-compute matches the night before**. 
*   *Advantage:* The Rider receives a push notification ("We found you a ride for tomorrow at 8:30 AM. Confirm?"), effectively locking in supply and demand before the commute begins and drastically reducing morning-of marketplace anxiety.

---

## 2. Market Balance: Supply vs. Demand Dynamics
A P2P mobility marketplace fails if there are far more riders than drivers (resulting in excessive wait times) or conversely, too many drivers (resulting in zero matches and churn).

Qmutez must implement an automated clearinghouse mechanism to maintain geographic equilibrium.

### Algorithmic Dynamic Pricing
When demand vastly outpaces supply in a specific TLC Taxi Zone, the system will apply a dynamic mathematical multiplier to the baseline `fare_amount` to incentivize latent drivers to activate their routes.

$$ Fare\_Multiplier = \frac{Active\_Rider\_Demand_{Zone}}{Active\_Driver\_Supply_{Zone}} $$

If $Demand > Supply$, the Multiplier increases $> 1.0$. The XGBoost `Acceptance Classifier` will instantly detect the higher potential profit and aggressively match drivers to that zone.

---

## 3. Environmental Impact Tracking
Shared rides inherently reduce carbon emissions. Expanding the ML architecture to include real-time CO₂ tracking validates the product's ESG (Environmental, Social, and Governance) impact for city grants or carbon-credit monetization.

### Carbon Footprint Estimate
Using standard EPA emission factors ($\approx 404$ grams of $CO_2$ per mile for an average passenger vehicle), the system can calculate exact emission savings per match. 

$$ CO_2\_Saved = \left( Distance_{Rider} \times 404g \right) - \left( Detour_{Driver} \times 404g \right) $$

By tracking this via an analytics dashboard, Qmutez can publicly quantify the exact metric tons of greenhouse gas the platform removes from the atmosphere weekly.
