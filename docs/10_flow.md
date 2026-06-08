# Qmutez — Complete Learning Path (Zero to Full Understanding)

> This document assumes you know **nothing** about Machine Learning, data science, or geographic algorithms.
> Follow each phase in order. Every section explains **what** the concept is, **why** it matters for this project, and **how** it connects to the next section.
> By the end, you will understand every line of code, every design decision, and every fix in this project.

---

## How This Document is Organized

```
Phase 1: Foundations         (What is ML? What tools exist? What vocabulary do I need?)
Phase 2: The Data Problem    (Where does our data come from? Why is it broken? How do we fix it?)
Phase 3: The Math            (What formulas power the matching engine?)
Phase 4: The Simulation      (How do we fake a carpooling marketplace from taxi data?)
Phase 5: The AI Models       (How does the computer learn to match rides?)
Phase 6: The Bottleneck Fixes (What was wrong and how did we fix it?)
Phase 7: The Startup Vision  (How does this become a real nationwide product?)
Phase 8: Reading Order       (Exact order to read every file)
```

---

# Phase 1: Foundations

## 1.1 — What is Machine Learning?

Machine Learning (ML) is when you give a computer **thousands of examples** of something, and it figures out the pattern by itself — instead of you writing explicit rules.

**Without ML (Rule-Based):**
You write: *"If detour < 3km AND fare > $10, match the ride."*
Problem: You have to think of every possible rule. What about a 4km detour on an 80km highway commute? Your rule rejects it, but a human would accept it.

**With ML:**
You show the computer 10,000 examples of rides that drivers accepted or rejected. It learns its OWN rules — rules that are often **better** than anything a human would write because it finds patterns across thousands of combinations that a human brain can't process.

**Connection to Qmutez:** Our entire project is about training a computer to predict: *"Will this driver accept this rider, given the detour distance, the fare, and the timing?"*

## 1.2 — Two Types of ML We Use

### Regression (Predicting a Number)
- **Question:** "How many kilometers of detour will this carpool add?"
- **Output:** A continuous number like `3.7 km`
- **Real-world analogy:** Predicting tomorrow's temperature — it could be 72.3 or 68.9, any number.
- **Our file:** `src/models/xgb_matching.py` → `self.regressor`

### Classification (Predicting a Category — Yes or No)
- **Question:** "Will this driver accept or reject this ride?"
- **Output:** `1` (accept) or `0` (reject)
- **Real-world analogy:** Email spam filter — "Is this spam? Yes or No."
- **Our file:** `src/models/xgb_matching.py` → `self.classifier`

**Why we use BOTH:** First we predict the detour (regression), then we predict acceptance (classification). Together they form a **Multi-Objective** model — that's the "Multi-Objective Matching" in our project title. Most research papers do one OR the other. Doing both and combining them into a single ranking score is one of our novel contributions.

## 1.3 — What are Features and Labels?

These are the two most important words in all of ML:

| Term | What It Means | Analogy | Example in Our Project |
|------|--------------|---------|----------------------|
| **Feature** | An input column the model uses to learn | The questions on an exam | `detour_distance_km`, `overlap_ratio`, `time_diff_mins` |
| **Label** | The answer we want the model to predict | The answer key | `label` column: `1` = accepted, `0` = rejected |
| **Training** | Showing examples to the model so it learns | Studying with flashcards | Feeding ~38,000 driver-rider pairs into XGBoost |
| **Testing** | Checking if the model learned correctly | The actual exam | Giving the model thousands of pairs it has NEVER seen |
| **Prediction** | Asking the trained model about something new | Answering a new question | "Here's a new driver and rider. Will they match?" |

**Key rule:** The model NEVER sees the test data during training. If it did, it would be like memorizing the answer key — it would score perfectly but learn nothing.

**Connection:** Phase 2 and Phase 4 are entirely about **generating** these features and labels from raw taxi data. Phase 5 is about **using** them to train the AI.

## 1.4 — What is a Train/Test Split?

We take our dataset (e.g., ~38,000 driver-rider pairs) and split it:
- **80% (~30,000 pairs)** → Training set. The model studies these.
- **20% (~7,600 pairs)** → Test set. The model is graded on these.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**`random_state=42`:** This is a "seed" that makes the random split reproducible. Every time you run the code with seed 42, you get the exact same split. This is critical for academic reproducibility — a professor should get the same results you did.

**Why 80/20?** It's the standard convention. Too much testing data = not enough to learn from. Too little testing data = unreliable grade.

## 1.5 — What is Overfitting?

Overfitting is when the model **memorizes** the training data instead of **learning** the pattern.

**Analogy:** A student who memorizes every practice exam answer verbatim but can't solve a slightly different problem on the real exam.

**How to detect it:** If training accuracy is 99% but test accuracy is 60%, the model overfit.

**How to prevent it:** Cross-validation (see 1.7), limiting model complexity, adding noise to labels (we do this with Gumbel noise — explained in Phase 4).

**Connection:** The original Qmutez code had a different overfitting problem — the labels were a **deterministic function** of the features (if detour < 3km → label = 1, always). XGBoost reverse-engineered the rule instantly and achieved 100% accuracy. This was Fix #1 in our bottleneck analysis.

## 1.6 — What is XGBoost?

XGBoost (eXtreme Gradient Boosting) is a specific ML algorithm. Think of it as the **gold standard** for tabular data (data in rows and columns, like spreadsheets or CSV files).

### How it works (simplified):
XGBoost builds **decision trees** — think of them as flowcharts:

```
Is overlap_ratio > 0.8?
├── YES → Is fare > $15?
│         ├── YES → ACCEPT (92% confident)
│         └── NO → Is time_diff < 5min?
│                   ├── YES → ACCEPT (75% confident)
│                   └── NO → REJECT (60% confident)
└── NO → Is detour < 2km?
          ├── YES → ACCEPT (55% confident)
          └── NO → REJECT (88% confident)
```

But one tree is weak. XGBoost builds **100 trees** in sequence. Each new tree focuses on correcting the mistakes of the previous trees. The final prediction is the combined vote of all 100 trees. That's the "Boosting" in the name.

### Why not Deep Learning (Neural Networks)?
Neural networks are great for images (recognizing faces) and text (ChatGPT). But for spreadsheet data (GPS coordinates, fares, distances), XGBoost consistently **wins**. It's:
- Faster to train (seconds vs hours)
- Easier to interpret (you can see which features matter)
- More accurate on structured, tabular data

**Read:** `docs/02_tech_stack_justification.md` — explains why every library was chosen.

## 1.7 — What is Cross-Validation?

A single 80/20 split might be "lucky" — maybe all the easy examples ended up in the test set. Cross-validation solves this.

**5-Fold Cross-Validation:**
1. Split data into 5 equal chunks (folds)
2. Train on folds 1-4, test on fold 5 → get accuracy score
3. Train on folds 1-3 + 5, test on fold 4 → get accuracy score
4. Repeat for all 5 combinations
5. Report the **average** accuracy across all 5 runs

If all 5 scores are similar (e.g., 78%, 81%, 79%, 80%, 77%), the model is **stable**. If they vary wildly (e.g., 95%, 45%, 88%, 52%, 71%), something is wrong.

**Connection:** `xgb_matching.py` now runs 5-fold cross-validation (via `cross_val_score`) alongside a `GridSearchCV` hyperparameter sweep — this resolved bottleneck issue #12.

## 1.8 — What are Evaluation Metrics?

"Accuracy" alone doesn't tell the full story. Here's what each metric means:

### For Classification (Accept/Reject prediction):

| Metric | What It Measures | Example |
|--------|-----------------|---------|
| **Accuracy** | % of predictions that were correct (overall) | "80% of the time, the model was right" |
| **Precision** | Of all the rides the model said "accept," how many actually were accepted? | High precision = few false alarms |
| **Recall** | Of all the rides that actually were accepted, how many did the model catch? | High recall = few missed matches |
| **F1-Score** | The harmonic mean of Precision and Recall | Balances both — the single best metric for imbalanced data |
| **ROC-AUC** | How well the model distinguishes between classes across ALL thresholds | 0.5 = random guessing, 1.0 = perfect. Good models: 0.75-0.95 |

### For Regression (Detour prediction):

| Metric | What It Measures | Example |
|--------|-----------------|---------|
| **MAE** (Mean Absolute Error) | Average km the prediction is off by | "On average, our detour prediction is off by 0.8 km" |
| **RMSE** (Root Mean Squared Error) | Same as MAE but penalizes large errors more | Large errors (off by 5km) are punished harder than small ones |

**Connection:** Our project proposal (Section 5) promises all of these metrics, and they are now computed — the regressor reports MAE and RMSE; the classifier reports accuracy, F1, ROC-AUC, and a full `classification_report` (this resolved bottleneck issue #13).

## 1.9 — What is Feature Engineering?

Raw data is rarely useful directly. **Feature engineering** is the art of transforming raw data into meaningful numbers that an ML model can learn from.

**Example from our project:**
- **Raw data:** Driver starts at (40.758, -73.985), ends at (40.712, -74.006). Rider starts at (40.752, -73.978).
- **Engineered features:** `detour_distance_km = 1.8`, `overlap_ratio = 0.82`, `time_diff_mins = 3.2`

The model never sees raw GPS coordinates. It only sees the **relationships** between coordinates (how far, how much overlap, how much time). This is called **Feature Abstraction** — and it's the reason our model can work in ANY city, not just NYC.

**Connection:** All feature engineering lives in `src/simulation/feature_eng.py`.

## 1.10 — What is a Bipartite Graph?

A bipartite graph is a structure with two groups of nodes where edges only connect nodes from different groups — never within the same group.

```
DRIVERS          RIDERS
  D1 ──────────── R1
  D1 ──────── R2
  D2 ──────────── R1
  D2 ────── R3
  D3 ──────────── R2
  D3 ──── R3
```

Each edge has a **weight** (the match score). The **Maximum Weight Bipartite Matching** algorithm finds the assignment where:
- Every driver is matched to at most ONE rider
- Every rider is matched to at most ONE driver
- The total weight (sum of all match scores) is **maximized**

This is solved by `networkx.max_weight_matching()` — a mathematical optimization algorithm (based on the Hungarian Algorithm).

**Connection:** This is the FINAL step in our pipeline. After XGBoost predicts scores for all possible pairs, the bipartite matcher picks the globally optimal 1:1 assignments. This happens at **inference time** (production), NOT during training.

---

# Phase 2: The Data Problem

## 2.1 — Where Does Our Data Come From?

We use the **NYC Taxi & Limousine Commission (TLC)** dataset — 3.5 million real taxi trips from **August 2025**.

**Why August 2025?**
1. **Economic reality:** Post-COVID taxi pricing gives accurate fare baselines (2015 prices are outdated).
2. **Volume:** August = summer = peak activity = densest dataset.

**Read:** `docs/03_dataset_selection_and_cleaning.md`
**Code:** `src/data/loader.py`

### The Big Problem: No Carpool Data Exists
Real-world P2P carpooling data (two people sharing a ride) is **proprietary** — Uber and Waze own it and don't share. The NYC TLC data contains single-passenger taxi rides, not shared rides.

**Our solution:** Take individual taxi rides and **simulate** what would happen if two of them tried to share a route. This is why Phase 4 (simulation) exists — we mathematically CREATE a carpooling dataset from scratch.

## 2.2 — What's In the Raw Dataset?

The NYC TLC Yellow Taxi dataset has 19 columns per trip. We only keep the essentials:

| Column | What It Is | Why We Keep It |
|--------|-----------|---------------|
| `tpep_pickup_datetime` | When the passenger was picked up | Calculate temporal overlap between drivers and riders |
| `tpep_dropoff_datetime` | When the passenger was dropped off | Trip duration context |
| `trip_distance` | Actual distance driven (miles) | Ground truth for synthesizing dropoff coordinates |
| `fare_amount` | What the ride cost ($) | Economic incentive for driver acceptance |
| `PULocationID` | Pickup Taxi Zone (integer, e.g., 161 = Midtown) | Used to synthesize pickup GPS coordinates |
| `DOLocationID` | Dropoff Taxi Zone (integer) | Used to compute corridor-aware bearing direction |

**What we discard:** `VendorID`, `RatecodeID`, `payment_type`, `tip_amount`, `tolls_amount`, `congestion_surcharge`, `airport_fee` — these are administrative metadata or post-trip variables that vary randomly per person. We strip the noise to focus purely on the geometric inconvenience vs. financial reward relationship.

## 2.3 — The Coordinate Scrub Problem

In July 2016, the NYC TLC **removed** exact GPS coordinates (latitude/longitude) from their data for privacy. Instead, they give us a `PULocationID` — an integer like `161` that means "Midtown Manhattan."

**The Dilemma:** You can't calculate "how far out of the way must a driver go" without exact coordinates. If the model doesn't have exact detour costs, it can't learn the threshold of human inconvenience.

## 2.4 — The Solution: Coordinate Synthesis

Since we don't have GPS coordinates, we **synthesize** them mathematically using the information we DO have. This happens in `src/data/loader.py` and involves 4 steps:

### Step 1: Download the Official Taxi Zone Shapefiles
NYC TLC provides official geographic boundary files (`.shp` shapefiles) that define the exact polygon shape of each Taxi Zone. We download these from the TLC servers using `geopandas`.

```python
gdf = gpd.read_file("taxi_zones.shp")  # Load all 263 zone polygons
gdf_4326 = gdf.to_crs("EPSG:4326")     # Convert to standard lat/lon coordinates
```

**What is EPSG:4326?** It's the standard coordinate system used by GPS (latitude/longitude in degrees). The shapefile might be stored in a different projection (like meters), so we convert.

### Step 2: Zonal Bounding Box Distribution (Pickup Synthesis)
For each trip, we know which zone the pickup was in (`PULocationID`). We extract the **bounding box** (the smallest rectangle that contains the zone polygon):

```
Zone 161 (Midtown):
  minx = -73.990 (westmost longitude)
  maxx = -73.975 (eastmost longitude)
  miny = 40.750  (southmost latitude)
  maxy = 40.765  (northmost latitude)
```

We then place the pickup at a **random point within this rectangle**:
```python
pickup_longitude = np.random.uniform(minx, maxx)
pickup_latitude  = np.random.uniform(miny, maxy)
```

**Why not use the zone centroid (center point)?**
If you map every trip to the exact center, you cluster 10,000 trips within a 200m radius. This creates artificially high match rates (everyone is standing next to each other) and prevents the model from learning edge cases (someone standing 1.5km away from the center who still needs a ride).

### Step 3: Point-in-Polygon Refinement (Rejection Sampling)
Bounding boxes are **rectangles**, but taxi zones are **irregular polygons**. A coastal zone's bounding box extends over water. An L-shaped zone's box includes neighboring zones.

After the initial scatter, we check every point against the actual polygon:
```python
import shapely
outside = ~shapely.contains(polygon, shapely.points(lons, lats))
# Resample any points that landed outside the polygon
```

Points outside are resampled within the bounding box and re-checked, up to 10 rounds. This is called **rejection sampling** — keep sampling until the point meets the criteria.

**Why this matters for nationwide scaling:** NYC zones are fairly rectangular. But suburban census tracts and rural county polygons can be highly irregular (following rivers, wrapping around lakes). Without this validation, synthesized points in Texas could land in a lake.

### Step 4: Corridor-Aware Dropoff Synthesis
For the dropoff, we can't just pick another random point. Real commuters travel in **directional corridors** — the Bronx toward Manhattan in the morning, Brooklyn toward Midtown, etc.

We use the `DOLocationID` (dropoff zone) to compute the **true bearing** (compass direction) from the pickup zone centroid to the dropoff zone centroid:

```
Forward Azimuth Formula:
x = sin(Δλ) · cos(φ₂)
y = cos(φ₁) · sin(φ₂) - sin(φ₁) · cos(φ₂) · cos(Δλ)
θ = atan2(x, y)    ← the true bearing in radians
```

Then we add Gaussian noise (σ = 0.3 radians ≈ 17 degrees) to simulate street-level variation — not every driver takes the exact same highway:
```python
θ_final = θ + N(0, 0.3)
```

Finally, we project a destination coordinate at **exactly** the `trip_distance` away using **Spherical Trigonometry** (because the Earth is a sphere, not a flat map):
```
lat2 = arcsin(sin(lat1) · cos(d/R) + cos(lat1) · sin(d/R) · cos(θ))
lon2 = lon1 + atan2(sin(θ) · sin(d/R) · cos(lat1), cos(d/R) - sin(lat1) · sin(lat2))
```

Where `R = 3,958.8 miles` (Earth's radius) and `d = trip_distance`.

**The Result:** The Haversine distance between our synthesized pickup and dropoff **perfectly matches** the real taxi meter distance. The direction preserves the actual commute corridor. The economic AND geographic reality of the original trip is mathematically preserved.

For trips with unknown `DOLocationID`, we fall back to a random angle (uniform 0 to 2π).

## 2.5 — Data Cleaning

After synthesis, we apply strict filters:

1. **Geographic Bounding Box:** Pickup coordinates must fall within NYC bounds (Lat: 40.5-41.2, Lon: -74.3 to -73.7). This implicitly removes any 0.0, NaN, or out-of-range values.
2. **Financial Integrity:** Drop rows with `trip_distance ≤ 0` or `fare_amount < 0`.

The cleaned dataset is saved to `data/processed/cleaned_yellow_tripdata_2025-08.parquet`.

## 2.6 — What is Parquet?

Parquet is a **columnar storage format** — like a super-compressed, super-fast version of a CSV file.

| Feature | CSV | Parquet |
|---------|-----|---------|
| Size | 800 MB | 60 MB |
| Read speed | Slow (reads every cell) | Fast (reads only needed columns) |
| Data types | Everything is text | Preserves integers, floats, dates natively |

With 3.5 million rows, Parquet is essential. CSV would be painfully slow.

## 2.7 — What is YAML Configuration?

YAML is a human-readable config format. Instead of hardcoding values in Python:
```python
max_detour_km = 3.0  # BAD — changing this requires editing code
```

We put them in a config file:
```yaml
# config/prod_config.yaml
simulation:
  max_detour_km: 3.0
```

And read them:
```python
max_detour_km = self.sim_config["max_detour_km"]  # GOOD — change the YAML, not the code
```

We have two configs:
- **`dev_config.yaml`:** Downsamples to 25% of the commute-window data for fast iteration, while keeping the same NYC-appropriate matching thresholds as prod (3km detour)
- **`prod_config.yaml`:** Full dataset (100%), same 3km detour thresholds, for final evaluation

Both configs share the matching thresholds (so training behaves consistently); they differ only in `sample_ratio`.

---

# Phase 3: The Math

## 3.1 — Haversine Distance

The Earth is a sphere. If you calculate the distance between two GPS points using flat geometry (Pythagorean theorem), you get the wrong answer — especially over longer distances where the curvature matters.

The **Haversine formula** calculates the "Great Circle" distance — the shortest path along the surface of a sphere:

```
a = sin²(Δlat/2) + cos(lat1) · cos(lat2) · sin²(Δlon/2)
c = 2 · arcsin(√a)
distance = R · c     where R = 6,371 km (Earth's radius)
```

**In our code:** We use `geopy.distance.geodesic()` instead of our own Haversine implementation. Geodesic is slightly more accurate (it accounts for Earth being an oblate spheroid, not a perfect sphere), but both give nearly identical results for the distances we work with (1-50 km).

**Note:** All actual distance calculations use `geodesic()`. An unused `calculate_haversine()` function previously lived in `feature_eng.py` as dead code; it has since been removed (cleanup issue #15, resolved).

## 3.2 — Detour Cost

The core question: *"How far out of the way does the driver have to go?"*

```
Original route:  Driver Start → Driver End
Carpool route:   Driver Start → Rider Pickup → Rider Dropoff → Driver End
Detour = Carpool route distance - Original route distance
```

In code (`feature_eng.py`):
```python
original_dist = geodesic(driver_start, driver_end).km
new_dist = geodesic(driver_start, rider_start).km
         + geodesic(rider_start, rider_end).km
         + geodesic(rider_end, driver_end).km
detour = new_dist - original_dist
```

**Can detour be negative?** Yes! If the rider's destination is directly between the driver's start and end, the carpool route might actually be shorter. A negative detour means the rider is "on the way." We handle this in the overlap ratio (see 3.3).

## 3.3 — Overlap Ratio

*"What percentage of the trip was the driver's original intent?"*

```
overlap_ratio = original_dist / (original_dist + detour)
```

| Overlap | What It Means |
|---------|--------------|
| **0.95** | Rider is basically on the way (5% detour) |
| **0.80** | Decent match — 20% extra driving |
| **0.50** | Driver is going 50% out of their way (bad) |
| **1.00** | Rider is directly on the route (negative detour, clamped to 1.0) |

**Why this is the most important feature:** It's a **ratio**, not an absolute distance. This means:
- A 3km detour on a 30km commute → overlap = 0.91 (acceptable)
- A 3km detour on a 5km commute → overlap = 0.63 (terrible)

The same absolute detour means very different things depending on the total trip. This ratio is what allows the AI to generalize across cities — a 10km detour on an 80km Texas highway (overlap = 0.89) is treated the same as a 1km detour on a 8km NYC ride (overlap = 0.89).

**Clamping to [0, 1]:** When detour is negative, the formula can produce values > 1.0 (e.g., `5.0 / (5.0 + (-0.5)) = 1.11`). We clamp: `min(1.0, max(0.0, ratio))`.

## 3.4 — Time Overlap

*"How many minutes apart are the driver and rider requesting?"*

```python
time_diff = abs(driver_pickup_time - rider_pickup_time) in minutes
```

A time difference of 2 minutes means they're nearly simultaneous. A time difference of 14 minutes means the rider is waiting a long time or the driver has to wait. The threshold is typically 15 minutes — beyond that, it's not a viable commute match.

## 3.5 — The BallTree Spatial Index

**The problem:** If you have 10,000 drivers and 10,000 riders, comparing every driver to every rider requires 100,000,000 distance calculations. This is O(N²) complexity — it crashes your computer.

**The solution:** A BallTree is a data structure that organizes points in a tree so you can find all neighbors within a radius in **O(log N)** time instead of **O(N)**.

```python
from sklearn.neighbors import BallTree

# Build the tree from rider coordinates (once per batch)
tree = BallTree(rider_coords_radians, metric='haversine')

# For each driver, find all riders within 5km — in O(log N) time
neighbor_indices = tree.query_radius(driver_coords_radians, r=5.0/6371.0)
```

**Why Haversine metric?** The BallTree needs to know how to measure distance. We use Haversine (spherical distance) instead of Euclidean (flat distance) because GPS coordinates are on a sphere. The radius must be in radians: `5.0 km / 6371.0 km = 0.000785 radians`.

**Connection:** This replaced the old approach that used a crude rectangular bounding box (`lat ± 0.05`). The rectangle was wrong because 0.05 degrees of longitude ≈ 4.2km at NYC latitude, but 0.05 degrees of latitude ≈ 5.5km. The box was rectangular, not circular — it missed diagonal neighbors and included corner outliers.

---

# Phase 4: The Simulation

## 4.1 — Why Simulate?

We don't have real carpool data. We have single-passenger taxi trips. To create training data for our ML model, we must:
1. Pretend some trips are "Drivers" (people willing to share their ride)
2. Pretend other trips are "Riders" (people looking for a ride)
3. Calculate the features if they were paired
4. Decide whether each pair would be accepted or rejected (the label)

This all happens in `src/simulation/pair_generator.py`.

## 4.2 — Step 1: Temporal Filtering

We only care about peak commutes — both the morning rush (7:00–10:00 AM) and the evening rush (4:00–7:00 PM / 16:00–19:00). Random 2 AM rides are not commuters. The windows come from the `time_windows` list in config, and a trip is kept if it falls in *any* of them:

```python
mask = pd.Series(False, index=self.df.index)
for w in self.sim_config["time_windows"]:   # e.g. [{7,10}, {16,19}]
    mask |= (
        (self.df['tpep_pickup_datetime'].dt.hour >= w['start_hour']) &
        (self.df['tpep_pickup_datetime'].dt.hour < w['end_hour'])
    )
self.df = self.df[mask]
```

In dev mode, we also downsample to 25% of trips for fast iteration:
```python
self.df = self.df.sample(frac=0.25, random_state=42)
```

## 4.3 — Step 2: Driver/Rider Split

We randomly assign 50% of trips as "Drivers" and 50% as "Riders":
```python
np.random.seed(42)  # Reproducible
msk = np.random.rand(len(self.df)) < 0.5
self.drivers_df = self.df[msk]
self.riders_df = self.df[~msk]
```

**Why 50/50?** In a real P2P marketplace, any commuter can be either. A balanced split ensures the model doesn't overfit to one side. (Note: In reality, suburbs are driver-heavy and urban cores are rider-heavy. This is a known simplification that will be addressed with configurable `driver_ratio` in the startup phase.)

## 4.4 — Step 3: Temporal Batching

Real-world matching apps don't search the entire city. They batch requests over short windows. We simulate 5-minute windows:

```
7:00-7:05: Match drivers from this window with riders from this window
7:05-7:10: Next batch
7:10-7:15: Next batch
...
```

This is realistic — a production app would batch for 30-60 seconds, but 5 minutes works for the density of historical taxi data.

## 4.5 — Step 4: BallTree Spatial Matching

For each batch window:
1. Build a BallTree from all rider pickup coordinates
2. For each driver, query the tree: "Find all riders within 5km"
3. For each nearby rider, calculate detour, overlap, time difference
4. If detour is within the threshold, this pair is **feasible**

```python
tree = BallTree(rider_coords_rad, metric='haversine')
neighbor_indices = tree.query_radius(driver_start_rad, r=5.0/6371.0)
```

## 4.6 — Step 5: Multi-Factor Stochastic Labeling

This is the **most critical design decision** in the entire project.

We need to assign a label (1 = accepted, 0 = rejected) to each feasible pair. If we just say "detour < 3km → label = 1," XGBoost will reverse-engineer the rule trivially and get 100% accuracy. Useless.

Instead, we model acceptance as a **composite of 4 independent behavioral factors:**

```python
# Factor 1: Detour severity (30% weight)
# "How bad is the detour relative to the maximum allowed?"
detour_score = 1.0 - (detour / max_detour_km)  # 1.0 = no detour, 0.0 = at threshold

# Factor 2: Fare profitability (25% weight)
# "Is the rider's fare worth the driver's trouble?"
fare_ratio = rider_fare / max(driver_fare, 1.0)
fare_score = min(fare_ratio, 1.5) / 1.5  # Normalized to [0, 1]

# Factor 3: Temporal inconvenience (25% weight)
# "How long does the driver have to wait for the rider?"
time_score = max(0, 1.0 - (time_diff_mins / 15.0))  # 1.0 at 0min, 0.0 at 15min

# Factor 4: Trip length ratio (20% weight)
# "Is this a short detour on a long trip (acceptable) or a long detour on a short trip (painful)?"
trip_ratio = original_dist / max(original_dist + detour, 0.1)

# Weighted composite
acceptance_prob = 0.30 * detour_score + 0.25 * fare_score + 0.25 * time_score + 0.20 * trip_ratio
```

Then we add **Gumbel noise** to simulate human irrationality:
```python
noise = np.random.gumbel(0, 0.1)
is_accepted = 1 if (acceptance_prob + noise) > 0.45 else 0
```

### What is Gumbel Noise?
The Gumbel distribution has an asymmetric heavy right tail, meaning occasional large positive spikes can push borderline matches over the acceptance threshold (or vice versa). This is more realistic than symmetric Gaussian noise for modeling human decisions — a driver might randomly reject a perfectly good ride because they're tired, or randomly accept a borderline one because they need the money today.

### Why 4 factors instead of 1?
The model must learn the **interactions** between features, not just one threshold. A high fare can compensate for a long detour. A short time wait can compensate for a low overlap. The 4-factor composite forces XGBoost to learn these trade-offs.

## 4.7 — Step 6: Negative Sample Generation

The steps above only produce **feasible pairs** (detour within threshold) — mostly label 1. For the model to learn what rejection looks like, we need realistic label-0 examples.

We randomly sample drivers and riders from the full dataset and compute real features. Only pairs that **genuinely exceed** the detour threshold are kept as negatives:

```python
detour, new_dist, d_orig_dist = calculate_detour_cost(d_start, d_end, r_start, r_end)
if detour <= max_detour_km:
    continue  # Not a negative — skip
# Compute REAL features (not hardcoded zeros!)
overlap_ratio = compute_spatial_overlap(detour, d_orig_dist)
```

**Why this matters:** The original code generated fake negatives with hardcoded `overlap_ratio=0.0` and `time_diff_mins=60.0`. XGBoost didn't need to learn real rejection patterns — it just learned "if overlap == 0.0, predict 0." This was Fix #2.

## 4.8 — Step 7: Training vs Inference Separation

A critical architectural decision:

| Phase | What Happens | Which Pairs |
|-------|-------------|-------------|
| **Training** | Collect ALL feasible pairs, assign stochastic labels | Every pair within detour threshold |
| **Inference** (Production) | Model predicts scores → bipartite matching selects optimal 1:1 | Only the "winning" assignments |

**Why this matters:** If you only train on "winning" pairs (the bipartite-optimal matches), the model never learns what a mediocre-but-feasible match looks like. It can't assign appropriately low scores to them. Training on ALL pairs gives the model the complete distribution.

The bipartite matching algorithm (`networkx.max_weight_matching`) is only used at inference time, where the model's predicted scores serve as edge weights.

## 4.9 — The Output

The final dataset is saved to `data/simulated/candidate_pairs.parquet` with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `driver_id` | int | Index of the driver trip |
| `rider_id` | int | Index of the rider trip |
| `driver_orig_dist` | float | Driver's original solo trip distance (km) |
| `detour_distance_km` | float | Extra km added by the carpool |
| `overlap_ratio` | float | [0, 1] — how much of the trip aligns with driver's intent |
| `time_diff_mins` | float | Minutes between driver and rider pickup times |
| `driver_approx_fare` | float | Driver's taxi meter fare ($) |
| `rider_approx_fare` | float | Rider's taxi meter fare ($) |
| `fare_ratio` | float | `rider_fare / max(driver_fare, 1)` — economic-incentive signal |
| `pickup_hour` | int | Hour of day of the driver's trip (AM/PM context) |
| `pickup_dow` | int | Day of week of the driver's trip (0 = Monday) |
| `shareability_count` | int | Riders within the driver's search radius (corridor density; 0 for negatives) |
| `label` | int | 0 = rejected, 1 = accepted |

---

# Phase 5: The AI Models

## 5.1 — Why Baselines Matter

Before using a fancy AI, you must prove it's **necessary**. If a simple rule ("accept if detour < 3km") gets 85% accuracy, why bother with XGBoost? Baselines establish the **floor** that the advanced model must beat.

**Code:** `src/models/baselines.py`

### Rule-Based Baseline
```python
predictions = (
    (df['detour_distance_km'] <= 3.0) &  # Hard threshold
    (df['time_diff_mins'] <= 15.0)        # Hard threshold
).astype(int)
```
No learning — just rigid cutoffs. If XGBoost can't beat this, something is wrong.

**Note:** When run through the pipeline, `RuleBasedBaseline` is constructed with the thresholds from the YAML config (`max_km=sim["max_detour_km"]`, `max_mins=sim["max_detour_mins"]`) rather than fixed values, so it is evaluated on the same threshold that generated the data (bottleneck #9, resolved).

### Logistic Regression Baseline
A simple statistical model that draws a **straight line** (linear boundary) through the feature space. It uses the same core features as the advanced model (10 in total — the 9 the XGBoost classifier uses, plus `detour_distance_km`) for a fair comparison.

It struggles because the relationship between detour, fare, and acceptance is **non-linear** — a 3km detour is fine on a 30km trip but terrible on a 4km trip. Logistic Regression can't capture this interaction. XGBoost can.

## 5.2 — The XGBoost Dual-Model Architecture

**Code:** `src/models/xgb_matching.py`

The core innovation: two separate XGBoost models working together.

```
Input Features ──→ XGBRegressor  ──→ Predicted Detour (km)        ─┐
                                                                     ├──→ Combined Match Score
Input Features ──→ XGBClassifier ──→ Match Probability (0.0-1.0)  ─┘
```

### Model A: Detour Regressor
Predicts exactly how many kilometers the detour will cost:
```python
self.regressor = xgb.XGBRegressor(
    objective='reg:squarederror',  # Minimize squared error
    n_estimators=100,              # 100 trees
    learning_rate=0.1,             # How much each tree corrects the previous
    max_depth=5                    # Maximum tree depth (prevents overfitting)
)
```

**Features used (9):** `driver_orig_dist`, `overlap_ratio`, `time_diff_mins`, `driver_approx_fare`, `rider_approx_fare`, `fare_ratio`, `pickup_hour`, `pickup_dow`, `shareability_count`.

**Note:** `detour_distance_km` is NOT a feature — it's the regression **target**. Including it as a feature would be circular (predicting detour from detour).

### Model B: Acceptance Classifier
Predicts the probability that the driver will accept:
```python
self.classifier = xgb.XGBClassifier(
    objective='binary:logistic',   # Output probability between 0 and 1
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    eval_metric='logloss'          # Log loss penalizes confident wrong predictions
)
```

**Same 9 features** as the regressor. To counter class imbalance, training also sets `scale_pos_weight` to the negative/positive ratio, and the shown hyperparameters are refined by a 5-fold `GridSearchCV` sweep before final fitting.

### The Unified Match Score
The `compute_match_score()` function combines both predictions:
```python
score = match_probability - (predicted_detour * 0.1)
```

A high acceptance probability increases the score. A high predicted detour decreases it (penalty weight = 0.1). This single number can rank all candidate pairs from best to worst.

## 5.3 — The Driver-Level Train/Test Split

Both models share the **exact same** train/test split, and that split is done at the **driver level** to prevent entity leakage. We partition the unique `driver_id`s 80/20 (not the individual pairs), then assign every pair to train or test based on its driver:
```python
unique_drivers = df['driver_id'].unique()
train_drivers, test_drivers = train_test_split(
    unique_drivers, test_size=0.2, random_state=42
)
train_mask = df['driver_id'].isin(train_drivers)
train_df, test_df = df[train_mask], df[~train_mask]
```

**Why this matters:** If the regressor and classifier used different random splits, their metrics would be evaluated on different test rows, making them incomparable (this was Fix #5). Splitting on *drivers* rather than *pairs* additionally guarantees the model never sees a test driver during training — otherwise a driver could appear in both sets and inflate the test score. *(The baselines use a simpler random pair-level split.)*

## 5.4 — The Jupyter Notebook

**File:** `notebooks/03_model_evaluation.ipynb`

This notebook ties everything together:
1. Loads the simulated data from `data/simulated/candidate_pairs.parquet`
2. Trains the baselines (Rule-Based → LogReg)
3. Trains the XGBoost models (Regressor + Classifier)
4. Generates comparison tables and scatter plots
5. Visualizes the **detour vs. acceptance probability trade-off**

**Connection:** This is what you present to a professor or investor. The notebook is the visual "proof" that the entire pipeline works.

**Note:** With the 3km NYC-appropriate thresholds (now used in both dev and prod) and the multi-factor stochastic labeling, the models report realistic metrics rather than the trivially-separable 100% accuracy seen with the old wide-threshold dev config.

---

# Phase 6: The Bottleneck Fixes

## 6.1 — Why We Needed Fixes

The original code worked but had **23 issues** (22 original + #23) ranging from data leakage to performance problems. 18 have been fixed so far; 5 remain (#7, #17, #18, #20, #22).

**Full list:** `deep_bottleneck_analysis.md` (22 original + #23 discovered later)
**Detailed fix writeups:** `docs/09_resolved_bottleneck_analysis.md`

## 6.2 — The Resolved Fixes (Summary)

| # | What Was Wrong | How It Was Fixed | Why It Mattered |
|---|---------------|-----------------|----------------|
| **#2** | Negative samples had hardcoded fake features (`overlap=0.0`, `time=60.0`) | Compute real features from genuinely infeasible pairs | Model learned a trivial shortcut instead of real rejection patterns |
| **#1** | Labels were a deterministic function of input features (detour < 3km → label = 1, always) | Replaced with 4-factor composite + Gumbel noise | XGBoost reverse-engineered the threshold → 100% accuracy with zero generalization |
| **#8** | Bipartite graph edge weight was unbalanced (`overlap*10 - detour`) | Replaced with normalized acceptance probability | Negative weights caused matcher to avoid high-detour matches even when profitable |
| **#10** | LogReg used 4 features, XGBoost used 5 different features | Aligned onto a shared core set (later expanded to 9 for XGB / 10 for LogReg; XGB excludes the regression target) | Performance gap could be due to different features, not algorithmic advantage |
| **#11** | Regressor and classifier used separate train/test splits | Single unified split for both | Metrics were evaluated on different test rows — incomparable |
| **#14** | Negative detour produced overlap ratio > 1.0 | Clamped to [0, 1] | Values > 1.0 confused XGBoost's feature space |
| **#5** | Inner loops used `.iterrows()` (100-1000x slower than alternatives) | Replaced with `.itertuples()` | 5-10x speedup with zero behavioral change |
| **#4** | O(N²) rectangular bounding box for spatial pre-filtering | BallTree with Haversine metric — O(log N) per query | Correctly handles Earth curvature, finds true circular neighbors |
| **#6** | Dropoff coordinates synthesized with random angles | Corridor-aware forward azimuth from DOLocationID centroids + Gaussian noise | Random angles destroyed commute corridor patterns → unrealistically sparse matches |
| **#3** | Bipartite matching was used to filter training data (only "winning" pairs entered the dataset) | Training collects ALL feasible pairs; bipartite reserved for inference | Selection bias — model only saw the best pairs, never learned mediocre ones |
| **#23** | Pickup coordinates scattered across bounding box (rectangle) instead of actual zone polygon | Added point-in-polygon rejection sampling with shapely.contains() | Coastal zone points landed in water; critical for irregular suburban/rural polygons |
| **#15/#16** | Dead `calculate_haversine()` + redundant boolean return in `feature_eng.py` | Removed the function; `calculate_time_overlap()` now returns a single float | Cleaner, leaner feature-engineering module |
| **#12/#13/#19** | No cross-validation/tuning; missing ROC-AUC/RMSE/F1; no feature-importance plot | Added 5-fold CV + GridSearchCV, full metric suite, feature-importance plot, plus ablation, learning curves, and system-level bipartite evaluation | Trustworthy, interpretable, system-aware evaluation |

### Recommended reading order for the fixes:
1. **Fix #2** (Fabricated negatives) → Understand why fake data = fake accuracy
2. **Fix #1** (Label leakage) → Understand why deterministic labels = trivial problem
3. **Fix #6** (Corridor bearings) → Understand the azimuth math
4. **Fix #4** (BallTree) → Understand O(N²) vs O(N log N) and Haversine
5. **Fix #3** (Bipartite decoupling) → Understand training vs inference separation

## 6.3 — The Remaining Issues

These are the issues still open (mostly minor cleanup):

| # | Issue | Planned Fix |
|---|-------|------------|
| #7 | Dropoffs can land in the ocean (no polygon validation on dropoffs) | Same rejection sampling as pickups |
| #17 | `limit` parameter counts drivers, not pairs | Rename or fix semantics |
| #18 | `np.random.seed(42)` called multiple times | Consolidate to single RandomState |
| #20 | No `__init__.py` files in src/ packages | Add them + `setup.py` |
| #22 | Filter order wasteful (synthesizes coords before filtering invalid zones) | Filter first, then synthesize |

*(Previously-listed issues #9, #12, #13, #15, #16, #19, and #21 have since been resolved — see `docs/09_resolved_bottleneck_analysis.md`.)*

---

# Phase 7: The Startup Vision

## 7.1 — From Academic to Production

The academic project proves the **algorithm** works. The startup needs:
- A mobile app (iOS/Android) that captures real GPS
- A backend server running `xgb_matching.py` in real-time
- A **Bipartite Matching** layer that assigns optimal 1:1 driver-rider pairs
- Real-time retraining on actual accept/reject decisions

**Read:** `docs/07_startup_roadmap_and_future_work.md`

## 7.2 — Feature Abstraction: Why One Model Works Everywhere

The model trained on NYC data works **anywhere** because it uses **ratios** (overlap_ratio), not absolute coordinates.

**NYC Commuter:** 5km trip, 5km detour → overlap = 50% → REJECT
**Texas Commuter:** 80km trip, 10km detour → overlap = 89% → ACCEPT

Same model. Same weights. Different city. The human behavior ("was the money worth the hassle?") is mathematically universal.

**Read:** `docs/08_nationwide_scaling_architecture.md`

## 7.3 — The Three Americas Problem

The USA has three fundamentally different matching challenges:

| Tier | Example | Detour Tolerance | Search Radius | Batch Window |
|------|---------|-----------------|---------------|-------------|
| **Tier 1: Urban Dense** | Manhattan, SF, Chicago Loop | 3 km | 5 km | 30-60 sec |
| **Tier 2: Suburban** | Atlanta suburbs, Denver metro | 7-8 km | 15 km | 2 min |
| **Tier 3: Rural/Highway** | Maine, Texas I-10, Midwest | 20-25 km | 50 km | 5-15 min |

**Why Tier 3 allows 20km:** A 20km detour at 70 mph takes only 10.5 minutes. The detour must be measured in **time**, not just distance. Highway speeds make huge distances tolerable.

`search_radius_km`, `batch_window_mins`, and `driver_ratio` are already read from the YAML config (with sensible defaults). What the startup phase adds is **tier-specific** config files that swap these values per geographic density tier.

## 7.4 — Research Paper Connections

6 academic papers were analyzed. The key ones:

| Paper | What We Borrowed | What We Added |
|-------|-----------------|--------------|
| **Santi et al. (PNAS 2014)** | NYC taxi data + bipartite matching for ride-pooling | ML prediction instead of fixed thresholds; acceptance modeling |
| **Bjorck et al. (AAAI 2021)** | Formal proof that ML + bipartite provably outperforms either alone | Our specific dual-model (regression + classification) pipeline |
| **Anagnostopoulos (2025)** | Multi-context matching (spatial + temporal + behavioral) | Zone-level behavioral features (planned) |
| **Hsieh (2017)** | Trajectory-based matching with real routing | OSRM integration (startup phase) |

**Our novel contributions:**
1. Synthetic P2P data pipeline from single-passenger taxi trips (corridor-aware bearings + polygon validation)
2. Dual-model architecture (regression + classification → combined score)
3. Multi-factor stochastic labeling with Gumbel noise
4. Training ALL pairs → inference with bipartite (learning-augmented matching)
5. Feature abstraction for nationwide scaling without retraining

**Read:** `docs/11_research_paper_analysis.md`

---

# Phase 8: Complete Reading Order

## For Understanding the Code (Start to Finish)

| Step | What to Read | What You'll Learn | Time |
|------|-------------|-------------------|------|
| 1 | **This document** (you're here) | Big picture of everything | 30 min |
| 2 | `docs/01_project_architecture.md` | Folder structure and design philosophy | 5 min |
| 3 | `docs/02_tech_stack_justification.md` | Why Python, XGBoost, Parquet, Conda | 5 min |
| 4 | `docs/03_dataset_selection_and_cleaning.md` | NYC TLC data and cleaning strategy | 10 min |
| 5 | `config/dev_config.yaml` | What each YAML parameter controls | 5 min |
| 6 | `src/data/loader.py` | The actual data loading and coordinate synthesis code | 15 min |
| 7 | `docs/06_data_synthesis_and_generalization.md` | Full mathematical justification for the synthesis | 15 min |
| 8 | `src/simulation/feature_eng.py` | The detour cost, overlap ratio, and time overlap functions | 10 min |
| 9 | `docs/04_p2p_simulation_logic.md` | How the simulation pipeline works conceptually | 10 min |
| 10 | `src/simulation/pair_generator.py` | The core simulation engine code | 20 min |
| 11 | `docs/05_multi_objective_matching_model.md` | The dual-model theory and bipartite matching | 10 min |
| 12 | `src/models/baselines.py` | The rule-based and logistic regression baselines | 10 min |
| 13 | `src/models/xgb_matching.py` | The XGBoost dual-model code | 15 min |
| 14 | `notebooks/03_model_evaluation.ipynb` | Run and see the visual results | 15 min |
| 15 | `deep_bottleneck_analysis.md` | All 23 issues identified | 15 min |
| 16 | `docs/09_resolved_bottleneck_analysis.md` | How the resolved issues were fixed (with math and code) | 20 min |
| 17 | `docs/08_nationwide_scaling_architecture.md` | How the model scales to all of USA | 10 min |
| 18 | `docs/07_startup_roadmap_and_future_work.md` | Future product vision | 10 min |
| 19 | `docs/11_research_paper_analysis.md` | Academic positioning and citations | 15 min |

**Total estimated reading time: ~4 hours**

---

## Quick Reference — File Connections Map

```
config/dev_config.yaml               (All simulation parameters — thresholds, windows, ratios)
    │
    ▼
loader.py                            (Downloads & synthesizes GPS data)
    │
    ├── Downloads: TLC Parquet        (3.5M raw trips from AWS)
    ├── Downloads: Taxi Zone Shapes   (Official polygon boundaries)
    ├── Synthesizes: Pickup coords    (Bounding box + point-in-polygon validation)
    ├── Synthesizes: Dropoff coords   (Corridor-aware azimuth + spherical projection)
    ├── Filters: Bounding box         (Remove out-of-NYC coordinates)
    ├── Output: cleaned_*.parquet     (3.5M trips with realistic coordinates)
    │
    ▼
pair_generator.py                    (Creates Driver-Rider training pairs)
    │
    ├── Step 1: Temporal filter       (Keep AM 7-10 + PM 16-19 commute windows)
    ├── Step 2: Driver/Rider split    (50/50 random assignment)
    ├── Step 3: BallTree indexing      (O(log N) spatial neighbor search)
    ├── Step 4: Feature computation    (Calls feature_eng.py for each pair)
    ├── Step 5: Stochastic labeling    (4-factor composite + Gumbel noise)
    ├── Step 6: Negative sampling      (Real features from genuinely infeasible pairs)
    ├── Output: candidate_pairs.parquet (Balanced labeled pairs)
    │
    ▼
feature_eng.py                       (Pure math — no state, no side effects)
    │
    ├── calculate_detour_cost()       (Geodesic distance: new route - original route)
    ├── compute_spatial_overlap()     (Overlap ratio, clamped [0, 1])
    ├── calculate_time_overlap()      (Temporal difference in minutes)
    │
    ▼
baselines.py                         (Simple benchmark models)
    │
    ├── RuleBasedBaseline             (Hard threshold: detour < 3km AND time < 15min)
    ├── LogisticRegressionBaseline    (Linear classifier, 10 features, StandardScaler)
    ├── Output: Accuracy/F1 floor     (The minimum bar XGBoost must beat)
    │
    ▼
xgb_matching.py                     (THE core AI model)
    │
    ├── XGBRegressor                  (Predicts detour_km — regression target)
    ├── XGBClassifier                 (Predicts accept/reject — scale_pos_weight + grid search)
    ├── Driver-level train/test split (Split on driver_id; both models share the partition)
    ├── compute_match_score()         (Combines both predictions into a single rank)
    ├── Output: MAE, RMSE, Accuracy, F1, ROC-AUC + ablation/learning-curve/system figures
    │
    ▼
03_model_evaluation.ipynb            (Visual proof — the academic deliverable)
    │
    ├── Loads all models
    ├── Compares baselines vs XGBoost
    ├── Generates trade-off scatter plots (detour vs acceptance probability)
    └── Output: The final visualization for professor/investor
```
