# 03 - Dataset Selection and Cleaning

## The Spatial Coordinate Constraint
The Qmutez project's core matching engine relies on exact geographic mathematics (Haversine distance) to compute realistic multi-objective pairings. It requires exact `pickup_latitude` and `pickup_longitude` floats.

**Why the Post-COVID August 2025 Dataset?**
Originally, this project utilized legacy 2015 data because the NYC TLC deprecated exact GPS coordinates in July 2016, moving to generic "Taxi Zones." However, because our `loader.py` script now successfully *synthesizes* perfect geometric coordinates based on the real `trip_distance` variable, we are no longer restricted by this privacy wall. 

By upgrading to the **August 2025 dataset**, Qmutez trains on modern, highly relevant realities:
1.  **Economic Inflation:** Post-COVID taxi pricing provides a vastly more accurate `fare_amount` baseline to calculate matching incentives than 2015 prices.
2.  **Summer Volume:** August represents stable weather and peak tourism/activity, offering the densest and most reliable pool of commuter and casual trips to simulate P2P networking interactions.

**The Coordinate Synthesis Function:**
Even though the 2025 `.parquet` files only contain zone integers (`PULocationID`), the `src/data/loader.py` script actively detects this limitation. It utilizes the `geopandas` GIS library to map these Zones to their official Geographic Polygon Centroids (creating **Smart Pickup Hubs**). It then dynamically plots an exact drop-off coordinate executing a **Vectorized Spherical Haversine Projection** strictly scaled to the true `trip_distance` variable. This allows our Haversine algorithms to train correctly without compromising the modern economic reality of the ride.

## Cleaning Strategy (`src/data/loader.py`)
1.  **Coordinate Validation:** Drops any row where latitude or longitude equals `0.0` or `NaN`.
2.  **Geographic Bounding Boxes:** NYC coordinates must exist within standard Tri-State Area bounding boxes:
    - Latitude: `40.5` to `41.2`
    - Longitude: `-74.3` to `-73.7`
    This aggressively prevents simulation failures caused by anomalous GPS readings that place trips in the middle of the Atlantic Ocean.
3.  **Financial Integrity:** Drops rows with invalid `trip_distance` or negative `fare_amount` parameters, ensuring the baseline models learn true economic incentives.
