# 03 - Dataset Selection and Cleaning

## The Spatial Coordinate Constraint
The Qmutez project's core matching engine relies on exact geographic mathematics (geodesic distance, via `geopy.distance.geodesic`) to compute realistic multi-objective pairings. It requires exact `pickup_latitude` and `pickup_longitude` floats.

**Why the Post-COVID August 2025 Dataset?**
Originally, this project utilized legacy 2015 data because the NYC TLC deprecated exact GPS coordinates in July 2016, moving to generic "Taxi Zones." However, because our `loader.py` script now successfully *synthesizes* perfect geometric coordinates based on the real `trip_distance` variable, we are no longer restricted by this privacy wall. 

By upgrading to the **August 2025 dataset**, Qmutez trains on modern, highly relevant realities:
1.  **Economic Inflation:** Post-COVID taxi pricing provides a vastly more accurate `fare_amount` baseline to calculate matching incentives than 2015 prices.
2.  **Summer Volume:** August represents stable weather and peak tourism/activity, offering the densest and most reliable pool of commuter and casual trips to simulate P2P networking interactions.

**The Coordinate Synthesis Function:**
Even though the 2025 `.parquet` files only contain zone integers (`PULocationID` / `DOLocationID`), the `src/data/loader.py` script actively detects this limitation. It utilizes the `geopandas` GIS library to map these Zones into their exact **Zonal Bounding Box Distribution**. Rather than clustering all trips at the exact center (which creates artificial density), pickups are scattered uniformly across the true geographic boundary of their respective zones, drastically improving simulation realism, especially for edge cases. A subsequent **Point-in-Polygon Refinement** step uses vectorized `shapely.contains()` to validate that every synthesized coordinate falls within the actual zone polygon — any points that land outside (e.g., in water for coastal zones or across a boundary for irregular shapes) are automatically resampled via rejection sampling. This ensures geographic validity even for highly irregular zone geometries, making the synthesis robust for nationwide scaling beyond NYC. It then computes **corridor-aware bearings** using the forward azimuth from pickup to dropoff zone centroids, and projects an exact drop-off coordinate via **Spherical Haversine Projection** strictly scaled to the true `trip_distance` variable. Gaussian noise (σ=0.3 rad ≈ 17°) adds street-level variation while preserving the commute corridor direction. This allows our geodesic distance algorithms to train correctly on geographically realistic commute patterns without compromising the modern economic reality of the ride.

## Cleaning Strategy (`src/data/loader.py`)
1.  **Geographic Bounding Boxes:** After coordinate synthesis, NYC pickup coordinates must fall within standard Tri-State Area bounding boxes (which implicitly removes any `0.0`, `NaN`, or out-of-range values):
    - Latitude: `40.5` to `41.2`
    - Longitude: `-74.3` to `-73.7`
    This aggressively prevents simulation failures caused by anomalous GPS readings that place trips in the middle of the Atlantic Ocean.
2.  **Financial Integrity:** Drops rows with invalid `trip_distance` or negative `fare_amount` parameters, ensuring the baseline models learn true economic incentives.
