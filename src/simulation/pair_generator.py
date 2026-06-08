import pandas as pd
import numpy as np
import yaml

from src.simulation.feature_eng import calculate_detour_cost, compute_spatial_overlap, calculate_time_overlap

class PairGenerator:
    """
    Core Machine Learning Simulator Engine.
    
    Why this class is needed:
    Real-world open source datasets containing completed P2P carpools do not exist.
    This class algorithmically forces single-passenger NYC taxi rides into overlapping 
    Commuter ('Driver') and Requestor ('Rider') pairings, engineered specifically 
    to train our XGBoost models on acceptable vs unacceptable geographic detours.
    """
    def __init__(self, processed_data_path, config_path="config/dev_config.yaml"):
        # Utilizing YAML parameters 
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        print(f"Loading data from {processed_data_path}...")
        self.df = pd.read_parquet(processed_data_path)
        
        # Standardize date types for mathematical time-window comparisons
        self.df['tpep_pickup_datetime'] = pd.to_datetime(self.df['tpep_pickup_datetime'])
        self.df['tpep_dropoff_datetime'] = pd.to_datetime(self.df['tpep_dropoff_datetime'])

        self.sim_config = self.config["simulation"]
        
    def filter_commuting_window(self):
        """
        Isolates commute windows. Supports multiple windows (AM + PM rush hours)
        via the time_windows list in config, or falls back to a single time_window.
        """
        windows = self.sim_config.get("time_windows")

        if windows:
            mask = pd.Series(False, index=self.df.index)
            labels = []
            for w in windows:
                mask |= (
                    (self.df['tpep_pickup_datetime'].dt.hour >= w['start_hour']) &
                    (self.df['tpep_pickup_datetime'].dt.hour < w['end_hour'])
                )
                labels.append(f"{w['start_hour']}:00-{w['end_hour']}:00")
            self.df = self.df[mask]
            print(f"Filtering for commute windows: {' + '.join(labels)}")
        else:
            # backward compatibility: single time_window key
            start_hr = self.sim_config["time_window"]["start_hour"]
            end_hr   = self.sim_config["time_window"]["end_hour"]
            print(f"Filtering for commute window: {start_hr}:00 - {end_hr}:00")
            self.df = self.df[
                (self.df['tpep_pickup_datetime'].dt.hour >= start_hr) &
                (self.df['tpep_pickup_datetime'].dt.hour < end_hr)
            ]
        
        # Apply deterministic downsampling 
        sample_ratio = self.sim_config.get("sample_ratio", 1.0)
        if sample_ratio < 1.0:
            print(f"Downsampling to {sample_ratio*100}% for development.")
            self.df = self.df.sample(frac=sample_ratio, random_state=42)
            
        print(f"Total trips in window: {len(self.df)}")
        return self.df

    def split_drivers_and_riders(self, driver_ratio=None):
        """
        Randomly tags a boolean assignment to trips establishing the two sides 
        of the matchmaking marketplace.
        """
        if driver_ratio is None:
            driver_ratio = self.sim_config.get("driver_ratio", 0.5)
        np.random.seed(42)
        msk = np.random.rand(len(self.df)) < driver_ratio
        
        self.drivers_df = self.df[msk].copy()
        self.riders_df = self.df[~msk].copy()
        
        print(f"Assigned {len(self.drivers_df)} Drivers and {len(self.riders_df)} Riders.")

    def generate_candidate_pairs(self, limit=1000):
        """
        Generates training data by collecting ALL feasible driver-rider pairs.
        
        Design Principle (Training vs Inference separation):
        During training, we collect every pair within the detour threshold and assign
        labels independently via multi-factor stochastic scoring. This gives the model
        the full distribution of feasible matches — both great and mediocre.
        
        At inference time (production), the trained model's predictions serve as edge
        weights in a Bipartite Graph, where networkx.max_weight_matching selects the
        globally optimal 1:1 assignments. This separation ensures the model learns
        general acceptance patterns, not just optimal-pair patterns.
        """
        import networkx as nx  # Retained for production inference (bipartite matching with ML weights)
        print("Starting candidate pair generation...")
        
        max_detour_km = self.sim_config["max_detour_km"]
        
        # Sort both dataframes chronologically to simulate a rolling time window
        self.drivers_df = self.drivers_df.sort_values('tpep_pickup_datetime')
        self.riders_df = self.riders_df.sort_values('tpep_pickup_datetime')
        
        final_pairs = []
        d_count = 0
        
        batch_window_mins = self.sim_config.get("batch_window_mins", 5)
        window_size = pd.Timedelta(minutes=batch_window_mins)
        start_time = self.drivers_df['tpep_pickup_datetime'].min()
        end_time = self.drivers_df['tpep_pickup_datetime'].max()
        
        current_time = start_time
        
        while current_time < end_time and d_count < limit:
            window_end = current_time + window_size
            
            # Extract the discrete batch of users who requested a ride in this 5-minute window
            batch_drivers = self.drivers_df[
                (self.drivers_df['tpep_pickup_datetime'] >= current_time) & 
                (self.drivers_df['tpep_pickup_datetime'] < window_end)
            ]
            
            batch_riders = self.riders_df[
                (self.riders_df['tpep_pickup_datetime'] >= current_time) & 
                (self.riders_df['tpep_pickup_datetime'] < window_end)
            ]
            
            if batch_drivers.empty or batch_riders.empty:
                current_time = window_end
                continue
            
            # Build a BallTree spatial index from rider pickup coordinates (once per batch).
            # Uses Haversine metric on the unit sphere — requires radians.
            # Query complexity: O(log N) per driver instead of O(N) linear scan.
            from sklearn.neighbors import BallTree
            rider_coords_rad = np.radians(
                batch_riders[['pickup_latitude', 'pickup_longitude']].values
            )
            tree = BallTree(rider_coords_rad, metric='haversine')
            EARTH_RADIUS_KM = 6371.0
            SEARCH_RADIUS_KM = self.sim_config.get("search_radius_km", 5.0)
            
            # Collect ALL feasible pairs in this batch — not just bipartite-optimal ones.
            # Why: Training data must represent the full distribution of feasible matches
            # so the model learns to score mediocre pairs low AND great pairs high.
            # Bipartite matching is an INFERENCE-TIME optimization (using the trained
            # model's predictions as edge weights), not a training-time data filter.
            for driver in batch_drivers.itertuples():
                if d_count >= limit: break
                d_idx = driver.Index
                d_count += 1
                
                d_start = (driver.pickup_latitude, driver.pickup_longitude)
                d_end = (driver.dropoff_latitude, driver.dropoff_longitude)
                
                # Spatial pre-filter: query BallTree for riders within 5km radius
                d_start_rad = np.radians([[d_start[0], d_start[1]]])
                neighbor_indices = tree.query_radius(
                    d_start_rad, r=SEARCH_RADIUS_KM / EARTH_RADIUS_KM
                )[0]
                
                if len(neighbor_indices) == 0:
                    continue

                # shareability_count: how many riders are within search radius of this driver.
                # High count = driver is in a dense corridor (Santi et al., PNAS 2014).
                shareability_count = len(neighbor_indices)
                potential_riders = batch_riders.iloc[neighbor_indices]
                
                for rider in potential_riders.itertuples():
                    r_idx = rider.Index
                    
                    r_start = (rider.pickup_latitude, rider.pickup_longitude)
                    r_end = (rider.dropoff_latitude, rider.dropoff_longitude)
                    
                    detour, new_dist, d_orig_dist = calculate_detour_cost(d_start, d_end, r_start, r_end)
                    
                    if detour <= max_detour_km:
                        overlap_ratio = compute_spatial_overlap(detour, d_orig_dist)
                        diff_mins = calculate_time_overlap(driver.tpep_pickup_datetime, rider.tpep_pickup_datetime)
                        
                        # Multi-factor Ground Truth Labeling
                        # Models acceptance as a composite of independent behavioral
                        # factors the ML model must learn to combine.
                        
                        # Factor 1: Fare profitability, is the rider's fare worth the detour?
                        fare_ratio = rider.fare_amount / max(driver.fare_amount, 1.0)
                        fare_score = min(fare_ratio, 1.5) / 1.5  # Normalize to [0, 1]
                        
                        # Factor 2: Temporal inconvenience, how disruptive is the timing?
                        time_score = max(0, 1.0 - (diff_mins / 15.0))  # 0 at 15min, 1 at 0min
                        
                        # Factor 3: Trip length ratio, short detour on long trip = acceptable
                        trip_ratio = d_orig_dist / max(d_orig_dist + detour, 0.1)
                        
                        # Factor 4: Detour severity, normalized by threshold
                        detour_score = 1.0 - (detour / max_detour_km)
                        
                        # Composite acceptance probability with weighted factors
                        acceptance_prob = (
                            0.30 * detour_score +    # Physical inconvenience
                            0.25 * fare_score +      # Economic incentive  
                            0.25 * time_score +      # Temporal compatibility
                            0.20 * trip_ratio         # Relative effort
                        )
                        
                        # Add Gumbel noise to simulate human irrationality
                        noise = np.random.gumbel(0, 0.1)
                        is_accepted = 1 if (acceptance_prob + noise) > 0.45 else 0
                        
                        # Pre-computed interaction features — hand the model direct signals
                        # instead of making it discover the ratio relationship through splits
                        fare_ratio = rider.fare_amount / max(driver.fare_amount, 1.0)
                        pickup_hour = driver.tpep_pickup_datetime.hour
                        pickup_dow  = driver.tpep_pickup_datetime.weekday()

                        final_pairs.append({
                            'driver_id': d_idx,
                            'rider_id': r_idx,
                            'driver_orig_dist': d_orig_dist,
                            'detour_distance_km': detour,
                            'overlap_ratio': overlap_ratio,
                            'time_diff_mins': diff_mins,
                            'driver_approx_fare': driver.fare_amount,
                            'rider_approx_fare': rider.fare_amount,
                            'fare_ratio': fare_ratio,
                            'pickup_hour': pickup_hour,
                            'pickup_dow': pickup_dow,
                            'shareability_count': shareability_count,
                            'label': is_accepted
                        })
                
            current_time = window_end
            
            if d_count % 100 == 0:
                print(f"Processed {d_count} drivers across batches. Current feasible pairs: {len(final_pairs)}")

        # Generate negative labels: real pairs that genuinely fail the matching criteria
        print("Generating negative labels from genuinely infeasible pairs...")
        rng = np.random.RandomState(42)
        negative_pairs = []
        target_negatives = len(final_pairs)  # Balance the dataset ~50/50
        attempts = 0
        max_attempts = target_negatives * 10  # Safety valve
        
        while len(negative_pairs) < target_negatives and attempts < max_attempts:
            attempts += 1
            d_idx = self.drivers_df.sample(random_state=rng).index[0]
            r_idx = self.riders_df.sample(random_state=rng).index[0]
            
            d_start = (self.drivers_df.loc[d_idx, 'pickup_latitude'], self.drivers_df.loc[d_idx, 'pickup_longitude'])
            d_end = (self.drivers_df.loc[d_idx, 'dropoff_latitude'], self.drivers_df.loc[d_idx, 'dropoff_longitude'])
            r_start = (self.riders_df.loc[r_idx, 'pickup_latitude'], self.riders_df.loc[r_idx, 'pickup_longitude'])
            r_end = (self.riders_df.loc[r_idx, 'dropoff_latitude'], self.riders_df.loc[r_idx, 'dropoff_longitude'])
            
            detour, new_dist, d_orig_dist = calculate_detour_cost(d_start, d_end, r_start, r_end)
            
            # Only keep pairs that genuinely exceed the detour threshold
            if detour <= max_detour_km:
                continue
            
            # Compute REAL features
            overlap_ratio = compute_spatial_overlap(detour, d_orig_dist)
            driver_time = self.drivers_df.loc[d_idx, 'tpep_pickup_datetime']
            rider_time = self.riders_df.loc[r_idx, 'tpep_pickup_datetime']
            time_diff = calculate_time_overlap(driver_time, rider_time)
            
            d_fare = self.drivers_df.loc[d_idx, 'fare_amount']
            r_fare = self.riders_df.loc[r_idx, 'fare_amount']
            negative_pairs.append({
                'driver_id': d_idx,
                'rider_id': r_idx,
                'driver_orig_dist': d_orig_dist,
                'detour_distance_km': detour,
                'overlap_ratio': overlap_ratio,
                'time_diff_mins': time_diff,
                'driver_approx_fare': d_fare,
                'rider_approx_fare': r_fare,
                'fare_ratio': r_fare / max(d_fare, 1.0),
                'pickup_hour': driver_time.hour,
                'pickup_dow': driver_time.weekday(),
                'shareability_count': 0,  # random pair, not from BallTree batch
                'label': 0
            })
        
        print(f"Generated {len(negative_pairs)} negative pairs from {attempts} attempts.")
            
        results_df = pd.DataFrame(final_pairs + negative_pairs)
        print(f"Generated {len(results_df)} total labeled pairs (Balanced 1s and 0s).")
        
        # Save output for offline baseline testing
        out_path = "data/simulated/candidate_pairs.parquet"
        results_df.to_parquet(out_path, index=False)
        print(f"Saved to {out_path}")
        
        return results_df

if __name__ == "__main__":
    try:
        # Standard entrypoint targeting the explicit 2025 dataset
        sim = PairGenerator("data/processed/cleaned_yellow_tripdata_2025-08.parquet", "config/dev_config.yaml")
        sim.filter_commuting_window()
        sim.split_drivers_and_riders()
        sim.generate_candidate_pairs(limit=500)
    except FileNotFoundError:
        print("Processed data not found. Please run src/data/loader.py first.")
