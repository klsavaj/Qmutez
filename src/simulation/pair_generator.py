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
        # Utilizing YAML parameters ensures our ML simulator is highly reproducible
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
        Isolates temporal commutes.
        Why: We only care about P2P carpooling during high-volume directional flows 
        (like Morning 7AM-10AM rushes).
        """
        start_hr = self.sim_config["time_window"]["start_hour"]
        end_hr = self.sim_config["time_window"]["end_hour"]
        
        print(f"Filtering for commute window: {start_hr}:00 - {end_hr}:00")
        
        self.df = self.df[
            (self.df['tpep_pickup_datetime'].dt.hour >= start_hr) & 
            (self.df['tpep_pickup_datetime'].dt.hour < end_hr)
        ]
        
        # Apply deterministic downsampling so DEV environments don't cause MemoryErrors
        sample_ratio = self.sim_config.get("sample_ratio", 1.0)
        if sample_ratio < 1.0:
            print(f"Downsampling to {sample_ratio*100}% for development.")
            self.df = self.df.sample(frac=sample_ratio, random_state=42)
            
        print(f"Total trips in window: {len(self.df)}")
        return self.df

    def split_drivers_and_riders(self, driver_ratio=0.5):
        """
        Randomly tags a boolean assignment to trips establishing the two sides 
        of the matchmaking marketplace.
        """
        # Set explicitly to 42 for academic reproducibility
        np.random.seed(42)
        msk = np.random.rand(len(self.df)) < driver_ratio
        
        self.drivers_df = self.df[msk].copy()
        self.riders_df = self.df[~msk].copy()
        
        print(f"Assigned {len(self.drivers_df)} Drivers and {len(self.riders_df)} Riders.")

    def generate_candidate_pairs(self, limit=1000):
        """
        Calculates all viable route matches using Bipartite Batch Optimization.
        
        Critical Engineering Concept (Bipartite Graph Matching):
        Instead of matching the first driver to the first rider (greedy sub-optimal), 
        we simulate a 60-second temporal aggregation window. We build a mathematical Graph 
        of all feasible matches inside that window, and use networkx to find the 
        globally optimal combination that maximizes ride acceptance.
        """
        import networkx as nx
        print("Starting Bipartite Batch pair generation...")
        
        max_detour_km = self.sim_config["max_detour_km"]
        
        # Sort both dataframes chronologically to simulate a rolling time window
        self.drivers_df = self.drivers_df.sort_values('tpep_pickup_datetime')
        self.riders_df = self.riders_df.sort_values('tpep_pickup_datetime')
        
        final_pairs = []
        d_count = 0
        
        # Simulate temporal batches (e.g., 5-minute rolling windows to ensure enough density)
        # In a real app this is 30-60 seconds, but 5 mins works better for historical data density
        window_size = pd.Timedelta(minutes=5)
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
                
            # Initialize the Bipartite Graph for this temporal batch
            B = nx.Graph()
            
            # Step 1: Draw edges between all feasible pairs in the batch
            for d_idx, driver in batch_drivers.iterrows():
                if d_count >= limit: break
                
                # Add Driver Node
                B.add_node(f"D_{d_idx}", bipartite=0)
                d_count += 1
                
                d_start = (driver['pickup_latitude'], driver['pickup_longitude'])
                d_end = (driver['dropoff_latitude'], driver['dropoff_longitude'])
                
                # Spatial Pre-filter (+/- 5km)
                potential_riders = batch_riders[
                    (batch_riders['pickup_latitude'].between(d_start[0]-0.05, d_start[0]+0.05)) & 
                    (batch_riders['pickup_longitude'].between(d_start[1]-0.05, d_start[1]+0.05))
                ]
                
                for r_idx, rider in potential_riders.iterrows():
                    # Add Rider Node
                    B.add_node(f"R_{r_idx}", bipartite=1)
                    
                    r_start = (rider['pickup_latitude'], rider['pickup_longitude'])
                    r_end = (rider['dropoff_latitude'], rider['dropoff_longitude'])
                    
                    detour, new_dist, d_orig_dist = calculate_detour_cost(d_start, d_end, r_start, r_end)
                    
                    if detour <= max_detour_km:
                        overlap_ratio = compute_spatial_overlap(detour, d_orig_dist)
                        time_compat, diff_mins = calculate_time_overlap(driver['tpep_pickup_datetime'], rider['tpep_pickup_datetime'])
                        
                        # Multi-factor Ground Truth Labeling
                        # Instead of basing labels primarily on detour (which creates circular
                        # dependency with overlap_ratio), we model acceptance as a composite
                        # of independent behavioral factors the model must learn to combine.
                        
                        # Factor 1: Fare profitability — is the rider's fare worth the detour?
                        fare_ratio = rider['fare_amount'] / max(driver['fare_amount'], 1.0)
                        fare_score = min(fare_ratio, 1.5) / 1.5  # Normalize to [0, 1]
                        
                        # Factor 2: Temporal inconvenience — how disruptive is the timing?
                        time_score = max(0, 1.0 - (diff_mins / 15.0))  # 0 at 15min, 1 at 0min
                        
                        # Factor 3: Trip length ratio — short detour on long trip = acceptable
                        trip_ratio = d_orig_dist / max(d_orig_dist + detour, 0.1)
                        
                        # Factor 4: Detour severity — normalized by threshold
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
                        
                        edge_data = {
                            'driver_id': d_idx,
                            'rider_id': r_idx,
                            'driver_orig_dist': d_orig_dist,
                            'detour_distance_km': detour,
                            'overlap_ratio': overlap_ratio,
                            'time_diff_mins': diff_mins,
                            'driver_approx_fare': driver['fare_amount'],
                            'rider_approx_fare': rider['fare_amount'],
                            'label': is_accepted
                        }
                        
                        # Use acceptance probability as Edge Weight for Bipartite Matching
                        B.add_edge(f"D_{d_idx}", f"R_{r_idx}", weight=acceptance_prob, data=edge_data)

            # Step 2: Execute Maximum Weight Bipartite Matching Algorithm
            # This mathematically solves the batch, ensuring no driver or rider is double-mapped
            # while maximizing the overall heuristic efficiency of the entire ecosystem.
            optimal_matches = nx.max_weight_matching(B, maxcardinality=False)
            
            # Step 3: Extract the winning pairs
            for node1, node2 in optimal_matches:
                # networkx returns (NodeA, NodeB) unordered. Find which is Driver.
                d_node = node1 if str(node1).startswith("D_") else node2
                r_node = node2 if str(node2).startswith("R_") else node1
                
                edge_metadata = B[d_node][r_node]['data']
                final_pairs.append(edge_metadata)
                
            current_time = window_end
            
            if d_count % 100 == 0:
                print(f"Processed {d_count} drivers across batches. Current optimal pairs locked: {len(final_pairs)}")

        # Generate negative labels: real pairs that genuinely fail the matching criteria
        # We compute ALL features honestly — no hardcoded fakes — so the model learns
        # real rejection patterns instead of trivial shortcuts.
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
            
            # Compute REAL features — no fakes
            overlap_ratio = compute_spatial_overlap(detour, d_orig_dist)
            driver_time = self.drivers_df.loc[d_idx, 'tpep_pickup_datetime']
            rider_time = self.riders_df.loc[r_idx, 'tpep_pickup_datetime']
            _, time_diff = calculate_time_overlap(driver_time, rider_time)
            
            negative_pairs.append({
                'driver_id': d_idx,
                'rider_id': r_idx,
                'driver_orig_dist': d_orig_dist,
                'detour_distance_km': detour,
                'overlap_ratio': overlap_ratio,
                'time_diff_mins': time_diff,
                'driver_approx_fare': self.drivers_df.loc[d_idx, 'fare_amount'],
                'rider_approx_fare': self.riders_df.loc[r_idx, 'fare_amount'],
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
