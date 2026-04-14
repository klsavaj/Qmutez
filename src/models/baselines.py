import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

class RuleBasedBaseline:
    """
    A simple zero-intelligence benchmarking model.
    
    Why it is needed:
    A machine learning model must mathematically justify its own existence. This class 
    simulates a 'dumb' app that simply accepts any ride strictly under X km. 
    It provides our absolute minimum performance floor.
    """
    def __init__(self, max_km=3.0, max_mins=15.0):
        self.max_km = max_km
        self.max_mins = max_mins
        
    def predict(self, df):
        """Returns 1 (accepted) or 0 (rejected) explicitly based on rigid cutoffs."""
        predictions = (
            (df['detour_distance_km'] <= self.max_km) & 
            (df['time_diff_mins'] <= self.max_mins)
        ).astype(int)
        return predictions
    
    def evaluate(self, y_true, y_pred):
        """Exposes raw viability metrics for comparisons."""
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0)
        }

class LogisticRegressionBaseline:
    """
    Standard Linear Prediction Baseline.
    
    Why it is needed:
    While RuleBased gives us the logical floor, LogisticRegression provides 
    the statistical floor. Logistic Regression struggles with non-linear feature 
    spaces (like the relationship between spatial overlaps and time), proving 
    why we eventually graduate to XGBoost Decision Trees.
    """
    def __init__(self):
        # Balanced prevents class-imbalance from zero-crushing predictions
        self.model = LogisticRegression(class_weight='balanced')
        # Scaling is required for distance math, otherwise KM vastly outweighs Ratio logic
        self.scaler = StandardScaler()
        # Explicit feature definition
        self.features = ['driver_orig_dist', 'detour_distance_km', 'overlap_ratio', 'time_diff_mins', 'driver_approx_fare', 'rider_approx_fare', 'fare_ratio', 'pickup_hour', 'pickup_dow']
        
    def prepare_data(self, df):
        X = df[self.features]
        y = df['label'] 
        return X, y
        
    def train_and_evaluate(self, df):
        """Pipelines splitting, scaling, and scoring into one traceable execution layer."""
        X, y = self.prepare_data(df)
        
        # 80/20 train test split locked randomly for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Prevent Data Leakage by fitting standard scaler exclusively on training subset
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dev_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    sim = config["simulation"]

    try:
        print("Loading simulated pairs...")
        df = pd.read_parquet("data/simulated/candidate_pairs.parquet")

        print("\n--- Testing Rule-Based Baseline ---")
        rb = RuleBasedBaseline(
            max_km=sim["max_detour_km"],
            max_mins=sim["max_detour_mins"]
        )
        y_pred_rb = rb.predict(df)
        print(rb.evaluate(df['label'], y_pred_rb))

        print("\n--- Testing Logistic Regression Baseline ---")
        lr = LogisticRegressionBaseline()
        lr_metrics = lr.train_and_evaluate(df)
        print(lr_metrics)

    except FileNotFoundError:
        print("Required datasets not found. Run simulation pipeline first.")
