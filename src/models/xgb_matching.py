import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

class MultiObjectiveMatcher:
    """
    Advanced Gradient Boosted Model (XGBoost) for Multi-Objective Matching.
    
    Why this class is the core of the Qmutez Application:
    Peer-to-peer carpooling is not a single problem. You must predict how heavy the detour 
    will physically be (Regression), and then calculate the psychological intent of 
    whether the driver will accept that specific detour given the fare (Classification).
    
    XGBoost is utilized because tabular coordinate geometry heavily requires non-linear 
    decision trees (which XGBoost calculates natively faster than deep learning models).
    """
    def __init__(self):
        # Sub-Model A: Predicts physical distance deviation
        self.regressor = xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5
        )
        
        # Sub-Model B: Predicts boolean acceptance logic
        self.classifier = xgb.XGBClassifier(
            objective='binary:logistic', 
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5,
            eval_metric='logloss'
        )
        
        # The exact parameter columns the tree utilizes to compute splits
        self.features = ['driver_orig_dist', 'overlap_ratio', 'time_diff_mins', 'driver_approx_fare', 'rider_approx_fare']
        
    def prepare_data(self, df):
        X = df[self.features]
        y_reg = df['detour_distance_km'] # Target 1 (Continuous)
        y_clf = df['label'] # Target 2 (Categorical/Boolean)
        return X, y_reg, y_clf

    def train(self, df):
        """
        Executes dual-layer algorithm training ensuring strict Train/Test boundary isolations.
        """
        X, y_reg, y_clf = self.prepare_data(df)
        print(f"Training on {len(X)} samples with {len(self.features)} features.")
        
        # Train Detour Regressor
        X_train, X_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        self.regressor.fit(X_train, yr_train)
        pred_detour = self.regressor.predict(X_test)
        mae = mean_absolute_error(yr_test, pred_detour)
        print(f"Regression MAE (Detour km): {mae:.2f}")

        # Train Match Probability Classifier
        X_train, X_test, yc_train, yc_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
        self.classifier.fit(X_train, yc_train)
        pred_match = self.classifier.predict(X_test)
        acc = accuracy_score(yc_test, pred_match)
        print(f"Classification Accuracy: {acc:.2f}")
        print("Classification Report:")
        print(classification_report(yc_test, pred_match, zero_division=0))
        
        return X_test, yr_test, yc_test, pred_detour, pred_match

    def compute_match_score(self, feature_row):
        """
        Translates raw computational output into the single 'Ranking Score' string
        used by the startup's matchmaking server queue.
        
        Logic: Combine predictions into a singular integer representation. Less detour 
        increases ranking mathematically, while higher probability increases it logarithmically.
        """
        # Formulate pandas DataFrame single entry safely
        df_feats = pd.DataFrame([feature_row], columns=self.features)
        
        predicted_detour = self.regressor.predict(df_feats)[0]
        match_probability = self.classifier.predict_proba(df_feats)[0][1] # Pull exact float percentage
        
        # Strict algorithmic penalty system:
        # Subtracting heavily weighted detour values directly from probabilistic intent bounds.
        score = match_probability - (predicted_detour * 0.1) 
        return score

if __name__ == "__main__":
    try:
        print("Loading simulated candidate pairs...")
        df = pd.read_parquet("data/simulated/candidate_pairs.parquet")
        matcher = MultiObjectiveMatcher()
        matcher.train(df)
    except FileNotFoundError:
        print("Run the pair generator before training the XGBoost model.")
