import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, accuracy_score,
    classification_report, f1_score, roc_auc_score, roc_curve
)

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
        self.features = ['driver_orig_dist', 'overlap_ratio', 'time_diff_mins', 'driver_approx_fare', 'rider_approx_fare', 'fare_ratio', 'pickup_hour', 'pickup_dow']
        
    def prepare_data(self, df):
        X = df[self.features]
        y_reg = df['detour_distance_km'] # Target 1 (Continuous)
        y_clf = df['label'] # Target 2 (Categorical/Boolean)
        return X, y_reg, y_clf

    def train(self, df):
        """
        Executes dual-layer algorithm training with a SINGLE unified Train/Test split.
        Both models evaluate on the exact same test set for consistent scoring.
        Includes 5-fold cross-validation and hyperparameter tuning via grid search.
        """
        X, y_reg, y_clf = self.prepare_data(df)
        print(f"Training on {len(X)} samples with {len(self.features)} features.")

        # Driver-level train/test split to prevent entity leakage.
        # Splitting by pair (random) allows the same driver to appear in both
        # train and test, inflating test accuracy. Splitting by driver_id ensures
        # the model never sees a test driver during training.
        unique_drivers = df['driver_id'].unique()
        train_drivers, test_drivers = train_test_split(
            unique_drivers, test_size=0.2, random_state=42
        )
        train_mask = df['driver_id'].isin(train_drivers)
        train_df = df[train_mask].reset_index(drop=True)
        test_df  = df[~train_mask].reset_index(drop=True)

        X_train = train_df[self.features]
        X_test  = test_df[self.features]
        yr_train = train_df['detour_distance_km']
        yr_test  = test_df['detour_distance_km']
        yc_train = train_df['label']
        yc_test  = test_df['label']

        print(f"Driver-level split: {len(train_df)} train pairs, {len(test_df)} test pairs.")
        print(f"No driver_id overlap between train and test: "
              f"{len(set(train_df['driver_id']) & set(test_df['driver_id'])) == 0}")

        # --- Hyperparameter tuning via grid search on the classifier ---
        print("\nRunning grid search for classifier hyperparameters...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        }
        grid_search = GridSearchCV(
            xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss'),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, yc_train)
        best_params = grid_search.best_params_
        print(f"Best classifier params: {best_params}")

        # Compute scale_pos_weight to handle class imbalance (mirrors LogReg class_weight='balanced')
        pos_count = (yc_train == 1).sum()
        neg_count = (yc_train == 0).sum()
        scale_pos_weight = neg_count / max(pos_count, 1)
        print(f"Class balance — pos: {pos_count}, neg: {neg_count}, scale_pos_weight: {scale_pos_weight:.3f}")

        # Rebuild both models with tuned params
        self.classifier = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            **best_params
        )
        self.regressor = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate']
        )

        # --- 5-fold cross-validation on training set ---
        print("\nRunning 5-fold cross-validation...")
        cv_clf_scores = cross_val_score(self.classifier, X_train, yc_train, cv=5, scoring='accuracy')
        cv_reg_scores = cross_val_score(self.regressor, X_train, yr_train, cv=5, scoring='neg_mean_absolute_error')

        print(f"Classifier CV Accuracy: {cv_clf_scores.mean():.4f} (+/- {cv_clf_scores.std():.4f})")
        print(f"  Fold scores: {np.round(cv_clf_scores, 4)}")
        print(f"Regressor CV MAE: {-cv_reg_scores.mean():.4f} (+/- {cv_reg_scores.std():.4f})")
        print(f"  Fold scores: {np.round(-cv_reg_scores, 4)}")

        # --- Final training on full train set with best params ---
        print("\nTraining final models with best hyperparameters...")
        self.regressor.fit(X_train, yr_train)
        pred_detour = self.regressor.predict(X_test)
        mae = mean_absolute_error(yr_test, pred_detour)
        print(f"Regression MAE (Detour km): {mae:.2f}")

        self.classifier.fit(X_train, yc_train)
        pred_match = self.classifier.predict(X_test)
        pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        acc = accuracy_score(yc_test, pred_match)
        print(f"Classification Accuracy: {acc:.2f}")
        print("Classification Report:")
        print(classification_report(yc_test, pred_match, zero_division=0))

        # --- Additional metrics ---
        rmse = np.sqrt(mean_squared_error(yr_test, pred_detour))
        f1 = f1_score(yc_test, pred_match, zero_division=0)
        roc_auc = roc_auc_score(yc_test, pred_proba)
        print(f"\nRegression RMSE (Detour km): {rmse:.4f}")
        print(f"Classification F1: {f1:.4f}")
        print(f"Classification ROC-AUC: {roc_auc:.4f}")

        # --- Save plots ---
        os.makedirs("figures", exist_ok=True)

        # Feature importance
        fig, ax = plt.subplots(figsize=(8, 5))
        xgb.plot_importance(self.classifier, ax=ax, importance_type='gain')
        ax.set_title("Feature Importance (Classifier — Gain)")
        fig.tight_layout()
        fig.savefig("figures/feature_importance.png", dpi=150)
        plt.close(fig)
        print("Saved figures/feature_importance.png")

        # ROC curve
        fpr, tpr, _ = roc_curve(yc_test, pred_proba)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Match Acceptance Classifier")
        ax.legend()
        fig.tight_layout()
        fig.savefig("figures/roc_curve.png", dpi=150)
        plt.close(fig)
        print("Saved figures/roc_curve.png")

        # --- Feature ablation study ---
        # Remove each feature one at a time, retrain, measure accuracy drop.
        # The bigger the drop, the more important the feature.
        print("\nRunning feature ablation study...")
        ablation_results = {}
        for feat in self.features:
            ablation_feats = [f for f in self.features if f != feat]
            clf_ablation = xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='logloss',
                scale_pos_weight=scale_pos_weight, **best_params
            )
            clf_ablation.fit(X_train[ablation_feats], yc_train)
            ablation_acc = accuracy_score(yc_test, clf_ablation.predict(X_test[ablation_feats]))
            ablation_results[feat] = round(acc - ablation_acc, 4)

        print("Accuracy drop when feature removed (higher = more important):")
        for feat, drop in sorted(ablation_results.items(), key=lambda x: -x[1]):
            print(f"  {feat}: {drop:+.4f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        feats_sorted = sorted(ablation_results, key=lambda x: ablation_results[x])
        drops_sorted = [ablation_results[f] for f in feats_sorted]
        ax.barh(feats_sorted, drops_sorted, color='steelblue')
        ax.set_xlabel("Accuracy drop when removed")
        ax.set_title("Feature Ablation Study")
        ax.axvline(0, color='black', linewidth=0.8)
        fig.tight_layout()
        fig.savefig("figures/ablation_study.png", dpi=150)
        plt.close(fig)
        print("Saved figures/ablation_study.png")

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
