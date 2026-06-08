import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
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
        self.features = ['driver_orig_dist', 'overlap_ratio', 'time_diff_mins', 'driver_approx_fare', 'rider_approx_fare', 'fare_ratio', 'pickup_hour', 'pickup_dow', 'shareability_count']
        
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

        # Feature importance — manual bar chart to avoid long decimal labels
        importance_dict = self.classifier.get_booster().get_score(importance_type='gain')
        feats_sorted = sorted(importance_dict, key=importance_dict.get)
        scores_sorted = [importance_dict[f] for f in feats_sorted]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(feats_sorted, scores_sorted, color='steelblue')
        ax.set_xlabel("Importance Score (Gain)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
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
        ax.legend()
        fig.tight_layout()
        fig.savefig("figures/roc_curve.png", dpi=150)
        plt.close(fig)
        print("Saved figures/roc_curve.png")

        # --- Feature ablation study ---
        # Remove each feature one at a time, retrain, measure accuracy drop.
        # The bigger the drop, the more important the feature.
        print("\nRunning feature ablation study (ROC-AUC drop)...")
        ablation_results = {}
        for feat in self.features:
            ablation_feats = [f for f in self.features if f != feat]
            clf_ablation = xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='logloss',
                scale_pos_weight=scale_pos_weight, **best_params
            )
            clf_ablation.fit(X_train[ablation_feats], yc_train)
            abl_proba = clf_ablation.predict_proba(X_test[ablation_feats])[:, 1]
            ablation_roc = roc_auc_score(yc_test, abl_proba)
            ablation_results[feat] = round(roc_auc - ablation_roc, 4)

        print("ROC-AUC drop when feature removed (higher = more important):")
        for feat, drop in sorted(ablation_results.items(), key=lambda x: -x[1]):
            print(f"  {feat}: {drop:+.4f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        feats_sorted = sorted(ablation_results, key=lambda x: ablation_results[x])
        drops_sorted = [ablation_results[f] for f in feats_sorted]
        ax.barh(feats_sorted, drops_sorted, color='steelblue')
        ax.set_xlabel("ROC-AUC drop when removed (higher = more important)")
        ax.axvline(0, color='black', linewidth=0.8)
        fig.tight_layout()
        fig.savefig("figures/ablation_study.png", dpi=150)
        plt.close(fig)
        print("Saved figures/ablation_study.png")

        # --- Learning curves ---
        # Diagnoses overfitting vs underfitting by plotting train/CV accuracy
        # across increasing training set sizes.
        print("\nGenerating learning curves...")
        lc_clf = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='logloss',
            scale_pos_weight=scale_pos_weight, **best_params
        )
        train_sizes, train_scores, cv_scores = learning_curve(
            lc_clf, X_train, yc_train, cv=5,
            train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
            scoring='accuracy', n_jobs=-1
        )
        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        cv_mean    = cv_scores.mean(axis=1)
        cv_std     = cv_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Training accuracy')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color='steelblue')
        ax.plot(train_sizes, cv_mean, 'o-', color='darkorange', label='CV accuracy (5-fold)')
        ax.fill_between(train_sizes, cv_mean - cv_std, cv_mean + cv_std,
                        alpha=0.15, color='darkorange')
        ax.set_xlabel('Training set size')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0.4, 1.02])
        ax.legend()
        fig.tight_layout()
        fig.savefig("figures/learning_curve.png", dpi=150)
        plt.close(fig)
        print("Saved figures/learning_curve.png")

        # --- System-level bipartite matching evaluation ---
        # Fulfills proposal Section 5: match rate, avg driver detour,
        # avg driver earnings, avg rider cost reduction, trade-off curve.
        print("\nRunning system-level bipartite matching evaluation...")
        import networkx as nx

        def _run_bipartite(pairs_df, score_threshold=None):
            B = nx.Graph()
            for _, row in pairs_df.iterrows():
                score = self.compute_match_score(row[self.features])
                if score_threshold is not None and score < score_threshold:
                    continue
                B.add_edge(
                    f"D_{row['driver_id']}", f"R_{row['rider_id']}",
                    weight=max(score, 0), data=row
                )
            return nx.max_weight_matching(B, maxcardinality=True)

        optimal = _run_bipartite(test_df)

        # Identify matched pair rows
        matched_keys = set()
        for u, v in optimal:
            d = u if str(u).startswith('D_') else v
            r = v if str(v).startswith('R_') else u
            matched_keys.add((d, r))

        matched_df = test_df[test_df.apply(
            lambda row: (f"D_{row['driver_id']}", f"R_{row['rider_id']}") in matched_keys,
            axis=1
        )]

        n_test_drivers = test_df['driver_id'].nunique()
        match_rate     = len(optimal) / max(n_test_drivers, 1) * 100
        avg_detour     = matched_df['detour_distance_km'].mean() if len(matched_df) else 0
        # Rider pays 30% of their fare to the driver as the incentive split
        avg_driver_earn  = matched_df['rider_approx_fare'].clip(lower=0).mean() * 0.3 if len(matched_df) else 0
        avg_rider_saving = matched_df['rider_approx_fare'].clip(lower=0).mean() * 0.3 if len(matched_df) else 0
        # CO2: rider avoided a solo taxi (rider_fare / $2.50/mile = miles, * 404g/mile)
        avg_co2_saved_g  = (matched_df['rider_approx_fare'].clip(lower=0) / 2.5 * 404).mean() if len(matched_df) else 0

        print(f"\nSystem-Level Results (Bipartite Matching on Test Set):")
        print(f"  Unique test drivers  : {n_test_drivers}")
        print(f"  Optimal matches      : {len(optimal)}")
        print(f"  Match rate           : {match_rate:.1f}%")
        print(f"  Avg driver detour    : {avg_detour:.2f} km")
        print(f"  Avg driver earnings  : +${avg_driver_earn:.2f} per ride")
        print(f"  Avg rider savings    : -${avg_rider_saving:.2f} per ride")
        print(f"  Avg CO2 saved        : {avg_co2_saved_g:.0f} g per match")

        # Trade-off curve: match rate vs avg detour across acceptance probability thresholds.
        # Use predict_proba thresholds directly (not score thresholds) so the curve
        # reflects a meaningful operator knob: "only match if model is >X% confident."
        prob_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        tc_rates, tc_detours, tc_labels = [], [], []

        proba_test = self.classifier.predict_proba(X_test)[:, 1]
        test_df_reset = test_df.reset_index(drop=True)

        for thresh in prob_thresholds:
            # Keep only pairs where model is confident enough
            confident = test_df_reset[proba_test >= thresh].copy()
            if confident.empty:
                continue
            B2 = nx.Graph()
            for _, row in confident.iterrows():
                prob = proba_test[row.name] if row.name < len(proba_test) else 0
                B2.add_edge(f"D_{row['driver_id']}", f"R_{row['rider_id']}", weight=prob)
            m2 = nx.max_weight_matching(B2, maxcardinality=True)
            rate = len(m2) / max(n_test_drivers, 1) * 100
            mk2 = set()
            for u, v in m2:
                d2 = u if str(u).startswith('D_') else v
                r2 = v if str(v).startswith('R_') else u
                mk2.add((d2, r2))
            mdf2 = confident[confident.apply(
                lambda row: (f"D_{row['driver_id']}", f"R_{row['rider_id']}") in mk2, axis=1
            )]
            avg_det = mdf2['detour_distance_km'].mean() if len(mdf2) else 0
            tc_rates.append(rate)
            tc_detours.append(avg_det)
            tc_labels.append(f">{int(thresh*100)}%")

        # Prepend the no-threshold baseline so the full range is visible
        all_detours = [avg_detour] + tc_detours
        all_rates   = [match_rate]  + tc_rates
        all_labels  = ['all pairs'] + tc_labels

        if all_rates:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(all_detours, all_rates, 'o-', color='steelblue', linewidth=2, markersize=8)
            for x, y, lbl in zip(all_detours, all_rates, all_labels):
                ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
            # Mark the no-threshold baseline in orange
            ax.plot(avg_detour, match_rate, 'o', color='#f77f00', markersize=11, zorder=5)
            ax.set_xlabel('Avg Driver Detour (km)', fontsize=11)
            ax.set_ylabel('Match Rate (%)', fontsize=11)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig("figures/system_tradeoff_curve.png", dpi=150)
            plt.close(fig)
            print("Saved figures/system_tradeoff_curve.png")

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