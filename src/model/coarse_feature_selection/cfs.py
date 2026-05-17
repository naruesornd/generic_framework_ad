"""
Feature Selection Model using Random Forest Regressor and SHAP
"""

import os
import time
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm


def random_forest_regressor(
    dp,
    target_colums,
    features,
    plant_name,
    test_size=0.2,
    random_state=42,
    top_num=5,
    max_rows=10000,
    n_estimators=200,
    rf_chunk=25,
    shap_rows=400,
    shap_batch=50
):
    """
    Performs feature selection using Random Forest Regressor and SHAP values.

    Args:
        dp: DataProcessor object
        target_colums: Target column name
        features: List of feature column names
        plant_name: Name of the plant (for file naming)
        test_size: Test set fraction
        random_state: Random seed
        top_num: Number of top features to select
        max_rows: Maximum rows to use for training
        n_estimators: Number of trees in forest
        rf_chunk: Trees per progress update
        shap_rows: Rows to use for SHAP calculation
        shap_batch: Batch size for SHAP calculation

    Returns:
        List of top-k selected features
    """
    steps = [
        "Data Splitting & Sampling",
        "Training Full RandomForest (tree progress)",
        "Feature Selection (Top 4×K)",
        "Training Refined RF Model (tree progress)",
        "Calculating SHAP Values (row progress)",
        "Saving Files & Plotting"
    ]

    features = [f for f in features if f not in ["cycle_id", "cycle_time"]]

    with tqdm(total=len(steps), desc="Processing", colour="grey") as pbar:
        t0 = time.time()

        # Step 1: Data Splitting & Sampling
        pbar.set_description(f"Processing: {steps[0]}")

        all_cols = features + [target_colums]
        df = dp.df[all_cols].dropna()

        if len(df) > max_rows:
            selected_data = df.sample(n=max_rows, random_state=random_state)
            pbar.write(f"Sampled {max_rows} rows from {len(df)} total.")
        else:
            selected_data = df
            pbar.write(f"Using {len(df)} rows (no sampling).")

        X_train, X_test, y_train, y_test = train_test_split(
            selected_data[features],
            selected_data[target_colums],
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        pbar.write(f"✅ {steps[0]} Complete | Train={len(X_train)}, Test={len(X_test)}")
        pbar.update(1)

        # Step 2: Training Full RandomForest
        pbar.set_description(f"Processing: {steps[1]}")

        rf = RandomForestRegressor(
            n_estimators=0,
            warm_start=True,
            random_state=random_state,
            n_jobs=-1
        )

        built = 0
        with tqdm(total=n_estimators, desc="Full RF: trees", leave=False) as t_rf:
            while built < n_estimators:
                nxt = min(built + rf_chunk, n_estimators)
                rf.set_params(n_estimators=nxt)
                rf.fit(X_train, y_train.values.ravel())
                t_rf.update(nxt - built)
                built = nxt

        pbar.write(f"✅ {steps[1]} Complete | Trees={n_estimators}")
        pbar.update(1)

        # Step 3: Feature Selection
        pbar.set_description(f"Processing: {steps[2]}")

        importances = rf.feature_importances_
        fi = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
        top_features = fi.head(4 * top_num).index.tolist()

        pbar.write(f"✅ {steps[2]} Complete | Selected={len(top_features)}")
        pbar.update(1)

        # Step 4: Training Refined RF Model
        pbar.set_description(f"Processing: {steps[3]}")

        X_sub = X_train[top_features]
        y_sub = y_train

        rf_small = RandomForestRegressor(
            n_estimators=0,
            warm_start=True,
            random_state=random_state,
            n_jobs=-1
        )

        built = 0
        with tqdm(total=n_estimators, desc="Refined RF: trees", leave=False) as t_rf2:
            while built < n_estimators:
                nxt = min(built + rf_chunk, n_estimators)
                rf_small.set_params(n_estimators=nxt)
                rf_small.fit(X_sub, y_sub.values.ravel())
                t_rf2.update(nxt - built)
                built = nxt

        pbar.write(f"✅ {steps[3]} Complete | Trees={n_estimators}")
        pbar.update(1)

        # Step 5: Calculating SHAP Values
        pbar.set_description(f"Processing: {steps[4]}")

        explainer = shap.TreeExplainer(rf_small)

        n_shap = min(shap_rows, len(X_sub))
        X_sub_sample = X_sub.sample(n=n_shap, random_state=random_state)

        shap_chunks = []
        with tqdm(total=n_shap, desc="SHAP: rows", leave=False) as t_shap:
            for start in range(0, n_shap, shap_batch):
                end = min(start + shap_batch, n_shap)
                batch = X_sub_sample.iloc[start:end]
                shap_vals_batch = explainer.shap_values(batch)
                shap_chunks.append(shap_vals_batch)
                t_shap.update(end - start)

        shap_values = np.vstack(shap_chunks)

        shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X_sub.columns)
        top_k_features = shap_importance.sort_values(ascending=False).head(top_num).index.tolist()

        pbar.write(f"✅ {steps[4]} Complete | Top-{top_num}: {top_k_features}")
        pbar.update(1)

        # Step 6: Saving Files & Plotting
        pbar.set_description(f"Processing: {steps[5]}")

        os.makedirs("../data/sads/", exist_ok=True)

        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X_sub_sample, feature_names=X_sub.columns, max_display=top_num, show=False)
        plt.title(f"Top Features Influencing {target_colums}")
        plt.tight_layout()
        plt.savefig(f"../data/sads/top_features_{plant_name}_{target_colums}.png", dpi=300)

        pd.DataFrame(top_k_features, columns=["feature_name"]).to_csv(
            f"../data/sads/top_k_features_{plant_name}_{target_colums}.csv", index=False
        )
        dp.df[top_k_features].to_csv(
            f"../data/sads/top_k_features_data_{plant_name}_{target_colums}.csv", index=False
        )

        pbar.write(f"✅ {steps[5]} Complete | Saved CSV + PNG")
        pbar.update(1)

        pbar.set_description(f"All Tasks Finished ({time.time() - t0:.1f}s)")

    return top_k_features, top_features, fi
