"""
Root Cause Analysis (RCA) Core Functions
Includes anomaly investigation and automated RCA using Z-scores.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# For FADS ######
def run_automated_rca(evaluated_df, target_col, feature_cols,
                      lr_model, baseline_means,   # ← pass from FADS
                      time_col='timestamp'):

    d_cols = [f'd_{f}' for f in feature_cols]
    coefficients = lr_model.coef_              # βᵢ for each feature
    print(coefficients)

    anomaly_df = evaluated_df[evaluated_df['Dynamic_Anomaly'] == 1].copy()
    normal_df  = evaluated_df[evaluated_df['Dynamic_Anomaly'] == 0]

    # Baseline: mean derivative of each feature during normal operation
    baseline_d_mean = normal_df[d_cols].mean()

    rca_results = []
    for idx, row in anomaly_df.iterrows():

        contributions = {}
        for feat, d_col, coef in zip(feature_cols, d_cols, coefficients):
            deviation      = row[d_col] - baseline_d_mean[d_col]
            contributions[feat] = coef * deviation   # signed: + means drove target up

        # Sort by absolute contribution
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        rca_results.append({
            'timestamp':       row[time_col],
            'Root_Cause':      sorted_contrib[0][0],
            'Secondary_Cause': sorted_contrib[1][0] if len(sorted_contrib) > 1 else None,
            'Contribution':    sorted_contrib[0][1],        # signed magnitude
            'Direction':       '↑ drove target UP' if sorted_contrib[0][1] > 0 else '↓ drove target DOWN',
            'All_Contributions': {f: float(round(c, 4)) for f, c in sorted_contrib}
        })

    return pd.DataFrame(rca_results)


def run_magnitude_rca(evaluated_df, d_cols, normal_df):
    # 1. Calculate the normal baseline mean AND standard deviation
    baseline_mean = normal_df[d_cols].mean()
    baseline_std = normal_df[d_cols].std()
    
    # Avoid division by zero for perfectly flat sensors
    baseline_std = baseline_std.replace(0, 1e-9) 
    
    rca_results = []
    
    # Filter only anomalies
    anomalies = evaluated_df[evaluated_df['Dynamic_Anomaly'] == True]
    
    for idx, row in anomalies.iterrows():
        z_scores = {}
        
        # 2. Calculate the Standardized Deviation (Z-Score) for each feature
        for col in d_cols:
            current_delta = row[col]
            z_score = abs((current_delta - baseline_mean[col]) / baseline_std[col])
            z_scores[col] = z_score
            
        # Sort features by highest Z-score descending
        sorted_causes = sorted(z_scores.items(), key=lambda item: item[1], reverse=True)
        
        primary_cause, primary_score = sorted_causes[0]
        secondary_cause, secondary_score = sorted_causes[1]
        
        # 3. "Prominent" Logic: Is the 2nd cause at least 75% as severe as the 1st?
        if secondary_score >= (0.75 * primary_score) and secondary_score > 3.0: 
            # Note: Also checking if it's > 3.0 to ensure it's actually an anomaly itself
            final_secondary = secondary_cause
        else:
            final_secondary = "None"
            
        rca_results.append({
            'Timestamp': idx,
            'Root_Cause': primary_cause,
            'Root_Cause_ZScore': round(primary_score, 2),
            'Secondary_Cause': final_secondary
        })
        
    return pd.DataFrame(rca_results)


#------------------------------------------------------------------------------


#### For SADS #######
def run_automated_rca_sads(
    dp,
    feature_name,
    sensor_cols,
    anomaly_col=None,
    timestamp_col='timestamp',
    baseline_window_hrs=168,
    top_k=5,
    save_dir='../data/sads',
):
    """
    Per-timestamp, model-weighted root cause analysis for SADS anomalies.

    Scoring follows the same logic as FADS RCA:
        weighted_score_i = |β_i| × |Z_score_i|

    β_i are proxy sensitivity weights from a linear regression fitted on
    normal (non-anomaly) rows:  feature_name ~ sensor_cols.
    This ensures sensors with stronger influence on the target get amplified,
    not just any sensor that happens to deviate.

    Args:
        dp:                   DataProcessor with dp.df containing sensor data and
                              the anomaly flag column written by lstm_hybrid_model.
        feature_name:         SADS target name, e.g. 'DifferentialPressure'.
        sensor_cols:          Input sensor columns to rank.
        anomaly_col:          Anomaly flag column name. Defaults to
                              'Anomaly_{feature_name}'.
        timestamp_col:        Timestamp column name.
        baseline_window_hrs:  Hours of pre-event history used as normal baseline
                              for Z-score computation.
        top_k:                Number of top sensors to report per timestamp.
        save_dir:             Directory to save the CSV result.

    Returns:
        pd.DataFrame with one row per anomaly timestamp:
            timestamp, target_value,
            cause_1_sensor, cause_1_score, cause_1_zscore, cause_1_direction,
            ... up to cause_{top_k}_*
            Root_Cause, Contribution
    """
    if anomaly_col is None:
        anomaly_col = f'Anomaly_{feature_name}'

    df = dp.df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # Validate columns
    if anomaly_col not in df.columns:
        existing = [c for c in df.columns if 'anomaly' in c.lower()]
        hint = (f"\n  Anomaly-like columns found: {existing}" if existing else
                "\n  No anomaly columns found — run lstm_hybrid_model() first.")
        raise ValueError(f"Column '{anomaly_col}' not in dataframe.{hint}")

    missing_s = [c for c in sensor_cols if c not in df.columns]
    if missing_s:
        raise ValueError(f"Sensor columns not found: {missing_s}")

    # ── Fit linear proxy weights on normal rows ───────────────────────────────
    # Normal = flagged as 0 (evaluated, not anomalous); exclude -1 (not evaluated)
    normal_df = df[df[anomaly_col] == 0].dropna(subset=sensor_cols + [feature_name])
    proxy_weights = np.ones(len(sensor_cols))   # fallback: uniform weights

    if len(normal_df) >= 20 and feature_name in df.columns:
        X_norm = normal_df[sensor_cols].values
        y_norm = normal_df[feature_name].values
        lr = LinearRegression().fit(X_norm, y_norm)
        proxy_weights = np.abs(lr.coef_)
        # Normalise so weights sum to 1 — keeps scores comparable across targets
        w_sum = proxy_weights.sum()
        if w_sum > 0:
            proxy_weights = proxy_weights / w_sum
        print(f"  Proxy weights calibrated on {len(normal_df)} normal rows.")
        for s, w in sorted(zip(sensor_cols, proxy_weights), key=lambda x: -x[1]):
            print(f"    {s:<28} β-weight = {w:.4f}")
    else:
        print(f"  ⚠ Not enough normal rows ({len(normal_df)}) to fit proxy — "
              "falling back to uniform weights (pure Z-score ranking).")

    weight_map = dict(zip(sensor_cols, proxy_weights))
    # ─────────────────────────────────────────────────────────────────────────

    anomaly_df = df[df[anomaly_col] == 1].copy()

    if anomaly_df.empty:
        print(f"[{feature_name}] No anomalies found in '{anomaly_col}'.")
        return pd.DataFrame()

    print(f"\n=== SADS RCA: {feature_name} ===")
    print(f"  Anomaly timestamps : {len(anomaly_df)}")
    print(f"  Baseline window    : {baseline_window_hrs}h")
    print(f"  Top-k sensors      : {top_k}")
    print()

    results = []
    for pos_idx in anomaly_df.index:
        ts = df[timestamp_col].iloc[pos_idx]

        # Rolling baseline: rows within baseline_window_hrs before this timestamp
        window_start = ts - pd.Timedelta(hours=baseline_window_hrs)
        baseline_slice = df.loc[
            (df[timestamp_col] >= window_start) & (df[timestamp_col] < ts),
            sensor_cols
        ].dropna()

        if len(baseline_slice) < 10:
            baseline_slice = df.iloc[:pos_idx][sensor_cols].dropna()

        if len(baseline_slice) < 5:
            continue

        baseline_mean = baseline_slice.mean()
        baseline_std  = baseline_slice.std().replace(0, 1e-6)

        current_vals = df.loc[pos_idx, sensor_cols]
        z_scores = (current_vals - baseline_mean) / baseline_std

        # Model-weighted score: |β_i| × |Z_i|
        weighted = pd.Series(
            {s: weight_map[s] * abs(float(z_scores[s])) for s in sensor_cols}
        )
        ranked = weighted.sort_values(ascending=False).head(top_k)

        row = {
            'timestamp':    ts,
            'target_value': df.loc[pos_idx, feature_name] if feature_name in df.columns else np.nan,
        }

        for rank, sensor in enumerate(ranked.index, start=1):
            z = float(z_scores[sensor])
            row[f'cause_{rank}_sensor']    = sensor
            row[f'cause_{rank}_score']     = round(float(weighted[sensor]), 4)
            row[f'cause_{rank}_zscore']    = round(z, 3)
            row[f'cause_{rank}_direction'] = 'HIGH' if z > 0 else 'LOW'

        results.append(row)

    if not results:
        print(f"[{feature_name}] All anomaly timestamps skipped (insufficient baseline).")
        return pd.DataFrame()

    rca_df = pd.DataFrame(results)

    # Alias columns compatible with plot_rca_dashboard
    rca_df['Root_Cause']            = rca_df['cause_1_sensor']
    rca_df['Contribution']          = rca_df['cause_1_score']
    rca_df['Root_Cause_Direction']  = rca_df['cause_1_direction']

    # Secondary cause — only populate if top_k >= 2
    if 'cause_2_sensor' in rca_df.columns:
        rca_df['Secondary_Cause']           = rca_df['cause_2_sensor']
        rca_df['Secondary_Contribution']    = rca_df['cause_2_score']
        rca_df['Secondary_Direction']       = rca_df['cause_2_direction']

    os.makedirs(save_dir, exist_ok=True)

    # Full detail CSV
    out_path = os.path.join(save_dir, f'sads_rca_{feature_name}.csv')
    rca_df.to_csv(out_path, index=False)

    # Summary CSV — timestamp + primary root cause only
    summary_cols = ['timestamp', 'target_value', 'Root_Cause', 'Contribution',
                    'cause_1_direction']
    summary_cols = [c for c in summary_cols if c in rca_df.columns]
    summary_path = os.path.join(save_dir, f'sads_rca_{feature_name}_summary.csv')
    rca_df[summary_cols].to_csv(summary_path, index=False)

    print(f"✓ {len(rca_df)} anomaly timestamps analysed.")
    print(f"✓ Full detail → {out_path}")
    print(f"✓ Summary     → {summary_path}")

    # Most frequent primary cause across all anomaly timestamps
    print(f"\nMost frequent primary cause:")
    print(rca_df['Root_Cause'].value_counts().head(5).to_string())

    return rca_df