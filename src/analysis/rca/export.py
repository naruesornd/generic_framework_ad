"""
Export functions for RCA results to Excel and CSV
"""

import pandas as pd
import os


def export_rca_to_excel(original_df, evaluated_df, rca_df, target_col, feature_cols, filename="Plant_Anomaly_Report.xlsx"):
    """
    Merges absolute sensor values, anomaly flags, and root cause analysis into a single,
    business-ready Excel report.

    Args:
        original_df: Original DataFrame with sensor values.
        evaluated_df: DataFrame with anomaly detection results.
        rca_df: Root cause analysis results.
        target_col: Target column name.
        feature_cols: List of feature column names.
        filename: Output filename.

    Returns:
        Merged DataFrame with all information.
    """
    print("--- Generating Master Excel Report ---")

    original_df = original_df.copy()
    evaluated_df = evaluated_df.copy()

    original_df['timestamp'] = pd.to_datetime(original_df['timestamp'])
    evaluated_df['timestamp'] = pd.to_datetime(evaluated_df['timestamp'])

    master_df = pd.merge(
        original_df[['timestamp', target_col] + feature_cols],
        evaluated_df[['timestamp', 'predicted_d_target', 'error', 'Dynamic_Anomaly']],
        on='timestamp',
        how='inner'
    )

    if rca_df is not None and not rca_df.empty:
        rca_df = rca_df.copy()
        rca_df['timestamp'] = pd.to_datetime(rca_df['timestamp'])

        master_df = pd.merge(
            master_df,
            rca_df[['timestamp', 'Root_Cause', 'Max_Z_Score']],
            left_on='timestamp',
            right_on='imestamp',
            how='left'
        )
        master_df = master_df.drop(columns=['Timestamp'])
    else:
        master_df['Root_Cause'] = None
        master_df['Max_Z_Score'] = None

    master_df['System_Status'] = master_df['Dynamic_Anomaly'].apply(lambda x: 'ANOMALY' if x == 1 else 'Normal')

    master_df['Root_Cause'] = master_df['Root_Cause'].fillna(' ')
    master_df['Max_Z_Score'] = master_df['Max_Z_Score'].fillna(0.0)

    column_order = [
        'timestamp',
        'System_Status',
        'Root_Cause',
        'Max_Z_Score',
        target_col,
        'predicted_d_target',
        'error'
    ] + feature_cols

    master_df = master_df[column_order]

    master_df = master_df.rename(columns={
        'timestamp': 'Timestamp',
        target_col: f'Actual Output ({target_col})',
        'predicted_d_target': 'Predicted Target Change (dy/dt)',
        'Max_Z_Score': 'Severity (Z-Score)'
    })

    try:
        os.makedirs('../data/physics/', exist_ok=True)
        master_df.to_excel(f'../data/physics/{filename}', index=False, engine='openpyxl')
        print(f"✅ Successfully exported {len(master_df)} rows to '../data/physics/{filename}'!")
        print("Columns included:")
        for col in master_df.columns:
            print(f"  - {col}")
    except (ModuleNotFoundError, ImportError):
        print("Warning: You need the 'openpyxl' library to save to Excel.")
        print("Run: pip install openpyxl")

        fallback_name = filename.replace('.xlsx', '.csv')
        master_df.to_csv(f'../data/physics/{fallback_name}', index=False)
        print(f"✅ Fallback: Exported to CSV instead -> '../data/physics/{fallback_name}'")

    return master_df
