"""
Multivariate anomaly detection and gradient relationship analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression



 # Define high-resolution download settings
export_config = {
            'responsive': True,
            'scrollZoom': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'feature_plot',
                # 'height': 1080,
                # 'width': 1920,
                'scale': 3
        }
    }




def plot_gradient_relationship(df, target_col, feature_col, time_col='timestamp', smoothing_window=3):
    """
    Plots d(Target)/dt vs d(Feature)/dt to visualize the physical relationship.

    Args:
        df: DataFrame containing the data.
        target_col: Target column name.
        feature_col: Feature column name.
        time_col: Timestamp column name.
        smoothing_window: Window size for moving average smoothing.
    """
    print(f"--- Analyzing Physical Relationship: d({target_col}) vs d({feature_col}) ---")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col)

    anom_col = f'Anomaly_{target_col}'
    has_anomalies = anom_col in df.columns
    if not has_anomalies:
        print(f"Note: '{anom_col}' not found. Plotting all data as 'Normal'.")
        df[anom_col] = 0

    dt_hours = df[time_col].diff().dt.total_seconds() / 3600.0
    dt_hours = dt_hours.replace(0, 0.0001)

    df['d_target'] = (df[target_col].diff() / dt_hours).rolling(window=smoothing_window, min_periods=1).mean()
    df['d_feature'] = (df[feature_col].diff() / dt_hours).rolling(window=smoothing_window, min_periods=1).mean()

    plot_df = df.dropna(subset=['d_target', 'd_feature', anom_col]).copy()

    normal_df = plot_df[plot_df[anom_col] == 0]
    anom_df = plot_df[plot_df[anom_col] == 1]

    lr_model = LinearRegression()

    X_normal = plot_df['d_feature'].values.reshape(-1, 1)
    y_normal = plot_df['d_target'].values

    if len(X_normal) > 0:
        lr_model.fit(X_normal, y_normal)

        x_range = np.linspace(plot_df['d_feature'].min(), plot_df['d_feature'].max(), 100).reshape(-1, 1)
        y_line = lr_model.predict(x_range)
        r2_score = lr_model.score(X_normal, y_normal)
    else:
        print("Warning: Not enough normal data to fit a regression line.")
        x_range, y_line, r2_score = [], [], 0

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df['d_feature'],
        y=plot_df['d_target'],
        mode='markers',
        name='Data Points',
        marker=dict(color='rgba(100, 100, 100, 0.3)', size=4),
        hovertemplate=f"<b>Normal</b><br>d({feature_col}): %{{x:.2f}}<br>d({target_col}): %{{y:.2f}}<extra></extra>"
    ))

    if len(x_range) > 0:
        fig.add_trace(go.Scatter(
            x=x_range.flatten(),
            y=y_line,
            mode='lines',
            name=f'Physics Baseline (R={r2_score:.3f})',
            line=dict(color='green', width=3, dash='dash')
        ))

    if not anom_df.empty:
        fig.add_trace(go.Scatter(
            x=anom_df['d_feature'],
            y=anom_df['d_target'],
            mode='markers',
            name='Anomaly Triggered',
            marker=dict(color='yellow', size=10, line=dict(color='red', width=2)),
            hovertemplate=f"<b>ANOMALY</b><br>d({feature_col}): %{{x:.2f}}<br>d({target_col}): %{{y:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        title=f"Rate of Change Relationship Analysis",
        xaxis_title=f"Rate of Change: {feature_col} (Δ/hr)",
        yaxis_title=f"Rate of Change: {target_col} (Δ/hr)",
        template="plotly_white",
        height=600,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.8)')
    )

    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.2)
    fig.add_vline(x=0, line_dash="solid", line_color="black", opacity=0.2)

    fig.show(config={'scrollZoom': True, 'responsive': True})


def plot_dynamic_multivariate_anomalies(df, target_col, feature_cols, time_col='timestamp', smoothing_window=1, sigma=3):
    """
    Fits a Multivariate Linear Regression and automatically flags anomalies.

    Args:
        df: DataFrame containing the data.
        target_col: Target column name.
        feature_cols: List of feature column names.
        time_col: Timestamp column name.
        smoothing_window: Window size for smoothing.
        sigma: Standard deviation multiplier for threshold.

    Returns:
        DataFrame with Dynamic_Anomaly column added.
    """
    print(f"--- Dynamic Fast System Anomaly Detection for {target_col} ---")
    print(f"Features: {', '.join(feature_cols)}")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col)

    dt_hours = df[time_col].diff().dt.total_seconds() / 3600.0
    dt_hours = dt_hours.replace(0, 0.0001)

    df['d_target'] = (df[target_col].diff() / dt_hours).rolling(window=smoothing_window, min_periods=1).mean()

    d_feature_names = []
    for col in feature_cols:
        d_col_name = f'd_{col}'
        df[d_col_name] = (df[col].diff() / dt_hours).rolling(window=smoothing_window, min_periods=1).mean()
        d_feature_names.append(d_col_name)

    plot_df = df.dropna(subset=['d_target'] + d_feature_names).copy()

    lr_model = LinearRegression()
    X_all = plot_df[d_feature_names].values
    y_actual = plot_df['d_target'].values

    if len(X_all) == 0:
        print("Warning: Not enough data to fit the model.")
        return df

    lr_model.fit(X_all, y_actual)
    plot_df['predicted_d_target'] = lr_model.predict(X_all)
    r2 = lr_model.score(X_all, y_actual)

    plot_df['error'] = np.abs(plot_df['d_target'] - plot_df['predicted_d_target'])

    mean_error = plot_df['error'].mean()
    std_error  = plot_df['error'].std()
    threshold  = mean_error + (sigma * std_error)

    plot_df['Dynamic_Anomaly'] = (plot_df['error'] > threshold).astype(int)

    print(f"R² Score:               {r2:.3f}")
    print(f"Average Error Distance: {mean_error:.3f}")
    print(f"Anomaly Threshold:      ±{threshold:.3f}")

    normal_df = plot_df[plot_df['Dynamic_Anomaly'] == 0]
    anom_df   = plot_df[plot_df['Dynamic_Anomaly'] == 1]

    print(f"Automatically detected {len(anom_df)} anomalies out of {len(plot_df)} points!")

    # ── Axis range — give a little padding ───────────────────────────────────
    all_x = plot_df['predicted_d_target']
    all_y = plot_df['d_target']
    min_val = min(all_x.min(), all_y.min())
    max_val = max(all_x.max(), all_y.max())
    pad = (max_val - min_val) * 0.05
    axis_min = min_val - pad
    axis_max = max_val + pad
    line_x = [axis_min, axis_max]

    fig = go.Figure()

    # Normal points
    fig.add_trace(go.Scatter(
        x=normal_df['predicted_d_target'], y=normal_df['d_target'],
        mode='markers', name='Normal Data Points',
        marker=dict(color='rgba(100, 100, 100, 0.4)', size=5),
        text=normal_df[time_col].dt.strftime('%Y-%m-%d %H:%M:%S'),
        hovertemplate="<b>Time: %{text}</b><br>Predicted: %{x:.2f}<br>Actual: %{y:.2f}<br>Error: %{customdata:.2f}<extra></extra>",
        customdata=normal_df['error']
    ))

    # Ideal fit line y = x  (with R²)
    fig.add_trace(go.Scatter(
        x=line_x, y=line_x,
        mode='lines',
        name=f'Ideal Fit (R²={r2:.3f})',
        line=dict(color='green', width=2, dash='dash')
    ))

    # Upper threshold band:  y = x + threshold
    fig.add_trace(go.Scatter(
        x=line_x, y=[v + threshold for v in line_x],
        mode='lines',
        name=f'Upper Limit (+{threshold:.1f})',
        line=dict(color='red', width=1.5, dash='dash')
    ))

    # Lower threshold band:  y = x - threshold
    fig.add_trace(go.Scatter(
        x=line_x, y=[v - threshold for v in line_x],
        mode='lines',
        name=f'Lower Limit (-{threshold:.1f})',
        line=dict(color='red', width=1.5, dash='dash')
    ))

    # Anomaly points
    if not anom_df.empty:
        fig.add_trace(go.Scatter(
            x=anom_df['predicted_d_target'], y=anom_df['d_target'],
            mode='markers', name='Anomaly',
            marker=dict(color='yellow', size=10, line=dict(color='red', width=2)),
            text=anom_df[time_col].dt.strftime('%Y-%m-%d %H:%M:%S'),
            hovertemplate="<b>ANOMALY</b><br>Time: %{text}<br>Predicted: %{x:.2f}<br>Actual: %{y:.2f}<br>Error: %{customdata:.2f}<extra></extra>",
            customdata=anom_df['error']
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"Dynamic Anomaly Detection: {target_col}<br>"
                f"<sup>Flagging points that fall outside the {sigma}-Sigma Safe Zone</sup>"
            ),
            x=0.5, xanchor='center'
        ),
        xaxis_title="PREDICTED Rate of Change (Δ/hr)",
        yaxis_title="ACTUAL Rate of Change (Δ/hr)",
        template="plotly_white",
        height=700,
        hovermode="closest",
        annotations=[
            dict(
                text=f"Total Anomalies Detected: {len(anom_df)}",
                xref="paper", yref="paper",
                x=0.01, y=1.0,
                showarrow=False,
                font=dict(size=13, color="black"),
                xanchor="left", yanchor="bottom"
            )
        ],
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='lightgrey',
            borderwidth=1
        ),
        xaxis=dict(range=[axis_min, axis_max], zeroline=True, zerolinecolor='lightgrey'),
        yaxis=dict(range=[axis_min, axis_max], zeroline=True, zerolinecolor='lightgrey'),
    )

    fig.show(config={'scrollZoom': True, 'responsive': True})

    # ── Write results back into df so the caller gets Dynamic_Anomaly ─────────
    # plot_df is a dropna() subset of df — use index alignment to map back.
    # Rows that were dropped (NaN) get Dynamic_Anomaly = 0 and error = NaN.
    df['Dynamic_Anomaly']      = plot_df['Dynamic_Anomaly'].reindex(df.index, fill_value=0)
    df['predicted_d_target']   = plot_df['predicted_d_target'].reindex(df.index)
    df['error']                = plot_df['error'].reindex(df.index)
    # ─────────────────────────────────────────────────────────────────────────

    return df

def fast_anomaly_detection_system(df, target_col, feature_cols, time_col='timestamp',
                                   smoothing_window=1, sigma=3, train_frac=0.3,
                                   directional=False, cycle_col='cycle_id',
                                   cycle_buffer=5):
    """
    ...
    Args:
        cycle_col:     Column name for cycle ID. Set to None to disable masking.
        cycle_buffer:  Number of timestamps to suppress anomaly detection at
                       the start and end of each cycle (transition dead zone).
    """
    print(f"--- Dynamic Fast System Anomaly Detection for {target_col} ---")
    print(f"Features: {', '.join(feature_cols)}")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col)

    dt_hours = df[time_col].diff().dt.total_seconds() / 3600.0
    dt_hours = dt_hours.replace(0, 0.0001)

    df['d_target'] = (df[target_col].diff() / dt_hours).rolling(window=smoothing_window, min_periods=1).mean()

    d_feature_names = []
    for col in feature_cols:
        d_col_name = f'd_{col}'
        df[d_col_name] = (df[col].diff() / dt_hours).rolling(window=smoothing_window, min_periods=1).mean()
        d_feature_names.append(d_col_name)

    plot_df = df.dropna(subset=['d_target'] + d_feature_names).copy()

    # ── Build cycle boundary mask ─────────────────────────────────────────────
    plot_df['_in_buffer'] = False
    if cycle_col and cycle_col in plot_df.columns and cycle_buffer > 0:
        for _, group in plot_df.groupby(cycle_col):
            if len(group) <= cycle_buffer * 2:
                # Entire cycle is within buffer — mask it all
                plot_df.loc[group.index, '_in_buffer'] = True
            else:
                head_idx = group.index[:cycle_buffer]
                tail_idx = group.index[-cycle_buffer:]
                plot_df.loc[head_idx, '_in_buffer'] = True
                plot_df.loc[tail_idx, '_in_buffer'] = True
        n_masked = plot_df['_in_buffer'].sum()
        print(f"🔇 Cycle buffer active: masking {n_masked} boundary timestamps "
              f"(±{cycle_buffer} pts per cycle) from anomaly detection.")
    # ─────────────────────────────────────────────────────────────────────────

    # Train only on baseline (first train_frac of data)
    n_train = max(10, int(len(plot_df) * train_frac))
    train_df = plot_df.iloc[:n_train]

    lr_model = LinearRegression()
    X_all = train_df[d_feature_names].values
    y_actual = train_df['d_target'].values

    if len(X_all) == 0:
        print("Warning: Not enough data to fit the model.")
        return df

    lr_model.fit(X_all, y_actual)
    plot_df['predicted_d_target'] = lr_model.predict(plot_df[d_feature_names].values)
    r2 = lr_model.score(X_all, y_actual)

    train_df = train_df.copy()
    train_df['predicted_d_target'] = lr_model.predict(train_df[d_feature_names].values)
    train_residuals = train_df['d_target'] - train_df['predicted_d_target']
    mean_res = train_residuals.mean()
    std_res  = train_residuals.std()
    threshold = mean_res + sigma * std_res

    plot_df['residual'] = plot_df['d_target'] - plot_df['predicted_d_target']
    plot_df['error']    = plot_df['residual'].abs()

    if directional:
        plot_df['Dynamic_Anomaly'] = (plot_df['residual'] > threshold).astype(int)
    else:
        plot_df['Dynamic_Anomaly'] = (plot_df['residual'].abs() > abs(threshold)).astype(int)

    # ── Suppress anomalies inside the buffer zone ─────────────────────────────
    plot_df.loc[plot_df['_in_buffer'], 'Dynamic_Anomaly'] = 0
    # ─────────────────────────────────────────────────────────────────────────

    normal_df = plot_df[plot_df['Dynamic_Anomaly'] == 0]
    anom_df   = plot_df[plot_df['Dynamic_Anomaly'] == 1]
    buffer_df = plot_df[plot_df['_in_buffer']]          # for visualization

    print(f"Automatically detected {len(anom_df)} anomalies out of {len(plot_df)} points "
          f"({len(buffer_df)} buffer pts excluded).")

    all_x = plot_df['predicted_d_target']
    all_y = plot_df['d_target']
    min_val = min(all_x.min(), all_y.min())
    max_val = max(all_x.max(), all_y.max())
    pad = (max_val - min_val) * 0.05
    axis_min = min_val - pad
    axis_max = max_val + pad
    line_x = [axis_min, axis_max]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=normal_df['predicted_d_target'], y=normal_df['d_target'],
        mode='markers', name='Normal',
        marker=dict(color='rgba(100, 100, 100, 0.4)', size=5),
        text=normal_df[time_col].dt.strftime('%Y-%m-%d %H:%M:%S'),
        hovertemplate="<b>Time: %{text}</b><br>Predicted: %{x:.2f}<br>Actual: %{y:.2f}<br>Error: %{customdata:.2f}<extra></extra>",
        customdata=normal_df['error']
    ))

    # ── Buffer zone points shown distinctly ───────────────────────────────────
    if not buffer_df.empty:
        fig.add_trace(go.Scatter(
            x=buffer_df['predicted_d_target'], y=buffer_df['d_target'],
            mode='markers', name=f'Cycle Buffer (±{cycle_buffer} pts)',
            marker=dict(color='rgba(180, 180, 255, 0.5)', size=5, symbol='diamond'),
            text=buffer_df[time_col].dt.strftime('%Y-%m-%d %H:%M:%S'),
            hovertemplate="<b>Buffer Zone</b><br>Time: %{text}<br>Predicted: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>",
        ))
    # ─────────────────────────────────────────────────────────────────────────

    fig.add_trace(go.Scatter(
        x=line_x, y=line_x,
        mode='lines', name=f'Ideal Fit (R²={r2:.3f})',
        line=dict(color='green', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=line_x, y=[v + threshold for v in line_x],
        mode='lines', name=f'Upper Limit (+{threshold:.1f})',
        line=dict(color='red', width=1.5, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=line_x, y=[v - threshold for v in line_x],
        mode='lines', name=f'Lower Limit (-{threshold:.1f})',
        line=dict(color='red', width=1.5, dash='dash')
    ))

    if not anom_df.empty:
        fig.add_trace(go.Scatter(
            x=anom_df['predicted_d_target'], y=anom_df['d_target'],
            mode='markers', name='Anomaly',
            marker=dict(color='yellow', size=10, line=dict(color='red', width=2)),
            text=anom_df[time_col].dt.strftime('%Y-%m-%d %H:%M:%S'),
            hovertemplate="<b>ANOMALY</b><br>Time: %{text}<br>Predicted: %{x:.2f}<br>Actual: %{y:.2f}<br>Error: %{customdata:.2f}<extra></extra>",
            customdata=anom_df['error']
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"Dynamic Anomaly Detection: {target_col}<br>"
                f"<sup>Flagging points outside the {sigma}-Sigma Safe Zone "
                f"| Cycle buffer: ±{cycle_buffer} pts</sup>"
            ),
            x=0.5, xanchor='center'
        ),
        xaxis_title="PREDICTED Rate of Change (Δ/hr)",
        yaxis_title="ACTUAL Rate of Change (Δ/hr)",
        template="plotly_white",
        height=700,
        hovermode="closest",
        annotations=[
            dict(
                text=f"Total Anomalies Detected: {len(anom_df)}  |  Buffer pts suppressed: {len(buffer_df)}",
                xref="paper", yref="paper",
                x=0.01, y=1.0,
                showarrow=False,
                font=dict(size=13, color="black"),
                xanchor="left", yanchor="bottom"
            )
        ],
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='lightgrey', borderwidth=1
        ),
        xaxis=dict(range=[axis_min, axis_max], zeroline=True, zerolinecolor='lightgrey'),
        yaxis=dict(range=[axis_min, axis_max], zeroline=True, zerolinecolor='lightgrey'),
    )

    fig.show(config=export_config)

    df['Dynamic_Anomaly']    = plot_df['Dynamic_Anomaly'].reindex(df.index, fill_value=0)
    df['predicted_d_target'] = plot_df['predicted_d_target'].reindex(df.index)
    df['error']              = plot_df['error'].reindex(df.index)

    # Create a clean export dataframe with just the requested columns
    export_cols = [time_col, 'Dynamic_Anomaly'] + [target_col] + feature_cols
    # We use .intersection to ensure we don't crash if a column is missing
    final_export_df = df[df.columns.intersection(export_cols)]

    return df, lr_model, train_df[d_feature_names].mean(), final_export_df
