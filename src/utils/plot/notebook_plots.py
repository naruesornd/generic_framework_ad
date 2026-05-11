"""
Interactive plotting functions for Jupyter notebooks including range sliders,
cycle visualization, and anomaly detection plots.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display
import re
import matplotlib.pyplot as plt



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



def get_unit(target_col):
    """Auto-detect unit based on column name ending patterns."""
    UNIT_PATTERNS = [
        (r'Flow$', 'm<sup>3</sup>/h'),
        (r'Pressure$', 'psi'),
        (r'Conductivity$', 'µS/cm'),
        (r'Temperature$', '°F'),
        (r'Recovery$', '%'),
        (r'TDS$', 'ppm'),
        (r'pH$', 'pH'),
        (r'ORP$', 'mV'),
    ]
    for pattern, unit in UNIT_PATTERNS:
        if re.search(pattern, target_col):
            return unit
    return ''


def format_yaxis_label(target_col):
    """Format column name with its unit for axis labels."""
    unit = get_unit(target_col)
    return f"{target_col} ({unit})" if unit else target_col


def plot_interactive_features(df, features, time_col='timestamp', title=None, 
                              separate_graphs=False, height=None, show_rangeslider=True, show_rangeselector=True):
    """
    Plots one or multiple features with interactive Selectors and an optional Range Slider.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        features (str or list): Column name(s) to plot.
        time_col (str): The name of the time column (default: 'timestamp').
        title (str): Title of the chart.
        separate_graphs (bool): If True, plot each feature in a separate subplot. Default: False.
        height (int, optional): Custom height for the entire figure in pixels. Default: None (auto-calculates).
        show_rangeslider (bool): If True, displays the interactive range slider below the x-axis. Default: True.
    """

    if isinstance(features, str):
        features = [features]

    if title is None:
        feature_names = ", ".join(features)
        title = f"Feature Analysis: {feature_names}"

    if separate_graphs:
        fig = make_subplots(
            rows=len(features), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[f"{f}" for f in features]
        )

        for i, col in enumerate(features):
            if col not in df.columns:
                print(f"⚠️ Warning: Column '{col}' not found in DataFrame. Skipping.")
                continue

            fig.add_trace(
                go.Scatter(
                    x=df[time_col],
                    y=df[col],
                    mode='lines',
                    name=col,
                    opacity=0.8,
                    showlegend=(i == 0)
                ),
                row=i+1, col=1
            )

        final_height = height if height is not None else 300 * len(features)

        fig.update_layout(
            title=title,
            hovermode="x unified",
            template="plotly_white",
            height=final_height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Apply the show_rangeslider parameter here
        fig.update_xaxes(
            rangeslider=dict(visible=show_rangeslider),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            type="date",
            row=len(features), col=1
        )

        for i, col in enumerate(features):
            fig.update_yaxes(
                title_text=format_yaxis_label(col),
                autorange=True,
                fixedrange=False,
                row=i+1, col=1
            )
    else:
        fig = go.Figure()

        for col in features:
            if col not in df.columns:
                print(f"⚠️ Warning: Column '{col}' not found in DataFrame. Skipping.")
                continue

            fig.add_trace(
                go.Scatter(
                    x=df[time_col],
                    y=df[col],
                    mode='lines',
                    name=col,
                    opacity=0.8
                )
            )

        if len(features) == 1:
            yaxis_label = format_yaxis_label(features[0])
        else:
            units_list = [get_unit(f) for f in features]
            non_empty_units = [u for u in units_list if u]

            if len(non_empty_units) == 0:
                yaxis_label = "Value"
            elif all(u == non_empty_units[0] for u in non_empty_units):
                yaxis_label = f"Value ({non_empty_units[0]})"
            else:
                yaxis_label = "Value (Mixed Units)"

        final_height = height if height is not None else 600

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=yaxis_label,
            hovermode="x unified",
            template="plotly_white",
            height=final_height,

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),

            yaxis=dict(
                autorange=True,
                fixedrange=False
            ),

            xaxis=dict(
                rangeslider=dict(visible=show_rangeslider),
                rangeselector=dict(
                    visible=show_rangeselector,
                    buttons=list([
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                type="date"
            )
        )

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

        fig.show(config=export_config)
        return

    fig.show(config={'responsive': True, 'scrollZoom': True})

def plot_interactive_secondary_axis(df, feature_primary, feature_secondary, 
                          time_col='timestamp', title=None,
                          primary_color='#1f77b4', secondary_color='#d62728',
                          show_correlation=False):
    """
    Plots two selected variables on the same graph with a dual y-axis
    (primary on left, secondary on right), with interactive Range Slider
    and Selectors.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        feature_primary (str): Column name for the primary (left y-axis) variable.
        feature_secondary (str): Column name for the secondary (right y-axis) variable.
        time_col (str): The name of the time column (default: 'timestamp').
        title (str): Title of the chart. Auto-generated if None.
        primary_color (str): Line color for the primary variable. Default: blue.
        secondary_color (str): Line color for the secondary variable. Default: red.
        show_correlation (bool): If True, appends Pearson correlation to the title.
    """

    # --- Validate columns ---
    for col in [feature_primary, feature_secondary, time_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # --- Compute correlation (optional) ---
    corr_text = ""
    if show_correlation:
        valid = df[[feature_primary, feature_secondary]].dropna()
        if len(valid) > 1:
            r = valid[feature_primary].corr(valid[feature_secondary])
            corr_text = f"  |  Pearson r = {r:.4f}"

    # --- Auto title ---
    if title is None:
        title = f"{feature_primary}  vs  {feature_secondary}{corr_text}"
    else:
        title = f"{title}{corr_text}"

    # --- Create figure with secondary y-axis ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- Primary trace (left y-axis) ---
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[feature_primary],
            mode='lines',
            name=feature_primary,
            line=dict(color=primary_color, width=1.5),
            opacity=0.85
        ),
        secondary_y=False
    )

    # --- Secondary trace (right y-axis) ---
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[feature_secondary],
            mode='lines',
            name=feature_secondary,
            line=dict(color=secondary_color, width=1.5),
            opacity=0.85
        ),
        secondary_y=True
    )

    # --- Layout ---
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title="Time",
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1,  label="1h", step="hour",  stepmode="backward"),
                    dict(count=1,  label="1d", step="day",   stepmode="backward"),
                    dict(count=7,  label="1w", step="day",   stepmode="backward"),
                    dict(count=1,  label="1m", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            type="date"
        )
    )

    # --- Y-axis labels ---
    fig.update_yaxes(
        title_text=feature_primary,
        title_font=dict(color=primary_color),
        tickfont=dict(color=primary_color),
        autorange=True,
        fixedrange=False,
        secondary_y=False
    )

    fig.update_yaxes(
        title_text=feature_secondary,
        title_font=dict(color=secondary_color),
        tickfont=dict(color=secondary_color),
        autorange=True,
        fixedrange=False,
        secondary_y=True
    )

    fig.show(config={'responsive': True, 'scrollZoom': True})

def interactive_chronological_selector(df, parameters, time_col='timestamp'):
    """
    Creates a responsive interactive dashboard using TRUE calendar dates.
    Uses FigureWidget to guarantee no duplicate graphs.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        parameters (str or list): Parameter column name(s) to plot.
        time_col (str): Name of timestamp column.
    """
    if isinstance(parameters, str):
        parameters = [parameters]

    df = df.copy()
    df['clean_time_str'] = pd.to_datetime(df[time_col]).dt.strftime('%Y-%m-%d %H:%M:%S')

    available_cycles = sorted(df['Cycle_ID'].unique())
    if len(available_cycles) == 0:
        print("⚠️ No cycles found in this dataframe.")
        return

    cycle_selector = widgets.SelectMultiple(
        options=available_cycles,
        value=[available_cycles[0]],
        description='Select Cycles:',
        disabled=False,
        layout=widgets.Layout(height='120px', width='80%')
    )

    help_text = widgets.HTML("<i>* Hold <b>Ctrl</b> (Windows) or <b>Cmd</b> (Mac) to select multiple cycles. <b>Click and drag</b> a box on the graph to zoom in. Double-click to zoom out.</i>")

    fig = make_subplots(
        rows=len(parameters), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[f"Timeline: {p}" for p in parameters]
    )

    fw = go.FigureWidget(fig)
    fw.layout.autosize = True

    def update_plot(change):
        selected_cycles = change['new']

        with fw.batch_update():
            fw.data = []

            for i, param in enumerate(parameters):
                for cycle in selected_cycles:
                    cycle_data = df[df['Cycle_ID'] == cycle]

                    show_legend = True if i == 0 else False

                    fw.add_trace(
                        go.Scatter(
                            x=cycle_data['clean_time_str'],
                            y=cycle_data[param],
                            mode='lines',
                            name=f"Cycle {cycle}",
                            showlegend=show_legend,
                            hovertemplate=f"<b>Cycle: {cycle}</b><br>Time: %{{x}}<br>{param}: %{{y:.2f}}<extra></extra>"
                        ),
                        row=i+1, col=1
                    )
                fw.layout.annotations[i].text = f"Timeline: {param} (Showing {len(selected_cycles)} Cycles)"

    fw.update_layout(
        height=400 * len(parameters),
        title="Interactive Chronological Cycle Viewer",
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    fw.update_xaxes(type='date', title_text="Date & Time")

    cycle_selector.observe(update_plot, names='value')
    update_plot({'new': cycle_selector.value})

    ui = widgets.VBox([help_text, cycle_selector])
    dashboard = widgets.VBox([ui, fw], layout=widgets.Layout(width='100%'))

    display(dashboard)


def plot_residual_threshold(df, feature_name, time_col='timestamp'):
    """
    Plots the raw data alongside the calculated residual and threshold
    to visually explain why the model flags certain points as anomalies.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        feature_name (str): Name of the feature to analyze.
        time_col (str): Name of timestamp column.
    """
    pred_col = f'Total_Prediction_{feature_name}'
    anom_col = f'Anomaly_{feature_name}'

    if pred_col not in df.columns:
        print(f"⚠️ Error: '{pred_col}' not found in DataFrame.")
        return

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col)

    diff = np.abs(df[feature_name] - df[pred_col])
    threshold = np.mean(diff) + 3 * np.std(diff)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Raw vs Predicted", "Residual & Threshold"))

    fig.add_trace(go.Scatter(x=df[time_col], y=df[feature_name], mode='lines', name='Actual', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[time_col], y=df[pred_col], mode='lines', name='Predicted', line=dict(color='red')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df[time_col], y=diff, mode='lines', name='Residual', line=dict(color='orange')), row=2, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold: {threshold:.2f}", row=2, col=1)

    fig.update_layout(height=700, title=f"Residual Analysis for {feature_name}", hovermode="x unified", template="plotly_white")
    fig.show()


# ----------------------------------------
# Cycle Visualization Functions 
#-----------------------------------------

import plotly.express as px

def plot_cycle(df, target_col, show_rangeslider=True, show_rangeselector=True, show_legend=True):
    unit = get_unit(target_col)
    yaxis_label = f"{target_col} ({unit})" if unit else target_col

    fig = px.line(df, x='timestamp', y=target_col, 
                color='cycle_id', title=f'{target_col} by Cycle')
    
    fig.update_layout(
        template="plotly_white",
        xaxis_title='Timestamp',
        yaxis_title=yaxis_label,
        legend_title='Cycle ID',
        showlegend=show_legend,
        xaxis=dict(
            rangeselector=dict(
                visible=show_rangeselector,
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=show_rangeslider),
            type="date"
        ),
        yaxis=dict(
            autorange=True,
            fixedrange=False   # allows y-axis zoom/pan as well
        )
    )
    fig.show(config=export_config)
    
def plot_features_with_cycle_boundaries(df, features, time_col='timestamp', cycle_col='Cycle_ID', title=None):
    """
    Plots one or multiple features with vertical lines marking cycle boundaries.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        features (str or list): Column name(s) to plot.
        time_col (str): The name of the time column (default: 'timestamp').
        cycle_col (str): The name of the cycle ID column (default: 'Cycle_ID').
        title (str): Title of the chart.
    """

    if isinstance(features, str):
        features = [features]

    if title is None:
        title = f"Features with Cycle Boundaries: {', '.join(features)}"

    fig = go.Figure()

    # Add traces for each feature
    for col in features:
        if col not in df.columns:
            print(f"⚠️ Warning: Column '{col}' not found in DataFrame. Skipping.")
            continue

        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[col],
                mode='lines',
                name=col,
                opacity=0.8
            )
        )

    # Add vertical lines at cycle boundaries
    if cycle_col in df.columns:
        cycle_boundaries = df[df[cycle_col] > 0].groupby(cycle_col)[time_col].min()

        for cycle_id, start_time in cycle_boundaries.items():
            fig.add_vline(
                x=start_time,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                annotation_text=f"Cycle {cycle_id}",
                annotation_position="top right"
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            autorange=True,
            fixedrange=False
        ),
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            type="date"
        )
    )

    fig.show(config={'responsive': True, 'scrollZoom': True})


def plot_cycle_overview(full_df, summary_df=None, feature='PermeateFlow', time_col='timestamp', title=None):
    """
    Visualizes RO cycle segmentation on a sensor timeline.

    Plots the full dataset with offline periods in grey and each active cycle
    in a distinct color, using the output of CycleProcessor.extract_hybrid_cycles().

    Args:
        full_df (pd.DataFrame): The third return value from extract_hybrid_cycles()
                                (full data with Cycle_ID and Cycle_Time columns added).
        summary_df (pd.DataFrame): The second return value from extract_hybrid_cycles()
                                   (one row per cycle). If provided, adds cycle-start
                                   annotations with duration info.
        feature (str): Sensor column to plot. Default: 'PermeateFlow'.
        time_col (str): Name of the timestamp column. Default: 'timestamp'.
        title (str): Chart title. Auto-generated if None.
    """
    import plotly.colors as pc

    full_df = full_df.copy()
    full_df[time_col] = pd.to_datetime(full_df[time_col])
    full_df = full_df.sort_values(by=time_col).reset_index(drop=True)

    if feature not in full_df.columns:
        print(f"⚠️ Column '{feature}' not found in DataFrame.")
        return

    if title is None:
        title = f"Cycle Segmentation Overview — {feature}"

    active_cycle_ids = sorted([c for c in full_df['Cycle_ID'].unique() if c > 0])
    n_cycles = len(active_cycle_ids)

    # Build a color palette that cycles through plotly's qualitative colors
    palette = pc.qualitative.Light24 + pc.qualitative.Dark24
    color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(active_cycle_ids)}

    fig = go.Figure()

    # 1. Offline / NaN periods — plot as a single grey background trace
    offline_df = full_df[full_df['Cycle_ID'] == 0]
    if not offline_df.empty:
        fig.add_trace(go.Scatter(
            x=offline_df[time_col],
            y=offline_df[feature],
            mode='lines',
            name='Offline / Invalid',
            line=dict(color='lightgrey', width=1),
            opacity=0.6,
            showlegend=True,
            hovertemplate='<b>Offline</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))

    # 2. One trace per cycle (colored)
    for cid in active_cycle_ids:
        cycle_df = full_df[full_df['Cycle_ID'] == cid]
        color = color_map[cid]

        # Build hover text with Cycle_Time if available
        if 'Cycle_Time' in cycle_df.columns:
            hover = (
                f'<b>Cycle {cid}</b><br>'
                'Time: %{x}<br>'
                f'{feature}: ' + '%{y:.2f}<br>'
                'Cycle Age: %{customdata:.1f} h<extra></extra>'
            )
            custom = cycle_df['Cycle_Time'].values
        else:
            hover = f'<b>Cycle {cid}</b><br>Time: %{{x}}<br>{feature}: %{{y:.2f}}<extra></extra>'
            custom = None

        trace_kwargs = dict(
            x=cycle_df[time_col],
            y=cycle_df[feature],
            mode='lines',
            name=f'Cycle {cid}',
            line=dict(color=color, width=1.5),
            showlegend=(n_cycles <= 30),   # hide legend if too many cycles
            hovertemplate=hover,
        )
        if custom is not None:
            trace_kwargs['customdata'] = custom

        fig.add_trace(go.Scatter(**trace_kwargs))

    # 3. Cycle-start markers from summary_df
    if summary_df is not None and not summary_df.empty:
        valid_summary = summary_df[summary_df['Cycle_ID'].isin(active_cycle_ids)]
        # Get the y-value at each cycle start for marker placement
        marker_y = []
        for _, row in valid_summary.iterrows():
            match = full_df[full_df['Cycle_ID'] == row['Cycle_ID']]
            marker_y.append(match[feature].iloc[0] if not match.empty else None)

        fig.add_trace(go.Scatter(
            x=valid_summary['Start_Time'],
            y=marker_y,
            mode='markers+text',
            marker=dict(symbol='triangle-up', size=10, color='black', opacity=0.7),
            text=[f"C{int(r.Cycle_ID)}<br>{r.Raw_Duration_Hours:.0f}h" for _, r in valid_summary.iterrows()],
            textposition='top center',
            textfont=dict(size=8),
            name='Cycle Start',
            hovertemplate=(
                '<b>Cycle %{customdata[0]}</b><br>'
                'Start: %{x}<br>'
                'Duration: %{customdata[1]:.0f} h<br>'
                'Downtime before: %{customdata[2]:.1f} h<extra></extra>'
            ),
            customdata=list(zip(
                valid_summary['Cycle_ID'],
                valid_summary['Raw_Duration_Hours'],
                valid_summary['Downtime_Before_Start_Hours']
            )),
            showlegend=True
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Time",
        yaxis_title=feature,
        hovermode="x unified",
        template="plotly_white",
        height=550,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9)
        ),
        yaxis=dict(autorange=True, fixedrange=False),
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=7,  label="1w",  step="day",   stepmode="backward"),
                    dict(count=1,  label="1m",  step="month", stepmode="backward"),
                    dict(count=3,  label="3m",  step="month", stepmode="backward"),
                    dict(count=6,  label="6m",  step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ]
            ),
            type="date"
        )
    )

    n_valid = n_cycles
    n_offline = len(offline_df)
    print(f"  Plotted {n_valid} active cycles | {n_offline} offline rows (grey)")
    fig.show(config={'responsive': True, 'scrollZoom': True})


def plot_interactive_dydt(df, parameter, time_col='timestamp', smoothing_window=3):
    """
    Plots Absolute value (Row 1), dy/dt (Row 2), and a Combined Overlay (Row 3).
    Includes built-in smoothing for cleaner derivatives.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        parameter (str): Parameter column name to analyze.
        time_col (str): Name of timestamp column.
        smoothing_window (int): Window size for moving average smoothing.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)

    dt_hours = (df[time_col].diff().dt.total_seconds() / 3600.0).values
    dt_hours = np.where(dt_hours <= 0, 1.0, dt_hours)

    dy_dt_raw = np.gradient(df[parameter].values, dt_hours)
    dy_dt = pd.Series(dy_dt_raw).rolling(window=smoothing_window, center=True).mean().values

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(f"1. {parameter} (Absolute)", "2. dy/dt (Rate of Change)", "3. Combined Overlay"))

    fig.add_trace(go.Scatter(x=df[time_col], y=df[parameter], mode='lines', name=parameter, line=dict(color='blue')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df[time_col], y=dy_dt, mode='lines', name='dy/dt (Smoothed)', line=dict(color='green')), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.add_trace(go.Scatter(x=df[time_col], y=df[parameter], mode='lines', name=parameter, line=dict(color='blue'), opacity=0.5), row=3, col=1)
    fig.add_trace(go.Scatter(x=df[time_col], y=dy_dt, mode='lines', name='dy/dt', line=dict(color='green')), row=3, col=1)

    fig.update_layout(height=900, title=f"Comprehensive dy/dt Analysis for {parameter}", hovermode="x unified", template="plotly_white")
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.show()




def plot_resid_for_ml(dp,true_col, phys_col):
    # Reconstruct the residual
    resid = dp.df[true_col] - dp.df[phys_col]
    resid = resid.dropna().reset_index(drop=True)

    n = len(resid)
    train_end = int(n * 0.64)
    val_end   = int(n * 0.80)

    # Plot residual over time with split markers
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    axes[0].plot(resid, lw=0.5, alpha=0.7)
    axes[0].axvline(train_end, color='green', lw=1.5, label='train→val')
    axes[0].axvline(val_end,   color='red',   lw=1.5, label='val→test')
    axes[0].set_title('Residual over time — look for regime changes at the split lines')
    axes[0].legend()

    # Distribution comparison per split
    axes[1].hist(resid.iloc[:train_end],     bins=60, alpha=0.5, label='Train', density=True)
    axes[1].hist(resid.iloc[train_end:val_end], bins=60, alpha=0.5, label='Val',   density=True)
    axes[1].hist(resid.iloc[val_end:],       bins=60, alpha=0.5, label='Test',  density=True)
    axes[1].set_title('Residual distribution per split — if shapes differ, the problem is confirmed')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print(f"Train resid — mean: {resid.iloc[:train_end].mean():.3f}  std: {resid.iloc[:train_end].std():.3f}")
    print(f"Val   resid — mean: {resid.iloc[train_end:val_end].mean():.3f}  std: {resid.iloc[train_end:val_end].std():.3f}")
    print(f"Test  resid — mean: {resid.iloc[val_end:].mean():.3f}  std: {resid.iloc[val_end:].std():.3f}")
