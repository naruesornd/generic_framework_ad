"""
Root Cause Analysis Dashboard Functions
Interactive and static dashboards for visualizing anomalies and root causes.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output
import re


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
    """Auto-detect unit based on column name ending patterns.
    
    Skips cross-features like FeedFlow_x_FeedPressure to avoid
    assigning units to pairwise interaction terms.
    """
    # ── Skip cross/interaction features ────────────────────────────────────
    CROSS_JOINERS = ['_x_', '_X_', '_cross_', '_interact_', '_mul_', '_times_']
    if any(joiner in target_col for joiner in CROSS_JOINERS):
        return ''
    # ───────────────────────────────────────────────────────────────────────
    
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


def plot_rca_dashboard(df, rca_df, target_col, feature_cols, time_col='timestamp', show_rangeslider=True):
    """
    Builds a vertically stacked dashboard.
    Top Row: Target variable with all anomalies.
    Subsequent Rows: Input features, highlighting the specific feature that caused the anomaly.

    Args:
        df: DataFrame with actual sensor data.
        rca_df: DataFrame with root cause analysis results.
        target_col: Target column name.
        feature_cols: List of feature column names.
        time_col: Timestamp column name.
    """
    print(f"--- Generating Stacked XAI Root Cause Dashboard for {target_col} ---")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    if rca_df is None or rca_df.empty:
        print("✅ No anomalies to plot. The system is healthy!")
        plot_rca = pd.DataFrame(columns=[time_col, target_col, 'Root_Cause', 'Contribution'] + feature_cols)
    else:
        rca_df = rca_df.copy()
        rca_df[time_col] = pd.to_datetime(rca_df[time_col])
        plot_rca = pd.merge(rca_df, df[[time_col, target_col] + feature_cols], left_on=time_col, right_on=time_col, how='left')

    n_features = len(feature_cols)
    total_rows = n_features + 1

    fig = make_subplots(
        rows=total_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
    [f"Target Output: {format_yaxis_label(target_col)}"] +
    [f"Input Feature: {format_yaxis_label(feat)}" for feat in feature_cols])
    )

    fig.add_trace(go.Scattergl(
        x=df[time_col], y=df[target_col],
        mode='lines', name=f'Actual {target_col}', line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)

    if not plot_rca.empty:
        fig.add_trace(go.Scattergl(
            x=plot_rca[time_col], y=plot_rca[target_col],
            mode='markers', name='Anomaly Detected',
            marker=dict(color='yellow', size=10, line=dict(color='red', width=2)),
            hovertemplate="<b>Anomaly</b><br>Time: %{x}<br>Value: %{y:.2f}<br><b>Root Cause: %{customdata}</b><extra></extra>",
            customdata=plot_rca['Root_Cause']
        ), row=1, col=1)

    for i, feature in enumerate(feature_cols):
        current_row = i + 2

        fig.add_trace(go.Scattergl(
            x=df[time_col], y=df[feature],
            mode='lines', name=feature, line=dict(color='gray', width=1), showlegend=False
        ), row=current_row, col=1)

        if not plot_rca.empty:
            feature_anomalies = plot_rca[plot_rca['Root_Cause'] == feature]

            if not feature_anomalies.empty:
                fig.add_trace(go.Scattergl(
                    x=feature_anomalies[time_col], y=feature_anomalies[feature],
                    mode='markers', name=f'{feature} Fault',
                    marker=dict(color='red', size=8, symbol='x'),
                    hovertemplate=f"<b>ROOT CAUSE: {feature}</b><br>Time: %{{x}}<br>Value: %{{y:.2f}}<br>Contribution: %{{customdata:.2f}}<extra></extra>",
                    customdata=feature_anomalies['Contribution'].abs()
                ), row=current_row, col=1)

    # --- Time range buttons ---
    rangeselector_buttons = [
        dict(count=1,  label="1D", step="day",   stepmode="backward"),
        dict(count=7,  label="1W", step="day",   stepmode="backward"),
        dict(count=1,  label="1M", step="month", stepmode="backward"),
        dict(count=3,  label="3M", step="month", stepmode="backward"),
        dict(step="all", label="All"),
    ]

    fig.update_layout(
        title=dict(
            text="Root Cause Analysis (RCA) Diagnostic Dashboard",
            y=0.99,
            yanchor="top"
        ),
        template="plotly_white",
        height=250 * total_rows,
        hovermode="x unified",
        # margin=dict(t=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5, bgcolor='rgba(255,255,255,0.8)',font=dict(size=10)),
        # Attach the range selector to the shared x-axis (xaxis)
        xaxis=dict(
            rangeselector=dict(
                buttons=rangeselector_buttons,
                bgcolor='rgba(240,240,240,0.9)',
                activecolor='#1f77b4',
                y=1.02,        # sit just above the top subplot
                yanchor='bottom',
            ),
            rangeslider=dict(visible=False),  # hide slider on top axis to avoid clutter
        ),
    )

    fig.update_xaxes(matches='x')
    fig.update_xaxes(rangeslider=dict(visible=show_rangeslider, thickness=0.02), row=total_rows, col=1)
    fig.update_yaxes(fixedrange=False, autorange=True)
    fig.update_yaxes(title_text=format_yaxis_label(target_col), row=1, col=1)
    for i, feat in enumerate(feature_cols):
        fig.update_yaxes(title_text=format_yaxis_label(feat), row=i + 2, col=1)


    fig.show(config=export_config)



def plot_rca_by_cycle(df, rca_df, target_col, feature_cols, time_col='timestamp',
                      cycle_col='cycle_id', cycle_time_col='cycle_time', row_height=400, show_rangeslider=True):
    print(f"--- Generating Per-Cycle RCA Selector for {target_col} ---")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # ── Valid cycles only ──────────────────────────────────────────────────
    cycle_df = df[df[cycle_col] != -1].copy()
    cycles   = sorted(cycle_df[cycle_col].unique())

    # ── Merge RCA results ─────────────────────────────────────────────────
    if rca_df is not None and not rca_df.empty:
        rca_df = rca_df.copy()
        rca_df[time_col] = pd.to_datetime(rca_df[time_col])
        pull_cols  = [time_col, cycle_col, target_col] + [f for f in feature_cols if f in df.columns]
        rca_merged = pd.merge(rca_df, df[pull_cols], on=time_col, how='left')
        rca_merged = rca_merged[rca_merged[cycle_col].notna() & (rca_merged[cycle_col] != -1)]
    else:
        rca_merged = pd.DataFrame()

    # ── Global y-range (1st–99th pct) ─────────────────────────────────────
    feat_cols_present = [f for f in feature_cols if f in df.columns]
    if feat_cols_present:
        all_feat_vals = pd.concat([df[f].dropna() for f in feat_cols_present])
        gmin, gmax    = all_feat_vals.quantile(0.01), all_feat_vals.quantile(0.99)
        pad           = (gmax - gmin) * 0.05
        feat_yrange   = [gmin - pad, gmax + pad]
    else:
        feat_yrange = None

    # ── Pre-compute per-cycle statistics ──────────────────────────────────
    all_cols   = [target_col] + list(feature_cols)
    cycle_stats = {}
    for cycle_id in cycles:
        cdata = cycle_df[cycle_df[cycle_col] == cycle_id]
        cycle_stats[cycle_id] = {}
        for col in all_cols:
            if col in cdata.columns:
                vals = cdata[col].dropna()
                if len(vals) > 0:
                    cycle_stats[cycle_id][col] = {
                        'mean' : float(vals.mean()),
                        'std'  : float(vals.std()),
                        'min'  : float(vals.min()),
                        'max'  : float(vals.max()),
                        'count': int(len(vals)),
                    }

    # ── Layout constants ───────────────────────────────────────────────────
    n_features = len(feature_cols)
    total_rows = n_features + 1
    VS         = 0.03   # vertical_spacing (must match make_subplots below)

    # Paper-coordinate y of the top edge of subplot row_idx (0-indexed from top)
    subplot_h = (1.0 - (total_rows - 1) * VS) / total_rows
    def subplot_ytop(row_idx):
        return 1.0 - row_idx * (subplot_h + VS)

    # ── Figure ────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=total_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=VS,
        subplot_titles=(
            [f"Target: {format_yaxis_label(target_col)}"] +
            [f"Feature: {format_yaxis_label(f)}" for f in feature_cols]
        )
    )

    # Capture subplot-title annotations BEFORE adding anything else
    # so we can re-inject them when the dropdown updates annotations.
    subplot_title_anns = list(fig.layout.annotations)

    # ── Helper: stats annotation boxes for one cycle ───────────────────────
    def make_stats_annotations(cycle_id):
        """Returns subplot-title anns + per-subplot stats boxes."""
        anns = list(subplot_title_anns)          # always keep subplot titles
        for row_idx, col in enumerate(all_cols):
            stats = cycle_stats.get(cycle_id, {}).get(col)
            if not stats:
                continue
            unit     = get_unit(col)
            u        = f" {unit}" if unit else ""
            n_anoms  = (
                len(rca_merged[(rca_merged[cycle_col] == cycle_id) &
                               (rca_merged['Root_Cause'] == col)])
                if (not rca_merged.empty and row_idx > 0) else
                len(rca_merged[rca_merged[cycle_col] == cycle_id])
                if not rca_merged.empty else 0
            )
            anom_str = f"<br><b style='color:red'>⚠ Faults: {n_anoms}</b>" if n_anoms > 0 else ""
            text = (
                f"<b>μ</b>={stats['mean']:.3f}{u}  "
                f"<b>σ</b>={stats['std']:.3f}<br>"
                f"Min={stats['min']:.3f}  "
                f"Max={stats['max']:.3f}"
                f"{anom_str}"
            )
            anns.append(dict(
                xref='paper', yref='paper',
                x=0.995, y=subplot_ytop(row_idx) - 0.004,
                xanchor='right', yanchor='top',
                text=text,
                showarrow=False,
                align='right',
                bgcolor='rgba(255,255,255,0.88)',
                bordercolor='#cccccc',
                borderwidth=1,
                font=dict(size=9, color='#333333'),
            ))
        return anns

    # ── Static legend traces ───────────────────────────────────────────────
    LEGEND_TRACES = 5
    for name, kw in [
        (target_col,          dict(mode='lines',   line=dict(color='#1f77b4', width=2))),
        ('Anomaly Detected',  dict(mode='markers', marker=dict(color='yellow', size=10, line=dict(color='red', width=2)))),
        ('Root Cause',        dict(mode='markers', marker=dict(color='red', size=9, symbol='x-thin', line=dict(width=2.5)))),
        ('Cycle Mean',        dict(mode='lines',   line=dict(color='orange', width=1.5, dash='dash'))),
        ('Mean ± 1σ',         dict(mode='lines',   fill='toself', fillcolor='rgba(255,165,0,0.12)',
                                   line=dict(color='rgba(255,165,0,0)', width=0))),
    ]:
        fig.add_trace(go.Scatter(x=[None], y=[None], name=name, showlegend=True, **kw), row=1, col=1)

    # ── Per-cycle traces ───────────────────────────────────────────────────
    trace_cycle_map = []

    def _mean_band_traces(cycle_id, col, cdata, row, show):
        """Add mean line + ±1σ band for one column. Returns 2 traces."""
        stats = cycle_stats.get(cycle_id, {}).get(col)
        if stats and len(cdata) > 0:
            m, s  = stats['mean'], stats['std']
            x0, x1 = cdata[time_col].iloc[0], cdata[time_col].iloc[-1]
            x_band  = [x0, x1, x1, x0]
            y_band  = [m+s, m+s, m-s, m-s]
            fig.add_trace(go.Scatter(
                x=x_band, y=y_band, fill='toself',
                fillcolor='rgba(255,165,0,0.12)',
                line=dict(color='rgba(255,165,0,0)', width=0),
                visible=show, showlegend=False,
                hoverinfo='skip', name='std band'
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[m, m], mode='lines',
                line=dict(color='orange', width=1.5, dash='dash'),
                visible=show, showlegend=False,
                hovertemplate=f"Mean {col}: {m:.3f}<extra></extra>",
                name='mean'
            ), row=row, col=1)
        else:
            fig.add_trace(go.Scatter(x=[], y=[], visible=show, showlegend=False), row=row, col=1)
            fig.add_trace(go.Scatter(x=[], y=[], visible=show, showlegend=False), row=row, col=1)
        trace_cycle_map.extend([cycle_idx, cycle_idx])

    for cycle_idx, cycle_id in enumerate(cycles):
        cdata    = cycle_df[cycle_df[cycle_col] == cycle_id].sort_values(time_col)
        show     = (cycle_idx == 0)
        c_anoms  = rca_merged[rca_merged[cycle_col] == cycle_id] if not rca_merged.empty else pd.DataFrame()

        # ── Target line ───────────────────────────────────────────────────
        fig.add_trace(go.Scattergl(
            x=cdata[time_col], y=cdata[target_col],
            mode='lines', name=target_col,
            line=dict(color='#1f77b4', width=2),
            visible=show, showlegend=False,
            hovertemplate=f"{target_col}: %{{y:.3f}}<br>%{{x}}<extra></extra>"
        ), row=1, col=1)
        trace_cycle_map.append(cycle_idx)

        # ── Target anomaly markers ─────────────────────────────────────────
        if not c_anoms.empty and target_col in c_anoms.columns:
            fig.add_trace(go.Scattergl(
                x=c_anoms[time_col], y=c_anoms[target_col],
                mode='markers', name='Anomaly',
                marker=dict(color='yellow', size=10, line=dict(color='red', width=2)),
                visible=show, showlegend=False,
                customdata=c_anoms[['Root_Cause', 'Contribution']].values,
                hovertemplate=(
                    "<b>⚠ Anomaly</b><br>%{x}<br>"
                    f"{target_col}: %{{y:.3f}}<br>"
                    "Root Cause: %{customdata[0]}<br>"
                    "Contribution: %{customdata[1]:.4f}<extra></extra>"
                )
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scattergl(x=[], y=[], visible=show, showlegend=False), row=1, col=1)
        trace_cycle_map.append(cycle_idx)

        # ── Target mean + std band ─────────────────────────────────────────
        _mean_band_traces(cycle_id, target_col, cdata, row=1, show=show)

        # ── Feature rows ──────────────────────────────────────────────────
        for feat_idx, feat in enumerate(feature_cols):
            feat_row = feat_idx + 2

            fig.add_trace(go.Scattergl(
                x=cdata[time_col],
                y=cdata[feat] if feat in cdata.columns else [],
                mode='lines', name=feat,
                line=dict(color='steelblue', width=1.2),
                visible=show, showlegend=False,
                hovertemplate=f"{feat}: %{{y:.3f}}<br>%{{x}}<extra></extra>"
            ), row=feat_row, col=1)
            trace_cycle_map.append(cycle_idx)

            feat_rc = c_anoms[c_anoms['Root_Cause'] == feat] if not c_anoms.empty else pd.DataFrame()
            if not feat_rc.empty and feat in feat_rc.columns:
                fig.add_trace(go.Scattergl(
                    x=feat_rc[time_col], y=feat_rc[feat],
                    mode='markers', name=f'{feat} root cause',
                    marker=dict(color='red', size=9, symbol='x-thin', line=dict(width=2.5)),
                    visible=show, showlegend=False,
                    customdata=feat_rc[['Contribution']].abs().values,
                    hovertemplate=(
                        f"<b>ROOT CAUSE: {feat}</b><br>%{{x}}<br>"
                        f"{feat}: %{{y:.3f}}<br>"
                        "Contribution: %{customdata[0]:.4f}<extra></extra>"
                    )
                ), row=feat_row, col=1)
            else:
                fig.add_trace(go.Scattergl(x=[], y=[], visible=show, showlegend=False), row=feat_row, col=1)
            trace_cycle_map.append(cycle_idx)

            # Feature mean + std band
            _mean_band_traces(cycle_id, feat, cdata, row=feat_row, show=show)

    # ── Dropdown buttons ───────────────────────────────────────────────────
    total_data_traces = len(trace_cycle_map)
    buttons = []

    for sel_idx, cycle_id in enumerate(cycles):
        c_subset = cycle_df[cycle_df[cycle_col] == cycle_id]
        x_start  = str(c_subset[time_col].min())
        x_end    = str(c_subset[time_col].max())
        n_anoms  = len(rca_merged[rca_merged[cycle_col] == cycle_id]) if not rca_merged.empty else 0
        label    = f"Cycle {int(cycle_id)}  {'⚠ (' + str(n_anoms) + ')' if n_anoms > 0 else '✅'}"
        subtitle = f"⚠ {n_anoms} anomalies detected" if n_anoms > 0 else "✅ No anomalies — normal cycle"

        visible_mask = (
            [True] * LEGEND_TRACES
            + [trace_cycle_map[t] == sel_idx for t in range(total_data_traces)]
        )

        buttons.append(dict(
            label=label,
            method='update',
            args=[
                {'visible': visible_mask},
                {
                    'title.text': (
                        f"RCA Cycle View — {target_col}  |  "
                        f"Cycle {int(cycle_id)}  —  {subtitle}"
                    ),
                    'xaxis.range':     [x_start, x_end],
                    'xaxis.autorange': False,
                    'annotations':     make_stats_annotations(cycle_id),  # ← updates stats boxes
                }
            ]
        ))

    # ── Layout ────────────────────────────────────────────────────────────
    first_cycle_id = int(cycles[0])
    first_n_anoms  = len(rca_merged[rca_merged[cycle_col] == cycles[0]]) if not rca_merged.empty else 0
    first_data     = cycle_df[cycle_df[cycle_col] == cycles[0]]

    fig.update_layout(
        title=dict(
            text=(
                f"RCA Cycle View — {target_col}  |  Cycle {first_cycle_id}  —  "
                f"{'⚠ ' + str(first_n_anoms) + ' anomalies' if first_n_anoms else '✅ No anomalies'}"
            ),
            x=0.5, xanchor='center',
            y=0.99, yanchor='top'
        ),
        updatemenus=[dict(
            active=0, buttons=buttons, direction='down', showactive=True,
            x=0.0, xanchor='left', y=1.08, yanchor='top',
            bgcolor='rgba(240,240,240,0.95)', bordercolor='#aaaaaa', font=dict(size=12)
        )],
        template='plotly_white',
        height=row_height * total_rows,
        hovermode='x unified',
        legend=dict(
            orientation='h', yanchor='bottom', y=1.04,
            xanchor='center', x=0.5,
            bgcolor='rgba(255,255,255,0.8)', font=dict(size=14),
        ),
        xaxis=dict(
            range=[str(first_data[time_col].min()), str(first_data[time_col].max())],
            type='date', autorange=False
        ),
        annotations=make_stats_annotations(cycles[0])   # ← initial stats boxes
    )

    fig.update_xaxes(
        tickfont=dict(size=14),
        title_font=dict(size=16),
    )
    fig.update_yaxes(
        tickfont=dict(size=14),
        title_font=dict(size=16),
    )

    fig.update_xaxes(matches='x')
    fig.update_xaxes(
        title_text="Timestamp",
        rangeslider=dict(visible=show_rangeslider, thickness=0.025),
        row=total_rows, col=1
    )

    fig.update_yaxes(title_text=format_yaxis_label(target_col), fixedrange=False, autorange=True, row=1, col=1)
    for feat_idx, feat in enumerate(feature_cols):
        update_kwargs = dict(title_text=format_yaxis_label(feat), fixedrange=False, row=feat_idx + 2, col=1)
        if feat_yrange:
            update_kwargs['range'] = feat_yrange
        fig.update_yaxes(**update_kwargs)



    fig.show(config=export_config) 



# def plot_rca_by_cycle(df, rca_df, target_col, feature_cols, time_col='timestamp',
#                       cycle_col='cycle_id', cycle_time_col='cycle_time', row_height=400, show_rangeslider=True):
#     print(f"--- Generating Per-Cycle RCA Selector for {target_col} ---")

#     df = df.copy()
#     df[time_col] = pd.to_datetime(df[time_col])

#     # ── Valid cycles only ──────────────────────────────────────────────────
#     cycle_df = df[df[cycle_col] != -1].copy()
#     cycles   = sorted(cycle_df[cycle_col].unique())

#     # ── Merge RCA results ─────────────────────────────────────────────────
#     if rca_df is not None and not rca_df.empty:
#         rca_df = rca_df.copy()
#         rca_df[time_col] = pd.to_datetime(rca_df[time_col])
#         pull_cols  = [time_col, cycle_col, target_col] + [f for f in feature_cols if f in df.columns]
#         rca_merged = pd.merge(rca_df, df[pull_cols], on=time_col, how='left')
#         rca_merged = rca_merged[rca_merged[cycle_col].notna() & (rca_merged[cycle_col] != -1)]
#     else:
#         rca_merged = pd.DataFrame()

#     # ── Global y-range (1st–99th pct) ─────────────────────────────────────
#     feat_cols_present = [f for f in feature_cols if f in df.columns]
#     if feat_cols_present:
#         all_feat_vals = pd.concat([df[f].dropna() for f in feat_cols_present])
#         gmin, gmax    = all_feat_vals.quantile(0.01), all_feat_vals.quantile(0.99)
#         pad           = (gmax - gmin) * 0.05
#         feat_yrange   = [gmin - pad, gmax + pad]
#     else:
#         feat_yrange = None

#     # ── Pre-compute per-cycle statistics ──────────────────────────────────
#     all_cols   = [target_col] + list(feature_cols)
#     cycle_stats = {}
#     for cycle_id in cycles:
#         cdata = cycle_df[cycle_df[cycle_col] == cycle_id]
#         cycle_stats[cycle_id] = {}
#         for col in all_cols:
#             if col in cdata.columns:
#                 vals = cdata[col].dropna()
#                 if len(vals) > 0:
#                     cycle_stats[cycle_id][col] = {
#                         'mean' : float(vals.mean()),
#                         'std'  : float(vals.std()),
#                         'min'  : float(vals.min()),
#                         'max'  : float(vals.max()),
#                         'count': int(len(vals)),
#                     }

#     # ── Layout constants ───────────────────────────────────────────────────
#     n_features = len(feature_cols)
#     total_rows = n_features + 1
#     VS         = 0.03   # vertical_spacing (must match make_subplots below)

#     # Paper-coordinate y of the top edge of subplot row_idx (0-indexed from top)
#     subplot_h = (1.0 - (total_rows - 1) * VS) / total_rows
#     def subplot_ytop(row_idx):
#         return 1.0 - row_idx * (subplot_h + VS)

#     # ── Figure ────────────────────────────────────────────────────────────
#     fig = make_subplots(
#         rows=total_rows, cols=1,
#         shared_xaxes=True,
#         vertical_spacing=VS,
#         subplot_titles=(
#             [f"Target: {format_yaxis_label(target_col)}"] +
#             [f"Feature: {format_yaxis_label(f)}" for f in feature_cols]
#         )
#     )

#     # Capture subplot-title annotations BEFORE adding anything else
#     # so we can re-inject them when the dropdown updates annotations.
#     subplot_title_anns = list(fig.layout.annotations)

#     # ── Helper: stats annotation boxes for one cycle ───────────────────────
#     def make_stats_annotations(cycle_id):
#         """Returns subplot-title anns + per-subplot stats boxes."""
#         anns = list(subplot_title_anns)          # always keep subplot titles
#         for row_idx, col in enumerate(all_cols):
#             stats = cycle_stats.get(cycle_id, {}).get(col)
#             if not stats:
#                 continue
#             unit     = get_unit(col)
#             u        = f" {unit}" if unit else ""
#             n_anoms  = (
#                 len(rca_merged[(rca_merged[cycle_col] == cycle_id) &
#                                (rca_merged['Root_Cause'] == col)])
#                 if (not rca_merged.empty and row_idx > 0) else
#                 len(rca_merged[rca_merged[cycle_col] == cycle_id])
#                 if not rca_merged.empty else 0
#             )
#             anom_str = f"<br><b style='color:red'>⚠ Faults: {n_anoms}</b>" if n_anoms > 0 else ""
#             text = (
#                 f"<b>μ</b>={stats['mean']:.3f}{u}  "
#                 f"<b>σ</b>={stats['std']:.3f}<br>"
#                 f"Min={stats['min']:.3f}  "
#                 f"Max={stats['max']:.3f}"
#                 f"{anom_str}"
#             )
#             anns.append(dict(
#                 xref='paper', yref='paper',
#                 x=0.995, y=subplot_ytop(row_idx) - 0.004,
#                 xanchor='right', yanchor='top',
#                 text=text,
#                 showarrow=False,
#                 align='right',
#                 bgcolor='rgba(255,255,255,0.88)',
#                 bordercolor='#cccccc',
#                 borderwidth=1,
#                 font=dict(size=9, color='#333333'),
#             ))
#         return anns

#     # ── Static legend traces ───────────────────────────────────────────────
#     LEGEND_TRACES = 5
#     for name, kw in [
#         (target_col,          dict(mode='lines',   line=dict(color='#1f77b4', width=2))),
#         ('Anomaly Detected',  dict(mode='markers', marker=dict(color='yellow', size=10, line=dict(color='red', width=2)))),
#         ('Root Cause',        dict(mode='markers', marker=dict(color='red', size=9, symbol='x-thin', line=dict(width=2.5)))),
#         ('Cycle Mean',        dict(mode='lines',   line=dict(color='orange', width=1.5, dash='dash'))),
#         ('Mean ± 1σ',         dict(mode='lines',   fill='toself', fillcolor='rgba(255,165,0,0.12)',
#                                    line=dict(color='rgba(255,165,0,0)', width=0))),
#     ]:
#         fig.add_trace(go.Scatter(x=[None], y=[None], name=name, showlegend=True, **kw), row=1, col=1)

#     # ── Per-cycle traces ───────────────────────────────────────────────────
#     trace_cycle_map = []

#     def _mean_band_traces(cycle_id, col, cdata, row, show):
#         """Add mean line + ±1σ band for one column. Returns 2 traces."""
#         stats = cycle_stats.get(cycle_id, {}).get(col)
#         if stats and len(cdata) > 0:
#             m, s  = stats['mean'], stats['std']
#             x0, x1 = cdata[time_col].iloc[0], cdata[time_col].iloc[-1]
#             x_band  = [x0, x1, x1, x0]
#             y_band  = [m+s, m+s, m-s, m-s]
#             fig.add_trace(go.Scatter(
#                 x=x_band, y=y_band, fill='toself',
#                 fillcolor='rgba(255,165,0,0.12)',
#                 line=dict(color='rgba(255,165,0,0)', width=0),
#                 visible=show, showlegend=False,
#                 hoverinfo='skip', name='std band'
#             ), row=row, col=1)
#             fig.add_trace(go.Scatter(
#                 x=[x0, x1], y=[m, m], mode='lines',
#                 line=dict(color='orange', width=1.5, dash='dash'),
#                 visible=show, showlegend=False,
#                 hovertemplate=f"Mean {col}: {m:.3f}<extra></extra>",
#                 name='mean'
#             ), row=row, col=1)
#         else:
#             fig.add_trace(go.Scatter(x=[], y=[], visible=show, showlegend=False), row=row, col=1)
#             fig.add_trace(go.Scatter(x=[], y=[], visible=show, showlegend=False), row=row, col=1)
#         trace_cycle_map.extend([cycle_idx, cycle_idx])

#     for cycle_idx, cycle_id in enumerate(cycles):
#         cdata    = cycle_df[cycle_df[cycle_col] == cycle_id].sort_values(time_col)
#         show     = (cycle_idx == 0)
#         c_anoms  = rca_merged[rca_merged[cycle_col] == cycle_id] if not rca_merged.empty else pd.DataFrame()

#         # ── Target line ───────────────────────────────────────────────────
#         fig.add_trace(go.Scattergl(
#             x=cdata[time_col], y=cdata[target_col],
#             mode='lines', name=target_col,
#             line=dict(color='#1f77b4', width=2),
#             visible=show, showlegend=False,
#             hovertemplate=f"{target_col}: %{{y:.3f}}<br>%{{x}}<extra></extra>"
#         ), row=1, col=1)
#         trace_cycle_map.append(cycle_idx)

#         # ── Target anomaly markers ─────────────────────────────────────────
#         if not c_anoms.empty and target_col in c_anoms.columns:
#             fig.add_trace(go.Scattergl(
#                 x=c_anoms[time_col], y=c_anoms[target_col],
#                 mode='markers', name='Anomaly',
#                 marker=dict(color='yellow', size=10, line=dict(color='red', width=2)),
#                 visible=show, showlegend=False,
#                 customdata=c_anoms[['Root_Cause', 'Contribution']].values,
#                 hovertemplate=(
#                     "<b>⚠ Anomaly</b><br>%{x}<br>"
#                     f"{target_col}: %{{y:.3f}}<br>"
#                     "Root Cause: %{customdata[0]}<br>"
#                     "Contribution: %{customdata[1]:.4f}<extra></extra>"
#                 )
#             ), row=1, col=1)
#         else:
#             fig.add_trace(go.Scattergl(x=[], y=[], visible=show, showlegend=False), row=1, col=1)
#         trace_cycle_map.append(cycle_idx)

#         # ── Target mean + std band ─────────────────────────────────────────
#         _mean_band_traces(cycle_id, target_col, cdata, row=1, show=show)

#         # ── Feature rows ──────────────────────────────────────────────────
#         for feat_idx, feat in enumerate(feature_cols):
#             feat_row = feat_idx + 2

#             fig.add_trace(go.Scattergl(
#                 x=cdata[time_col],
#                 y=cdata[feat] if feat in cdata.columns else [],
#                 mode='lines', name=feat,
#                 line=dict(color='steelblue', width=1.2),
#                 visible=show, showlegend=False,
#                 hovertemplate=f"{feat}: %{{y:.3f}}<br>%{{x}}<extra></extra>"
#             ), row=feat_row, col=1)
#             trace_cycle_map.append(cycle_idx)

#             feat_rc = c_anoms[c_anoms['Root_Cause'] == feat] if not c_anoms.empty else pd.DataFrame()
#             if not feat_rc.empty and feat in feat_rc.columns:
#                 fig.add_trace(go.Scattergl(
#                     x=feat_rc[time_col], y=feat_rc[feat],
#                     mode='markers', name=f'{feat} root cause',
#                     marker=dict(color='red', size=9, symbol='x-thin', line=dict(width=2.5)),
#                     visible=show, showlegend=False,
#                     customdata=feat_rc[['Contribution']].abs().values,
#                     hovertemplate=(
#                         f"<b>ROOT CAUSE: {feat}</b><br>%{{x}}<br>"
#                         f"{feat}: %{{y:.3f}}<br>"
#                         "Contribution: %{customdata[0]:.4f}<extra></extra>"
#                     )
#                 ), row=feat_row, col=1)
#             else:
#                 fig.add_trace(go.Scattergl(x=[], y=[], visible=show, showlegend=False), row=feat_row, col=1)
#             trace_cycle_map.append(cycle_idx)

#             # Feature mean + std band
#             _mean_band_traces(cycle_id, feat, cdata, row=feat_row, show=show)

#     # ── Dropdown buttons ───────────────────────────────────────────────────
#     total_data_traces = len(trace_cycle_map)
#     buttons = []

#     for sel_idx, cycle_id in enumerate(cycles):
#         c_subset = cycle_df[cycle_df[cycle_col] == cycle_id]
#         x_start  = str(c_subset[time_col].min())
#         x_end    = str(c_subset[time_col].max())
#         n_anoms  = len(rca_merged[rca_merged[cycle_col] == cycle_id]) if not rca_merged.empty else 0
#         label    = f"Cycle {int(cycle_id)}  {'⚠ (' + str(n_anoms) + ')' if n_anoms > 0 else '✅'}"
#         subtitle = f"⚠ {n_anoms} anomalies detected" if n_anoms > 0 else "✅ No anomalies — normal cycle"

#         visible_mask = (
#             [True] * LEGEND_TRACES
#             + [trace_cycle_map[t] == sel_idx for t in range(total_data_traces)]
#         )

#         buttons.append(dict(
#             label=label,
#             method='update',
#             args=[
#                 {'visible': visible_mask},
#                 {
#                     'title.text': (
#                         f"RCA Cycle View — {target_col}  |  "
#                         f"Cycle {int(cycle_id)}  —  {subtitle}"
#                     ),
#                     'xaxis.range':     [x_start, x_end],
#                     'xaxis.autorange': False,
#                     'annotations':     make_stats_annotations(cycle_id),  # ← updates stats boxes
#                 }
#             ]
#         ))

#     # ── Layout ────────────────────────────────────────────────────────────
#     first_cycle_id = int(cycles[0])
#     first_n_anoms  = len(rca_merged[rca_merged[cycle_col] == cycles[0]]) if not rca_merged.empty else 0
#     first_data     = cycle_df[cycle_df[cycle_col] == cycles[0]]

#     fig.update_layout(
#         title=dict(
#             text=(
#                 f"RCA Cycle View — {target_col}  |  Cycle {first_cycle_id}  —  "
#                 f"{'⚠ ' + str(first_n_anoms) + ' anomalies' if first_n_anoms else '✅ No anomalies'}"
#             ),
#             x=0.5, xanchor='center',
#             y=0.99, yanchor='top',
#             font=dict(size=18)  # ← ENLARGED: title font
#         ),
#         updatemenus=[dict(
#             active=0, buttons=buttons, direction='down', showactive=True,
#             x=0.0, xanchor='left', y=1.08, yanchor='top',
#             bgcolor='rgba(240,240,240,0.95)', bordercolor='#aaaaaa', font=dict(size=14)  # ← ENLARGED: dropdown font
#         )],
#         template='plotly_white',
#         height=row_height * total_rows,
#         hovermode='x unified',
#         legend=dict(
#             orientation='h', 
#             yanchor='bottom', 
#             y=1.04,
#             xanchor='center', 
#             x=0.5,
#             bgcolor='rgba(255,255,255,0.9)',   # ← slightly less transparent
#             bordercolor='#888888',              # ← darker border for visibility
#             borderwidth=2,                      # ← ADDED: thicker border
#             font=dict(size=16),
#         ),
#         xaxis=dict(
#             range=[str(first_data[time_col].min()), str(first_data[time_col].max())],
#             type='date', autorange=False
#         ),
#         annotations=make_stats_annotations(cycles[0])   # ← initial stats boxes
#     )

#     # ── ENLARGED AXIS TEXT ─────────────────────────────────────────────────
#     fig.update_xaxes(
#         tickfont=dict(size=16),      # ← was 14, now 16
#         title_font=dict(size=20),    # ← was 16, now 20
#     )
#     fig.update_yaxes(
#         tickfont=dict(size=16),      # ← was 14, now 16
#         title_font=dict(size=20),    # ← was 16, now 20
#     )
#     # ───────────────────────────────────────────────────────────────────────

#     fig.update_xaxes(matches='x')
#     fig.update_xaxes(
#         title_text="Timestamp",
#         title_font=dict(size=20),    # ← ENLARGED: bottom x-axis title
#         tickfont=dict(size=16),      # ← ENLARGED: bottom x-axis ticks
#         rangeslider=dict(visible=show_rangeslider, thickness=0.025),
#         row=total_rows, col=1
#     )

#     fig.update_yaxes(title_text=format_yaxis_label(target_col), fixedrange=False, autorange=True, row=1, col=1)
#     for feat_idx, feat in enumerate(feature_cols):
#         update_kwargs = dict(title_text=format_yaxis_label(feat), fixedrange=False, row=feat_idx + 2, col=1)
#         if feat_yrange:
#             update_kwargs['range'] = feat_yrange
#         fig.update_yaxes(**update_kwargs)

#     # ── ENLARGED SUBPLOT TITLES ───────────────────────────────────────────
#     fig.update_annotations(font=dict(size=16))  # ← was default (~12), now 18
#     # ───────────────────────────────────────────────────────────────────────

#     fig.show(config=export_config)