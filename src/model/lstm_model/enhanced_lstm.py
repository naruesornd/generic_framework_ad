"""
Hybrid LSTM Model: Physics + ML Residual Prediction
----------------------------------------------------
All plots are now interactive Plotly charts and use real timestamps
on the x-axis instead of integer time steps.

Usage:
    lstm_hybrid_model(
        dp                  = dp,
        selected_features   = ['feat1', 'feat2', ...],
        target_col          = 'PermeateConductivity',
        physics_col         = 'PermeateConductivity_Phys',
        feature_name        = 'PermeateConductivity',
        timestamp_col       = 'timestamp',       # column with datetime values
        cycle_col           = 'Cycle_ID',
        cycle_buffer        = 5,
    )
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import joblib
import re


# ── Model ─────────────────────────────────────────────────────────────────────

class EnhancedLSTM(nn.Module):
    """LSTM with additive attention for residual prediction."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def _make_loader(X_seq, y_seq, batch_size=64, shuffle=False, generator=None):
    ds = TensorDataset(
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(y_seq.reshape(-1, 1), dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _set_seed(seed: int):
    """Pin every source of randomness so training is reproducible."""
    import random as _py_random
    _py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_cycle_buffer_mask(cycle_ids: np.ndarray, seq_len: int,
                              cycle_buffer: int) -> np.ndarray:
    """Boolean mask (True = suppress) for output timesteps near cycle boundaries."""
    n_out = len(cycle_ids) - seq_len
    boundaries = np.where(cycle_ids[1:] != cycle_ids[:-1])[0] + 1

    mask = np.zeros(n_out, dtype=bool)
    for b in boundaries:
        out_b = b - seq_len
        lo = max(0, out_b - cycle_buffer)
        hi = min(n_out, out_b + cycle_buffer + 1)
        mask[lo:hi] = True
    return mask


# ── Training helpers ──────────────────────────────────────────────────────────

def _evaluate(model, loader, device, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for Xb, yb in loader:
            pred = model(Xb.to(device))
            total += criterion(pred, yb.to(device)).item() * Xb.size(0)
    return total / len(loader.dataset)


def _train(model, train_loader, val_loader, optimizer, criterion,
           device, num_epochs=100, patience=5):
    import copy
    best_loss     = float("inf")
    no_improve    = 0
    best_weights  = copy.deepcopy(model.state_dict())
    train_history = []
    val_history   = []

    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * Xb.size(0)

        avg_train = running / len(train_loader.dataset)
        val_loss  = _evaluate(model, val_loader, device, criterion)
        train_history.append(avg_train)
        val_history.append(val_loss)
        print(f"  Epoch {epoch+1:3d}/{num_epochs}  train={avg_train:.5f}  val={val_loss:.5f}")

        if val_loss < best_loss:
            best_loss    = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_weights)
    return model, train_history, val_history


# ── Helpers for plotting buffer regions on a time axis ───────────────────────

def _mask_to_intervals(mask: np.ndarray, timestamps: np.ndarray):
    """
    Convert a boolean mask into a list of (start, end) timestamp intervals.
    Used for shading buffer zones on time-axis plots.
    """
    if mask is None or not mask.any():
        return []
    diffs = np.diff(mask.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends   = np.where(diffs == -1)[0] + 1

    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])

    return [(timestamps[s], timestamps[min(e, len(timestamps) - 1)])
            for s, e in zip(starts, ends)]


def _add_buffer_shading(fig, intervals, row, col, label, fillcolor="rgba(150,150,150,0.18)"):
    """Add gray shaded vertical regions for cycle buffer zones."""
    for i, (x0, x1) in enumerate(intervals):
        kwargs = dict(x0=x0, x1=x1, fillcolor=fillcolor, line_width=0,
                      row=row, col=col, layer="below")
        if label and i == 0:
            kwargs.update(annotation_text=label, annotation_position="top left",
                          annotation=dict(font=dict(size=9)))
        fig.add_vrect(**kwargs)


# ── Plotly plotting ──────────────────────────────────────────────────────────

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


def _plot_learning_curves(train_history, val_history, feature_name, save_dir):
    """Interactive Plotly learning curves."""
    epochs = list(range(1, len(train_history) + 1))
    best_epoch = int(np.argmin(val_history)) + 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=train_history,
        mode='lines+markers', name='Train loss',
        line=dict(color='steelblue', width=2),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=val_history,
        mode='lines+markers', name='Validation loss',
        line=dict(color='tomato', width=2),
        marker=dict(size=7, symbol='square'),
    ))
    fig.add_vline(
        x=best_epoch, line_dash="dash", line_color="green",
        annotation_text=f"Best epoch ({best_epoch})",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"Learning curves: {feature_name}",
        xaxis_title="Epoch",
        yaxis_title="Huber loss",
        template="plotly_white",
        hovermode="x unified",
        height=450,
    )
    fig.show()


def _plot_hybrid(timestamps, y_true, y_phys, y_resid, y_total, anomalies,
                 buffer_mask, threshold, feature_name, r2, mae,
                 save_dir, cycle_buffer, show_rangeslider=False,
                 panels=None):   # ← NEW: optional panel selector
    """
    Interactive 4-panel Plotly figure. 
    panels: list of panel numbers to show, e.g. [1], [1,3], or None for all.
            1=Raw vs prediction, 2=Physics vs ML, 3=Error, 4=Flags
    """
    # Default: show all panels
    if panels is None:
        panels = [1, 2, 3, 4]
    
    errors = np.abs(y_true - y_total)
    anom_idx = np.where(anomalies)[0]
    buffer_intervals = _mask_to_intervals(buffer_mask, timestamps)

    # Build subplot specs dynamically
    n_rows = len(panels)
    row_heights = {1: 0.32, 2: 0.24, 3: 0.22, 4: 0.22}
    heights = [row_heights[p] for p in panels]
    
    # Normalize heights to sum to 1
    total = sum(heights)
    heights = [h/total for h in heights]

    specs = []
    subplot_titles = []
    for p in panels:
        specs.append([{"secondary_y": (p == 2)}])  # only panel 2 needs dual y
        titles = {
            1: "",
            2: "Panel B: Physics contribution vs ML residual",
            3: "Panel C: Prediction error and 3σ threshold",
            4: "Panel D: Anomaly flags",
        }
        subplot_titles.append(titles[p])

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=heights,
        subplot_titles=tuple(subplot_titles),
        specs=specs,
    )

    # Map original panel number to actual row in the figure
    panel_to_row = {p: i+1 for i, p in enumerate(panels)}

    # ── Panel 1 traces ──────────────────────────────────────────────────────
    if 1 in panels:
        r = panel_to_row[1]
        fig.add_trace(go.Scatter(
            x=timestamps, y=y_true, name="True sensor",
            line=dict(color="steelblue", width=1.4),
            hovertemplate="<b>True</b><br>%{x}<br>%{y:.3f}<extra></extra>",
        ), row=r, col=1)
        fig.add_trace(go.Scatter(
            x=timestamps, y=y_total, name="Total prediction",
            line=dict(color="tomato", width=1.4),
            hovertemplate="<b>Pred</b><br>%{x}<br>%{y:.3f}<extra></extra>",
        ), row=r, col=1)
        fig.add_trace(go.Scatter(
            x=timestamps, y=y_phys, name="Physics only",
            line=dict(color="darkorange", width=1, dash="dash"), opacity=0.6,
            hovertemplate="<b>Physics</b><br>%{x}<br>%{y:.3f}<extra></extra>",
        ), row=r, col=1)
        if len(anom_idx) > 0:
            fig.add_trace(go.Scatter(
                x=timestamps[anom_idx], y=y_true[anom_idx],
                mode="markers", name=f"Anomaly ({len(anom_idx)})",
                marker=dict(color="yellow", size=10,
                            line=dict(color="red", width=1.5)),
                hovertemplate="<b>Anomaly</b><br>%{x}<br>%{y:.3f}<extra></extra>",
            ), row=r, col=1)

    # ── Panel 2 traces ──────────────────────────────────────────────────────
    if 2 in panels:
        r = panel_to_row[2]
        fig.add_trace(go.Scatter(
            x=timestamps, y=y_phys, name="Physics",
            line=dict(color="darkorange", width=1.2), showlegend=False,
            hovertemplate="<b>Physics</b><br>%{x}<br>%{y:.3f}<extra></extra>",
        ), row=r, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=timestamps, y=y_resid, name="ML residual",
            line=dict(color="purple", width=1.2), showlegend=False,
            hovertemplate="<b>ML residual</b><br>%{x}<br>%{y:.3f}<extra></extra>",
        ), row=r, col=1, secondary_y=True)

    # ── Panel 3 traces ──────────────────────────────────────────────────────
    if 3 in panels:
        r = panel_to_row[3]
        fig.add_trace(go.Scatter(
            x=timestamps, y=errors, name="|error|",
            line=dict(color="gray", width=1), showlegend=False,
            hovertemplate="<b>|error|</b><br>%{x}<br>%{y:.3f}<extra></extra>",
        ), row=r, col=1)
        fig.add_hline(
            y=threshold, line_dash="dash", line_color="red",
            annotation_text=f"3σ threshold ({threshold:.3f})",
            annotation_position="top right",
            row=r, col=1,
        )

    # ── Panel 4 traces ──────────────────────────────────────────────────────
    if 4 in panels:
        r = panel_to_row[4]
        fig.add_trace(go.Scatter(
            x=timestamps, y=anomalies.astype(int),
            name="Anomaly flag", fill="tozeroy",
            line=dict(color="red", width=0.5),
            fillcolor="rgba(220,40,40,0.45)", showlegend=False,
            hovertemplate="<b>%{x}</b><br>flag=%{y}<extra></extra>",
        ), row=r, col=1)

    # Buffer shading — apply to panels that are visible (skip panel 2 if busy)
    buffer_target_rows = [p for p in panels if p != 2]
    for p in buffer_target_rows:
        r = panel_to_row[p]
        _add_buffer_shading(
            fig, buffer_intervals, row=r, col=1,
            label=f"Buffer (±{cycle_buffer})" if p == 1 else None,
        )

    # Dummy legend trace
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        name=f"Buffer (±{cycle_buffer})",
        marker=dict(size=12, color="rgba(150,150,150,0.35)", symbol="square"),
        showlegend=True,
    ))

    # Layout
    base_height = {1: 350, 2: 250, 3: 220, 4: 200}
    total_height = sum(base_height[p] for p in panels) + 150  # +150 for title/legend

    fig.update_layout(
        title=dict(
            text=f"<b>Hybrid model: {feature_name}</b>"
                 f"   |   R²={r2:.4f}   MAE={mae:.4f}",
            font=dict(size=16), pad=dict(t=30),
        ),
        height=total_height,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.03,
            xanchor="right", x=1, font=dict(size=16),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="lightgray", borderwidth=1,
            entrywidthmode="fraction", entrywidth=0.15,
        ),
        margin=dict(l=80, t=140, r=40, b=60),
    )

    # Y-axes updates
    if 1 in panels:
        fig.update_yaxes(
            title_text=format_yaxis_label(feature_name),
            tickfont=dict(size=14), title_font=dict(size=16),
            row=panel_to_row[1], col=1
        )
    if 2 in panels:
        fig.update_yaxes(
            title_text="Physics value",
            title_font=dict(color="darkorange", size=16),
            tickfont=dict(color="darkorange", size=14),
            row=panel_to_row[2], col=1, secondary_y=False
        )
        fig.update_yaxes(
            title_text="ML residual",
            title_font=dict(color="purple", size=16),
            tickfont=dict(color="purple", size=14),
            row=panel_to_row[2], col=1, secondary_y=True
        )
    if 3 in panels:
        fig.update_yaxes(
            title_text="Absolute error",
            tickfont=dict(size=14), title_font=dict(size=16),
            row=panel_to_row[3], col=1
        )
    if 4 in panels:
        fig.update_yaxes(
            title_text="Flag",
            tickvals=[0, 1], ticktext=["Normal", "Anomaly"],
            tickfont=dict(size=14), title_font=dict(size=16),
            row=panel_to_row[4], col=1
        )

    # X-axis on last row
    last_row = panel_to_row[panels[-1]]
    fig.update_xaxes(
        title_text="Timestamp",
        tickfont=dict(size=14), title_font=dict(size=16),
        row=last_row, col=1,
        rangeslider_visible=show_rangeslider
    )

    # Save
    os.makedirs(f"{save_dir}/../img", exist_ok=True)
    panel_tag = "_".join(f"p{p}" for p in panels)
    out_html = f"{save_dir}/../img/hybrid_{feature_name}_{panel_tag}.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Interactive plot saved → {out_html}")

    export_config = {
        'responsive': True, 'scrollZoom': True,
        'toImageButtonOptions': {
            'format': 'png', 'filename': 'feature_plot', 'scale': 3
        }
    }
    fig.show(config=export_config)



# ── Main public function ──────────────────────────────────────────────────────

def lstm_hybrid_model(
    dp,
    selected_features: list,
    target_col: str,
    physics_col: str,
    feature_name: str,
    seq_len: int     = 12,
    test_size: float = 0.2,
    num_epochs: int  = 100,
    patience: int    = 5,
    save_dir: str    = "../data/sads",
    cycle_col: str   = None,
    cycle_buffer: int = 5,
    timestamp_col: str = "timestamp",   # column holding datetime values
    seed: int = 42,                     # set None for stochastic training
    show_rangeslider: bool = False,
    panels= None,
):
    """
    Train an LSTM to predict the physics residual, build the hybrid total
    prediction, and visualise everything as interactive Plotly charts on a
    real timestamp x-axis. Threshold calibrated on validation set.

    Set seed=None to get a different random initialisation every call (useful
    for assessing model robustness). With a fixed seed the run is fully
    reproducible: identical R², identical anomaly count, identical threshold.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── 0. Pin all randomness for reproducibility ────────────────────────────
    if seed is not None:
        _set_seed(seed)
        loader_gen = torch.Generator().manual_seed(seed)
        print(f"Using fixed seed={seed} (reproducible run).")
    else:
        loader_gen = None
        print("No seed set — results will vary between runs.")

    # ── 1. Pull data ──────────────────────────────────────────────────────────
    needed = selected_features + [target_col, physics_col]
    if cycle_col is not None:
        if cycle_col not in dp.df.columns:
            raise ValueError(f"cycle_col='{cycle_col}' not found in dataframe.")
        needed = needed + [cycle_col]
    if timestamp_col not in dp.df.columns:
        raise ValueError(f"timestamp_col='{timestamp_col}' not found in dataframe.")
    needed = needed + [timestamp_col]
    needed = list(dict.fromkeys(needed))   # de-dup just in case

    df = dp.df[needed].dropna().copy()

    # Make sure timestamps are real datetime values
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).reset_index(drop=True)

    MAX_SAMPLES = 15_000
    if len(df) > MAX_SAMPLES:
        df = df.iloc[-MAX_SAMPLES:].reset_index(drop=True)

    X_raw       = df[selected_features].values
    y_true_raw  = df[target_col].values
    y_phys_raw  = df[physics_col].values
    y_resid_raw = y_true_raw - y_phys_raw
    timestamps  = df[timestamp_col].values   # numpy datetime64 array

    # ── 2. Train / test split (time-ordered) ─────────────────────────────────
    split = int(len(X_raw) * (1 - test_size))
    X_train_raw,  X_test_raw  = X_raw[:split],       X_raw[split:]
    yr_train_raw, yr_test_raw = y_resid_raw[:split],  y_resid_raw[split:]
    yt_test_raw               = y_true_raw[split:]
    yp_test_raw               = y_phys_raw[split:]
    timestamps_test           = timestamps[split:]

    cycle_ids_test = df[cycle_col].values[split:] if cycle_col is not None else None

    # ── 3. Val split from train (time-ordered) ───────────────────────────────
    X_tr, X_val, yr_tr, yr_val = train_test_split(
        X_train_raw, yr_train_raw, test_size=0.2, shuffle=False
    )
    val_start  = int(len(X_train_raw) * 0.8)
    yt_val_raw = y_true_raw[:split][val_start:]
    yp_val_raw = y_phys_raw[:split][val_start:]

    # ── 4. Export raw splits ─────────────────────────────────────────────────
    for split_name, Xs, ys in [
        ("train", X_tr,       yr_tr),
        ("val",   X_val,      yr_val),
        ("test",  X_test_raw, yr_test_raw),
    ]:
        pd.DataFrame(
            np.concatenate([Xs, ys.reshape(-1, 1)], axis=1),
            columns=selected_features + [f"{target_col}_residual"],
        ).to_csv(f"{save_dir}/{split_name}_data_{feature_name}.csv", index=False)

    # ── 5. Scale (fit only on train) ─────────────────────────────────────────
    scaler_x = StandardScaler().fit(X_tr)
    scaler_y = StandardScaler().fit(yr_tr.reshape(-1, 1))
    joblib.dump(scaler_x, f"{save_dir}/scaler_x_{feature_name}.pkl")
    joblib.dump(scaler_y, f"{save_dir}/scaler_y_{feature_name}.pkl")
    joblib.dump(seq_len,  f"{save_dir}/seq_len_{feature_name}.pkl")
    print(f"Scalers and seq_len saved to {save_dir}/")

    X_tr_s  = scaler_x.transform(X_tr)
    X_val_s = scaler_x.transform(X_val)
    X_te_s  = scaler_x.transform(X_test_raw)

    yr_tr_s  = scaler_y.transform(yr_tr.reshape(-1, 1)).flatten()
    yr_val_s = scaler_y.transform(yr_val.reshape(-1, 1)).flatten()
    yr_te_s  = scaler_y.transform(yr_test_raw.reshape(-1, 1)).flatten()

    # ── 6. Sequence loaders ──────────────────────────────────────────────────
    X_tr_seq,  yr_tr_seq  = _make_sequences(X_tr_s,  yr_tr_s,  seq_len)
    X_val_seq, yr_val_seq = _make_sequences(X_val_s, yr_val_s, seq_len)
    X_te_seq,  yr_te_seq  = _make_sequences(X_te_s,  yr_te_s,  seq_len)

    train_loader = _make_loader(X_tr_seq,  yr_tr_seq,  shuffle=True, generator=loader_gen)
    val_loader   = _make_loader(X_val_seq, yr_val_seq, shuffle=False)
    test_loader  = _make_loader(X_te_seq,  yr_te_seq,  shuffle=False)

    # ── 7. Cycle buffer mask for test ─────────────────────────────────────────
    if cycle_ids_test is not None:
        buffer_mask = _build_cycle_buffer_mask(cycle_ids_test, seq_len, cycle_buffer)
        print(f"Cycle buffer: {buffer_mask.sum()} test timesteps suppressed "
              f"(±{cycle_buffer} pts per boundary)")
    else:
        buffer_mask = None
        print("No cycle_col provided — cycle buffer masking disabled.")

    # ── 8. Build & train ─────────────────────────────────────────────────────
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model     = EnhancedLSTM(
        input_dim=X_tr_s.shape[1], hidden_dim=64, output_dim=1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.HuberLoss()

    print(f"\n--- Training LSTM for [{feature_name}] residual ---")
    model, train_history, val_history = _train(
        model, train_loader, val_loader, optimizer, criterion,
        device, num_epochs, patience
    )
    _plot_learning_curves(train_history, val_history, feature_name, save_dir)

    weights_path = f"{save_dir}/model_weights_{feature_name}.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"Model saved → {weights_path}")

    # ── 9. Threshold calibration on validation ───────────────────────────────
    yt_val_aligned = yt_val_raw[seq_len:]
    yp_val_aligned = yp_val_raw[seq_len:]

    model.eval()
    val_preds = []
    with torch.no_grad():
        for Xb, _ in val_loader:
            val_preds.append(model(Xb.to(device)).cpu().numpy())
    val_pred_resid = scaler_y.inverse_transform(
        np.concatenate(val_preds).reshape(-1, 1)
    ).flatten()

    val_total   = yp_val_aligned + val_pred_resid
    val_errors  = np.abs(yt_val_aligned - val_total)
    thresh_mean = float(np.mean(val_errors))
    thresh_std  = float(np.std(val_errors))
    threshold   = thresh_mean + 3 * thresh_std

    joblib.dump(
        {"mean": thresh_mean, "std": thresh_std, "threshold": threshold},
        f"{save_dir}/anomaly_threshold_{feature_name}.pkl",
    )
    print(f"Threshold (val, 3σ): mean={thresh_mean:.4f}  "
          f"std={thresh_std:.4f}  bound={threshold:.4f}")

    # ── 10. Evaluate on test ──────────────────────────────────────────────────
    yt_test_aligned        = yt_test_raw[seq_len:]
    yp_test_aligned        = yp_test_raw[seq_len:]
    timestamps_test_aligned = timestamps_test[seq_len:]

    test_preds = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            test_preds.append(model(Xb.to(device)).cpu().numpy())
    test_pred_resid = scaler_y.inverse_transform(
        np.concatenate(test_preds).reshape(-1, 1)
    ).flatten()

    total_pred = yp_test_aligned + test_pred_resid
    errors     = np.abs(yt_test_aligned - total_pred)

    anomalies = errors > threshold
    if buffer_mask is not None:
        anomalies = anomalies & ~buffer_mask

    r2  = r2_score(yt_test_aligned, total_pred)
    mae = mean_absolute_error(yt_test_aligned, total_pred)
    print(f"\nTest performance  R²={r2:.4f}  MAE={mae:.4f}")
    print(f"Anomalies detected (after buffer): {anomalies.sum()} / {len(anomalies)}")
    print(f"Physics only R²: {r2_score(yt_test_aligned, yp_test_aligned):.4f}")
    print(f"Hybrid R²:       {r2_score(yt_test_aligned, total_pred):.4f}")

    # ── 11. Plot ──────────────────────────────────────────────────────────────
    _plot_hybrid(
        timestamps   = timestamps_test_aligned,
        y_true       = yt_test_aligned,
        y_phys       = yp_test_aligned,
        y_resid      = test_pred_resid,
        y_total      = total_pred,
        anomalies    = anomalies,
        buffer_mask  = buffer_mask,
        threshold    = threshold,
        feature_name      = feature_name,
        r2                = r2,
        mae               = mae,
        save_dir          = save_dir,
        cycle_buffer      = cycle_buffer,
        show_rangeslider  = show_rangeslider,
        panels = panels,
    )

    # ── 12. Write anomaly flags back into dp.df ───────────────────────────────
    # Column is initialised to -1 (= "not evaluated by SADS").
    # Test rows get 0 (normal) or 1 (anomaly) keyed by timestamp.
    anomaly_col = f"Anomaly_{feature_name}"
    dp.df[anomaly_col] = -1

    ts_aligned_series = pd.to_datetime(timestamps_test_aligned)
    anom_map = dict(zip(ts_aligned_series, anomalies.astype(int)))
    dp.df[timestamp_col] = pd.to_datetime(dp.df[timestamp_col])
    flags = dp.df[timestamp_col].map(anom_map)
    mask  = flags.notna()
    dp.df.loc[mask, anomaly_col] = flags[mask].astype(int)

    n_anom = int((dp.df[anomaly_col] == 1).sum())
    print(f"Anomaly flags written to dp.df['{anomaly_col}']  "
          f"(test rows: {mask.sum()}, anomalies: {n_anom})")

    return model, scaler_x, scaler_y, threshold


def export_anomaly_report_to_excel(dp, selected_features, target_col, feature_name, timestamp_col="timestamp", save_dir="../data/reports"):
    """
    Exports an Excel report with the exact sequence: 
    Timestamp -> Output -> Top Features -> Individual Features -> Anomaly Flag
    """
    os.makedirs(save_dir, exist_ok=True)
    anomaly_col = f"Anomaly_{feature_name}"
    
    if anomaly_col not in dp.df.columns:
        print(f"⚠️ Error: {anomaly_col} not found in dataframe. Run lstm_hybrid_model first.")
        return
        
    # 1. Filter only the evaluated test rows
    report_df = dp.df[dp.df[anomaly_col] >= 0].copy()
    
    # 2. SEQUENCE START: Add Timestamp and Output
    ordered_cols = [timestamp_col, target_col]
    
    # 3. Add Top Features
    for feat in selected_features:
        if feat not in ordered_cols and feat in dp.df.columns:
            ordered_cols.append(feat)
            
    # 4. Find and Add Individual Features (from interaction terms like "_x_")
    individual_features = set()
    for feat in selected_features:
        if "_x_" in feat:
            parts = feat.split("_x_")
            for part in parts:
                if part in dp.df.columns:
                    individual_features.add(part)
                    
    # Sort them alphabetically just to keep the Excel sheet looking clean
    for feat in sorted(list(individual_features)):
        if feat not in ordered_cols:
            ordered_cols.append(feat)
            
    # 5. Add Anomaly Flag at the very end
    ordered_cols.append(anomaly_col)
            
    # 6. Extract final dataframe using the exact sequence
    final_df = report_df[ordered_cols].copy()
    
    # Rename the anomaly column to be more readable
    final_df = final_df.rename(columns={anomaly_col: "Anomaly"})
    
    # Sort chronologically
    final_df = final_df.sort_values(by=timestamp_col)
    
    # Save to Excel
    save_path = os.path.join(save_dir, f"{feature_name}_Anomaly_Report.xlsx")
    final_df.to_excel(save_path, index=False, engine='openpyxl')
    
    print(f"✅ Excel report successfully exported to: {save_path}")
    print(f"   Column Sequence: {', '.join(final_df.columns)}")








# """
# Enhanced LSTM Model with Attention Mechanism for RO System Anomaly Detection
# """

# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score, mean_absolute_error
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


# class EnhancedLSTM(nn.Module):
#     """
#     LSTM model with Attention Mechanism for sequence prediction.
#     """
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.lstm = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1),
#             nn.Softmax(dim=1)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim//2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim//2, output_dim)
#         )

#     def forward(self, x):
#         """
#         Forward pass through LSTM with attention mechanism.

#         Args:
#             x: Input tensor of shape [batch_size, seq_len, input_dim]

#         Returns:
#             Output tensor of shape [batch_size, output_dim]
#         """
#         lstm_out, _ = self.lstm(x)

#         attn_weights = self.attention(lstm_out)
#         context = torch.sum(attn_weights * lstm_out, dim=1)
#         context = self.dropout(context)

#         return self.fc(context)


# def run_interactive_forensic_analysis(timestamps, y_true, y_phys, y_ml_resid, y_total, feature_name, start_idx=0, end_idx=None, threshold=None, buffer_mask=None, cycle_buffer=0):
#     """
#     Generate interactive forensic analysis plots comparing actual vs predicted values.

#     Args:
#         timestamps: Array of timestamps
#         y_true: Actual sensor values
#         y_phys: Physics model predictions
#         y_ml_resid: ML model residual predictions
#         y_total: Total predictions (physics + ML residual)
#         feature_name: Name of the feature being analyzed
#         start_idx: Starting index for time window (optional)
#         end_idx: Ending index for time window (optional)
#     """
#     if end_idx is None:
#         end_idx = len(timestamps)

#     start = max(0, start_idx)
#     end = min(len(timestamps), end_idx)

#     t_slice = timestamps[start:end]
#     true_slice = y_true[start:end]
#     phys_slice = y_phys[start:end]
#     ml_slice = y_ml_resid[start:end]
#     total_slice = y_total[start:end]

#     # Graph 1: Anomaly Detection
#     diff = np.abs(true_slice - total_slice)
#     if threshold is None:
#         threshold = np.mean(diff) + 3 * np.std(diff) # fallback only if no threshold provided in analyzed_saved_model
#     anomalies_idx = np.where(diff > threshold)[0]
#     anom_dates = t_slice[anomalies_idx]
#     anom_values = true_slice[anomalies_idx]


#     fig = go.Figure()
#      # Show buffer zone as shaded band or distinct markers in Graph 1
#     if buffer_mask is not None:
#         buffer_idx = np.where(buffer_mask[start:end])[0]
#         fig.add_trace(go.Scatter(
#             x=t_slice[buffer_idx], y=true_slice[buffer_idx],
#             mode='markers', name=f'Cycle Buffer (±{cycle_buffer} pts)',
#             marker=dict(color='rgba(180,180,255,0.5)', size=5, symbol='diamond'),
#             hovertemplate="<b>Buffer Zone</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>"
#         ))

    
#     fig.add_trace(go.Scatter(x=t_slice, y=true_slice, mode='lines', name='True Sensor Values', line=dict(color='blue', width=2), opacity=0.7))
#     fig.add_trace(go.Scatter(x=t_slice, y=total_slice, mode='lines', name='Total Prediction Values', line=dict(color='red', width=2), opacity=0.7))

#     if len(anomalies_idx) > 0:
#         fig.add_trace(go.Scatter(x=anom_dates, y=anom_values, mode="markers", name='Anomaly Detected', marker=dict(color='yellow', size=8, line=dict(width=1, color='red'))))

#     fig.update_layout(autosize=True, title=f"Anomaly Detection for {feature_name}", xaxis_title='Timestamp', yaxis_title='Value', hovermode='x unified', template='plotly_white', height=600, yaxis=dict(autorange=True, fixedrange=False))
#     fig.update_xaxes(rangeslider_visible=True)
#     fig.show(config={'responsive': True, 'scrollZoom': True})

#     # Graph 2: Scale-Locked Analysis
#     fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=(f"1. Total Result", "2. Physics Model", "3. ML"))

#     fig2.add_trace(go.Scatter(x=t_slice, y=true_slice, name="Actual", line=dict(color='blue', width=0.8)), row=1, col=1)
#     fig2.add_trace(go.Scatter(x=t_slice, y=total_slice, name="Total Pred", line=dict(color='red', width=0.8)), row=1, col=1)
#     fig2.add_trace(go.Scatter(x=t_slice, y=phys_slice, name="Physics Model", line=dict(color='orange', width=0.8)), row=1, col=1)
#     fig2.add_trace(go.Scatter(x=t_slice, y=phys_slice, name="Physics Model", line=dict(color='orange', width=0.8)), row=2, col=1)
#     fig2.add_trace(go.Scatter(x=t_slice, y=ml_slice, name="ML", line=dict(color='purple', width=0.8)), row=3, col=1)

#     fig2.update_layout(
#         title_text=f"Graphs Analysis: {feature_name} (Unified Zoom Level)", height=900, hovermode="x unified",
#         yaxis=dict(title="Value", autorange=True),
#         yaxis2=dict(title="Value", scaleanchor="y", scaleratio=1),
#         yaxis3=dict(title="Value", scaleanchor="y", scaleratio=1)
#     )
#     fig2.show(config={'responsive': True, 'scrollZoom': True})

#     # Graph 3: Secondary Axis
#     fig3 = make_subplots(specs=[[{"secondary_y": True}]])
#     fig3.add_trace(go.Scatter(x=t_slice, y=phys_slice, name="Physics", line=dict(color='orange', width=0.8)), secondary_y=False)
#     fig3.add_trace(go.Scatter(x=t_slice, y=ml_slice, name="ML", line=dict(color='purple', width=0.8)), secondary_y=True)

#     fig3.update_layout(
#         title_text=f'Physics Model VS ML', hovermode="x unified", xaxis_title='Timestamp',
#         yaxis=dict(title=dict(text="Physics Value", font=dict(color="blue")), tickfont=dict(color="blue")),
#         yaxis2=dict(title=dict(text="ML", font=dict(color="red")), tickfont=dict(color="red"), anchor="x", overlaying="y", side="right")
#     )
#     fig3.show(config={'responsive': True, 'scrollZoom': True})




# """
# Hybrid LSTM Model: Physics + ML Residual Prediction
# ----------------------------------------------------
# Usage:
#     lstm_hybrid_model(
#         dp                  = dp,
#         selected_features   = ['feat1', 'feat2', ...],
#         target_col          = 'PermeateConductivity',
#         physics_col         = 'PermeateConductivity_Phys',
#         feature_name        = 'PermeateConductivity',
#         cycle_col           = 'Cycle_ID',   # optional — enables buffer masking
#         cycle_buffer        = 5,            # timesteps suppressed at each boundary
#     )
# """

# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from tqdm import tqdm
# import joblib


# # ── Model ─────────────────────────────────────────────────────────────────────

# class EnhancedLSTM(nn.Module):
#     """2-layer LSTM with additive attention."""
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.4):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0,
#         )
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1),
#             nn.Softmax(dim=1),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, output_dim),
#         )

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)                           # [B, T, H]
#         attn_weights = self.attention(lstm_out)              # [B, T, 1]
#         context = torch.sum(attn_weights * lstm_out, dim=1)  # [B, H]
#         return self.fc(context)                              # [B, out_dim]


# # ── Data helpers ──────────────────────────────────────────────────────────────

# def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
#     """Slide a window over X/y and return stacked arrays."""
#     Xs, ys = [], []
#     for i in range(len(X) - seq_len):
#         Xs.append(X[i : i + seq_len])
#         ys.append(y[i + seq_len])
#     return np.array(Xs), np.array(ys)


# def _make_loader(X_seq, y_seq, batch_size=64, shuffle=False):
#     ds = TensorDataset(
#         torch.tensor(X_seq, dtype=torch.float32),
#         torch.tensor(y_seq.reshape(-1, 1), dtype=torch.float32),
#     )
#     return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# def _build_cycle_buffer_mask(cycle_ids: np.ndarray, seq_len: int,
#                               cycle_buffer: int) -> np.ndarray:
#     """
#     Build a boolean mask (True = suppress) for output timesteps that fall
#     within cycle_buffer steps of any cycle boundary.

#     How it works:
#       - The output at position i corresponds to input index (i + seq_len).
#       - A cycle boundary at full-array index b maps to output index (b - seq_len).
#       - We mark all output indices within cycle_buffer of that point as True.

#     Args:
#         cycle_ids:    1-D array of cycle IDs, same length as the raw (pre-sequence) split.
#         seq_len:      sequence length used to build windows.
#         cycle_buffer: number of output steps to suppress on each side of a boundary.

#     Returns:
#         Boolean numpy array of length (len(cycle_ids) - seq_len).
#     """
#     n_out = len(cycle_ids) - seq_len

#     # Positions in the full array where cycle ID changes
#     boundaries = np.where(cycle_ids[1:] != cycle_ids[:-1])[0] + 1

#     mask = np.zeros(n_out, dtype=bool)
#     for b in boundaries:
#         out_b = b - seq_len          # corresponding output index
#         lo = max(0, out_b - cycle_buffer)
#         hi = min(n_out, out_b + cycle_buffer + 1)
#         mask[lo:hi] = True

#     return mask


# # ── Training helpers ──────────────────────────────────────────────────────────

# def _evaluate(model, loader, device, criterion):
#     model.eval()
#     total = 0.0
#     with torch.no_grad():
#         for Xb, yb in loader:
#             pred = model(Xb.to(device))
#             total += criterion(pred, yb.to(device)).item() * Xb.size(0)
#     return total / len(loader.dataset)


# def _train(model, train_loader, val_loader, optimizer, criterion,
#            device, num_epochs=100, patience=5):
#     import copy
#     best_loss    = float("inf")
#     no_improve   = 0
#     best_weights = copy.deepcopy(model.state_dict())

#     train_history = []   # NEW
#     val_history   = []   # NEW

#     for epoch in range(num_epochs):
#         model.train()
#         running = 0.0
#         for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
#             Xb, yb = Xb.to(device), yb.to(device)
#             optimizer.zero_grad()
#             loss = criterion(model(Xb), yb)
#             loss.backward()
#             optimizer.step()
#             running += loss.item() * Xb.size(0)

#         avg_train = running / len(train_loader.dataset)
#         val_loss  = _evaluate(model, val_loader, device, criterion)

#         train_history.append(avg_train)   # NEW
#         val_history.append(val_loss)      # NEW

#         print(f"  Epoch {epoch+1:3d}/{num_epochs}  "
#               f"train={avg_train:.5f}  val={val_loss:.5f}")

#         if val_loss < best_loss:
#             best_loss    = val_loss
#             best_weights = copy.deepcopy(model.state_dict())
#             no_improve   = 0
#         else:
#             no_improve += 1

#         if no_improve >= patience:
#             print(f"  Early stopping at epoch {epoch+1}")
#             break

#     model.load_state_dict(best_weights)
#     return model, train_history, val_history   # CHANGED: return histories

# def _plot_learning_curves(train_history, val_history, feature_name, save_dir):
#     """Plot training and validation loss curves with best-epoch marker."""
#     epochs = np.arange(1, len(train_history) + 1)
#     best_epoch = int(np.argmin(val_history)) + 1

#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.plot(epochs, train_history, marker='o', linewidth=1.5,
#             color='steelblue', label='Train loss')
#     ax.plot(epochs, val_history,   marker='s', linewidth=1.5,
#             color='tomato',    label='Validation loss')
#     ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7,
#                label=f'Best epoch ({best_epoch})')

#     # Highlight overfitting region (val rising while train falling)
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Huber loss")
#     ax.set_title(f"Learning curves: {feature_name}")
#     ax.legend()
#     ax.grid(True, linestyle="--", alpha=0.4)

#     out_path = f"{save_dir}/../img/learning_curves_{feature_name}.png"
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.show()
#     print(f"Learning curves saved → {out_path}")


# # ── Plotting ──────────────────────────────────────────────────────────────────

# def _plot_hybrid(y_true, y_phys, y_resid, y_total, anomalies,
#                  buffer_mask, threshold, feature_name, r2, mae,
#                  save_dir, cycle_buffer):
#     """
#     4-panel plot:
#       1. Raw sensor vs total prediction + anomaly markers + buffer zones
#       2. Physics contribution vs ML residual (dual axis)
#       3. Absolute error with 3σ threshold + buffered zones shaded
#       4. Anomaly flag timeline with buffer zones shown
#     """
#     n      = len(y_true)
#     t      = np.arange(n)
#     anom_t = t[anomalies]

#     fig = plt.figure(figsize=(16, 12))
#     fig.suptitle(
#         f"Hybrid model: {feature_name}   |   R²={r2:.4f}   MAE={mae:.4f}",
#         fontsize=13, fontweight="bold", y=0.98,
#     )
#     gs = gridspec.GridSpec(4, 1, hspace=0.45)

#     # ── Panel 1: True vs Total prediction ────────────────────────────────────
#     ax1 = fig.add_subplot(gs[0])
#     ax1.plot(t, y_true,  color="steelblue",  lw=1.2, alpha=0.85, label="True sensor")
#     ax1.plot(t, y_total, color="tomato",     lw=1.2, alpha=0.85, label="Total prediction")
#     ax1.plot(t, y_phys,  color="darkorange", lw=0.8, alpha=0.55,
#              linestyle="--", label="Physics only")
#     if buffer_mask is not None and buffer_mask.any():
#         # Shade buffer zones — use ymin/ymax after lines are drawn
#         ymin, ymax = ax1.get_ylim()
#         ax1.fill_between(t, ymin, ymax, where=buffer_mask,
#                          color="gray", alpha=0.15,
#                          label=f"Cycle buffer (±{cycle_buffer})")
#     if anomalies.any():
#         ax1.scatter(anom_t, y_true[anomalies],
#                     c="yellow", edgecolors="red", s=40, zorder=4,
#                     label=f"Anomaly ({anomalies.sum()})")
#     ax1.set_ylabel(feature_name)
#     ax1.set_title("True sensor vs hybrid prediction")
#     ax1.legend(fontsize=8, loc="upper right")
#     ax1.grid(True, linestyle="--", alpha=0.4)

#     # ── Panel 2: Physics vs ML residual (dual axis) ───────────────────────────
#     ax2 = fig.add_subplot(gs[1])
#     ax2.plot(t, y_phys,  color="darkorange", lw=0.9, alpha=0.8, label="Physics")
#     ax2.set_ylabel("Physics value", color="darkorange")
#     ax2.tick_params(axis="y", labelcolor="darkorange")
#     ax2r = ax2.twinx()
#     ax2r.plot(t, y_resid, color="purple", lw=0.9, alpha=0.7, label="ML residual")
#     ax2r.set_ylabel("ML residual", color="purple")
#     ax2r.tick_params(axis="y", labelcolor="purple")
#     ax2r.axhline(0, color="purple", lw=0.5, linestyle=":")
#     ax2.set_title("Physics contribution vs ML residual")
#     lines1, lbl1 = ax2.get_legend_handles_labels()
#     lines2, lbl2 = ax2r.get_legend_handles_labels()
#     ax2.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc="upper right")
#     ax2.grid(True, linestyle="--", alpha=0.3)

#     # ── Panel 3: Absolute error + threshold ───────────────────────────────────
#     errors = np.abs(y_true - y_total)
#     ax3 = fig.add_subplot(gs[2])
#     ax3.plot(t, errors, color="gray", lw=0.8, alpha=0.7, label="|error|")
#     ax3.axhline(threshold, color="red", lw=1.2, linestyle="--",
#                 label=f"3σ threshold ({threshold:.4f})")
#     ax3.fill_between(t, 0, errors,
#                      where=(errors > threshold),
#                      color="red", alpha=0.25, label="Exceeds threshold")
#     if buffer_mask is not None and buffer_mask.any():
#         ax3.fill_between(t, 0, errors, where=buffer_mask,
#                          color="gray", alpha=0.25,
#                          label="Buffered (suppressed)")
#     ax3.set_ylabel("Absolute error")
#     ax3.set_title("Prediction error and anomaly threshold")
#     ax3.legend(fontsize=8, loc="upper right")
#     ax3.grid(True, linestyle="--", alpha=0.4)

#     # ── Panel 4: Anomaly flag ─────────────────────────────────────────────────
#     ax4 = fig.add_subplot(gs[3])
#     ax4.fill_between(t, 0, anomalies.astype(int),
#                      step="mid", color="red", alpha=0.5, label="Anomaly flag")
#     if buffer_mask is not None and buffer_mask.any():
#         ax4.fill_between(t, 0, buffer_mask.astype(float) * 0.5,
#                          step="mid", color="gray", alpha=0.4,
#                          label=f"Buffer zone (±{cycle_buffer})")
#     ax4.set_yticks([0, 1])
#     ax4.set_yticklabels(["Normal", "Anomaly"])
#     ax4.set_xlabel("Time step")
#     ax4.set_title("Anomaly flags")
#     ax4.legend(fontsize=8, loc="upper right")
#     ax4.grid(True, linestyle="--", alpha=0.3)

#     os.makedirs(f"{save_dir}/../img", exist_ok=True)
#     out_path = f"{save_dir}/../img/hybrid_{feature_name}.png"
#     plt.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.show()
#     print(f"Plot saved → {out_path}")


# # ── Main public function ──────────────────────────────────────────────────────

# def lstm_hybrid_model(
#     dp,
#     selected_features: list,
#     target_col: str,           # raw sensor column name (string)
#     physics_col: str,          # physics model prediction column name
#     feature_name: str,
#     seq_len: int    = 12,
#     test_size: float = 0.2,
#     num_epochs: int  = 100,
#     patience: int    = 5,
#     save_dir: str    = "../data/model_data",
#     cycle_col: str   = 'cycle_id ',   # column holding cycle ID e.g. 'Cycle_ID'
#     cycle_buffer: int = 5,     # timesteps to suppress at each cycle boundary
# ):
#     """
#     Train an LSTM to predict the physics residual, then:
#       total_prediction = physics_pred + lstm_residual_pred
#     Anomalies flagged where |true - total| > mean + 3σ, with optional
#     suppression of points near cycle boundaries (backwash transitions).
#     Threshold calibrated on validation set, never on test set.
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # ── 1. Pull data ──────────────────────────────────────────────────────────
#     needed = selected_features + [target_col, physics_col]
#     if cycle_col is not None:
#         if cycle_col not in dp.df.columns:
#             raise ValueError(f"cycle_col='{cycle_col}' not found in dataframe.")
#         needed = needed + [cycle_col]

#     df = dp.df[needed].dropna().copy()

#     MAX_SAMPLES = 15_000
#     if len(df) > MAX_SAMPLES:
#         df = df.iloc[-MAX_SAMPLES:]

#     X_raw       = df[selected_features].values
#     y_true_raw  = df[target_col].values
#     y_phys_raw  = df[physics_col].values
#     y_resid_raw = y_true_raw - y_phys_raw

#     # ── 2. Train / test split (time-ordered) ─────────────────────────────────
#     split = int(len(X_raw) * (1 - test_size))
#     X_train_raw,  X_test_raw  = X_raw[:split],       X_raw[split:]
#     yr_train_raw, yr_test_raw = y_resid_raw[:split],  y_resid_raw[split:]
#     yt_test_raw               = y_true_raw[split:]
#     yp_test_raw               = y_phys_raw[split:]

#     # Preserve cycle IDs aligned with test split
#     cycle_ids_test = df[cycle_col].values[split:] if cycle_col is not None else None

#     # ── 3. Val split from train (time-ordered) ───────────────────────────────
#     X_tr, X_val, yr_tr, yr_val = train_test_split(
#         X_train_raw, yr_train_raw, test_size=0.2, shuffle=False
#     )
#     val_start  = int(len(X_train_raw) * 0.8)
#     yt_val_raw = y_true_raw[:split][val_start:]
#     yp_val_raw = y_phys_raw[:split][val_start:]

#     # ── 4. Export raw splits ─────────────────────────────────────────────────
#     for split_name, Xs, ys in [
#         ("train", X_tr,       yr_tr),
#         ("val",   X_val,      yr_val),
#         ("test",  X_test_raw, yr_test_raw),
#     ]:
#         pd.DataFrame(
#             np.concatenate([Xs, ys.reshape(-1, 1)], axis=1),
#             columns=selected_features + [f"{target_col}_residual"],
#         ).to_csv(f"{save_dir}/{split_name}_data_{feature_name}.csv", index=False)

#     # ── 5. Scale (fit ONLY on train) ─────────────────────────────────────────
#     scaler_x = StandardScaler().fit(X_tr)
#     scaler_y = StandardScaler().fit(yr_tr.reshape(-1, 1))

#     joblib.dump(scaler_x, f"{save_dir}/scaler_x_{feature_name}.pkl")
#     joblib.dump(scaler_y, f"{save_dir}/scaler_y_{feature_name}.pkl")
#     joblib.dump(seq_len,  f"{save_dir}/seq_len_{feature_name}.pkl")
#     print(f"Scalers and seq_len saved to {save_dir}/")

#     X_tr_s  = scaler_x.transform(X_tr)
#     X_val_s = scaler_x.transform(X_val)
#     X_te_s  = scaler_x.transform(X_test_raw)

#     yr_tr_s  = scaler_y.transform(yr_tr.reshape(-1, 1)).flatten()
#     yr_val_s = scaler_y.transform(yr_val.reshape(-1, 1)).flatten()
#     yr_te_s  = scaler_y.transform(yr_test_raw.reshape(-1, 1)).flatten()

#     # ── 6. Build sequence loaders ─────────────────────────────────────────────
#     X_tr_seq,  yr_tr_seq  = _make_sequences(X_tr_s,  yr_tr_s,  seq_len)
#     X_val_seq, yr_val_seq = _make_sequences(X_val_s, yr_val_s, seq_len)
#     X_te_seq,  yr_te_seq  = _make_sequences(X_te_s,  yr_te_s,  seq_len)

#     train_loader = _make_loader(X_tr_seq,  yr_tr_seq,  shuffle=True)
#     val_loader   = _make_loader(X_val_seq, yr_val_seq, shuffle=False)
#     test_loader  = _make_loader(X_te_seq,  yr_te_seq,  shuffle=False)

#     # ── 7. Build cycle buffer mask for test set ───────────────────────────────
#     if cycle_ids_test is not None:
#         buffer_mask = _build_cycle_buffer_mask(cycle_ids_test, seq_len, cycle_buffer)
#         print(f"Cycle buffer: {buffer_mask.sum()} test timesteps suppressed "
#               f"(±{cycle_buffer} pts per boundary)")
#     else:
#         buffer_mask = None
#         print("No cycle_col provided — cycle buffer masking disabled.")

#     # ── 8. Build and train model ──────────────────────────────────────────────
#     device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model     = EnhancedLSTM(
#         input_dim=X_tr_s.shape[1], hidden_dim=16, output_dim=1
#     ).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
#     criterion = nn.HuberLoss()

#     print(f"\n--- Training LSTM for [{feature_name}] residual ---")
#     # model = _train(model, train_loader, val_loader, optimizer, criterion,
#     #                device, num_epochs, patience)
#     model, train_history, val_history = _train(model, train_loader, val_loader, optimizer, criterion,device, num_epochs, patience)
#     # Plot learning curves
#     _plot_learning_curves(train_history, val_history, feature_name, save_dir)




#     weights_path = f"{save_dir}/model_weights_{feature_name}.pth"
#     torch.save(model.state_dict(), weights_path)
#     print(f"Model saved → {weights_path}")

#     # ── 9. Calibrate threshold on VALIDATION set ─────────────────────────────
#     yt_val_aligned = yt_val_raw[seq_len:]
#     yp_val_aligned = yp_val_raw[seq_len:]

#     model.eval()
#     val_preds = []
#     with torch.no_grad():
#         for Xb, _ in val_loader:
#             val_preds.append(model(Xb.to(device)).cpu().numpy())
#     val_pred_resid = scaler_y.inverse_transform(
#         np.concatenate(val_preds).reshape(-1, 1)
#     ).flatten()

#     val_total   = yp_val_aligned + val_pred_resid
#     val_errors  = np.abs(yt_val_aligned - val_total)
#     thresh_mean = float(np.mean(val_errors))
#     thresh_std  = float(np.std(val_errors))
#     threshold   = thresh_mean + 3 * thresh_std

#     joblib.dump(
#         {"mean": thresh_mean, "std": thresh_std, "threshold": threshold},
#         f"{save_dir}/anomaly_threshold_{feature_name}.pkl",
#     )
#     print(f"Threshold (val, 3σ): mean={thresh_mean:.4f}  "
#           f"std={thresh_std:.4f}  bound={threshold:.4f}")

#     # ── 10. Evaluate on test set ──────────────────────────────────────────────
#     yt_test_aligned = yt_test_raw[seq_len:]
#     yp_test_aligned = yp_test_raw[seq_len:]

#     test_preds = []
#     with torch.no_grad():
#         for Xb, _ in test_loader:
#             test_preds.append(model(Xb.to(device)).cpu().numpy())
#     test_pred_resid = scaler_y.inverse_transform(
#         np.concatenate(test_preds).reshape(-1, 1)
#     ).flatten()

#     total_pred = yp_test_aligned + test_pred_resid
#     errors     = np.abs(yt_test_aligned - total_pred)

#     # Flag anomalies first, then clear buffer zones
#     anomalies = errors > threshold
#     if buffer_mask is not None:
#         anomalies = anomalies & ~buffer_mask

#     r2  = r2_score(yt_test_aligned, total_pred)
#     mae = mean_absolute_error(yt_test_aligned, total_pred)
#     print(f"\nTest performance  R²={r2:.4f}  MAE={mae:.4f}")
#     print(f"Anomalies detected (after buffer): {anomalies.sum()} / {len(anomalies)}")


#     print("Physics only:", r2_score(yt_test_aligned, yp_test_aligned))
#     print("Hybrid:      ", r2_score(yt_test_aligned, total_pred))

#     # ── 11. Plot ──────────────────────────────────────────────────────────────
#     _plot_hybrid(
#         y_true       = yt_test_aligned,
#         y_phys       = yp_test_aligned,
#         y_resid      = test_pred_resid,
#         y_total      = total_pred,
#         anomalies    = anomalies,
#         buffer_mask  = buffer_mask,
#         threshold    = threshold,
#         feature_name = feature_name,
#         r2           = r2,
#         mae          = mae,
#         save_dir     = save_dir,
#         cycle_buffer = cycle_buffer,
#     )

#     return model, scaler_x, scaler_y, threshold
