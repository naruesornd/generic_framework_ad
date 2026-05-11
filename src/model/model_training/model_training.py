"""
Model Training Module for LSTM-based Residual Prediction
Includes data preparation, model training, and analysis functions.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import os
import copy
import joblib
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def prepare_data(dp, selected_features, target_col_resid, target_col_true, target_col_phys, test_size=0.1, max_samples=15000):
    """
    Prepare data for LSTM training with cycle awareness.
    """
    time_col = 'timestamp'
    all_cols = selected_features + [target_col_resid, target_col_true, target_col_phys, time_col, 'cycle_id']

    df_clean = dp.df[all_cols].dropna().copy()

    if len(df_clean) > max_samples:
        df_clean = df_clean.iloc[-max_samples:]

    df_clean[target_col_resid] = df_clean[target_col_true] - df_clean[target_col_phys]

    split_idx = int(len(df_clean) * (1 - test_size))

    train_df = df_clean.iloc[:split_idx].copy()
    test_df = df_clean.iloc[split_idx:].copy()

    data = {
        "train_df": train_df,
        "test_df": test_df,
        "input_dim": len(selected_features)
    }

    print(f"Data Prepared. Train Rows: {len(train_df)}, Test Rows: {len(test_df)}")
    return data


def create_cycle_aware_loaders(df, selected_features, target_col, seq_len=12, batch_size=64, shuffle=True):
    """
    Create data loaders that safely slide a window only within individual cycles.
    """
    X_list, y_list = [], []

    for cycle_id, cycle_data in df.groupby('cycle_id'):
        if len(cycle_data) <= seq_len:
            continue

        X_cycle = cycle_data[selected_features].values
        y_cycle = cycle_data[target_col].values

        for i in range(len(X_cycle) - seq_len):
            X_list.append(X_cycle[i : i + seq_len])
            y_list.append(y_cycle[i + seq_len])

    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list).reshape(-1, 1), dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


def evaluate_model(model, val_loader, device, criterion):
    """
    Evaluate model on validation set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            y_pred = model(X_val)
            loss = criterion(y_pred, y_val)
            total_loss += loss.item() * X_val.size(0)
    return total_loss / len(val_loader.dataset)


def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=100, patience=5):
    """
    Train model with early stopping.
    """
    best_val_loss = float('inf')
    epoch_without_improvement = 0
    best_model_weights = model.state_dict()

    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_train, y_train in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            X_train, y_train = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_train.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        val_loss = evaluate_model(model, val_loader, device, criterion)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}  Test Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epoch_without_improvement = 0
        else:
            epoch_without_improvement += 1

        if epoch_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_weights)
            break

    return model, train_loss_history, val_loss_history


def train_lstm_model(dp, selected_features, target_col_resid, target_col_true, target_col_phys, feature_name,
                     test_size=0.1, num_epochs=50, patience=5, seq_len=12):
    """
    Train LSTM model for residual prediction.
    """
    print(f"\n--- STARTING TRAINING FOR {feature_name} ---")

    from model.lstm_model.enhanced_lstm import EnhancedLSTM

    data = prepare_data(dp, selected_features, target_col_resid, target_col_true, target_col_phys, test_size)
    train_df = data['train_df'].copy()

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    train_df[selected_features] = scaler_x.fit_transform(train_df[selected_features])
    train_df[[target_col_resid]] = scaler_y.fit_transform(train_df[[target_col_resid]])

    os.makedirs('../data/physics/', exist_ok=True)
    joblib.dump(scaler_x, f'../data/physics/scaler_x_{feature_name}.pkl')
    joblib.dump(scaler_y, f'../data/physics/scaler_y_{feature_name}.pkl')

    unique_cycles = train_df['cycle_id'].unique()
    rng = np.random.default_rng(seed=42)
    shuffled_cycles = rng.permutation(unique_cycles)
    split_cycle_idx = int(len(shuffled_cycles) * 0.8)

    train_cycles = shuffled_cycles[:split_cycle_idx]
    val_cycles = shuffled_cycles[split_cycle_idx:]

    sub_train_df = train_df[train_df['cycle_id'].isin(train_cycles)]
    val_df = train_df[train_df['cycle_id'].isin(val_cycles)]

    train_loader = create_cycle_aware_loaders(sub_train_df, selected_features, target_col_resid, seq_len=seq_len, batch_size=64, shuffle=True)
    val_loader = create_cycle_aware_loaders(val_df, selected_features, target_col_resid, seq_len=seq_len, batch_size=64, shuffle=False)
    
    # val_split_idx = int(len(train_df) * 0.8)
    # sub_train_df = train_df.iloc[:val_split_idx]
    # val_df = train_df.iloc[val_split_idx:]

    # train_loader = create_cycle_aware_loaders(sub_train_df, selected_features, target_col_resid, seq_len=seq_len, batch_size=64, shuffle=True)
    # val_loader = create_cycle_aware_loaders(val_df, selected_features, target_col_resid, seq_len=seq_len, batch_size=64, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EnhancedLSTM(input_dim=data['input_dim'], hidden_dim=128, output_dim=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    criterion = nn.HuberLoss()

    model, train_history, val_history = train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, patience)

    model_path = f"../data/physics/model_weights_{feature_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"SUCCESS: Model saved to {model_path}")

    # ── Calibrate anomaly threshold on validation residuals ──────────────────
    # We run the trained model over the val split (known-normal data) and
    # compute mean + 3σ of |residual|. This threshold is then applied at
    # inference time so the classifier is never influenced by the test window.
    model.eval()
    device_cpu = torch.device("cpu")
    model.to(device_cpu)
    val_preds = []
    val_true  = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch.to(device_cpu)).cpu().numpy()
            val_preds.append(preds)
            val_true.append(y_batch.numpy())
    model.to(device)   # restore original device

    val_pred_scaled  = np.concatenate(val_preds).reshape(-1, 1)
    val_true_scaled  = np.concatenate(val_true).reshape(-1, 1)
    val_pred_resid   = scaler_y.inverse_transform(val_pred_scaled).flatten()
    val_true_resid   = scaler_y.inverse_transform(val_true_scaled).flatten()

    val_abs_errors   = np.abs(val_true_resid - val_pred_resid)
    threshold_mean   = float(np.mean(val_abs_errors))
    threshold_std    = float(np.std(val_abs_errors))
    anomaly_threshold = threshold_mean + 3 * threshold_std

    threshold_path = f"../data/physics/anomaly_threshold_{feature_name}.pkl"
    joblib.dump({"mean": threshold_mean, "std": threshold_std, "threshold": anomaly_threshold}, threshold_path)
    print(f"   Threshold calibrated on val residuals → mean={threshold_mean:.4f}, "
          f"3σ bound={anomaly_threshold:.4f}")
    print(f"   Threshold saved to {threshold_path}")
    # ─────────────────────────────────────────────────────────────────────────

    fig = go.Figure()
    epochs_ran = list(range(1, len(train_history) + 1))

    fig.add_trace(go.Scatter(x=epochs_ran, y=train_history, mode='lines+markers', name='Train Loss', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=epochs_ran, y=val_history, mode='lines+markers', name='Validation (Test) Loss', line=dict(color='orange')))

    best_epoch = val_history.index(min(val_history)) + 1
    fig.add_vline(x=best_epoch, line_dash="dash", line_color="green", annotation_text=f"Best Model (Epoch {best_epoch})")

    fig.update_layout(
        title=f"LSTM Learning Curves: {feature_name}",
        xaxis_title="Epoch",
        yaxis_title="Huber Loss",
        template="plotly_white",
        hovermode="x unified",
        height=400
    )
    fig.show()


def analyze_saved_model(dp, selected_features, target_col_resid, target_col_true, target_col_phys, feature_name, test_size=0.1, cycle_buffer=5):
    """
    Analyze saved LSTM model and generate predictions with anomaly detection.
    """
    print(f"\n--- ANALYZING SAVED MODEL FOR {feature_name} ---")

    from model.lstm_model.enhanced_lstm import EnhancedLSTM

    data = prepare_data(dp, selected_features, target_col_resid, target_col_true, target_col_phys, test_size)
    test_df = data['test_df'].copy()

    scaler_x = joblib.load(f'../data/physics/scaler_x_{feature_name}.pkl')
    scaler_y = joblib.load(f'../data/physics/scaler_y_{feature_name}.pkl')

    test_df[selected_features] = scaler_x.transform(test_df[selected_features])

    SEQ_LEN = 12
    X_list, t_list, true_list, phys_list, buffer_list = [], [], [], [], []

    for cycle_id, cycle_data in test_df.groupby('cycle_id'):
        if len(cycle_data) <= SEQ_LEN:
            continue

        X_cycle = cycle_data[selected_features].values
        t_cycle = cycle_data['timestamp'].values
        true_cycle = cycle_data[target_col_true].values
        phys_cycle = cycle_data[target_col_phys].values
        n = len(X_cycle)

        # for i in range(len(X_cycle) - SEQ_LEN):
        #     X_list.append(X_cycle[i : i + SEQ_LEN])
        #     t_list.append(t_cycle[i + SEQ_LEN])
        #     true_list.append(true_cycle[i + SEQ_LEN])
        #     phys_list.append(phys_cycle[i + SEQ_LEN])
        
        # Indices within this cycle that are in the buffer zone
        # Output index is i + SEQ_LEN, so buffer covers:
        #   head: i + SEQ_LEN < cycle_buffer + SEQ_LEN  →  i < cycle_buffer
        #   tail: i + SEQ_LEN > n - cycle_buffer        →  i > n - cycle_buffer - 1
        for i in range(n - SEQ_LEN):
            output_pos = i + SEQ_LEN          # position inside this cycle
            in_buffer  = (output_pos < cycle_buffer) or (output_pos >= n - cycle_buffer)

            X_list.append(X_cycle[i : i + SEQ_LEN])
            t_list.append(t_cycle[output_pos])
            true_list.append(true_cycle[output_pos])
            phys_list.append(phys_cycle[output_pos])
            buffer_list.append(in_buffer)

    if len(X_list) == 0:
        print("⚠️ Not enough test data to form sequences.")
        return

    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    test_dataset = TensorDataset(X_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EnhancedLSTM(input_dim=data['input_dim'], hidden_dim=128, output_dim=1).to(device)
    model.load_state_dict(torch.load(f"../data/physics/model_weights_{feature_name}.pth", map_location=device))
    model.eval()

    y_preds = []
    with torch.no_grad():
        for X_batch, in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            y_preds.append(preds)

    y_pred_resid_scaled = np.concatenate(y_preds, axis=0).reshape(-1, 1)
    y_pred_resid = scaler_y.inverse_transform(y_pred_resid_scaled).flatten()

    timestamps_plot = np.array(t_list)
    y_true_plot = np.array(true_list)
    y_phys_plot = np.array(phys_list)

    y_total_pred = y_phys_plot + y_pred_resid
    real_residual_gap = y_true_plot - y_phys_plot

    # ── Load pre-calibrated threshold from training ───────────────────────────
    # The threshold was computed on validation residuals (known-normal data)
    # during train_lstm_model(). Using it here avoids the circularity of
    # computing mean+3σ from the same window being classified, which would
    # suppress detections when the test period itself is anomalous.
    threshold_path = f"../data/physics/anomaly_threshold_{feature_name}.pkl"
    if os.path.exists(threshold_path):
        threshold_info = joblib.load(threshold_path)
        threshold = threshold_info["threshold"]
        print(f"   Loaded calibrated threshold: {threshold:.4f} "
              f"(val mean={threshold_info['mean']:.4f}, val 3σ bound)")
    else:
        # Fallback: compute from test set and warn — re-run train_lstm_model()
        # to generate a proper calibrated threshold.
        diff_fallback = np.abs(y_true_plot - y_total_pred)
        threshold = np.mean(diff_fallback) + 3 * np.std(diff_fallback)
        print(f"   ⚠️  No saved threshold found for '{feature_name}'. "
              f"Falling back to test-set threshold ({threshold:.4f}). "
              f"Re-run train_lstm_model() to fix this.")
    # ─────────────────────────────────────────────────────────────────────────

    diff = np.abs(y_true_plot - y_total_pred)
    anomalies_binary = (diff > threshold).astype(int)

    buffer_mask    = np.array(buffer_list)                      # True = suppress
    anomalies_binary = (diff > threshold).astype(int)
    anomalies_binary[buffer_mask] = 0                           # zero-out buffer zone

    n_masked = buffer_mask.sum()
    n_anomalies = anomalies_binary.sum()
    print(f"🔇 Cycle buffer: {n_masked} timestamps suppressed (±{cycle_buffer} pts per cycle boundary)")
    print(f"✅ Final anomalies after masking: {n_anomalies} / {len(anomalies_binary)}")

    anom_col_name = f'Anomaly_{feature_name}'
    pred_col_name = f'Total_Prediction_{feature_name}'
    

    print(f"Saving results for {feature_name} to CSV...")
    df_results = pd.DataFrame({
        'timestamp': timestamps_plot,
        'Physics_Model': y_phys_plot,
        'ML_Residual_Pred': y_pred_resid,
        'Real_Residual_Gap': real_residual_gap,
        'Raw_Sensor_Value': y_true_plot,
        pred_col_name: y_total_pred,
        anom_col_name: anomalies_binary
    })

    csv_path = f"../data/physics/analysis_outputs_{feature_name}.csv"
    os.makedirs('../data/physics/', exist_ok=True)
    df_results.to_csv(csv_path, index=False)

    if anom_col_name in dp.df.columns:
        dp.df = dp.df.drop(columns=[anom_col_name, pred_col_name])

    dp.df = pd.merge(
        dp.df,
        df_results[['timestamp', pred_col_name, anom_col_name]],
        on='timestamp',
        how='left'
    )

    dp.df[anom_col_name] = dp.df[anom_col_name].fillna(0).astype(int)
    print(f"✅ Successfully injected '{anom_col_name}' back into main dataframe!")

    r2 = r2_score(y_true_plot, y_total_pred)
    mae = mean_absolute_error(y_true_plot, y_total_pred)
    correlation = np.corrcoef(y_true_plot, y_total_pred)[0, 1]
    print(f"Performance: R2={r2:.3f}, MAE={mae:.3f}, Corr={correlation:.4f}")

    from model.lstm_model.enhanced_lstm import run_interactive_forensic_analysis
    run_interactive_forensic_analysis(
        timestamps=timestamps_plot,
        y_true=y_true_plot,
        y_phys=y_phys_plot,
        y_ml_resid=y_pred_resid,
        y_total=y_total_pred,
        feature_name=feature_name,
        threshold=threshold,
        buffer_mask=buffer_mask,        # <-- new
        cycle_buffer=cycle_buffer       # <-- new
    )
