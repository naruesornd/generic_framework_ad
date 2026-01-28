import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)   # 1 target
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)          # [batch, seq_len, hidden_dim]
        last_hidden = lstm_out[:, -1, :]    # use last time step
        out = self.fc(last_hidden)          # [batch, 1]
        return out


def build_sequences(X, y, seq_len):
    """
    X: numpy array [N, num_features]
    y: numpy array [N] or [N, 1]
    seq_len: length of input sequence

    Returns:
        X_seq: [N - seq_len + 1, seq_len, num_features]
        y_seq: [N - seq_len + 1, 1]
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    X_seq = []
    y_seq = []
    N = len(X)
    for i in range(N - seq_len + 1):
        X_seq.append(X[i:i+seq_len, :])
        # predict value at the end of the window
        y_seq.append(y[i+seq_len-1])

    X_seq = np.stack(X_seq, axis=0)
    y_seq = np.stack(y_seq, axis=0)
    return X_seq, y_seq


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y_batch.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0).reshape(-1)
    all_trues = np.concatenate(all_trues, axis=0).reshape(-1)
    return total_loss / len(loader.dataset), all_trues, all_preds


def train_lstm_for_target(dp,
                          feature_cols,
                          target_col,
                          seq_len=12,
                          test_size=0.2,
                          val_size=0.2,
                          num_epochs=50,
                          batch_size=64,
                          hidden_dim=64,
                          lr=1e-3,
                          patience=5):
    """
    dp: DataProcessor with dp.df (inputs) and dp.outputs_df (targets)
    feature_cols: list of feature column names from dp.df
    target_col: string, column name in dp.outputs_df
    """
    # 1) Build a DataFrame with features + target, drop NaNs
    X_all = dp.df[feature_cols]
    if dp.outputs_df is not None and target_col in dp.outputs_df.columns:
        y_all = dp.outputs_df[target_col]
    else:
        # fallback: target is inside dp.df
        y_all = dp.df[target_col]

    df_all = pd.concat([X_all, y_all], axis=1).dropna()
    X = df_all[feature_cols].to_numpy()
    y = df_all[target_col].to_numpy()

    # 2) Train/test split in time order
    N = len(X)
    test_len = int(N * test_size)
    trainval_len = N - test_len

    X_trainval, X_test = X[:trainval_len], X[trainval_len:]
    y_trainval, y_test = y[:trainval_len], y[trainval_len:]

    # further split train/val
    val_len = int(trainval_len * val_size)
    X_train, X_val = X_trainval[:-val_len], X_trainval[-val_len:]
    y_train, y_val = y_trainval[:-val_len], y_trainval[-val_len:]

    # 3) Scale features and target
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled   = scaler_x.transform(X_val)
    X_test_scaled  = scaler_x.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled   = scaler_y.transform(y_val.reshape(-1, 1))
    y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1))

    # 4) Build sequences
    X_train_seq, y_train_seq = build_sequences(X_train_scaled, y_train_scaled, seq_len)
    X_val_seq,   y_val_seq   = build_sequences(X_val_scaled,   y_val_scaled,   seq_len)
    X_test_seq,  y_test_seq  = build_sequences(X_test_scaled,  y_test_scaled,  seq_len)

    # 5) DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_seq, dtype=torch.float32),
        torch.tensor(y_val_seq, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_seq, dtype=torch.float32),
        torch.tensor(y_test_seq, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # 6) Model + training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyLSTM(input_dim=len(feature_cols),
                   hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = eval_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 7) Final test evaluation (in original units)
    test_loss, y_true_scaled, y_pred_scaled = eval_one_epoch(model, test_loader, criterion, device)

    # inverse-transform
    y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).reshape(-1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Test MSE (scaled): {test_loss:.4f}")
    print(f"Test R2: {r2:.4f}, Test MAE: {mae:.4f} (in original units)")

    # 8) Plot first 200 points
    n_plot = min(200, len(y_true))
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:n_plot], label="True")
    plt.plot(y_pred[:n_plot], label="Pred")
    plt.title(f"{target_col} – True vs Predicted")
    plt.xlabel("Time index in test set")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return model, scaler_x, scaler_y, history
