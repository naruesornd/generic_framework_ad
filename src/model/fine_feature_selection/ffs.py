# ===================== 1. 定义模型 (适配LSTM) ========== #
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from data_loader import time_series_loader,test_time_series_loader
from utils.plot.fs_plot_test import plot_predictions_by_cycle
class LSTM_Dropout_Net(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.2, num_layers=2):
        super(LSTM_Dropout_Net, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        return self.fc1(lstm_out)

# ===================== 2. DFS 工具函数 (适配LSTM) ===================== #
def fine_feature_selection(dp, top_k_features, target_columns, test_size=0.2, random_state=42, dropout_threshold=0.01):
    if dp.df[top_k_features + target_columns + ["cycle_id", "cycle_time"]].isnull().sum().sum() > 0:
        df_dropna = dp.df[top_k_features + target_columns + ["cycle_id", "cycle_time"]].dropna()
    else:
        df_dropna = dp.df[top_k_features + target_columns + ["cycle_id", "cycle_time"]]

    feature_num = len(top_k_features)
    df_dropna = df_dropna[0:10000]
    # 先提取原始数据（未标准化的）
    X_raw = df_dropna[top_k_features].values
    y_raw = df_dropna[target_columns].values
    cycle_id_all = df_dropna["cycle_id"].values
    cycle_time_all = df_dropna["cycle_time"].values

    # 划分训练 / 测试
    split_idx = int(len(X_raw) * (1 - test_size))
    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train_raw, y_test_raw = y_raw[:split_idx], y_raw[split_idx:]
    cycle_id_train, cycle_id_test = cycle_id_all[:split_idx], cycle_id_all[split_idx:]
    cycle_time_train, cycle_time_test = cycle_time_all[:split_idx], cycle_time_all[split_idx:]

    # ✅ 现在才进行标准化：只用训练集 fit
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train_raw)
    X_test = scaler_x.transform(X_test_raw)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train_raw)
    y_test = scaler_y.transform(y_test_raw)

    train_loader = time_series_loader(X_train, y_train, seq_len=12, batch_size=64, shuffle=False)
    test_loader = test_time_series_loader(X_test, y_test,cycle_id_test,cycle_time_test, seq_len=12, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_Dropout_Net(input_size=feature_num, hidden_size=64, dropout=0.2, num_layers=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = model_training(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=30)
    plot_predictions_by_cycle(model, test_loader, device, scaler_y, target_columns)
    # ========== Drop Feature Selection: 逐个特征置零，比较验证集损失变化 ========== #
    print("\n\n[Drop Feature Selection]\n")
    baseline_loss = evaluate_model(model, test_loader, criterion, device)
    drop_results = {}
    print(f"Base Loss: {baseline_loss:.6f}")
    for idx, col in enumerate(top_k_features):
        X_test_dropped = X_test.copy()
        X_test_dropped[:, idx] = 0
        dropped_loader = test_time_series_loader(X_test_dropped, y_test,cycle_id_test,cycle_time_test, seq_len=12, batch_size=64, shuffle=False)
        drop_loss = evaluate_model(model, dropped_loader, criterion, device)
        delta_ratio = (drop_loss - baseline_loss) / baseline_loss
        drop_results[col] = delta_ratio
        print(f"Drop {col}: loss delta = {delta_ratio:.6f}")

    # 过滤贡献小于阈值的特征
    selected_features = [f for f, d in drop_results.items() if d > dropout_threshold]
    print(f"\n最终选定特征数: {len(selected_features)}")

    return selected_features

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch, _, _ in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            total_loss += criterion(y_pred, y_batch).item()
    return total_loss / len(dataloader)

def model_training(model, train_dataloader, test_dataloader, criterion, optimizer, device,
                   num_epochs=100, patience=10):
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = evaluate_model(model, test_dataloader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss/len(train_dataloader):.4f}  Test Loss: {val_loss:.4f}")

        # ========== Early Stopping Check ==========
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"🛑 No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"\n✅ Early stopping triggered at epoch {epoch+1}. Best val loss: {best_loss:.4f}")
            break

    model.load_state_dict(best_model_wts)
    return model


def dataset_to_tensor(dataset):
    samples = []
    for X, _ in dataset:
        samples.append(X)
    return torch.stack(samples)
