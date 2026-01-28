import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class CycleWindowDataset(Dataset):
    """
    Builds fixed-length windows inside each cycle_id.
    Each sample: X[t:t+seq_len] where all rows share same cycle_id.
    """
    def __init__(self, df, feature_cols, seq_len=24, stride=1, cycle_col="cycle_id"):
        self.df = df
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.stride = stride
        self.cycle_col = cycle_col

        self.samples = []  # list of (cycle_id, start_pos)
        self._build_index()

    def _build_index(self):
        self.samples.clear()
        for cid, g in self.df.groupby(self.cycle_col):
            g = g.sort_values("cycle_time") if "cycle_time" in g.columns else g
            n = len(g)
            if n < self.seq_len:
                continue
            for start in range(0, n - self.seq_len + 1, self.stride):
                self.samples.append((cid, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cid, start = self.samples[idx]
        g = self.df[self.df[self.cycle_col] == cid]
        g = g.sort_values("cycle_time") if "cycle_time" in g.columns else g
        window = g.iloc[start:start + self.seq_len][self.feature_cols].values.astype(np.float32)
        return torch.from_numpy(window), cid, start


import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.out = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: [B, T, F]
        enc_out, (h, c) = self.encoder(x)        # h: [L, B, H]
        # repeat final hidden across time
        z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)  # [B, T, H]
        dec_out, _ = self.decoder(z)             # [B, T, H]
        x_hat = self.out(dec_out)                # [B, T, F]
        return x_hat


from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch

def train_cycle_ae(df, feature_cols, seq_len=24, stride=1,
                   hidden_dim=64, lr=1e-3, epochs=30, batch_size=64,
                   device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # scale features
    scaler = StandardScaler()
    X = df[feature_cols].copy()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(X)

    ds = CycleWindowDataset(df_scaled, feature_cols, seq_len=seq_len, stride=stride)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = LSTMAutoencoder(n_features=len(feature_cols), hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        n = 0
        for xb, _, _ in dl:
            xb = xb.to(device)
            opt.zero_grad()
            xhat = model(xb)
            loss = criterion(xhat, xb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        print(f"Epoch {ep:03d} | loss={total/n:.6f}")

    return model, scaler, df_scaled


from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch

def train_cycle_ae(df, feature_cols, seq_len=24, stride=1,
                   hidden_dim=64, lr=1e-3, epochs=30, batch_size=64,
                   device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # scale features
    scaler = StandardScaler()
    X = df[feature_cols].copy()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(X)

    ds = CycleWindowDataset(df_scaled, feature_cols, seq_len=seq_len, stride=stride)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = LSTMAutoencoder(n_features=len(feature_cols), hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        n = 0
        for xb, _, _ in dl:
            xb = xb.to(device)
            opt.zero_grad()
            xhat = model(xb)
            loss = criterion(xhat, xb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        print(f"Epoch {ep:03d} | loss={total/n:.6f}")

    return model, scaler, df_scaled


import numpy as np
import pandas as pd
import torch

@torch.no_grad()
def score_cycles(df, feature_cols, model, scaler,
                 seq_len=24, stride=1, cycle_col="cycle_id",
                 agg="p95", device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])

    # prepare storage for row-level scores
    row_score = np.full(len(df_scaled), np.nan, dtype=float)
    row_count = np.zeros(len(df_scaled), dtype=int)

    # score windows cycle-by-cycle
    for cid, g in df_scaled.groupby(cycle_col):
        g = g.sort_values("cycle_time") if "cycle_time" in g.columns else g
        idxs = g.index.to_numpy()
        X = g[feature_cols].values.astype(np.float32)

        if len(X) < seq_len:
            continue

        for start in range(0, len(X) - seq_len + 1, stride):
            window = torch.from_numpy(X[start:start+seq_len]).unsqueeze(0).to(device)  # [1,T,F]
            xhat = model(window).cpu().numpy()[0]  # [T,F]
            err_t = np.mean((xhat - X[start:start+seq_len])**2, axis=1)  # [T]

            # add window timestep errors back to the original rows
            target_rows = idxs[start:start+seq_len]
            row_score[target_rows] = np.nan_to_num(row_score[target_rows], nan=0.0) + err_t
            row_count[target_rows] += 1

    # average overlapping windows
    valid = row_count > 0
    row_score[valid] = row_score[valid] / row_count[valid]

    df_out = df.copy()
    df_out["ae_recon_error"] = row_score

    # aggregate per cycle
    cycle_scores = []
    for cid, g in df_out.groupby(cycle_col):
        s = g["ae_recon_error"].dropna().values
        if len(s) == 0:
            continue
        if agg == "mean":
            score = float(np.mean(s))
        elif agg == "median":
            score = float(np.median(s))
        elif agg == "max":
            score = float(np.max(s))
        elif agg == "p95":
            score = float(np.percentile(s, 95))
        else:
            raise ValueError("agg must be mean/median/max/p95")
        cycle_scores.append({"cycle_id": cid, f"cycle_score_{agg}": score, "n_rows": len(g)})

    cycle_scores = pd.DataFrame(cycle_scores).sort_values(f"cycle_score_{agg}", ascending=False)
    return df_out, cycle_scores
