import matplotlib.pyplot as plt
import torch
import pandas as pd
import random
def plot_predictions_by_cycle(model, dataloader, device, scaler_y=None, target_name="Target", max_cycles=4):
    model.eval()
    preds = []
    trues = []
    cycle_ids = []
    cycle_times = []

    with torch.no_grad():
        for X_batch, y_batch, cid_batch, ctime_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)

            preds.append(y_pred.cpu())
            trues.append(y_batch.cpu())
            cycle_ids.extend(cid_batch)
            cycle_times.extend(ctime_batch)

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    print(preds.shape, trues.shape)
    # 反标准化（如果需要）
    if scaler_y is not None:
        preds = scaler_y.inverse_transform(preds)
        trues = scaler_y.inverse_transform(trues)

    # 组装 DataFrame
    df_plot = pd.DataFrame({
        "cycle_id": cycle_ids,
        "cycle_time": cycle_times,
        "true": trues[:, 0],
        "pred": preds[:, 0]
    })

    # 随机选取 max_cycles 个周期
    unique_cycles = df_plot["cycle_id"].unique()
    selected_cycles = random.sample(list(unique_cycles), min(max_cycles, len(unique_cycles)))

    for cid in selected_cycles:
        group = df_plot[df_plot["cycle_id"] == cid].sort_values("cycle_time")
        plt.figure(figsize=(10, 4))
        plt.plot(group["cycle_time"], group["true"], label="True", color="black")
        plt.plot(group["cycle_time"], group["pred"], label="Predicted", color="red", linestyle="--")
        plt.title(f"{target_name} - Cycle {cid}")
        plt.xlabel("Cycle Time")
        plt.ylabel(target_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
