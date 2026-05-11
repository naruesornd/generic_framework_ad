import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_loader import time_series_loader,test_time_series_loader
def plot_prediction(
    model, 
    X, 
    y, 
    device, 
    seq_len=12,  # 序列长度
    batch_size=64,  # 批大小
    n_points=500,  # 要绘制的点数
    start_idx=0,  # 起始索引
    save_path=None  # 图片保存路径
):
    """
    完整的预测和可视化函数，包含NaN处理、数据加载和评估
    
    参数:
        model: 训练好的PyTorch模型
        X: 输入特征 (DataFrame, numpy array 或 Tensor)
        y: 真实标签 (Series, numpy array 或 Tensor)
        device: 计算设备
        time_series_loader: 时间序列数据加载器函数
        seq_len: 序列长度
        batch_size: 批大小
        n_points: 要绘制的点数
        start_idx: 起始索引
        save_path: 图片保存路径
    """
    # 检查并处理 NaN 值
    if isinstance(X, (np.ndarray, torch.Tensor)):
        # 统一转换为 numpy 数组进行检查
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        if np.isnan(X_np).any():
            print("警告: 输入数据包含 NaN 值，请先处理")
            raise ValueError("输入数据包含 NaN 值")
    else:
        raise TypeError("输入数据类型必须是 numpy.ndarray 或 torch.Tensor")

    if isinstance(y, (np.ndarray, torch.Tensor)):
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        if np.isnan(y_np).any():
            print("警告: 标签数据包含 NaN 值，请先处理")
            raise ValueError("标签数据包含 NaN 值")
    else:
        raise TypeError("标签数据类型必须是 numpy.ndarray 或 torch.Tensor")

    # 转换为 Tensor（如果还不是 Tensor）
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    # 检查数据量是否足够
    total_samples = X.shape[0]
    if start_idx + n_points > total_samples:
        print(f"警告: 数据量不足，只能取 {total_samples - start_idx} 个点")
        n_points = total_samples - start_idx
    
    # 获取数据子集
    X_subset = X[start_idx : start_idx + n_points]
    y_subset = y[start_idx : start_idx + n_points]
    
    # 创建数据加载器
    loader = time_series_loader(X_subset, y_subset, seq_len=seq_len, 
                              batch_size=batch_size, shuffle=False)
    
    # 模型预测
    model.eval()
    y_preds = []
    y_trues = []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).cpu().numpy()
            y_preds.append(preds)
            y_trues.append(y_batch.numpy())
    
    # 合并结果
    y_pred = np.concatenate(y_preds, axis=0).flatten()
    y_true = np.concatenate(y_trues, axis=0).flatten()
    
    # 检查长度是否一致
    assert len(y_pred) == len(y_true), f"预测值({len(y_pred)})和真实值({len(y_true)})长度不一致"
    
    # 计算评估指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"评估指标 (基于索引 {start_idx} 到 {start_idx + n_points} 的 {len(y_true)} 个样本):")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # 创建画布
    plt.figure(figsize=(20, 8))
    
    # ================= 左图：散点对比图 =================
    plt.subplot(1, 2, 1)
    
    # 绘制散点图
    scatter = plt.scatter(y_true, y_pred, alpha=0.6, 
                         c=np.abs(y_true - y_pred),  # 颜色表示误差大小
                         cmap='viridis', 
                         label='Predicted vs True')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Absolute Error')
    
    # 绘制理想对角线
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
    
    # 美化设置
    plt.xlabel('True Value', fontsize=12)
    plt.ylabel('Predicted Value', fontsize=12)
    plt.title(f'Scatter Comparison\n(Index {start_idx}-{start_idx + n_points})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # ================= 右图：曲线对比图 =================
    plt.subplot(1, 2, 2)
    
    # 绘制前200个样本（可根据需要调整）
    plot_samples = min(400, len(y_true))
    x_axis = np.arange(plot_samples)
    
    # 绘制曲线
    plt.plot(x_axis, y_true[:plot_samples], 'b-', label='True Values', 
             alpha=0.7, linewidth=1.5)
    plt.plot(x_axis, y_pred[:plot_samples], 'r--', label='Predictions', 
             alpha=0.8, linewidth=1.2)
    
    # 标记异常点
    diff = np.abs(y_true[:plot_samples] - y_pred[:plot_samples])
    threshold = np.mean(diff) + 2 * np.std(diff)
    anomalies = np.where(diff > threshold)[0]
    plt.scatter(anomalies, y_true[anomalies], c='yellow', s=50,
                edgecolors='red', label='Large Errors (>2σ)', zorder=3)
    
    # 美化设置
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Target Value', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存至: {save_path}")
    else:
        plt.show()
    
    return y_pred