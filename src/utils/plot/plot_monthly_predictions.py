import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

def plot_monthly_predictions(y_true, y_pred, timestamps, window_months=3):
    """
    绘制每月的真实值和预测值，并标注基于前三个月残差IQR检测出的异常点。
    """
    # === 用历史3个月IQR检测异常 ===
    df_anomalies = detect_anomalies_iqr(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        timestamps=pd.to_datetime(timestamps),
        window_months=window_months
    )

    # 添加误差列用于计算 MAE、R²
    df_anomalies['error'] = df_anomalies['true_value'] - df_anomalies['predicted_value']
    df_anomalies['year_month'] = df_anomalies['timestamp'].dt.to_period('M')

    # 获取所有唯一的年月并按时间排序
    months = sorted(df_anomalies['year_month'].unique())

    # 创建包含多个子图的大图（最多12个）
    n_cols = 3
    n_rows = int(np.ceil(len(months) / n_cols))
    plt.figure(figsize=(6 * n_cols, 4.5 * n_rows))

    for i, month in enumerate(months, 1):
        ax = plt.subplot(n_rows, n_cols, i)
        month_data = df_anomalies[df_anomalies['year_month'] == month]

        # 绘制真实值和预测值
        ax.plot(month_data['timestamp'], month_data['true_value'], label='True', linewidth=1.5)
        ax.plot(month_data['timestamp'], month_data['predicted_value'], label='Predicted', linestyle='--', linewidth=1)

        # 异常点标红
        outliers = month_data[month_data['anomaly_flag'] == -1]
        ax.scatter(outliers['timestamp'], outliers['true_value'], color='red', label='Outlier', s=20, marker='x')

        # 当月指标
        r2 = r2_score(month_data['true_value'], month_data['predicted_value'])
        mae = mean_absolute_error(month_data['true_value'], month_data['predicted_value'])

        # 设置标题和图例
        ax.set_title(f'{month.strftime("%Y-%m")}\nR²={r2:.3f}, MAE={mae:.3f}', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    # 整体标题和指标
    plt.tight_layout()
    plt.suptitle('Monthly Prediction Performance with IQR-based Anomaly Detection', y=1.02, fontsize=14)
    plt.show()

    # 打印整体指标
    overall_r2 = r2_score(df_anomalies['true_value'], df_anomalies['predicted_value'])
    overall_mae = mean_absolute_error(df_anomalies['true_value'], df_anomalies['predicted_value'])
    print(f'\nOverall Performance: R²={overall_r2:.3f}, MAE={overall_mae:.3f}')

def detect_anomalies_iqr(y_true, y_pred, timestamps, window_months=3, factor=3):
    """使用前三个月的残差计算 IQR，滑动判断异常点"""
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'true_value': y_true.flatten(),
        'predicted_value': y_pred.flatten()
    })
    df['residual'] = np.abs(df['true_value'] - df['predicted_value'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    anomaly_flags = []

    for i in range(len(df)):
        current_time = df.loc[i, 'timestamp']
        window_start = current_time - pd.DateOffset(months=window_months)

        # 取前三个月的历史残差
        history = df[(df['timestamp'] >= window_start) & (df['timestamp'] < current_time)]

        if len(history) < 10:  # 太少的数据跳过异常检测
            anomaly_flags.append(1)
            continue
        
        q1 = np.percentile(history['residual'], 25)
        q3 = np.percentile(history['residual'], 75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        current_residual = df.loc[i, 'residual']
        if current_residual < lower_bound or current_residual > upper_bound:
            anomaly_flags.append(-1)
        else:
            anomaly_flags.append(1)

    df['anomaly_flag'] = anomaly_flags
    return df[['timestamp', 'true_value', 'predicted_value', 'anomaly_flag']]
