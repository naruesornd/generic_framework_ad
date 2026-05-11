import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from utils.plot.dynamic_plot import RealTimePlotter
class IndustrialDataFlowSimulator:
    def __init__(self, model, data_loader, device, feature_names, target_name, seq_length=12, 
                 max_points=300, output_path='../data/model_data/predictions.csv',
                 anomaly_window=20, iqr_multiplier=1.5, x_scaler=None, y_scaler=None):
        """
        工业数据流模拟器初始化（增加特征分析功能）
        
        参数:
            model: 训练好的LSTM模型
            data_loader: 数据生成器/迭代器
            device: 计算设备 (cpu/cuda)
            feature_names: 特征名称列表
            seq_length: 序列长度
            max_points: 最大处理数据点数
            output_path: 输出文件路径
            anomaly_window: 异常检测窗口大小
            iqr_multiplier: IQR乘数因子
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.feature_names = feature_names
        self.target_name = target_name
        self.num_features = len(feature_names)
        self.seq_length = seq_length
        self.max_points = max_points
        self.output_path = output_path
        self.anomaly_window = anomaly_window
        self.iqr_multiplier = iqr_multiplier
        self.x_scaler = x_scaler  # 特征的Scaler
        self.y_scaler = y_scaler  # 目标值的Scaler
        # 初始化数据结构
        self.window = deque(maxlen=seq_length)
        self.prediction_window = deque(maxlen=anomaly_window)
        
        # 结果存储
        self.all_predictions = []
        self.all_actuals = []
        self.timestamps = []
        self.anomalies = []
        self.anomaly_bounds = []
        self.anomaly_contributions = []  # 存储异常点的特征贡献分析
        
        # 状态变量
        self.point_count = 0
        self.lower_bound = None
        self.upper_bound = None

    def initialize_window(self):
        """填充初始窗口数据"""
        print(f"等待初始{self.seq_length}个数据点...")
        while len(self.window) < self.seq_length:
            try:
                data_point = next(self.data_loader)
                self.window.append(data_point)
            except StopIteration:
                raise ValueError("数据不足，无法填充初始窗口")

    def prepare_sequence(self):
        """准备当前序列数据"""
        current_seq = np.array(self.window)[:, :-1]  # 排除最后一列目标值
        seq_tensor = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)

        # 如果传入了x_scaler，则对当前输入进行反标准化
        if self.x_scaler:
            seq_tensor = torch.FloatTensor(self.x_scaler.inverse_transform(current_seq)).unsqueeze(0).to(self.device)

        return seq_tensor


    def make_prediction(self, seq_tensor):
        """使用模型进行预测"""
        with torch.no_grad():
            self.model.eval()
            return self.model(seq_tensor).cpu().numpy().flatten()[0]

    def analyze_feature_contributions(self, seq_tensor):
        """
        修复后的特征贡献分析
        """
        seq_tensor = seq_tensor.clone().detach().requires_grad_(True)
        if seq_tensor.grad is not None:
            seq_tensor.grad.zero_()
        
        prediction = self.model(seq_tensor)
        prediction.backward()
        
        gradients = seq_tensor.grad.data.cpu().numpy()
        
        # 计算每个特征的平均梯度（跨时间步）
        # 修改点：直接计算每个特征的平均值，保持向量形式
        feature_contributions = np.mean(np.abs(gradients), axis=(0, 1))  # 形状 (num_features,)
        
        # 归一化处理
        total_sum = np.sum(feature_contributions)
        if total_sum > 0:
            normalized_contributions = feature_contributions / total_sum
        else:
            normalized_contributions = np.zeros_like(feature_contributions)
        
        return normalized_contributions

    def detect_anomaly(self, prediction):
        """使用IQR方法检测异常点"""
        is_anomaly = False
        current_bound = (None, None)
        
        # 更新预测窗口
        self.prediction_window.append(prediction)
        
        # 当窗口填满后开始检测
        if len(self.prediction_window) >= self.anomaly_window:
            preds = np.array(self.prediction_window)
            q1 = np.percentile(preds, 25)
            q3 = np.percentile(preds, 75)
            iqr = q3 - q1
            
            # 计算异常边界
            self.lower_bound = q1 - self.iqr_multiplier * iqr
            self.upper_bound = q3 + self.iqr_multiplier * iqr
            current_bound = (self.lower_bound, self.upper_bound)
            
            # 检查当前预测值是否异常
            if prediction < self.lower_bound or prediction > self.upper_bound:
                is_anomaly = True
                print(f"⚠️ 异常点检测! 时间点: {self.point_count}, 预测值: {prediction:.3f}, "
                      f"边界: [{self.lower_bound:.3f}, {self.upper_bound:.3f}]")
        
        return is_anomaly, current_bound

    def update_state(self, prediction, actual_value, is_anomaly, current_bound):
        """更新系统状态"""
        self.all_predictions.append(prediction)
        self.all_actuals.append(actual_value)
        self.timestamps.append(self.point_count)
        self.anomalies.append(1 if is_anomaly else 0)
        self.anomaly_bounds.append(current_bound)
        
        # 更新窗口
        self.window.append(next(self.data_loader))
        
        # 更新计数器
        self.point_count += 1

    def print_progress(self, prediction, actual_value):
        """打印实时进度"""
        if self.point_count % 50 == 0:
            status = f"已处理 {self.point_count}/{self.max_points} 个点 | 最新预测: {prediction:.3f}, 实际值: {actual_value:.3f}"
            if self.lower_bound is not None:
                status += f" | 异常边界: [{self.lower_bound:.3f}, {self.upper_bound:.3f}]"
            print(status)

    def calculate_metrics(self):
        """计算评估指标"""
        mae = mean_absolute_error(self.all_actuals, self.all_predictions)
        mse = mean_squared_error(self.all_actuals, self.all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.all_actuals, self.all_predictions)
        
        # 异常检测统计
        anomaly_count = sum(self.anomalies)
        anomaly_rate = anomaly_count / len(self.anomalies) if len(self.anomalies) > 0 else 0
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_rate
        }

    def save_results(self):
        """保存结果到CSV文件"""
        results = pd.DataFrame({
            'timestamp': self.timestamps,
            'actual': self.all_actuals,
            'predicted': self.all_predictions,
            'is_anomaly': self.anomalies,
            'lower_bound': [b[0] for b in self.anomaly_bounds],
            'upper_bound': [b[1] for b in self.anomaly_bounds]
        })
        results.to_csv(self.output_path, index=False)
        print(f"预测结果已保存至 {self.output_path}")
        
        # 保存特征贡献分析结果
        if self.anomaly_contributions:
            anomaly_df = pd.DataFrame(self.anomaly_contributions, 
                                     columns=['timestamp', 'top_feature', 'contribution'])
            anomaly_path = self.output_path.replace('.csv', '_anomaly_contributions.csv')
            anomaly_df.to_csv(anomaly_path, index=False)
            print(f"异常点特征贡献分析已保存至 {anomaly_path}")
        
        return results

    def plot_results(self, results):
        """绘制结果图表"""
        plt.figure(figsize=(15, 10))
        
        # 主预测图
        plt.subplot(2, 1, 1)
        plt.plot(results['timestamp'], results['actual'], label='actual', alpha=0.7)
        plt.plot(results['timestamp'], results['predicted'], label='predicted', alpha=0.7)
        
        # 标记异常点
        anomalies_df = results[results['is_anomaly'] == 1]
        if not anomalies_df.empty:
            plt.scatter(anomalies_df['timestamp'], anomalies_df['predicted'], 
                        color='red', s=50, zorder=5, label='anomaly')
        
        # 绘制异常边界
        if not results['lower_bound'].isnull().all():
            plt.plot(results['timestamp'], results['lower_bound'], 
                     '--', color='orange', alpha=0.6, label='lower bound')
            plt.plot(results['timestamp'], results['upper_bound'], 
                     '--', color='orange', alpha=0.6, label='upper bound')
            plt.fill_between(results['timestamp'], results['lower_bound'], 
                             results['upper_bound'], color='orange', alpha=0.1)
        
        plt.title(f'Real-time Prediction and Anomaly Detection (Total {len(results)} points)')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # 残差图
        plt.subplot(2, 1, 2)
        residuals = np.array(self.all_actuals) - np.array(self.all_predictions)
        plt.plot(results['timestamp'], residuals, label='residual')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        
        # 标记异常点对应的残差
        if not anomalies_df.empty:
            anomaly_residuals = residuals[results['is_anomaly'] == 1]
            plt.scatter(anomalies_df['timestamp'], anomaly_residuals, 
                        color='red', s=50, zorder=5, label='anomaly residual')
        
        plt.title('Prediction Residual')
        plt.xlabel('Time step')
        plt.ylabel('Residual value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = self.output_path.replace('.csv', '.png')
        plt.savefig(plot_path, dpi=300)
        print(f"结果图表已保存至 {plot_path}")
        plt.close()

    def run_simulation(self, show_plot=True):
        self.initialize_window()
        print(f"开始实时预测(最多处理{self.max_points}个点)...")

        plotter = RealTimePlotter(target_name=self.target_name,max_points=200) if show_plot else None


        while self.point_count < self.max_points:
            try:
                seq_tensor = self.prepare_sequence()
                prediction = self.make_prediction(seq_tensor)

                next_data = next(self.data_loader)
                actual_value = next_data[-1]  # 实际目标值

                # 如果传入了y_scaler，则对实际值和预测值进行反标准化
                if self.y_scaler:
                    actual_value = self.y_scaler.inverse_transform(actual_value.reshape(1, -1))[0][0]
                    prediction = self.y_scaler.inverse_transform(np.array([[prediction]]))[0][0]

                is_anomaly, current_bound = self.detect_anomaly(prediction)

                if is_anomaly:
                    contributions = self.analyze_feature_contributions(seq_tensor)
                    top_index = np.argmax(contributions)
                    top_feature = self.feature_names[top_index]
                    top_contribution = contributions[top_index]
                    print(f"⚠️ 异常点贡献分析: {top_feature} ({top_contribution:.4f})")
                    self.anomaly_contributions.append((self.point_count, top_feature, top_contribution))

                self.update_state(prediction, actual_value, is_anomaly, current_bound)

                if show_plot:
                    plotter.update(self.point_count, actual_value, prediction, is_anomaly)

                self.print_progress(prediction, actual_value)
            except StopIteration:
                print("数据流结束")
                break
        if plotter is not None:
            # 假设 update 被调用多次后再保存 gif
            if plotter.frames:
                gif_filename=f"../img/{self.target_name}_realtime_prediction.gif"
                plotter.save_gif(gif_filename=gif_filename, fps=5)
            else:
                print("未生成任何帧，无法保存 GIF。请确认是否调用了 update()。")