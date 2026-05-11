import matplotlib.pyplot as plt
import imageio
import numpy as np
from PIL import Image
class RealTimePlotter:
    def __init__(self, target_name, title='Real-Time Prediction', max_points=200):
        self.max_points = max_points
        self.timestamps = []
        self.actuals = []
        self.predictions = []
        self.anomaly_flags = []

        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line_actual, = self.ax.plot([], [], label="true value", color="blue")
        self.line_pred, = self.ax.plot([], [], label="predicted value", color="green")
        self.scatter_anomaly = self.ax.scatter([], [], color="red", label="anomaly point")

        self.ax.set_title(title)
        self.ax.set_xlabel("time")
        self.ax.set_ylabel(f"{target_name}_value")
        self.ax.grid(True)
        self.ax.legend()

        self.frames = []  # 存储每一帧
        plt.show(block=False)

    def update(self, t, actual, predicted, is_anomaly=False):
        self.timestamps.append(t)
        self.actuals.append(actual)
        self.predictions.append(predicted)
        self.anomaly_flags.append(is_anomaly)

        # 控制最大显示范围
        if len(self.timestamps) > self.max_points:
            self.timestamps = self.timestamps[-self.max_points:]
            self.actuals = self.actuals[-self.max_points:]
            self.predictions = self.predictions[-self.max_points:]
            self.anomaly_flags = self.anomaly_flags[-self.max_points:]

        self.line_actual.set_data(self.timestamps, self.actuals)
        self.line_pred.set_data(self.timestamps, self.predictions)

        # 设置动态坐标轴范围
        self.ax.set_xlim(max(0, self.timestamps[0]), self.timestamps[-1] + 1)
        ymin = min(min(self.actuals), min(self.predictions)) - 1
        ymax = max(max(self.actuals), max(self.predictions)) + 1
        self.ax.set_ylim(ymin, ymax)

        # 更新异常点
        anomalies_x = [self.timestamps[i] for i, flag in enumerate(self.anomaly_flags) if flag]
        anomalies_y = [self.predictions[i] for i, flag in enumerate(self.anomaly_flags) if flag]
        self.scatter_anomaly.remove()
        self.scatter_anomaly = self.ax.scatter(anomalies_x, anomalies_y, color="red", s=50, label="error point")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.pause(0.01)

        # 保存当前帧为静态图像（RGBA）
        image_from_canvas = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        image_from_canvas = image_from_canvas.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
        image_from_canvas = image_from_canvas[:, :, [1, 2, 3, 0]]  # ARGB -> RGBA
        pil_image = Image.fromarray(image_from_canvas)
        self.frames.append(pil_image)

    def save_gif(self, gif_filename="../img/realtime_prediction.gif", fps=5):
        """保存所有收集的帧为 GIF 文件"""
        imageio.mimsave(gif_filename, self.frames, fps=fps)
        print(f"GIF 已保存至 {gif_filename}")
