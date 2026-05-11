import numpy as np
from collections import deque
from torch.utils.data import IterableDataset

class IndustrialStreamLoader(IterableDataset):
    def __init__(self, X_scaled, y_scaled, seq_length=12, delay_ms=100):
        """
        标准化数据流加载器（输入已标准化数据）
        
        参数:
            X_scaled: 已标准化的特征数据 (n_samples, n_features)
            y_scaled: 已标准化的目标数据 (n_samples,)
            seq_length: 序列长度 (默认12)
            delay_ms: 模拟实时延迟 (毫秒)
        """
        # 合并特征和目标数据
        self.data = np.column_stack([X_scaled, y_scaled.reshape(-1, 1)])
        
        # 配置参数
        self.seq_length = seq_length
        self.delay_ms = delay_ms
        
        # 初始化指针和窗口
        self.current_idx = seq_length
        self.window = deque(maxlen=seq_length)
        self._init_window()

    def _init_window(self):
        """用历史数据初始化窗口"""
        self.window.extend(self.data[:self.seq_length])

    def __iter__(self):
        return self

    def __next__(self):
        """返回下一个标准化数据点 [features..., target]"""
        if self.current_idx >= len(self.data):
            raise StopIteration
        
        # 模拟实时延迟
        if self.delay_ms > 0:
            import time
            time.sleep(self.delay_ms / 1000)
        
        # 获取并返回数据点 (保持标准化状态)
        point = self.data[self.current_idx]
        self.current_idx += 1
        return point.astype(np.float32)

    def get_current_window(self):
        """获取当前窗口数据 (用于预测)"""
        return np.array(self.window)