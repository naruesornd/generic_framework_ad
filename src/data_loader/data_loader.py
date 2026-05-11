import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .data_set import TimeSeriesDataset,TestTimeSeriesDataset
def time_series_loader(X, y, seq_len, batch_size=64, shuffle=False) -> DataLoader:

    dataset = TimeSeriesDataset(X=X, y=y, seq_len=seq_len)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

def test_time_series_loader(X, y, cycle_id, cycle_time, seq_len=12, batch_size=64, shuffle=False):
    dataset = TestTimeSeriesDataset(X, y, cycle_id, cycle_time, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)