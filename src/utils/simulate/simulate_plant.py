from .simulate_dataflow import IndustrialDataFlowSimulator
import pandas as pd
import numpy as np
import os
import torch
from model.load_model.load_model import load_model
from data_loader.industrialstreamloader import IndustrialStreamLoader
import joblib
import re
def simulate_plant(top_k_features, output_feature):

    test_data_path = f"../data/model_data/test_data_export_{output_feature}.csv"
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    
    test_df = pd.read_csv(test_data_path).dropna()

    # 处理目标列名形如 "['PermeateFlow']"
    # 处理目标列名，提取和匹配
    target_col_name = [col for col in test_df.columns if output_feature == col.strip("[]'")]

    print(f"Target column: {target_col_name}")

    if not target_col_name:
        raise ValueError(f"Cannot find target column matching {output_feature} in test data.")
    target_name = target_col_name[0].strip("[]'")
    target_col_name = target_col_name[0]

    print(f"Target column: {target_col_name}")
    clean_X = test_df[top_k_features]
    clean_y = test_df[target_col_name]

    x_scaler = joblib.load(f'../data/model_data/scaler_x_{output_feature}.pkl')
    y_scaler = joblib.load(f'../data/model_data/scaler_y_{output_feature}.pkl')
    X = x_scaler.transform(clean_X.values)
    y = y_scaler.transform(clean_y.values.reshape(-1, 1)).flatten()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = f"../model/model_weights_{output_feature}.pth"
    model = load_model(model_path=model_path, X_train=clean_X, device=device)

    dataloader = IndustrialStreamLoader(X_scaled=X, y_scaled=y, seq_length=12, delay_ms=100)
    output_path = f"../data/prediction/{output_feature}/predictions.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    simulator = IndustrialDataFlowSimulator(
        model=model,
        data_loader=dataloader,
        device=device,
        feature_names=top_k_features,
        target_name=target_name,
        seq_length=12,
        output_path=output_path,
        y_scaler=y_scaler
    )

    results = simulator.run_simulation()
    return results


    
