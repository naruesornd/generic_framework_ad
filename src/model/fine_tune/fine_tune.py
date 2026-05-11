import pandas as pd
import numpy as np
import sys
import os
from data_processor import DataProcessor
from data_processor import CycleProcessor
from feature_engineering import FeatureEngineering
import torch
from model.load_model.load_model import load_model
from data_loader.industrialstreamloader import IndustrialStreamLoader
import joblib
from data_loader.data_loader import time_series_loader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from utils.simulate.simulate_dataflow import IndustrialDataFlowSimulator
def fine_tune(target_col, data_num = 10000, train_test_split_ratio = 0.8):
    sys.path.append(os.path.abspath('../src'))

    file_path = "../data/raw/data_factory_2.csv"
    dp = DataProcessor(file_path)
    dp.change_pivot('site_date_tz','param_name','display_value')
    dp.drop_NA_with_feature(features=['PrimaryPressure','FeedTemperature'])
    dp.rename_column_to_timestamp('site_date_tz')
    dp.rename_column_to_feedflow('FeedFlowRate')
    dp.rename_column_to_permeateflow('PermeateFlowRate')
    dp.rename_column_to_feedpressure('PrimaryPressure')
    dp.rename_column_to_concentrateflow('ConcentrateFlowRate')
    
    cp = CycleProcessor(column_name='FeedFlow', df = dp.df, threshold=20)
    cp.identify_cycles()
    cp.assign_cycle_features()

    fe = FeatureEngineering(dp)
    fe.generate_cross_features(drop_features=['Recovery', 'PermeateFlow', 'PermeateConductivity', 'PermeatePressure'])
    fe.lag_engineer()

    dp.df = fe.df
    features =dp.df.columns.tolist()
    top_k_features = pd.read_csv(f"../data/temp_data/top_k_features_plant1_{target_col}.csv")
    top_k_features = top_k_features.iloc[:,1].tolist()
    target = [target_col]

    model_path = f"../model/model_weights_{target_col}.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train = dp.df[top_k_features]
    model = load_model(model_path=model_path, X_train=X_train, device=device)
    data = dp.df[top_k_features + target].dropna()
    clean_X = data[top_k_features]
    clean_y = data[target]


    # 验证一致性
    assert len(clean_X) == len(clean_y), "数据长度不一致！"
    x_scaler = joblib.load(f'../data/model_data/scaler_x_{target_col}.pkl')
    y_scaler = joblib.load(f'../data/model_data/scaler_y_{target_col}.pkl')
    X = x_scaler.transform(clean_X.values)
    y = y_scaler.transform(clean_y.values.reshape(-1,1)).flatten()

    # 划分训练集和测试集
    train_size = int(data_num * train_test_split_ratio)
    test_size = data_num - train_size
    X_train_raw, X_test = X[:train_size], X[train_size:]
    y_train_raw, y_test = y[:train_size], y[train_size:]
    X_train, X_val, y_train, y_val = train_test_split(X_train_raw, y_train_raw, test_size=0.2, random_state=42, shuffle=False)

    train_loader = time_series_loader(X_train, y_train, seq_len=12, batch_size=64, shuffle=False)
    val_loader = time_series_loader(X_val, y_val, seq_len=12, batch_size=64, shuffle=False)
    test_loader = IndustrialStreamLoader(X_test, y_test, seq_length=12, delay_ms=100)

    # 5. 微调训练参数
    LEARNING_RATE = 1e-5  # 使用非常小的学习率进行微调
    EPOCHS = 100
    EARLY_STOP_PATIENCE = 10  # 早停耐心值

    # 6. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. 微调训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # 训练模式
        model.train()
        epoch_train_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * inputs.size(0)
        
        # 计算平均训练损失
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # 验证模式
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item() * inputs.size(0)
        
        # 计算平均验证损失
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # 打印进度
        print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}')
        
        # 早停机制和模型保存
        if epoch_val_loss < best_val_loss and abs(epoch_val_loss - best_val_loss) > 2e-3:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f'../model/finetuned_model_{target_col}.pth')
            print(f'保存最佳模型 (Val Loss: {best_val_loss:.6f})')
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f'早停在第 {epoch+1} 轮触发')
                break
    
    # 8. 加载最佳模型
    model.load_state_dict(torch.load(f'../model/finetuned_model_{target_col}.pth'))
    
    # 9. 保存微调后的模型和结果
    # 保存模型权重
    torch.save(model.state_dict(), f'../model/finetuned_model_weights_{target_col}.pth')
    
    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses)+1)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    history_df.to_csv(f'../reports/finetune_history_{target_col}.csv', index=False)

    file_path = f"../data/prediction/{target_col}/predictions.csv"
    IDF = IndustrialDataFlowSimulator(model=model, data_loader=test_loader, device=device, feature_names=top_k_features,seq_length=12, output_path=file_path)
    results = IDF.run_simulation()
