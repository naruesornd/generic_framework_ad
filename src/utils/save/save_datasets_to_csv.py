import pandas as pd
import numpy as np

def save_datasets_to_csv(train_loader, val_loader, test_loader, feature_names=None, target_name='target'):
    """
    将DataLoader中的数据集转换为DataFrame并保存为CSV
    参数:
        train_loader, val_loader, test_loader: PyTorch DataLoader对象
        feature_names: 特征列名称列表(可选)
        target_name: 目标列名称(默认为'target')
    """
    def loader_to_df(loader, set_name):
        # 获取数据集
        dataset = loader.dataset
        
        # 处理TensorDataset或自定义Dataset
        if hasattr(dataset, 'tensors'):  # TensorDataset
            features = dataset.tensors[0].numpy()
            targets = dataset.tensors[1].numpy()
        elif hasattr(dataset, 'X') and hasattr(dataset, 'y'):  # 自定义Dataset(X/y属性)
            features = dataset.X if isinstance(dataset.X, np.ndarray) else dataset.X.numpy()
            targets = dataset.y if isinstance(dataset.y, np.ndarray) else dataset.y.numpy()
        else:
            raise ValueError("无法识别的数据集格式，需要TensorDataset或包含X/y属性的Dataset")
        
        # 确保targets是二维的
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        
        # 合并特征和目标
        data = np.concatenate([features, targets], axis=1)
        
        # 创建列名
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        columns = feature_names + [target_name]
        
        # 创建DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # 添加数据集类型标识
        df['dataset_type'] = set_name
        
        return df

    # 转换各数据集
    train_df = loader_to_df(train_loader, 'train')
    val_df = loader_to_df(val_loader, 'val')
    test_df = loader_to_df(test_loader, 'test')
    
    # 合并所有数据集(可选)
    combined_df = pd.concat([train_df, val_df, test_df], axis=0)
    
    # 保存为CSV
    train_df.to_csv('train_dataset.csv', index=False)
    val_df.to_csv('validation_dataset.csv', index=False)
    test_df.to_csv('test_dataset.csv', index=False)
    combined_df.to_csv('combined_dataset.csv', index=False)
    
    print("数据集已成功保存为CSV文件:")
    print(f"- 训练集: train_dataset.csv (样本数: {len(train_df)})")
    print(f"- 验证集: validation_dataset.csv (样本数: {len(val_df)})")
    print(f"- 测试集: test_dataset.csv (样本数: {len(test_df)})")
    print(f"- 合并集: combined_dataset.csv (样本数: {len(combined_df)})")
    
    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'combined_df': combined_df
    }

# 使用示例
data_frames = save_datasets_to_csv(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    feature_names=['temp', 'pressure', 'humidity'],  # 替换为实际特征名
    target_name='output'  # 替换为目标变量名
)