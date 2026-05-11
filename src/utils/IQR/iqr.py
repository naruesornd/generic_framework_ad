import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def iqr(df, features, factor=1.5,plot=True):
    Q1 = df[features].quantile(0.25)
    Q3 = df[features].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 检查每个特征是否超出边界
    outliers_iqr = df[
        (df[features] < lower_bound) | (df[features] > upper_bound)
    ].any(axis=1)
    outliers_iqr = df[outliers_iqr]

    # 清理异常点（删除IQR检测到的异常）
    clean_df = df[~df.index.isin(outliers_iqr.index)]

    print(f"Original shape: {df.shape}")
    print(f"Clean shape: {clean_df.shape}")
    print(f"Number of outliers removed: {len(df) - len(clean_df)}")

    # 可视化对比（箱线图）
    if plot:
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=clean_df[features])
        plt.xticks(rotation=45)
        plt.title('Boxplot of Features After Outlier Removal')
        plt.show()

