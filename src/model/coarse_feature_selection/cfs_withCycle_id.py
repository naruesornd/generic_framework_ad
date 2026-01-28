import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_processor import DataProcessor

def random_forest_regressor(dp: DataProcessor, target_colums, features, plant_name,test_size=0.2, random_state=42, top_num = 5):
    
    if target_colums not in dp.outputs_df.columns:
        raise ValueError(f"Target column '{target_colums}' not found in dp.outputs_df")
    # 1. 数据拆分
    # Remove cycle metadata (cycle_id, cycle_time) since these should not be inputs for RF
    features = [f for f in features if f not in ['cycle_time']]
    # 选出前10000个数据
    selected_data = dp.df[0:10000]
    selected_output_data = dp.outputs_df[0:10000]
    X_train, X_test, y_train, y_test = train_test_split(
       selected_data[features], selected_output_data[target_colums], test_size=test_size, random_state=random_state
    )

    # 2. 训练随机森林
    print("训练 RandomForest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train.values.ravel())

   # Gets feature importances from RF, sorts features from most to least important. Keeps 4 * top_num of them
    importances = rf.feature_importances_
    feature_importance_series = pd.Series(importances, index=X_train.columns)
    top_features = feature_importance_series.sort_values(ascending=False).head(4*top_num).index.tolist()

    X_sub = X_train[top_features]
    y_sub = y_train
    print("训练 simple RandomForest...")
    rf_small = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf_small.fit(X_sub, y_sub.values.ravel())
    explainer = shap.TreeExplainer(rf_small)
    X_sub_sample = X_sub.sample(n=200, random_state=42)
    shap_values = explainer.shap_values(X_sub_sample)

    # 3. 计算 SHAP 重要性
    shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X_sub.columns)
    top_k_features = shap_importance.sort_values(ascending=False).head(top_num*2).index.tolist()

    shap.summary_plot(shap_values, X_sub_sample, feature_names=X_sub.columns, max_display=top_num*2)

    import_features = pd.DataFrame(top_k_features)
    import_data = dp.df[top_k_features]
    import_data.to_csv(f"../data/processed/top_k_features_data_{plant_name}_{target_colums}.csv")
    import_features.to_csv(f"../data/processed/top_k_features_{plant_name}_{target_colums}.csv")
    return top_k_features



# ################## PREVIOUS #####################

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# import shap
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from sklearn.preprocessing import StandardScaler
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from data_processor import DataProcessor

# def random_forest_regressor(dp: DataProcessor, target_colums, features, plant_name,test_size=0.2, random_state=42, top_num = 5):
#     # 1. 数据拆分
#     features = [f for f in features if f not in ['cycle_id', 'cycle_time']]
#     # 选出前10000个数据
#     selected_data = dp.df[0:10000]
#     X_train, X_test, y_train, y_test = train_test_split(
#        selected_data[features], selected_data[target_colums], test_size=test_size, random_state=random_state
#     )

#     # 2. 训练随机森林
#     print("训练 RandomForest...")
#     rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
#     rf.fit(X_train, y_train.values.ravel())

#     importances = rf.feature_importances_
#     feature_importance_series = pd.Series(importances, index=X_train.columns)
#     top_features = feature_importance_series.sort_values(ascending=False).head(4*top_num).index.tolist()

#     X_sub = X_train[top_features]
#     y_sub = y_train
#     print("训练 simple RandomForest...")
#     rf_small = RandomForestRegressor(n_estimators=100, random_state=random_state)
#     rf_small.fit(X_sub, y_sub.values.ravel())
#     explainer = shap.TreeExplainer(rf_small)
#     X_sub_sample = X_sub.sample(n=200, random_state=42)
#     shap_values = explainer.shap_values(X_sub_sample)

#     # 3. 计算 SHAP 重要性
#     shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X_sub.columns)
#     top_k_features = shap_importance.sort_values(ascending=False).head(top_num*2).index.tolist()

#     shap.summary_plot(shap_values, X_sub_sample, feature_names=X_sub.columns, max_display=top_num*2)

#     import_features = pd.DataFrame(top_k_features)
#     import_data = dp.df[top_k_features]
#     import_data.to_csv(f"../data/temp_data/top_k_features_data_{plant_name}_{target_colums}.csv")
#     import_features.to_csv(f"../data/temp_data/top_k_features_{plant_name}_{target_colums}.csv")
#     return top_k_features
