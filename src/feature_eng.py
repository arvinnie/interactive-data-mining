import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def construct_safe_feature(df, new_col_name, col_a, operator, col_b):
    try:
        df_res = df.copy()
        val_a = pd.to_numeric(df_res[col_a], errors='coerce').fillna(0).values
        
        try:
            val_b_num = float(col_b)
            val_b = np.full_like(val_a, val_b_num)
        except ValueError:
            val_b = pd.to_numeric(df_res[col_b], errors='coerce').fillna(0).values
            
        if operator == "+": res_values = val_a + val_b
        elif operator == "-": res_values = val_a - val_b
        elif operator == "*": res_values = val_a * val_b
        elif operator == "/": res_values = np.where(val_b == 0, 0.0, val_a / np.where(val_b == 0, 1.0, val_b))
        else: return df, False, "不支持的运算符"
            
        df_res[new_col_name] = res_values
        return df_res, True, f"新特征列【{new_col_name}】安全创建并成功注入数据矩阵！"
    except Exception as e:
        return df, False, f"计算失败。错误详情: {e}"

def run_feature_engineering(df, target_col, feature_cols, onehot_cols, scale_strategy):
    # 💡 独热编码原理解析：确保只对特征集 X 进行变换，绝不污染目标响应列 Y
    cols_to_keep = list(feature_cols)
    if target_col and target_col in df.columns:
        X = df[cols_to_keep].copy()
        y = df[target_col]
    else:
        X = df[cols_to_keep].copy()
        y = None
    
    # 仅针对用户挑选的、且在特征集 X 中的文本列做独热拆解
    if onehot_cols:
        actual_onehot = [c for c in onehot_cols if c in X.columns]
        if actual_onehot:
            X = pd.get_dummies(X, columns=actual_onehot, drop_first=False)
        
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
            
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    real_num_cols = [c for c in num_cols if X[c].nunique() > 2]
    
    if real_num_cols and scale_strategy != "保持原始数据":
        if scale_strategy == "标准化":
            scaler = StandardScaler()
            X[real_num_cols] = scaler.fit_transform(X[real_num_cols].astype(float))
        elif scale_strategy == "归一化":
            scaler = MinMaxScaler()
            X[real_num_cols] = scaler.fit_transform(X[real_num_cols].astype(float))
            
    df_processed = X.copy()
    if y is not None:
        df_processed['__TARGET_LABEL__'] = y.reset_index(drop=True)
    return df_processed
