import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

def train_tree_model(df_processed, task_type, algo_choice, max_depth, max_leaf_nodes, min_samples_split=2, svm_kernel='rbf', svm_c=1.0, svm_degree=3, test_size=0.2):
    
    # ------------------ 【真实标签备份中心】 ------------------
    # 物理剥离目标列 Y，同时克隆一份最纯正的原始实际值（保留文本如 'Dropout'），用于最后并排对答案
    if '__TARGET_LABEL__' in df_processed.columns:
        y_raw_backup = df_processed['__TARGET_LABEL__'].copy()
        y = df_processed['__TARGET_LABEL__'].copy()
        X = df_processed.drop(columns=['__TARGET_LABEL__'])
    else:
        X = df_processed.iloc[:, :-1]
        y = df_processed.iloc[:, -1]
        y_raw_backup = y.copy()
    
    # 分类任务下的目标 Y 文本转数字防御锁
    le = None
    if "分类任务" in task_type:
        if y.dtype == 'object' or pd.api.types.is_string_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        else:
            y = y.astype(int).values
            
    # 特征集 X 铁血净化：只留数值和独热编码后的列
    X_numeric = X.select_dtypes(include=['number']).copy()
    if X_numeric.empty:
        X_numeric = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 标准数据切分
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=test_size, random_state=42)
    
    if "分类任务" in task_type:
        # 初始化分类算法
        if "支持向量机" in algo_choice:
            model = SVC(kernel=svm_kernel, C=svm_c, degree=svm_degree, probability=True, random_state=42)
        elif "朴素贝叶斯" in algo_choice:
            model = GaussianNB()
        else:
            model = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=42)
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # 计算评估矩阵
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        
        # 还原锁：反向解码数字 ID
        if le is not None:
            new_report = {}
            for k, v in report.items():
                if k.isdigit() or (k.startswith('-') and k[1:].isdigit()):
                    decoded_key = str(le.inverse_transform([int(k)])[0])
                    new_report[decoded_key] = v
                else:
                    new_report[k] = v
            report = new_report
            
            raw_all_preds = model.predict(X_numeric)
            final_preds = le.inverse_transform(raw_all_preds)
        else:
            final_preds = model.predict(X_numeric)
            
        # 💡 核心修复：在这里构建并排对答案的数据矩阵
        df_out = X.copy()
        # 强制将最原始的实际值（如文本 'Dropout'）写回倒数第二列
        df_out['真实实际值 (Y)'] = y_raw_backup.values
        # 将模型算出来的预测值写在最右侧最后一列
        df_out['模型预测值 (★)'] = final_preds
        
        return {"model_mode": "classification", "metrics": acc, "dataframe": df_out, "report": report}
        
    else:
        # ------------------ 【回归任务线并排修正】 ------------------
        if "支持向量机" in algo_choice:
            model = SVR(kernel=svm_kernel, C=svm_c, degree=svm_degree)
        elif "线性回归" in algo_choice:
            model = LinearRegression()
        else:
            model = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=42)
            
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        mse = mean_squared_error(y_test, model.predict(X_test))
        
        df_out = X.copy()
        # 回归任务同样将真实连续值与预测连续值进行并排并网
        df_out['真实实际值 (Y)'] = y_raw_backup.values
        df_out['模型预测值 (★)'] = model.predict(X_numeric)
        
        return {"model_mode": "regression", "r2": r2, "mse": mse, "dataframe": df_out}