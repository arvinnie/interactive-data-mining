import pandas as pd
from sklearn.ensemble import IsolationForest

def run_anomaly_detection(df_processed, contamination=0.05):
    X = df_processed.drop(columns=['__TARGET_LABEL__']) if '__TARGET_LABEL__' in df_processed.columns else df_processed
    X_num = X.select_dtypes(include=['number']).fillna(0)
    
    if contamination == '自动识别': contamination = 'auto'
    iforest = IsolationForest(contamination=contamination, random_state=42)
    preds = iforest.fit_predict(X_num)
    scores = iforest.decision_function(X_num)
    
    df_out = df_processed.copy()
    df_out['异常判定 (1正常/-1异常)'] = preds
    df_out['异常得分(越小越可疑)'] = scores
    return {"dataframe": df_out, "anomaly_count": (preds == -1).sum(), "scores": scores}
