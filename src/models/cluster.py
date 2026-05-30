import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

def run_clustering_engine(df_processed, algo_type, n_clusters=3, eps=0.5, min_samples=5):
    X = df_processed.drop(columns=['__TARGET_LABEL__']) if '__TARGET_LABEL__' in df_processed.columns else df_processed
    X_num = X.select_dtypes(include=['number']).fillna(0)
    
    if "K-Means" in algo_type:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = model.fit_predict(X_num)
        inertia = model.inertia_
    else:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_num)
        inertia = 0.0
        
    pca_df = pd.DataFrame(columns=['主成分维度_1', '主成分维度_2'])
    if X_num.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_num)
        pca_df = pd.DataFrame(X_pca, columns=['主成分维度_1', '主成分维度_2'])
    else:
        pca_df['主成分维度_1'] = X_num.iloc[:, 0] if X_num.shape[1] > 0 else 0
        pca_df['主成分维度_2'] = 0.0
        
    pca_df['聚类类别'] = [f"噪声异常点" if i == -1 else f"类别族_{i}" for i in labels]
    df_out = df_processed.copy()
    df_out['聚类类别标签'] = labels
    return {"dataframe": df_out, "inertia": inertia, "pca_df": pca_df, "labels": labels}
