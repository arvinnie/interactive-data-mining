import pandas as pd
def clean_dataframe(df, global_strategy, drop_duplicates):
    df_cleaned = df.copy()
    if drop_duplicates:
        df_cleaned = df_cleaned.drop_duplicates()
        
    if global_strategy == "剔除包含空值的整行":
        df_cleaned = df_cleaned.dropna()
    elif global_strategy == "使用均值填充(数值列)":
        num_cols = df_cleaned.select_dtypes(include=['number']).columns
        df_cleaned[num_cols] = df_cleaned[num_cols].fillna(df_cleaned[num_cols].mean())
    elif global_strategy == "使用中位数填充(数值列)":
        num_cols = df_cleaned.select_dtypes(include=['number']).columns
        df_cleaned[num_cols] = df_cleaned[num_cols].fillna(df_cleaned[num_cols].median())
    elif global_strategy == "使用众数填充(所有列)":
        for col in df_cleaned.columns:
            mode_series = df_cleaned[col].mode()
            if not mode_series.empty:
                df_cleaned[col] = df_cleaned[col].fillna(mode_series[0])
    elif global_strategy == "固定值0填充":
        df_cleaned = df_cleaned.fillna(0)
        
    return df_cleaned
