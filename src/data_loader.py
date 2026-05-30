import pandas as pd
def load_uploaded_file(uploaded_file):
    try:
        return pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(uploaded_file, encoding='gbk')
    except Exception:
        return None
