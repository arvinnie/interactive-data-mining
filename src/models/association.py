import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def clean_frozenset_string(fs_obj):
    return ", ".join([str(item) for item in list(fs_obj)])

def run_association_mining(df_processed, min_support=0.05, min_threshold=0.3):
    X = df_processed.drop(columns=['__TARGET_LABEL__']) if '__TARGET_LABEL__' in df_processed.columns else df_processed
    X_binary = X.copy()
    for col in X_binary.columns:
        if X_binary[col].nunique() > 2:
            med = X_binary[col].median()
            X_binary[col] = (X_binary[col] > med).astype(int)
        else:
            X_binary[col] = (X_binary[col] > 0).astype(int)
            
    frequent_itemsets = apriori(X_binary, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return {"rules": pd.DataFrame(), "msg": "未挖掘出满足最小支持度门槛的条件组合。"}
        
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
    if not rules.empty:
        rules['前置条件'] = rules['antecedents'].apply(clean_frozenset_string)
        rules['后置关联'] = rules['consequents'].apply(clean_frozenset_string)
        
        rules_output = pd.DataFrame()
        rules_output['关联前置项（列组合）'] = rules['前置条件']
        rules_output['关联后置项（预测结果）'] = rules['后置关联']
        rules_output['支持度'] = rules['support'].round(4)
        rules_output['置信度'] = rules['confidence'].round(4)
        rules_output['提升度'] = rules['lift'].round(4)
        
        rules_output = rules_output[rules_output['提升度'] > 1]
        rules_output = rules_output.sort_values(by='置信度', ascending=False)
        return {"rules": rules_output, "msg": f"成功挖掘出 {len(rules_output)} 条核心强关联决策规则！"}
    return {"rules": pd.DataFrame(), "msg": "未产生强关联规则。"}
