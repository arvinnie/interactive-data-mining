import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import load_uploaded_file
from src.data_cleaner import clean_dataframe
from src.feature_eng import run_feature_engineering, construct_safe_feature
from src.models.classical import train_tree_model
from src.models.cluster import run_clustering_engine
from src.models.anomaly import run_anomaly_detection
from src.models.association import run_association_mining

st.set_page_config(page_title="数据挖掘小工具", layout="wide")
st.title("📊 用户决策的数据挖掘")

if "step" not in st.session_state: st.session_state.step = 1
for key in ["df_raw", "df_clean", "df_processed", "mining_results", "raw_target_col"]:
    if key not in st.session_state: st.session_state[key] = None

if st.sidebar.button("🔄 重置系统状态"):
    for k in ["step", "df_raw", "df_clean", "df_processed", "mining_results", "raw_target_col"]: st.session_state[k] = None
    st.session_state.step = 1
    st.rerun()

# ==================== Step 1. 数据导入 ====================
st.header("📁 步骤 1: 原始数据导入与预览")
if st.session_state.df_raw is None:
    uploaded_file = st.file_uploader("请上传您的 CSV 文件：", type=["csv"])
    if uploaded_file:
        st.session_state.df_raw = load_uploaded_file(uploaded_file)
        st.session_state.step = max(st.session_state.step, 2)
        st.rerun()

if st.session_state.df_raw is not None:
    st.success(f"📈 数据读取成功！当前数据集包含： **{st.session_state.df_raw.shape[0]}** 行记录， **{st.session_state.df_raw.shape[1]}** 个数据列。")
    st.dataframe(st.session_state.df_raw.head(3))

# ==================== Step 2. 清洗前概览与智能清洗 ====================
if st.session_state.step >= 2 and st.session_state.df_raw is not None:
    st.write("---")
    st.header("🧼 步骤 2: 数据质量概览与清洗")
    
    raw_df = st.session_state.df_raw
    st.markdown("##### 🔍 模块 A：清洗前数据质量概览")
    
    duplicate_count = raw_df.duplicated().sum()
    nan_info = raw_df.isna().sum()
    
    # 💡 核心修正1：智能检查并过滤缺失值为 0 的无病灶列，全0不展示
    health_details = []
    for col in raw_df.columns:
        missing_cnt = nan_info[col]
        if missing_cnt > 0:  # 👈 只筛选真正有空值的列进行报警
            missing_pct = (missing_cnt / len(raw_df)) * 100
            health_details.append({
                "字段名称（列标题）": col,
                "数据类型": str(raw_df[col].dtype),
                "非空有效值数": len(raw_df) - missing_cnt,
                "缺失值(NaN)个数": missing_cnt,
                "当前缺失率占配比 (%)": f"{missing_pct:.2f}%"
            })
            
    hc1, hc2 = st.columns([1, 3])
    with hc1:
        st.metric("全表完全重复行数", f"{duplicate_count} 行")
    with hc2:
        if health_details:
            st.write("📋 **当前数据集中[存在缺失]的特征字段表：**")
            st.dataframe(pd.DataFrame(health_details), use_container_width=True)
        else:
            st.success("🎉 检查完毕：当前数据集完美无缺，所有数据列的缺失值均为 0，无需担忧空值漏洞！")
        
    st.markdown("##### ⚙️ 模块 B：清洗策略执行配置")
    if st.session_state.df_clean is None:
        col_c1, col_c2 = st.columns(2)
        with col_c1: 
            global_imp = st.selectbox("请指定全局空值(NaN)填充置换策略：", [
                "使用均值填充(数值列)", "使用中位数填充(数值列)", "使用众数填充(所有列)", "固定值0填充", "剔除包含空值的整行"
            ])
        with col_c2: drop_dup = st.checkbox("自动删除并合并全表完全重复记录", value=True)
        if st.button("🚀 确认为当前数据执行深度清洗"):
            st.session_state.df_clean = clean_dataframe(st.session_state.df_raw, global_imp, drop_dup)
            st.session_state.step = max(st.session_state.step, 3)
            st.rerun()

    if st.session_state.df_clean is not None:
        st.success(f"✅ 数据清洗与健康度优化完成！以下为清洗后的表格预览：")
        st.dataframe(st.session_state.df_clean.head(3))
        
        # 💡 核心修正2：清洗完成后，立刻在下方并网一键下载按钮
        st.download_button(
            label="📥 下载已完成清洗后的干净数据集 (CSV)",
            data=st.session_state.df_clean.to_csv(index=False).encode('utf-8'),
            file_name="cleaned_commercial_data.csv",
            mime="text/csv"
        )

# ==================== Step 3. 特征构造与预处理器配置 ====================
if st.session_state.step >= 3 and st.session_state.df_clean is not None:
    st.write("---")
    st.header("🧬 步骤 3: 特征构造与预处理")
    
    all_clean_cols = st.session_state.df_clean.columns.tolist()
    st.markdown("##### 📐 模块 A：特征工程")
    with st.expander("➕ 点击展开特征构造器：自由通过四则运算衍生新的特征"):
        ec1, ec2, ec3, ec4 = st.columns([3, 1, 3, 3])
        with ec1: col_a = st.selectbox("选择左变量列 (Column A)", all_clean_cols, key="fe_col_a")
        with ec2: operator = st.selectbox("算子", ["+", "-", "*", "/"])
        with ec3: 
            b_mode = st.radio("右变量类型 (Operand B)", ["绑定已有数据列", "输入固定常数数字"], horizontal=True)
            if b_mode == "绑定已有数据列":
                col_b = st.selectbox("选择右变量列 (Column B)", all_clean_cols, key="fe_col_b")
            else:
                col_b = st.text_input("请输入具体的常数数值：", value="1.0")
        with ec4:
            new_col_name = st.text_input("为新生成的复合特征列命名：", placeholder="例如: 资产负债率_衍生")
            
        if st.button("🛠️ 确认生成新的特征"):
            if new_col_name:
                updated_df, success, msg = construct_safe_feature(st.session_state.df_clean, new_col_name.strip(), col_a, operator, col_b.strip())
                if success:
                    st.session_state.df_clean = updated_df
                    st.toast(msg, icon="✅")
                    st.rerun()
                else: st.error(msg)
            else: st.warning("⚠️ 请先为新生成的特征列命名。")

    st.markdown("##### 🛠️ 模块 B：数据预处理")
    if st.session_state.df_processed is None:
        target_options = ["【做无监督任务 / 聚类 / 关联规则分析不选此项】"] + all_clean_cols
        f1, f2 = st.columns(2)
        with f1: 
            target_sel = st.selectbox("请指定目标响应列 (Y)：", target_options)
            if target_sel == "【做无监督任务 / 聚类 / 关联规则分析不选此项】":
                st.session_state.raw_target_col = None
            else:
                st.session_state.raw_target_col = target_sel
        with f2: 
            available_features = [c for c in all_clean_cols if c != st.session_state.raw_target_col]
            feature_cols = st.multiselect("请挑选参与后续计算的多维特征 (X)：", available_features, default=available_features[:6])
            
        # 💡 独热编码控制原理解析：前端多选组件中，自动将目标列 Y 摘除，确保不可能被误选污染
        text_features = [c for c in available_features if st.session_state.df_clean[c].dtype == 'object' and c in feature_cols]
        if text_features:
            onehot_cols = st.multiselect("🧼 请指定特征 (X) 中需要转化为 0/1 状态码的【离散文本变量】：", text_features, default=text_features)
        else:
            onehot_cols = []
            
        scale_choice = st.selectbox("连续数值列特征变换策略：", ["标准化", "归一化", "保持原始数据"])
        
        if st.button("💾 确认完成预处理") and feature_cols:
            st.session_state.df_processed = run_feature_engineering(st.session_state.df_clean, st.session_state.raw_target_col, feature_cols, onehot_cols, scale_choice)
            st.session_state.step = max(st.session_state.step, 4)
            st.rerun()

    if st.session_state.df_processed is not None:
        display_df = st.session_state.df_processed.copy()
        if '__TARGET_LABEL__' in display_df.columns and st.session_state.raw_target_col:
            display_df = display_df.rename(columns={'__TARGET_LABEL__': st.session_state.raw_target_col})
        st.success("✅ 数据预处理和特征工程就绪。")
        st.dataframe(display_df.head(3))
        st.download_button(
            label="📥 下载经预处理后的特征数据 (CSV)",
            data=display_df.to_csv(index=False).encode('utf-8'),
            file_name="preprocessed_feature_matrix.csv",
            mime="text/csv",
            key="download_preprocessed_df_btn" # 设置唯一 key 防止组件状态冲突
        )
        

# ==================== Step 4. 四大业务任务分流 ====================
if st.session_state.step >= 4 and st.session_state.df_processed is not None:
    st.write("---")
    st.header("🎯 步骤 4: 数据挖掘和模型训练")
    
    if st.session_state.mining_results is None:
        task_mode = st.radio("请选择您的数据挖掘任务：", [
            "🏆 监督预测任务",
            "🧩 无监督聚类任务",
            "🛍️ 商业关联分析",
            "🔍 异常检测任务"
        ])
        
        if "监督预测" in task_mode:
            if '__TARGET_LABEL__' not in st.session_state.df_processed.columns:
                st.error("❌ 启动失败：您当前未配置目标列(Y)，无法驱动监督模型。")
            else:
                y_val = st.session_state.df_processed['__TARGET_LABEL__']
                is_numeric_target = pd.api.types.is_numeric_dtype(y_val) and y_val.nunique() > 10
                
                if is_numeric_target: 
                    algo = st.selectbox("回归模型：", ["支持向量机回归模型", "多元线性回归模型", "决策树回归"])
                else: 
                    algo = st.selectbox("分类模型：", ["支持向量机分类模型", "高斯朴素贝叶斯模型", "决策树分类"])
                
                max_d, max_leaves, min_split = 6, 30, 2
                svm_kernel, svm_c_val, svm_degree_val = 'rbf', 1.0, 3
                
                if "决策树" in algo:
                    st.markdown("##### 🌲 树模型调优面板")
                    col_t1, col_t2 = st.columns(2)
                    with col_t1: max_d = st.slider("最大树模型深度", 2, 30, 6)
                    with col_t2: max_leaves = st.slider("最多叶子数", 5, 100, 30)
                elif "支持向量机" in algo:
                    st.markdown("##### 🛡️ 支持向量机参数调优面板")
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        kernel_ui_choice = st.selectbox("请选择核函数映射机制 (Kernel)：", ["rbf (高斯径向基核)", "linear (线性核)", "poly (多项式核)"])
                        kernel_mapping = {"rbf (高斯径向基核)": "rbf", "linear (线性核)": "linear", "poly (多项式核)": "poly"}
                        svm_kernel = kernel_mapping[kernel_ui_choice]
                    with sc2:
                        svm_c_val = st.slider("惩罚系数 (C)：", 0.01, 20.0, 1.0, step=0.1)
                    if svm_kernel == "poly":
                        svm_degree_val = st.slider("多项式核最高次数 (degree)：", 1, 6, 3)
                
                task_tag = "回归任务" if is_numeric_target else "分类任务"
                if st.button("🚀 驱动预测模型开始训练"):
                    st.session_state.mining_results = {"type": "Supervised", "data": train_tree_model(
                        st.session_state.df_processed, task_tag, algo, max_d, max_leaves, min_split, 
                        svm_kernel=svm_kernel, svm_c=svm_c_val, svm_degree=svm_degree_val
                    )}
                    st.session_state.step = max(st.session_state.step, 5)
                    st.rerun()
                    
        elif "无监督聚类" in task_mode:
            st.markdown("##### ⚙️ 聚类引擎配置")
            algo = st.selectbox("聚类机制：", ["K-Means 聚类 (数量细分)", "DBSCAN 聚类 (密度聚集)"])
            if "K-Means" in algo:
                k_val = st.slider("目标聚类簇数(K值)：", 2, 12, 3)
                if st.button("🚀 运行 K-Means 聚类驱动"):
                    st.session_state.mining_results = {"type": "Cluster", "algo": "K-Means", "data": run_clustering_engine(st.session_state.df_processed, "K-Means", n_clusters=k_val)}
                    st.session_state.step = max(st.session_state.step, 5)
                    st.rerun()
            else:
                c1, c2 = st.columns(2)
                with c1: eps_val = st.slider("局部高维搜索半径 (Eps)", 0.05, 5.0, 0.5, step=0.05)
                with c2: min_s = st.slider("核心样本密度邻近数", 2, 50, 5)
                if st.button("🚀 运行 DBSCAN 密度聚类驱动"):
                    st.session_state.mining_results = {"type": "Cluster", "algo": "DBSCAN", "data": run_clustering_engine(st.session_state.df_processed, "DBSCAN", eps=eps_val, min_samples=min_s)}
                    st.session_state.step = max(st.session_state.step, 5)
                    st.rerun()
                    
        elif "商业关联分析" in task_mode:
            st.markdown("##### ⚙️ 关联规则配置")
            ac1, ac2 = st.columns(2)
            with ac1: min_sup = st.slider("最小支持度门槛", 0.01, 0.5, 0.05, step=0.01)
            with ac2: min_conf = st.slider("最低置信度门槛", 0.1, 1.0, 0.3, step=0.05)
            if st.button("🚀 驱动 Apriori 强关联分析网络"):
                st.session_state.mining_results = {"type": "Association", "data": run_association_mining(st.session_state.df_processed, min_sup, min_conf)}
                st.session_state.step = max(st.session_state.step, 5)
                st.rerun()
                    
        elif "异常检测" in task_mode:
            st.markdown("##### ⚙️ 离群样本异常排查配置")
            contam = st.select_slider("预估异常样本污染率：", options=[0.01, 0.03, 0.05, 0.1], value=0.05)
            if st.button("🚀 驱动孤立森林排查风险点"):
                st.session_state.mining_results = {"type": "Anomaly", "data": run_anomaly_detection(st.session_state.df_processed, contam)}
                st.session_state.step = max(st.session_state.step, 5)
                st.rerun()

# ==================== Step 5. 最终交付物面板 ====================
if st.session_state.step == 5 and st.session_state.mining_results is not None:
    st.write("---")
    st.header("📊 步骤 5: 模型分析与结果展示")
    res = st.session_state.mining_results
    
    if res["type"] == "Supervised":
        md = res["data"]
        if md["model_mode"] == "classification":
            st.success(f"🎉 分类预测完成！")
            rep_dict = md['report']
            valid_rows = []
            for key, val in rep_dict.items():
                if key in ['accuracy', 'macro avg', 'weighted avg']: continue
                if val['support'] > 0:
                    valid_rows.append({
                        "业务分类类别": key,
                        "精准率 (Precision)": f"{val['precision']:.4f}",
                        "召回率 (Recall)": f"{val['recall']:.4f}",
                        "F1 调和得分": f"{val['f1-score']:.4f}",
                        "测试样本底数": int(val['support'])
                    })
            if valid_rows: st.dataframe(pd.DataFrame(valid_rows), use_container_width=True)
            st.metric("模型全表准确率 (Accuracy)", f"{rep_dict['accuracy']:.4f}")
        else:
            st.success("🎉 回归模型拟合完成！")
            rc1, rc2 = st.columns(2)
            rc1.metric("模型判定优度 (R² Score)", f"{md['r2']:.4f}")
            rc2.metric("均方预测误差 (MSE)", f"{md['mse']:.4f}")
            
        final_out_df = md['dataframe'].copy()
        if '__TARGET_LABEL__' in final_out_df.columns and st.session_state.raw_target_col:
            final_out_df = final_out_df.rename(columns={'__TARGET_LABEL__': st.session_state.raw_target_col})
        st.write("📋 **预测数据预览：**")
        st.dataframe(final_out_df.head(5))
        st.download_button("📥 导出全量预测结果报表 (CSV)", final_out_df.to_csv(index=False).encode('utf-8'), "supervised_deliverables.csv", "text/csv")
        
    elif res["type"] == "Cluster":
        st.success(f"🎉 聚类模型执行完毕！")
        st.markdown("**📊 PCA高维特征解耦降维图：**")
        st.scatter_chart(res['data']['pca_df'], x="主成分维度_1", y="主成分维度_2", color="聚类类别")
        final_out_df = res['data']['dataframe'].copy()
        st.dataframe(final_out_df.head(5))
        st.download_button("📥 导出带分类簇标签的数据 (CSV)", final_out_df.to_csv(index=False).encode('utf-8'), "cluster_deliverables.csv", "text/csv")
        
    elif res["type"] == "Association":
        rules_df = res["data"]["rules"]
        st.info(f"📢 关联分析报告：{res['data']['msg']}")
        if not rules_df.empty:
            st.dataframe(rules_df, use_container_width=True)
            st.download_button("📥 导出过滤0值后的纯正向关联规则表 (CSV)", rules_df.to_csv(index=False).encode('utf-8'), "association_rules.csv", "text/csv")
        
    elif res["type"] == "Anomaly":
        st.success(f"🎉 离群样本反常诊断结束。")
        st.metric("模型甄别出的高危异常可疑样本数", f"{res['data']['anomaly_count']} 个")
        final_out_df = res['data']['dataframe'].copy()
        st.dataframe(final_out_df.head(5))
        st.download_button("📥 导出带异常标记的核查数据表 (CSV)", final_out_df.to_csv(index=False).encode('utf-8'), "anomaly_deliverables.csv", "text/csv")
