import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 引入数据挖掘与机器学习库
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

# 页面配置
st.set_page_config(page_title="交互式数据挖掘平台", layout="wide")

# --- 初始化 Session State （确保全局数据安全与持久化） ---
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'history_log' not in st.session_state:
    st.session_state['history_log'] = []
if 'report_data' not in st.session_state:
    st.session_state['report_data'] = {}

# --- 侧边栏导航 ---
st.sidebar.title("🛠️ 数据挖掘工作流")
page = st.sidebar.radio(
    "请选择当前操作步骤：",
    ["1. 数据导入与画像", "2. 互动预处理", "3. 特征工程", "4. 模型训练与挖掘", "5. 导出报告"]
)

st.sidebar.markdown("---")
if st.session_state['history_log']:
    st.sidebar.subheader("📄 已执行的操作日志")
    for log in st.session_state['history_log']:
        st.sidebar.caption(f"• {log}")

# ==========================================
# 步骤 1：数据导入与画像
# ==========================================
if page == "1. 数据导入与画像":
    st.title("📂 数据导入与智能画像")
    uploaded_file = st.file_uploader("选择数据文件", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            if st.session_state['raw_data'] is None or not df.equals(st.session_state['raw_data']):
                st.session_state['raw_data'] = df
                st.session_state['processed_data'] = df.copy()
                st.session_state['history_log'] = ["成功导入原始数据集"]
                st.session_state['report_data'] = {} # 重置报告
            st.success("✅ 文件上传成功！")
        except Exception as e:
            st.error(f"读取文件时出错: {e}")

    if st.session_state['raw_data'] is not None:
        df = st.session_state['raw_data']
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="数据行数 (Rows)", value=df.shape[0])
            st.metric(label="数据列数 (Columns)", value=df.shape[1])
        with col2:
            missing_cells = df.isnull().sum().sum()
            st.metric(label="总缺失值率", value=f"{(missing_cells / df.size) * 100:.2f}%")

        st.subheader("👀 数据预览 (前 5 行)")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("🔍 缺失值智能诊断")
        missing_info = df.isnull().sum().reset_index()
        missing_info.columns = ['列名', '缺失值个数']
        missing_info['缺失率 (%)'] = (missing_info['缺失值个数'] / len(df)) * 100
        missing_cols = missing_info[missing_info['缺失值个数'] > 0]

        if not missing_cols.empty:
            st.warning("⚠️ 检测到数据中存在缺失值！")
            fig = px.bar(missing_cols, x='列名', y='缺失率 (%)', text='缺失值个数', title='各列缺失率')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 太棒了！您的数据非常完整，没有检测到任何缺失值。")

# ==========================================
# 步骤 2：互动预处理
# ==========================================
elif page == "2. 互动预处理":
    st.title("🛠️ 互动数据预处理")
    if st.session_state['processed_data'] is None:
        st.warning("请先回到第 1 步上传数据文件！")
    else:
        df_work = st.session_state['processed_data'].copy()
        missing_counts = df_work.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0].index.tolist()
        
        if not cols_with_missing:
            st.success("✨ 当前数据中已无缺失值，您可以直接进入下一步“特征工程”。")
            st.dataframe(df_work.head(), use_container_width=True)
        else:
            st.write("请针对有缺失值的列选择处理方法。系统已为您自动勾选了**推荐方法**。")
            with st.form("preprocessing_form"):
                user_choices = {}
                for col in cols_with_missing:
                    dtype = df_work[col].dtype
                    missing_num = missing_counts[col]
                    missing_rate = (missing_num / len(df_work)) * 100
                    
                    st.markdown(f"### 📍 列名: `{col}`")
                    if missing_rate > 50:
                        options = ["直接删除该列", "均值填充", "中位数填充", "众数填充"]
                        st.info(f"💡 App 推荐：缺失率过高，推荐 **直接删除**。")
                        default_idx = 0
                    elif dtype in ['float64', 'int64']:
                        options = ["中位数填充", "均值填充", "直接删除该列"]
                        st.info(f"💡 App 推荐：数值型特征，推荐 **中位数填充**。")
                        default_idx = 0
                    else:
                        options = ["众数填充", "直接删除该列"]
                        st.info(f"💡 App 推荐：文本类别特征，推荐 **众数填充**。")
                        default_idx = 0
                    
                    choice = st.selectbox(f"请选择 `{col}` 的处理方式：", options=options, index=default_idx, key=f"select_{col}")
                    user_choices[col] = {"method": choice}
                
                submit_button = st.form_submit_button("🚀 应用选择的处理方法")
            
            if submit_button:
                for col, action in user_choices.items():
                    method = action["method"]
                    if method == "直接删除该列":
                        df_work.drop(columns=[col], inplace=True)
                    elif method == "均值填充":
                        df_work[col] = df_work[col].fillna(df_work[col].mean())
                    elif method == "中位数填充":
                        df_work[col] = df_work[col].fillna(df_work[col].median())
                    elif method == "众数填充":
                        df_work[col] = df_work[col].fillna(df_work[col].mode()[0])
                    st.session_state['history_log'].append(f"对列 [{col}] 执行了 {method}")
                st.session_state['processed_data'] = df_work
                st.success("🎉 数据预处理成功！")
                st.rerun()

# ==========================================
# 步骤 3：互动特征工程
# ==========================================
elif page == "3. 特征工程":
    st.title("🧪 互动特征工程")
    if st.session_state['processed_data'] is None:
        st.warning("请先回到第 1 步上传数据文件！")
    else:
        df_fe = st.session_state['processed_data'].copy()
        
        st.header("1. 数值特征缩放 (Scaling)")
        numeric_cols = df_fe.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_cols:
            selected_scale_cols = st.multiselect("请选择需要进行缩放的数值型特征：", numeric_cols)
            if selected_scale_cols:
                scale_method = st.radio("选择缩放方法：", ["StandardScaler (Z-score)", "MinMaxScaler"])
                if st.button("⚡ 执行特征缩放"):
                    for col in selected_scale_cols:
                        if "StandardScaler" in scale_method:
                            df_fe[col] = (df_fe[col] - df_fe[col].mean()) / df_fe[col].std()
                        else:
                            df_fe[col] = (df_fe[col] - df_fe[col].min()) / (df_fe[col].max() - df_fe[col].min())
                    st.session_state['history_log'].append(f"对特征 {selected_scale_cols} 进行了 {scale_method.split(' ')[0]}")
                    st.session_state['processed_data'] = df_fe
                    st.success("🎉 缩放应用成功！")
        
        st.markdown("---")
        st.header("2. 类别型特征编码 (Encoding)")
        object_cols = df_fe.select_dtypes(include=['object', 'category']).columns.tolist()
        if object_cols:
            selected_encode_cols = st.multiselect("请选择需要进行独热编码的特征：", object_cols)
            if selected_encode_cols and st.button("⚡ 执行独热编码"):
                df_fe = pd.get_dummies(df_fe, columns=selected_encode_cols, dtype=int)
                st.session_state['processed_data'] = df_fe
                st.session_state['history_log'].append(f"对列 {selected_encode_cols} 进行了独热编码")
                st.success("🎉 编码成功！")
        else:
            st.success("✨ 所有特征都已数字化，可直接用于建模挖掘。")

# ==========================================
# 步骤 4：多算法模型训练与数据挖掘
# ==========================================
elif page == "4. 模型训练与挖掘":
    st.title("🤖 互动模型训练与数据挖掘")
    if st.session_state['processed_data'] is None:
        st.warning("请先回到第 1 步上传数据文件！")
    else:
        df_model = st.session_state['processed_data'].copy()
        unencoded_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
        if unencoded_cols:
            st.error(f"⚠️ 警告：检测到特征列 {unencoded_cols} 未被数字化，请先去第 3 步做特征编码。")
            st.stop()

        task_type = st.selectbox(
            "🔮 请选择您要进行的数据挖掘任务：",
            ["聚类分析 (Clustering - 无监督学习)", "分类任务 (Classification - 有监督学习)", "回归任务 (Regression - 有监督学习)"]
        )
        
        if "聚类分析" in task_type:
            all_cols = df_model.columns.tolist()
            features = st.multiselect("选择参与聚类的特征：", all_cols, default=all_cols)
            if features:
                X_cluster = df_model[features]
                cluster_algo = st.radio("请选择聚类算法：", ["K-Means", "DBSCAN"])
                
                if cluster_algo == "K-Means":
                    k_val = st.slider("请选择聚类簇数 (K值):", 2, 10, 3)
                    if st.button("🚀 运行 K-Means 聚类"):
                        kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
                        labels = kmeans.fit_predict(X_cluster)
                        
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_cluster)
                        df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
                        df_pca['Cluster'] = labels.astype(str)
                        fig = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', title="🏆 K-Means 聚类效果图")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.session_state['report_data'] = {"任务类型": "无监督聚类", "算法名称": "K-Means", "核心超参数": f"K = {k_val}", "评估反馈": f"成功归纳出 {k_val} 个核心数据群体。"}
                        st.session_state['history_log'].append(f"运行了 K-Means 聚类，K={k_val}")
                
                elif cluster_algo == "DBSCAN":
                    eps = st.slider("邻域半径 (eps):", 0.1, 5.0, 0.5, 0.1)
                    min_samples = st.slider("最小样本数 (MinSamples):", 2, 20, 5)
                    if st.button("🚀 运行 DBSCAN 聚类"):
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        labels = dbscan.fit_predict(X_cluster)
                        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                        n_noise_ = list(labels).count(-1)
                        
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_cluster)
                        df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
                        df_pca['Cluster'] = labels.astype(str)
                        fig = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', title="🏆 DBSCAN 聚类效果图")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.session_state['report_data'] = {"任务类型": "无监督聚类", "算法名称": "DBSCAN", "核心超参数": f"eps={eps}, min_samples={min_samples}", "评估反馈": f"自动识别出 {n_clusters_} 个密度分布群，并清洗剔除了 {n_noise_} 个边缘噪点。"}
                        st.session_state['history_log'].append(f"运行了 DBSCAN 聚类")

        elif "分类任务" in task_type:
            all_cols = df_model.columns.tolist()
            target = st.selectbox("请选择预测目标标签 (Target):", all_cols)
            features = [c for c in all_cols if c != target]
            X, y = df_model[features], df_model[target]
            
            clf_algo = st.radio("选择分类算法：", ["Random Forest", "Logistic Regression"])
            if "Random Forest" in clf_algo:
                n_est = st.slider("树的数量:", 10, 200, 100, 10)
                max_depth = st.slider("最大深度:", 1, 20, 5)
                model = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
                hyper_str = f"n_estimators={n_est}, max_depth={max_depth}"
            else:
                c_val = st.slider("正则化参数 C:", 0.01, 10.0, 1.0)
                model = LogisticRegression(C=c_val, max_iter=1000)
                hyper_str = f"C={c_val}"
                
            if st.button("🚀 开始训练分类模型"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                
                st.metric("🏆 模型测试集准确率 (Accuracy)", f"{acc:.2%}")
                cm = confusion_matrix(y_test, preds)
                st.plotly_chart(px.imshow(cm, text_auto=True, title="📊 混淆矩阵"), use_container_width=True)
                
                st.session_state['report_data'] = {"任务类型": "有监督分类", "算法名称": clf_algo, "核心超参数": hyper_str, "评估反馈": f"测试集预测准确率达到 {acc:.2%}。"}
                st.session_state['history_log'].append(f"训练了分类模型 [{clf_algo}]")

        elif "回归任务" in task_type:
            all_cols = df_model.columns.tolist()
            target = st.selectbox("请选择连续变量目标 (Target):", all_cols)
            features = [c for c in all_cols if c != target]
            X, y = df_model[features], df_model[target]
            
            reg_algo = st.radio("选择回归算法：", ["Linear Regression", "Decision Tree Regressor"])
            if "Decision Tree" in reg_algo:
                max_depth = st.slider("最大深度:", 1, 20, 5)
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                hyper_str = f"max_depth={max_depth}"
            else:
                model = LinearRegression()
                hyper_str = "无特定参数"
                
            if st.button("🚀 开始训练回归模型"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                
                st.metric("🏆 拟合优度 R²", f"{r2:.4f}")
                fig_reg = go.Figure()
                fig_reg.add_trace(go.Scatter(x=y_test, y=preds, mode='markers', name='预测点'))
                fig_reg.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', line=dict(color='red')))
                st.plotly_chart(fig_reg, use_container_width=True)
                
                st.session_state['report_data'] = {"任务类型": "有监督回归", "算法名称": reg_algo, "核心超参数": hyper_str, "评估反馈": f"模型的决定系数 R² = {r2:.4f}。"}
                st.session_state['history_log'].append(f"训练了回归模型 [{reg_algo}]")

# ==========================================
# 步骤 5：智能化报告导出
# ==========================================
elif page == "5. 导出报告":
    st.title("📄 动态生成数据挖掘实验报告")
    
    if st.session_state['raw_data'] is None:
        st.warning("暂无数据操作痕迹，请回到前几步执行完整的实验工作流！")
    else:
        st.success("🎉 数据挖掘工作流已全部走通！系统已为您自动沉淀出全流程的分析报告。")
        
        # 1. 动态拼装 Markdown 文本结构
        report_md = f"""# 📈 数据挖掘实验与分析报告
        
## 一、 数据集基本画像 (Data Profiling)
- **原始数据形态**：共包含 `{st.session_state['raw_data'].shape[0]}` 行数据，`{st.session_state['raw_data'].shape[1]}` 个原始特征列。
- **当前特征可用数量**：经处理后当前剩余 `{st.session_state['processed_data'].shape[1]}` 列。

## 二、 交互式数据清洗与特征工程步骤
本报告包含用户做出的所有关键数据治理决策。系统按执行时间线忠实记录如下：
"""
        # 填充日志历史
        for idx, log in enumerate(st.session_state['history_log'], 1):
            report_md += f"{idx}. {log}\n"
            
        # 填充模型挖掘阶段的结果
        report_md += "\n## 三、 数据挖掘与算法模型表现\n"
        if st.session_state['report_data']:
            res = st.session_state['report_data']
            report_md += f"""- **分析任务大类**：{res.get('任务类型', '未选择')}
- **所选挖掘算法**：`{res.get('算法名称', '暂无')}`
- **用户指定的核心超参数**：`{res.get('核心超参数', '暂无')}`
- **模型核心量化效果反馈**：**{res.get('评估反馈', '未捕获指标')}**
"""
        else:
            report_md += "> ⚠️ **注意**：您在第 4 步中还未点击运行任何训练/挖掘算法。请运行算法后再来本页导出核心指标。\n"
            
        report_md += """
## 四、 实验结论与后续优化建议
1. **控制反哺**：通过前端“人类决策”的介入，清洗逻辑完美避开了机械化删除带来的信息丢失问题。
2. **模型优化**：若想进一步提升模型指标，可尝试在第 3 步加入更加复杂的特征交叉，或在第 4 步加大算法树的深度进行更细致的超参数寻优。
---
*报告由 Interactive Data Mining Platform 自动生成 • 2026*
"""

        # 2. 在前端界面渲染 Markdown 报告，供人类肉眼检查
        st.info("📊 **报告效果实时预览如下：**")
        st.markdown(report_md)
        st.markdown("---")
        
        # 3. 提供一键导出下载按钮
        st.subheader("💾 报告一键导出")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.download_button(
                label="📥 下载为 Markdown 文件 (.md)",
                data=report_md,
                file_name="Data_Mining_Experiment_Report.md",
                mime="text/markdown"
            )
        with col_btn2:
            st.caption("💡 提示：Markdown 文件可以直接用 Notion、Obsidian、或者 Word 轻松打开并转换为优雅的排版印刷体。")