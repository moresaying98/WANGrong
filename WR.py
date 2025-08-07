import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

# 设置页面标题
st.title("LightGBM Prediction and SHAP Visualization")

# 加载模型
model = joblib.load("lgb_model.pkl")

# 特征列表（根据模型训练顺序调整）
model_feature_names = ['TG', 'ALT', 'HDL-C', 'Eosinophil', 'Ures',
                       'PA', 'RDW-SD', 'D-D', 'APTT', 'GLB', 'CCP',
                       'AST', 'ALB', 'UA', 'LCL-C', 'Arthralgia',
                       'P-LCR', 'WBC']

# 特征范围字典（可根据你数据具体范围调整）
feature_ranges = {
    'TG': (10, 500), 'ALT': (5, 300), 'HDL-C': (20, 120), 'Eosinophil': (0, 10),
    'Ures': (1, 20), 'PA': (50, 400), 'RDW-SD': (30, 60), 'D-D': (0, 20),
    'APTT': (15, 80), 'GLB': (10, 50), 'CCP': (0, 300), 'AST': (5, 300),
    'ALB': (20, 60), 'UA': (100, 900), 'LCL-C': (50, 200),
    'Arthralgia': (0, 2),  # 三分类变量：0/1/2
    'P-LCR': (5, 50), 'WBC': (1, 20)
}

# 输入区域
st.header("Input Feature Values")
user_input = []
for feature in model_feature_names:
    if feature == 'Arthralgia':
        val = st.selectbox(f"{feature} (0:无, 1:轻微, 2:明显)", [0, 1, 2])
    else:
        min_val, max_val = feature_ranges[feature]
        val = st.slider(f"{feature} ({min_val} - {max_val})", float(min_val), float(max_val),
                        float((min_val + max_val) / 2))
    user_input.append(val)

# 转为模型格式
X_input = pd.DataFrame([user_input], columns=model_feature_names)

# 预测与SHAP可视化
if st.button("Predict"):
    prediction = model.predict(X_input)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_input)
    else:
        proba = None

    # 显示预测结果
    if hasattr(model, 'classes_'):
        class_names = model.classes_
        pred_class = int(prediction[0])
        st.success(f"Predicted Class: {pred_class}")
        if proba is not None:
            for i, p in enumerate(proba[0]):
                st.write(f"Probability of class {class_names[i]}: {p:.4f}")
    else:
        st.success(f"Predicted value: {prediction[0]:.4f}")

    # SHAP 分析
    st.subheader("SHAP Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)

    # 判断是二分类还是多分类
    if isinstance(shap_values, list):  # 多分类
        st.write("Detected multiclass model")
        for i in range(len(shap_values)):
            st.write(f"SHAP summary plot for class {i}")
            plt.figure()
            shap.summary_plot(shap_values[i], X_input, plot_type="bar", show=False)
            st.pyplot(plt.gcf())
            plt.clf()
    else:  # 二分类或回归
        st.write("Detected binary or regression model")
        plt.figure()
        shap.summary_plot(shap_values, X_input, plot_type="bar", show=False)
        st.pyplot(plt.gcf())
        plt.clf()






