import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from streamlit_shap import st_shap

# 设置页面配置
st.set_page_config(page_title="Predictive Model App", layout="wide", initial_sidebar_state="expanded")

# 设置样式：字体、背景颜色、按钮颜色等
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        color: #333333;
    }
    h1, h2, h3 {
        color: #3a7bd5;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stTextInput {
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True
)

# 页面标题和说明
st.title("Predictive XGBoost Machine Learning App for Pressure Insury in ICU")
st.subheader("This app allows you to input clinical data and predicts outcomes using an XGBoost model.")
st.write("---")  # 分隔线

# 布局优化：使用列布局来分配输入字段
col1, col2, col3 = st.columns(3)

with col1:
    Days_in_ICU = st.number_input("Days in ICU (days)", min_value=0, value=3)
    Department_Transfer = st.selectbox("Department Transfer", ("Yes", "No"))
    
with col2:
    Glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, value=5.6)
    Neutrophil_Count = st.number_input("Neutrophil Count (*10^9/L)", min_value=0.0, value=0.2)
    Consciousness = st.selectbox("Consciousness (coma)", ("Yes", "No"))
    Serum_Albumin = st.number_input("Serum Albumin (g/dL)", min_value=0.0, value=35.0)

with col3:
    Sedatives = st.selectbox("Sedatives Used", ("Yes", "No"))
    Warming_Blanket = st.selectbox("Warming Blanket", ("Yes", "No"))
    Mechanical_Ventilation = st.selectbox("Mechanical Ventilation", ("Yes", "No"))
    Smoking_History = st.selectbox("Smoking History", ("Yes", "No"))

# 提交按钮和输出区域
if st.button("Predict"):
    
    # 载入模型
    clf = joblib.load("xgb_model.pkl")
    
    # 将选择框转换为0/1
    X = pd.DataFrame([[Department_Transfer, Consciousness, Mechanical_Ventilation, Sedatives, 
                       Warming_Blanket, Days_in_ICU, Serum_Albumin, Neutrophil_Count, Glucose, Smoking_History]],
                     columns=clf.feature_names_in_)

    X = X.replace({"Yes": 1, "No": 0})
    
    # 模型预测
    prediction = clf.predict(X)[0]
    prediction_probability = clf.predict_proba(X)[0, 1]

    # 根据预测值输出不同的文本
    if prediction == 0:
        prediction_text = "Prediction: Pressure Injury will not occur"
    else:
        prediction_text = "Prediction: Pressure Injury will occur"

    # 显示预测结果
    st.success(f"{prediction_text}")
    st.info(f"Predicted probability: **{round(prediction_probability * 100, 2)}%**")

    # SHAP值解释模型
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    
    # 显示SHAP force plot
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0, :]), height=140, width=1500)

# 添加脚注
st.write("---")
st.write("Developed by [Jie Xu]. Powered by Streamlit and XGBoost.")
