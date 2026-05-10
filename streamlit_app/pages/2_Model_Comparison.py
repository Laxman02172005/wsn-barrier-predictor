import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Model Comparison", layout="wide")

st.title("📊 Model Comparison Dashboard")

METRICS_PATH = os.getenv("METRICS_PATH", "models/metrics.json")

try:
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
        
    df_metrics = pd.DataFrame(metrics).T.reset_index()
    df_metrics.rename(columns={"index": "Model"}, inplace=True)
    
    st.subheader("Metrics Overview")
    st.dataframe(df_metrics)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("R² Score Comparison")
        fig_r2 = px.bar(df_metrics, x="Model", y="r2", color="Model", title="R² Score (Higher is better)")
        st.plotly_chart(fig_r2, use_container_width=True)
        
    with col2:
        st.subheader("RMSE Comparison")
        fig_rmse = px.bar(df_metrics, x="Model", y="rmse", color="Model", title="RMSE (Lower is better)")
        st.plotly_chart(fig_rmse, use_container_width=True)
        
    st.subheader("MAE Comparison")
    fig_mae = px.bar(df_metrics, x="Model", y="mae", color="Model", title="MAE (Lower is better)")
    st.plotly_chart(fig_mae, use_container_width=True)

except FileNotFoundError:
    st.warning("Metrics file not found. Please train the models first using `make train`.")
