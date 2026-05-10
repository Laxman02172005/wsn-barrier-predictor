import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="WSN MLOps Dashboard",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Wireless Sensor Network (WSN) - MLOps Dashboard")

st.markdown("""
Welcome to the production-grade MLOps Dashboard for predicting the **Number of Barriers** in a Wireless Sensor Network. 

### Features:
- **Prediction System**: Make real-time predictions using our FastAPI backend, complete with SHAP explainability.
- **Model Comparison**: Compare training metrics (R², RMSE, MAE) across multiple models.
- **Data Analytics**: Explore dataset distributions and correlations.

Use the sidebar to navigate between pages.
""")

DATA_PATH = os.getenv("DATA_PATH", "data/data.csv")

try:
    df = pd.read_csv(DATA_PATH)
    
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Max Area", df["Area"].max())
    col4.metric("Max Nodes", df["Number of Sensor nodes"].max())
    
    st.subheader("Data Sample")
    st.dataframe(df.head(10))
    
    st.subheader("Feature Correlation with Target")
    correlation = df.corr()
    target_corr = correlation["Number of Barriers"].sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=target_corr.values[1:], y=target_corr.index[1:], ax=ax, palette="viridis")
    ax.set_xlabel("Correlation Coefficient")
    ax.set_title("Feature Importance (Correlation)")
    st.pyplot(fig)
    
except Exception as e:
    st.warning("Please ensure `data.csv` is placed in the `data/` directory to view the dataset overview.")
