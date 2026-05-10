import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Data Analytics", layout="wide")

st.title("📈 Data Analytics")

DATA_PATH = os.getenv("DATA_PATH", "data/data.csv")

try:
    df = pd.read_csv(DATA_PATH)
    
    st.subheader("Data Distributions")
    fig_hist = plt.figure(figsize=(10, 8))
    df.hist(ax=fig_hist.gca())
    plt.tight_layout()
    st.pyplot(fig_hist)
    
    st.subheader("Correlation Matrix")
    correlation_matrix = df.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)
    
    st.subheader("Pairplot")
    fig_pair = sns.pairplot(df)
    st.pyplot(fig_pair.fig)

except FileNotFoundError:
    st.warning("`data.csv` not found in the `data/` directory. Please add the dataset to view analytics.")
