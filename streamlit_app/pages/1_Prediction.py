import streamlit as st
import requests
import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.predict import load_artifacts, predict
from src.preprocess import feature_engineering
import os

st.set_page_config(page_title="Prediction & Explainability", layout="wide")

st.title("🔮 Barrier Prediction System")

st.sidebar.header("Input Features")
area = st.sidebar.number_input("Area", min_value=1000.0, max_value=100000.0, value=5000.0)
sensing_range = st.sidebar.number_input("Sensing Range", min_value=5.0, max_value=100.0, value=15.0)
transmission_range = st.sidebar.number_input("Transmission Range", min_value=10.0, max_value=200.0, value=30.0)
sensor_nodes = st.sidebar.number_input("Number of Sensor Nodes", min_value=10.0, max_value=1000.0, value=100.0)

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

if st.button("Predict Number of Barriers"):
    payload = {
        "Area": area,
        "Sensing_Range": sensing_range,
        "Transmission_Range": transmission_range,
        "Number_of_Sensor_nodes": sensor_nodes
    }
    
    with st.spinner("Predicting..."):
        try:
            try:
                response = requests.post(API_URL, json=payload, timeout=3)
                response.raise_for_status()
                prediction = response.json()["predicted_barriers"]
            except Exception as api_err:
                st.info("FastAPI backend unreachable. Using embedded model for local inference.")
                input_dict = {
                    "Area": area,
                    "Sensing Range": sensing_range,
                    "Transmission Range": transmission_range,
                    "Number of Sensor nodes": sensor_nodes
                }
                prediction = predict(input_dict)
                
            st.success("Prediction Complete!")
            st.metric(label="Predicted Number of Barriers", value=f"{prediction:.2f}")
            
            # Gauge visualization
            import plotly.graph_objects as go
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Predicted Barriers"},
                gauge = {'axis': {'range': [None, max(500, prediction*1.5)]},
                         'bar': {'color': "darkblue"},
                         'steps' : [
                             {'range': [0, 100], 'color': "lightgray"},
                             {'range': [100, 300], 'color': "gray"}],}
            ))
            st.plotly_chart(fig)
            
            # SHAP Explainability
            st.subheader("Model Explainability (SHAP)")
            try:
                model, scaler, feature_columns = load_artifacts()
                
                # Create input dataframe
                input_dict_shap = {
                    "Area": area,
                    "Sensing Range": sensing_range,
                    "Transmission Range": transmission_range,
                    "Number of Sensor nodes": sensor_nodes
                }
                df_input = pd.DataFrame([input_dict_shap])
                df_input = feature_engineering(df_input)
                df_input = df_input[feature_columns]
                
                X_scaled = scaler.transform(df_input)
                
                # Use TreeExplainer for Random Forest/GBM, LinearExplainer for LR
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.LinearExplainer(model, scaler.transform(pd.DataFrame(columns=feature_columns)))
                
                shap_values = explainer(X_scaled)
                
                fig, ax = plt.subplots()
                shap.waterfall_plot(shap_values[0], show=False)
                st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"SHAP visualization failed: {e}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
