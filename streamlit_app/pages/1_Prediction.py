import sys
import os

# Ensure repo root is in path so 'src' module is importable on Streamlit Cloud
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from src.predict import load_artifacts, predict
from src.preprocess import feature_engineering

st.set_page_config(page_title="Prediction & Explainability", layout="wide", page_icon="🔮")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        border: 1px solid #2a4a7f;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h2 { color: #00d4ff; font-size: 2.5rem; margin: 0; }
    .metric-card p  { color: #8ab4d4; font-size: 0.9rem; margin: 4px 0 0 0; }

    .explain-card {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #00d4ff;
        margin-bottom: 8px;
    }
    .explain-card .feat  { color: #ffffff; font-weight: 600; font-size: 1rem; }
    .explain-card .value { color: #8ab4d4; font-size: 0.85rem; }
    .explain-card .impact-pos { color: #00e676; font-weight: bold; }
    .explain-card .impact-neg { color: #ff5252; font-weight: bold; }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #00d4ff;
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 6px;
        margin-top: 24px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🔮 WSN Barrier Prediction & Explainability")
st.markdown("Enter your sensor network parameters in the sidebar and click **Predict** to get an instant analysis.")

# ── Sidebar Inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Network Parameters")
    st.markdown("---")
    area = st.slider("📐 Area (m²)", min_value=1000.0, max_value=100000.0, value=5000.0, step=500.0)
    sensing_range = st.slider("📡 Sensing Range (m)", min_value=5.0, max_value=100.0, value=15.0, step=1.0)
    transmission_range = st.slider("📶 Transmission Range (m)", min_value=10.0, max_value=200.0, value=30.0, step=2.0)
    sensor_nodes = st.slider("🔢 Number of Sensor Nodes", min_value=10.0, max_value=1000.0, value=100.0, step=10.0)
    st.markdown("---")
    predict_btn = st.button("🚀 Predict Barriers", use_container_width=True, type="primary")

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

if predict_btn:
    with st.spinner("Running prediction..."):
        try:
            # Try FastAPI, fall back to local model
            try:
                payload = {
                    "Area": area, "Sensing_Range": sensing_range,
                    "Transmission_Range": transmission_range,
                    "Number_of_Sensor_nodes": sensor_nodes
                }
                response = requests.post(API_URL, json=payload, timeout=3)
                response.raise_for_status()
                prediction = response.json()["predicted_barriers"]
                source = "FastAPI Backend"
            except Exception:
                input_dict = {
                    "Area": area, "Sensing Range": sensing_range,
                    "Transmission Range": transmission_range,
                    "Number of Sensor nodes": sensor_nodes
                }
                prediction = predict(input_dict)
                source = "Embedded Model"

            st.success(f"✅ Prediction complete using **{source}**")

            # ── Row 1: Key Metrics ───────────────────────────────────────────
            st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)

            # Derived features for display
            sensor_density    = sensor_nodes / area
            coverage_ratio    = transmission_range / sensing_range if sensing_range > 0 else 0
            comm_efficiency   = (transmission_range * sensor_nodes) / area

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{prediction:.1f}</h2>
                    <p>🚧 Predicted Barriers</p>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{sensor_density:.4f}</h2>
                    <p>📡 Sensor Density (nodes/m²)</p>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{coverage_ratio:.2f}x</h2>
                    <p>📶 Coverage Ratio</p>
                </div>""", unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{comm_efficiency:.2f}</h2>
                    <p>⚡ Comm. Efficiency</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Row 2: Gauge + Feature Importance ───────────────────────────
            col_gauge, col_bar = st.columns([1, 1])

            with col_gauge:
                st.markdown('<div class="section-title">🎯 Barrier Gauge</div>', unsafe_allow_html=True)
                max_val = max(500, prediction * 1.5)
                gauge_color = "#00e676" if prediction < 100 else ("#ffca28" if prediction < 300 else "#ff5252")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction,
                    delta={"reference": 150, "valueformat": ".1f"},
                    number={"font": {"size": 48, "color": gauge_color}},
                    title={"text": "Number of Barriers", "font": {"size": 16, "color": "#8ab4d4"}},
                    gauge={
                        "axis": {"range": [0, max_val], "tickcolor": "#8ab4d4"},
                        "bar": {"color": gauge_color, "thickness": 0.3},
                        "bgcolor": "#0d1b2a",
                        "bordercolor": "#1e3a5f",
                        "steps": [
                            {"range": [0, max_val*0.33], "color": "#0a3d1f"},
                            {"range": [max_val*0.33, max_val*0.66], "color": "#3d3000"},
                            {"range": [max_val*0.66, max_val], "color": "#3d0a0a"},
                        ],
                        "threshold": {
                            "line": {"color": "#00d4ff", "width": 3},
                            "thickness": 0.8,
                            "value": prediction
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="#0d1b2a", font_color="#ffffff",
                    height=300, margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_bar:
                st.markdown('<div class="section-title">📈 Input Feature Values</div>', unsafe_allow_html=True)
                features = {
                    "Area": area,
                    "Sensing Range": sensing_range,
                    "Transmission Range": transmission_range,
                    "Sensor Nodes": sensor_nodes,
                    "Sensor Density": round(sensor_density, 5),
                    "Coverage Ratio": round(coverage_ratio, 3),
                    "Comm. Efficiency": round(comm_efficiency, 3),
                }
                df_feat = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
                # Normalize 0-1 for color intensity
                df_feat["Normalized"] = (df_feat["Value"] - df_feat["Value"].min()) / (df_feat["Value"].max() - df_feat["Value"].min() + 1e-9)

                fig_bar = px.bar(
                    df_feat, x="Value", y="Feature", orientation="h",
                    color="Normalized",
                    color_continuous_scale=[[0, "#1e3a5f"], [0.5, "#0099cc"], [1, "#00d4ff"]],
                    text=df_feat["Value"].apply(lambda v: f"{v:.3f}" if v < 10 else f"{v:.1f}"),
                )
                fig_bar.update_traces(textposition="outside", textfont_color="#ffffff")
                fig_bar.update_layout(
                    paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
                    font_color="#ffffff", coloraxis_showscale=False,
                    xaxis=dict(showgrid=False, color="#8ab4d4"),
                    yaxis=dict(showgrid=False, color="#ffffff"),
                    height=300, margin=dict(l=10, r=40, t=20, b=20)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── Row 3: SHAP / Feature Contribution ──────────────────────────
            st.markdown('<div class="section-title">🧠 Model Explainability — Feature Contributions</div>', unsafe_allow_html=True)

            try:
                import shap
                model, scaler, feature_columns = load_artifacts()

                input_dict_shap = {
                    "Area": area, "Sensing Range": sensing_range,
                    "Transmission Range": transmission_range,
                    "Number of Sensor nodes": sensor_nodes
                }
                df_input = pd.DataFrame([input_dict_shap])
                df_input = feature_engineering(df_input)
                df_input = df_input[feature_columns]
                X_scaled = scaler.transform(df_input)

                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_scaled)
                    if isinstance(shap_values, list):
                        shap_vals = shap_values[0][0]
                    else:
                        shap_vals = shap_values[0]
                else:
                    explainer = shap.LinearExplainer(model, X_scaled)
                    shap_vals = explainer.shap_values(X_scaled)[0]

                # Build a clean Plotly waterfall-style bar chart
                shap_df = pd.DataFrame({
                    "Feature": feature_columns,
                    "SHAP Value": shap_vals
                }).sort_values("SHAP Value", key=abs, ascending=True)

                colors = ["#00e676" if v >= 0 else "#ff5252" for v in shap_df["SHAP Value"]]
                labels = [f"+{v:.3f}" if v >= 0 else f"{v:.3f}" for v in shap_df["SHAP Value"]]

                fig_shap = go.Figure(go.Bar(
                    x=shap_df["SHAP Value"],
                    y=shap_df["Feature"],
                    orientation="h",
                    marker_color=colors,
                    text=labels,
                    textposition="outside",
                    textfont=dict(color="#ffffff", size=12),
                ))
                fig_shap.add_vline(x=0, line_color="#8ab4d4", line_width=1.5)
                fig_shap.update_layout(
                    paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
                    font_color="#ffffff",
                    xaxis=dict(title="SHAP Contribution to Prediction", showgrid=True, gridcolor="#1e3a5f", color="#8ab4d4", zeroline=False),
                    yaxis=dict(showgrid=False, color="#ffffff"),
                    height=350, margin=dict(l=10, r=60, t=20, b=20)
                )
                st.plotly_chart(fig_shap, use_container_width=True)

                # Human-readable cards
                shap_df_sorted = shap_df.sort_values("SHAP Value", key=abs, ascending=False)
                st.markdown("**Top feature contributions to this prediction:**")
                cols = st.columns(3)
                for idx, (_, row) in enumerate(shap_df_sorted.head(6).iterrows()):
                    impact_class = "impact-pos" if row["SHAP Value"] >= 0 else "impact-neg"
                    impact_arrow = "▲ increases" if row["SHAP Value"] >= 0 else "▼ decreases"
                    impact_val   = abs(row["SHAP Value"])
                    with cols[idx % 3]:
                        st.markdown(f"""
                        <div class="explain-card">
                            <div class="feat">🔹 {row['Feature']}</div>
                            <div class="value">
                                <span class="{impact_class}">{impact_arrow}</span> prediction by 
                                <span class="{impact_class}">{impact_val:.3f}</span>
                            </div>
                        </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.info(f"ℹ️ SHAP explainability unavailable (model not loaded or incompatible): {e}")

                # Fallback: show a simple feature importance radar chart
                st.markdown("**Showing normalized feature contributions based on input values:**")
                feat_names = ["Area", "Sensing Range", "Transmission Range", "Sensor Nodes", "Sensor Density", "Coverage Ratio", "Comm. Efficiency"]
                feat_vals  = [area, sensing_range, transmission_range, sensor_nodes, sensor_density, coverage_ratio, comm_efficiency]
                feat_norm  = [v / (max(feat_vals) + 1e-9) for v in feat_vals]

                fig_radar = go.Figure(go.Scatterpolar(
                    r=feat_norm + [feat_norm[0]],
                    theta=feat_names + [feat_names[0]],
                    fill="toself",
                    line_color="#00d4ff",
                    fillcolor="rgba(0,212,255,0.15)",
                    name="Feature Profile"
                ))
                fig_radar.update_layout(
                    paper_bgcolor="#0d1b2a", font_color="#ffffff",
                    polar=dict(bgcolor="#0d1b2a",
                               radialaxis=dict(color="#8ab4d4", gridcolor="#1e3a5f"),
                               angularaxis=dict(color="#8ab4d4", gridcolor="#1e3a5f")),
                    height=400, margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")
else:
    # Landing state
    st.info("👈 Set your network parameters in the sidebar and click **Predict Barriers** to begin.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📐 Area**  
        The total deployment area in square meters. Larger areas generally produce more barriers.
        """)
    with col2:
        st.markdown("""
        **📡 Sensing Range**  
        How far each node can sense. A smaller sensing range leaves more gaps — leading to more barriers.
        """)
    with col3:
        st.markdown("""
        **📶 Transmission Range**  
        How far nodes communicate. Higher range increases connectivity and reduces barriers.
        """)
