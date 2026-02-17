"""
Streamlit Dashboard for AQI Predictions.
Displays 3-day forecasts, alerts, historical trends, and model explainability.
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CITY_NAME, AQI_CATEGORIES, MODELS_DIR

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title=f"AQI Predictor — {CITY_NAME}",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .aqi-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 5px;
    }
    .aqi-value {
        font-size: 48px;
        margin: 10px 0;
    }
    .aqi-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .alert-banner {
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ff4444;
        background-color: #fff3f3;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────────

def get_aqi_color(aqi: float) -> str:
    """Get color for AQI value."""
    for cat, info in AQI_CATEGORIES.items():
        if info["min"] <= aqi <= info["max"]:
            return info["color"]
    return "#7E0023"


def get_aqi_category(aqi: float) -> str:
    """Get category name for AQI value."""
    for cat, info in AQI_CATEGORIES.items():
        if info["min"] <= aqi <= info["max"]:
            return cat
    return "Hazardous"


def get_aqi_emoji(aqi: float) -> str:
    """Get emoji for AQI value."""
    if aqi <= 50:
        return "🟢"
    if aqi <= 100:
        return "🟡"
    if aqi <= 150:
        return "🟠"
    if aqi <= 200:
        return "🔴"
    if aqi <= 300:
        return "🟣"
    return "⚫"


@st.cache_data(ttl=3600)
def load_predictions(model_name=None):
    """Load predictions from inference pipeline."""
    try:
        from src.inference.predict import predict_next_3_days
        return predict_next_3_days(model_name=model_name)
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
        return None


@st.cache_data(ttl=3600)
def load_historical_data():
    """Load historical feature data."""
    try:
        from src.feature_pipeline.feature_store import get_features
        return get_features(use_hopsworks=False)
    except Exception as e:
        st.error(f"Could not load historical data: {e}")
        return pd.DataFrame()


def load_model_info():
    """Load model metadata."""
    best_meta_path = MODELS_DIR / "best" / "metadata.json"
    if best_meta_path.exists():
        with open(best_meta_path, "r") as f:
            return json.load(f)
    return None


# ─── Sidebar ──────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌍 AQI Predictor")
    st.markdown(f"**City:** {CITY_NAME}")
    st.markdown("---")

    # Model selector
    available_models = []
    if MODELS_DIR.exists():
        for d in MODELS_DIR.iterdir():
            if d.is_dir() and (d / "metadata.json").exists():
                available_models.append(d.name)

    selected_model = st.selectbox(
        "Select Model",
        options=["best"] + [m for m in available_models if m != "best"],
        index=0,
    )

    if st.button("🔄 Refresh Predictions"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### AQI Scale")
    for cat, info in AQI_CATEGORIES.items():
        st.markdown(
            f'<div style="background-color: {info["color"]}; color: white; '
            f'padding: 4px 8px; border-radius: 4px; margin: 2px 0; font-size: 12px;">'
            f'{info["min"]}-{info["max"]}: {cat}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "Built with Streamlit • [GitHub](https://github.com)",
        unsafe_allow_html=True,
    )


# ─── Main Content ────────────────────────────────────────────────

st.title(f"🏙️ Air Quality Index — {CITY_NAME}")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load predictions
model_to_use = None if selected_model == "best" else selected_model
predictions = load_predictions(model_to_use)

# ─── Tab Layout ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 3-Day Forecast",
    "📈 Historical Trends",
    "🔍 Feature Importance",
    "ℹ️ Model Info",
])

# ─── Tab 1: 3-Day Forecast ───────────────────────────────────────
with tab1:
    if predictions and "predictions" in predictions:
        # Alert Banner
        if predictions.get("alert"):
            st.markdown(
                f'<div class="alert-banner">'
                f'<strong>⚠️ Air Quality Alert</strong><br>'
                f'{predictions["alert_message"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Forecast Cards
        cols = st.columns(3)
        for i, pred in enumerate(predictions["predictions"]):
            with cols[i]:
                color = pred["color"]
                emoji = get_aqi_emoji(pred["predicted_aqi"])
                st.markdown(
                    f'<div class="aqi-card" style="background-color: {color};">'
                    f'<div class="aqi-label">Day {pred["day"]} — {pred["day_name"]}</div>'
                    f'<div class="aqi-value">{emoji} {pred["predicted_aqi"]}</div>'
                    f'<div class="aqi-label">{pred["category"]}</div>'
                    f'<div class="aqi-label" style="font-size: 11px;">{pred["date"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Forecast trend chart
        forecast_df = pd.DataFrame(predictions["predictions"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["predicted_aqi"],
            mode="lines+markers",
            marker=dict(
                size=15,
                color=[get_aqi_color(v) for v in forecast_df["predicted_aqi"]],
                line=dict(width=2, color="white"),
            ),
            line=dict(color="#3498db", width=3),
            text=forecast_df["category"],
            hovertemplate="<b>%{x}</b><br>AQI: %{y}<br>Category: %{text}<extra></extra>",
        ))

        # Add category background bands
        fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.08)
        fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.08)
        fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.08)
        fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.08)
        fig.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.08)

        fig.update_layout(
            title="3-Day AQI Forecast Trend",
            xaxis_title="Date",
            yaxis_title="AQI",
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig, width='stretch')

        # Health advisories
        st.subheader("🏥 Health Advisories")
        for pred in predictions["predictions"]:
            with st.expander(f"{get_aqi_emoji(pred['predicted_aqi'])} {pred['day_name']} ({pred['date']}) — {pred['category']}"):
                st.write(pred["health_advisory"])
                if pred["is_hazardous"]:
                    st.warning("⚠️ Consider limiting outdoor activities on this day.")

    else:
        st.info(
            "No predictions available. Please ensure:\n"
            "1. The feature pipeline has been run (`python -m src.feature_pipeline.backfill --days 30`)\n"
            "2. The training pipeline has been run (`python -m src.training_pipeline.train`)\n"
        )


# ─── Tab 2: Historical Trends ────────────────────────────────────
with tab2:
    hist_df = load_historical_data()

    if not hist_df.empty:
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            days_back = st.slider("Days to show", 7, 90, 30)

        # Filter data
        cutoff = datetime.now() - timedelta(days=days_back)
        filtered = hist_df[hist_df["datetime"] >= cutoff]

        if not filtered.empty:
            # AQI time series
            fig = px.line(
                filtered, x="datetime", y="aqi",
                title=f"AQI Trend — Last {days_back} Days",
                labels={"aqi": "AQI", "datetime": "Date"},
            )
            fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.05, annotation_text="Good")
            fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.05, annotation_text="Moderate")
            fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.05)
            fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.05)
            fig.update_layout(height=450, template="plotly_white")
            st.plotly_chart(fig, width='stretch')

            # Pollutant breakdown
            st.subheader("Pollutant Levels")
            pollutants = ["pm2_5", "pm10", "no2", "so2", "co", "o3"]
            available = [p for p in pollutants if p in filtered.columns]

            if available:
                fig = make_subplots(rows=2, cols=3,
                                    subplot_titles=[p.upper().replace("_", ".") for p in available])
                for i, pol in enumerate(available):
                    row, col = i // 3 + 1, i % 3 + 1
                    fig.add_trace(
                        go.Scatter(x=filtered["datetime"], y=filtered[pol],
                                   mode="lines", name=pol.upper()),
                        row=row, col=col,
                    )
                fig.update_layout(height=500, showlegend=False, template="plotly_white")
                st.plotly_chart(fig, width='stretch')

            # Statistics
            st.subheader("📊 Summary Statistics")
            stats_cols = ["aqi"] + available
            stats = filtered[stats_cols].describe().round(2)
            st.dataframe(stats, width='stretch')
        else:
            st.info(f"No data available for the last {days_back} days.")
    else:
        st.info("No historical data available. Run the backfill pipeline first.")


# ─── Tab 3: Feature Importance ────────────────────────────────────
with tab3:
    st.subheader("🔍 SHAP Feature Importance")

    # Check for pre-generated SHAP plots
    shap_summary_path = MODELS_DIR / "shap_plots" / "shap_summary.png"
    shap_bar_path = MODELS_DIR / "shap_plots" / "shap_bar.png"

    if shap_bar_path.exists():
        st.image(str(shap_bar_path), caption="Top Feature Importance (SHAP)", width='stretch')

    if shap_summary_path.exists():
        st.image(str(shap_summary_path), caption="SHAP Summary Plot", width='stretch')

    if not shap_bar_path.exists() and not shap_summary_path.exists():
        st.info(
            "SHAP plots not yet generated. They will be created after training.\n\n"
            "You can generate them by running:\n"
            "```python\n"
            "from src.explainability.shap_explain import explain_model\n"
            "# ... load model and data, then call explain_model()\n"
            "```"
        )

    # LIME section
    st.subheader("🍋 LIME Explanations")
    lime_dir = MODELS_DIR / "lime_plots"
    if lime_dir.exists():
        lime_plots = list(lime_dir.glob("*.png"))
        if lime_plots:
            for plot in lime_plots[:3]:
                st.image(str(plot), width='stretch')
        else:
            st.info("No LIME plots generated yet.")
    else:
        st.info("LIME explanations will be available after running the explainability pipeline.")


# ─── Tab 4: Model Info ───────────────────────────────────────────
with tab4:
    st.subheader("ℹ️ Model Information")

    model_info = load_model_info()

    if model_info:
        col1, col2, col3 = st.columns(3)
        metrics = model_info.get("metrics", {})

        with col1:
            st.metric("Model Type", model_info.get("model_name", "N/A").replace("_", " ").title())
        with col2:
            st.metric("RMSE", f"{metrics.get('rmse', 'N/A'):.2f}" if isinstance(metrics.get('rmse'), (int, float)) else "N/A")
        with col3:
            st.metric("R²", f"{metrics.get('r2', 'N/A'):.3f}" if isinstance(metrics.get('r2'), (int, float)) else "N/A")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("MAE", f"{metrics.get('mae', 'N/A'):.2f}" if isinstance(metrics.get('mae'), (int, float)) else "N/A")
        with col5:
            st.metric("Trained At", model_info.get("trained_at", "N/A")[:19])
        with col6:
            st.metric("Features", len(model_info.get("features", [])))

        # Model comparison table
        comparison_path = MODELS_DIR / "model_comparison.csv"
        if comparison_path.exists():
            st.subheader("Model Comparison")
            comparison_df = pd.read_csv(comparison_path, index_col=0)
            st.dataframe(comparison_df.style.highlight_min(subset=["rmse", "mae"], color="#d4edda")
                         .highlight_max(subset=["r2"], color="#d4edda"),
                         width='stretch')

        # Prediction vs Actual plots
        st.subheader("Prediction vs Actual")
        for img_name in ["ridge_predictions.png", "rf_predictions.png",
                         "xgb_predictions.png", "model_comparison.png"]:
            img_path = MODELS_DIR / img_name
            if img_path.exists():
                st.image(str(img_path), width='stretch')

    else:
        st.info("No model trained yet. Run the training pipeline first.")


# ─── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<center style='color: #888; font-size: 12px;'>"
    f"Pearls AQI Predictor • {CITY_NAME} • "
    f"Data: OpenWeatherMap • Model: Scikit-learn / TensorFlow"
    f"</center>",
    unsafe_allow_html=True,
)
