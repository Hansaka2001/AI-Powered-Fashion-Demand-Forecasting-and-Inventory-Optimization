import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from pathlib import Path
import scipy.stats as stats

# ==========================================
# Configuration & Setup
# ==========================================
st.set_page_config(
    page_title="Fashion Demand Forecasting & Inventory Optimization",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 14px;
        color: #c9d1d9;
    }
    .st-eb {
        background-color: #0d1117;
    }
</style>
""", unsafe_allow_html=True)

# Paths
ROOT_DIR = Path(__file__).parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
VIZ_DIR = OUTPUTS_DIR / "visualizations"

# ==========================================
# Data Loading Functions
# ==========================================
@st.cache_data
def load_data():
    try:
        eval_report = pd.read_csv(REPORTS_DIR / "model_evaluation_report.csv")
        forecast_7d = pd.read_csv(REPORTS_DIR / "forecast_7day.csv")
        forecast_30d = pd.read_csv(REPORTS_DIR / "forecast_30day.csv")
        inv_recs = pd.read_csv(REPORTS_DIR / "inventory_recommendations.csv")
        shap_feat = pd.read_csv(REPORTS_DIR / "shap_feature_importance.csv")
        return eval_report, forecast_7d, forecast_30d, inv_recs, shap_feat
    except Exception as e:
        st.error(f"Error loading data. Ensure the pipeline has been run first. ({e})")
        return None, None, None, None, None

@st.cache_data
def load_historical_stats(forecast_df):
    # Derive historical std dev approximation for interactive optimization
    # (Since we don't load the full 500k row historical dataset for the UI)
    stats_map = {}
    for store_id, group in forecast_df.groupby("Store"):
        # We approximate historical std as 15% of avg forecast
        avg_f = group["Forecast_Sales"].mean()
        stats_map[store_id] = avg_f * 0.15
    return stats_map

eval_report, forecast_7d, forecast_30d, inv_recs, shap_feat = load_data()

if eval_report is None:
    st.stop()

stores = sorted(forecast_7d["Store"].unique())
historical_std_map = load_historical_stats(forecast_30d)

# ==========================================
# Sidebar Navigation
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3014/3014436.png", width=60)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Global Dashboard", 
    "🏬 Store Forecaster & Inventory", 
    "🧠 Model Explainability",
    "📈 Exploratory Data Analysis"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status:** 🟢 Online")
st.sidebar.markdown(f"**Models Trained:** {len(eval_report)}")
st.sidebar.markdown(f"**Stores Tracked:** {len(stores)}")

# ==========================================
# Interactive Inventory Optimization Logic
# ==========================================
def recalculate_inventory(store_id, horizon, forecast_df, lead_time, service_level_pct, unit_cost, ordering_cost, holding_cost_rate):
    group = forecast_df[forecast_df["Store"] == store_id]
    total_forecast = group["Forecast_Sales"].sum()
    avg_daily_forecast = group["Forecast_Sales"].mean()
    
    # Calculate Z score
    z = stats.norm.ppf(service_level_pct / 100.0)
    hist_std = historical_std_map.get(store_id, avg_daily_forecast * 0.15)
    
    # Formulas
    safety_stock = z * hist_std * np.sqrt(lead_time)
    reorder_point = avg_daily_forecast * lead_time + safety_stock
    recommended_stock = total_forecast + safety_stock
    
    annual_demand = avg_daily_forecast * 365
    holding_cost = holding_cost_rate * unit_cost
    eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost) if holding_cost > 0 else 0
    
    # Stockout risk
    z_score_stockout = (recommended_stock - total_forecast) / (hist_std * np.sqrt(horizon))
    stockout_risk_pct = max(0.0, (1 - stats.norm.cdf(z_score_stockout)) * 100)
    
    if stockout_risk_pct < 5:
        risk_label = "LOW"
        risk_color = "green"
    elif stockout_risk_pct < 15:
        risk_label = "MEDIUM"
        risk_color = "orange"
    else:
        risk_label = "HIGH"
        risk_color = "red"
        
    return {
        "Safety Stock": int(safety_stock),
        "Reorder Point": int(reorder_point),
        "Recommended Stock": int(recommended_stock),
        "EOQ (Units)": int(eoq),
        "Stockout Risk": f"{stockout_risk_pct:.1f}%",
        "Risk Level": risk_label,
        "Risk Color": risk_color,
        "Total Demand": int(total_forecast)
    }

# ==========================================
# Page: Global Dashboard
# ==========================================
if page == "📊 Global Dashboard":
    st.title("📊 Global Operations Dashboard")
    st.markdown("Overview of the forecasting models and overall inventory health.")
    
    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    best_model = eval_report.iloc[0]
    
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Best Model</div><div class="metric-value" style="color: #3fb950;">{best_model["Model"]}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Overall MAPE</div><div class="metric-value">{best_model["MAPE (%)"]}%</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">R² Score</div><div class="metric-value">{best_model["R²"]}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Stores Tracked</div><div class="metric-value">{len(stores)}</div></div>', unsafe_allow_html=True)
        
    st.markdown("### Model Performance Comparison")
    st.dataframe(eval_report.style.highlight_min(subset=['MAE', 'RMSE', 'MAPE (%)', 'CV_RMSE'], color='#1e3a8a')
                                         .highlight_max(subset=['R²'], color='#1e3a8a'))
    
    st.markdown("### Evaluation Visualizations")
    col_viz1, col_viz2 = st.columns(2)
    with col_viz1:
        try:
            st.image(Image.open(VIZ_DIR / "09_model_comparison.png"), use_container_width=True)
        except Exception:
            st.info("Visualization not found.")
    with col_viz2:
        try:
            st.image(Image.open(VIZ_DIR / "10_actual_vs_predicted.png"), use_container_width=True)
        except Exception:
            st.info("Visualization not found.")

# ==========================================
# Page: Store Forecaster & Inventory
# ==========================================
elif page == "🏬 Store Forecaster & Inventory":
    st.title("🏬 Store-Level Demand & Inventory Planner")
    st.markdown("Select a store to view demand forecasts and interactively plan inventory requirements.")
    
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        selected_store = st.selectbox("Select Store ID", stores)
        horizon_opts = {7: "7 Days (Short-term)", 30: "30 Days (Medium-term)"}
        selected_horizon = st.radio("Forecast Horizon", options=[7, 30], format_func=lambda x: horizon_opts[x])
        
    # Get relevant data
    df_f = forecast_7d if selected_horizon == 7 else forecast_30d
    store_f = df_f[df_f["Store"] == selected_store].copy()
    store_f["Date"] = pd.to_datetime(store_f["Date"])
    
    # Interactive Input Parameters
    st.markdown("### ⚙️ Interactive Inventory Parameters")
    st.markdown("Adjust these parameters to see how they impact inventory recommendations in real-time.")
    
    with st.expander("Show/Hide Input Parameters", expanded=True):
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            lead_time = st.number_input("Supplier Lead Time (Days)", min_value=1, max_value=30, value=7, step=1)
        with col_p2:
            service_level = st.slider("Target Service Level (%)", min_value=80.0, max_value=99.9, value=95.0, step=0.1)
        with col_p3:
            unit_cost = st.number_input("Unit Cost (€)", min_value=1.0, value=15.0, step=1.0)
        with col_p4:
            ordering_cost = st.number_input("Fixed Order Cost (€)", min_value=0.0, value=50.0, step=5.0)
    
    # Recalculate
    holding_cost_rate = 0.25 # fixed assumption
    recs = recalculate_inventory(selected_store, selected_horizon, df_f, lead_time, service_level, unit_cost, ordering_cost, holding_cost_rate)
    
    # Display Results
    st.markdown(f"### 📈 Forecasted Demand & Inventory Plan ({selected_horizon} Days)")
    
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Forecasted Demand", f"€{recs['Total Demand']:,}")
    kpi2.metric("Safety Stock", f"{recs['Safety Stock']:,} units")
    kpi3.metric("Reorder Point", f"{recs['Reorder Point']:,} units")
    kpi4.metric("Economic Order Qty (EOQ)", f"{recs['EOQ (Units)']:,} units")
    
    risk_html = f"<span style='color: {recs['Risk Color']}; font-weight: bold;'>{recs['Risk Level']} ({recs['Stockout Risk']})</span>"
    kpi5.markdown(f"**Stockout Risk:**<br>{risk_html}", unsafe_allow_html=True)
    
    # Plotly interactive chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=store_f["Date"], y=store_f["Forecast_Sales"],
        mode='lines+markers', name='Forecasted Sales',
        line=dict(color='#58a6ff', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=store_f["Date"], y=store_f["Upper_CI_85"],
        mode='lines', line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=store_f["Date"], y=store_f["Lower_CI_85"],
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(88, 166, 255, 0.2)', name='85% Confidence Interval'
    ))
    
    fig.update_layout(
        title=f"Demand Forecast for Store {selected_store}",
        xaxis_title="Date",
        yaxis_title="Sales (€)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# Page: Model Explainability
# ==========================================
elif page == "🧠 Model Explainability":
    st.title("🧠 AI Model Explainability (SHAP)")
    st.markdown("Understand what features are driving the AI's predictions using SHapley Additive exPlanations.")
    
    col_feat1, col_feat2 = st.columns([1, 2])
    with col_feat1:
        st.markdown("#### Top 10 Drivers of Demand")
        st.dataframe(shap_feat.head(10).style.bar(subset=['mean_abs_shap'], color='#58a6ff'))
        
    with col_feat2:
        st.markdown("#### Feature Impact Distribution (Beeswarm)")
        try:
            st.image(Image.open(VIZ_DIR / "18_shap_beeswarm.png"), use_container_width=True)
        except Exception:
            st.info("Visualization not found.")
            
    st.markdown("#### Individual Prediction Explanation (Waterfall)")
    try:
        st.image(Image.open(VIZ_DIR / "19_shap_waterfall_sample.png"), use_container_width=True)
    except Exception:
        st.info("Visualization not found.")

# ==========================================
# Page: EDA
# ==========================================
elif page == "📈 Exploratory Data Analysis":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("Historical insights from the fashion retail dataset.")
    
    plots = [
        ("Sales Trend Over Time", "01_sales_trend.png"),
        ("Monthly Seasonality", "02_monthly_seasonality.png"),
        ("Day of Week Demand", "03_day_of_week_demand.png"),
        ("Sales Distribution", "04_sales_distribution.png"),
        ("Correlation Heatmap", "05_correlation_heatmap.png"),
        ("Promotion Impact", "07_promotion_impact.png")
    ]
    
    for i in range(0, len(plots), 2):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### {plots[i][0]}")
            try:
                st.image(Image.open(VIZ_DIR / plots[i][1]), use_container_width=True)
            except Exception:
                pass
        if i + 1 < len(plots):
            with c2:
                st.markdown(f"#### {plots[i+1][0]}")
                try:
                    st.image(Image.open(VIZ_DIR / plots[i+1][1]), use_container_width=True)
                except Exception:
                    pass
