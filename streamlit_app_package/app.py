import streamlit as st
import pandas as pd
from pathlib import Path

from models.forecast_model import (
    FEATURE_COLUMNS,
    prepare_product_data,
    train_evaluate_model,
    build_future_forecast,
)
from models.fuzzy_logic import fuzzy_reorder
from models.inventory import (
    calculate_safety_stock,
    calculate_reorder_point,
    classify_stock_status,
)
from utils.exporter import create_results_table

st.set_page_config(page_title="HD Smart Stock System", layout="wide")
st.title("📦 HD Smart Stock System")
st.caption("Two Core AI Subsystems: Random Forest Demand Forecasting + Fuzzy Logic Reorder Intelligence")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "data_test.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

st.sidebar.header("System Controls")
product_options = df["StockCode"].dropna().astype(str).unique().tolist()
product_options = sorted(product_options)
selected_code = st.sidebar.selectbox("Select Product (StockCode)", product_options, index=0)

product_names = df.loc[df["StockCode"].astype(str) == selected_code, "ProductName"].dropna().unique().tolist()
product_name = product_names[0] if product_names else "Unknown Product"

forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 3, 30, 7)
lead_time = st.sidebar.slider("Supplier Lead Time (days)", 1, 30, 7)
current_stock = st.sidebar.number_input("Current Stock On Hand", min_value=0, max_value=100000, value=100, step=10)
service_level = st.sidebar.selectbox("Service Level", [0.90, 0.95, 0.99], index=1)

# correct call
product_df = prepare_product_data(df, selected_code)

if product_df.empty or len(product_df) < 40:
    st.error("Selected product does not have enough clean records for modelling.")
    st.stop()

results = train_evaluate_model(product_df)
future_df, future_forecast = build_future_forecast(product_df, results["model"], forecast_horizon)

avg_forecast = float(future_forecast["PredictedDemand"].mean())
demand_std = float(product_df["DailyDemand"].std())
safety_stock = calculate_safety_stock(demand_std, lead_time, service_level)
reorder_point = calculate_reorder_point(avg_forecast, lead_time, safety_stock)
decision, score = fuzzy_reorder(current_stock, avg_forecast, lead_time, safety_stock, reorder_point)
stock_status = classify_stock_status(current_stock, reorder_point, safety_stock)

summary_results = create_results_table(
    selected_code,
    product_name,
    current_stock,
    lead_time,
    forecast_horizon,
    avg_forecast,
    safety_stock,
    reorder_point,
    decision,
    score,
    results
)

st.subheader("HD Results Table")
st.dataframe(summary_results, use_container_width=True)
