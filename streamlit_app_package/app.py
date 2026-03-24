import streamlit as st
import pandas as pd
from pathlib import Path

# Import your modules
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

# ✅ FIXED PATH (important)
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "data_test.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

# Load data
df = load_data()

# Sidebar
st.sidebar.header("System Controls")
product_options = df["StockCode"].dropna().astype(str).unique().tolist()
selected_product = st.sidebar.selectbox("Select Product", product_options)

# Filter data
product_df = df[df["StockCode"].astype(str) == selected_product]

# Prepare data
prepared_df = prepare_product_data(product_df)

# Train model
model, metrics = train_evaluate_model(prepared_df)

# Forecast
forecast_df = build_future_forecast(model, prepared_df)

# Inventory calculations
avg_demand = prepared_df["Quantity"].mean()
std_demand = prepared_df["Quantity"].std()

safety_stock = calculate_safety_stock(avg_demand, std_demand)
reorder_point = calculate_reorder_point(avg_demand, safety_stock)
stock_status = classify_stock_status(avg_demand, reorder_point)

# Fuzzy logic reorder decision
reorder_decision = fuzzy_reorder(avg_demand, std_demand)

# Results
st.subheader("📊 Forecast Results")
st.dataframe(forecast_df)

st.subheader("📦 Inventory Insights")
st.write(f"Safety Stock: {safety_stock}")
st.write(f"Reorder Point: {reorder_point}")
st.write(f"Stock Status: {stock_status}")
st.write(f"Fuzzy Reorder Decision: {reorder_decision}")

# Export
results_table = create_results_table(forecast_df)
st.download_button(
    label="Download Results",
    data=results_table.to_csv(index=False),
    file_name="forecast_results.csv",
    mime="text/csv",
)
