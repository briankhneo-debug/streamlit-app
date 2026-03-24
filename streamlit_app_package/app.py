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

st.set_page_config(
    page_title="HD Smart Stock System",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Sidebar controls
st.sidebar.header("System Controls")

product_options = df["StockCode"].dropna().astype(str).unique().tolist()
product_options = sorted(product_options)
selected_code = st.sidebar.selectbox("Select Product (StockCode)", product_options, index=0)

product_names = (
    df.loc[df["StockCode"].astype(str) == selected_code, "ProductName"]
    .dropna()
    .unique()
    .tolist()
)
product_name = product_names[0] if product_names else "Unknown Product"

forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 3, 30, 7)
lead_time = st.sidebar.slider("Supplier Lead Time (days)", 1, 30, 7)
current_stock = st.sidebar.number_input(
    "Current Stock On Hand",
    min_value=0,
    max_value=100000,
    value=100,
    step=10,
)
service_level = st.sidebar.selectbox("Service Level", [0.90, 0.95, 0.99], index=1)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select View",
    ["Overview", "Forecast Model", "Inventory Decision", "Results Export"],
    index=0,
)

# Core processing
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
decision, score = fuzzy_reorder(
    current_stock, avg_forecast, lead_time, safety_stock, reorder_point
)
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
    results,
)

# Page rendering
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Product", selected_code)
    c2.metric("Rows Used", len(product_df))
    c3.metric("Average Daily Demand", f"{product_df['DailyDemand'].mean():.2f}")
    c4.metric("Current Stock", f"{current_stock}")

    st.subheader(f"Selected Product: {product_name}")
    trend_df = product_df[["Date", "DailyDemand"]].set_index("Date")
    st.line_chart(trend_df)

    st.subheader("Dataset Preview")
    preview_cols = ["Date", "StockCode", "ProductName", "DailyDemand"] + FEATURE_COLUMNS[:6]
    st.dataframe(product_df[preview_cols].tail(15), use_container_width=True)

elif page == "Forecast Model":
    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{results['rmse']:.3f}")
    m2.metric("MAE", f"{results['mae']:.3f}")
    m3.metric("R²", f"{results['r2']:.3f}")

    st.subheader("Actual vs Predicted (Test Set)")
    compare_df = results["comparison_df"].set_index("Date")
    st.line_chart(compare_df[["ActualDemand", "PredictedDemand"]])

    st.subheader("Next Period Forecast")
    st.dataframe(future_forecast, use_container_width=True)
    st.line_chart(future_forecast.set_index("Date")[["PredictedDemand"]])

    st.subheader("Feature Importance")
    fi = results["feature_importance"].sort_values("Importance", ascending=False)
    st.bar_chart(fi.set_index("Feature").head(12))

    with st.expander("Model Notes"):
        st.write(
            """
            - Model: RandomForestRegressor
            - Split strategy: chronological train/test split
            - Inputs: lag demand, rolling averages, calendar features
            - Output: DailyDemand forecast
            """
        )

elif page == "Inventory Decision":
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Avg Forecast Demand", f"{avg_forecast:.2f}")
    a2.metric("Safety Stock", f"{safety_stock:.2f}")
    a3.metric("Reorder Point", f"{reorder_point:.2f}")
    a4.metric("Fuzzy Risk Score", f"{score:.2f}")

    st.subheader("Decision Summary")
    st.success(decision)

    if stock_status == "Critical":
        st.error("Critical stock status: current stock is below the reorder point and close to stockout risk.")
    elif stock_status == "Low":
        st.warning("Low stock status: monitor closely and consider replenishment.")
    elif stock_status == "Balanced":
        st.info("Balanced stock status: stock level is within a healthy operating range.")
    else:
        st.warning("Possible overstock: current stock is significantly above projected demand.")

    st.subheader("Inventory Rules")
    st.write(f"- Lead time demand estimate: **{avg_forecast * lead_time:.2f}**")
    st.write(f"- Safety stock buffer: **{safety_stock:.2f}**")
    st.write(f"- Reorder point: **{reorder_point:.2f}**")
    st.write(f"- Current stock: **{current_stock:.2f}**")

    st.subheader("Manager Interpretation")
    st.write(
        "The Random Forest subsystem predicts future demand using historical patterns and engineered time-series features. "
        "The Fuzzy Logic subsystem converts forecast demand, current stock, lead time, and safety stock into a practical reorder decision for operations."
    )

else:  # Results Export
    st.subheader("HD Results Table")
    st.dataframe(summary_results, use_container_width=True)

    csv_bytes = summary_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results as CSV",
        data=csv_bytes,
        file_name=f"smartstock_results_{selected_code}.csv",
        mime="text/csv",
    )

    st.subheader("Included Dataset")
    st.write("This system runs directly from the built-in dataset: `data/data_test.csv`")
