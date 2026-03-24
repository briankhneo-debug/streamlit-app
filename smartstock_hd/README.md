# HD Smart Stock System

This is a built-in Streamlit inventory intelligence system designed for higher-grade assignment submission.

## Core AI Subsystems
1. **Random Forest Demand Forecasting**
   - Predicts future daily demand using historical demand, lag features, rolling statistics, and calendar features.

2. **Fuzzy Logic Reorder Intelligence**
   - Converts current stock, forecast demand, lead time, safety stock, and reorder point into a practical reorder decision.

## Included Features
- Built-in dataset only (`data/data_test.csv`)
- Product selection by `StockCode`
- Forecast horizon control
- Model evaluation metrics: RMSE, MAE, R²
- Actual vs predicted demand visualization
- Feature importance visualization
- Safety stock and reorder point calculations
- CSV export of final results

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
