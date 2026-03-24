import pandas as pd

def create_results_table(stock_code, product_name, current_stock, lead_time, horizon, avg_forecast,
                         safety_stock, reorder_point, decision, score, model_results):
    return pd.DataFrame([{
        "StockCode": stock_code,
        "ProductName": product_name,
        "CurrentStock": round(current_stock, 2),
        "LeadTimeDays": lead_time,
        "ForecastHorizonDays": horizon,
        "AverageForecastDemand": round(avg_forecast, 3),
        "SafetyStock": round(safety_stock, 3),
        "ReorderPoint": round(reorder_point, 3),
        "Decision": decision,
        "FuzzyRiskScore": round(score, 3),
        "RMSE": round(model_results["rmse"], 3),
        "MAE": round(model_results["mae"], 3),
        "R2": round(model_results["r2"], 3),
        "TrainRows": model_results["train_rows"],
        "TestRows": model_results["test_rows"],
    }])
