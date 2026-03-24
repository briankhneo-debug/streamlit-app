import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_COLUMNS = [
    "Year", "Month", "Day", "DayOfWeek", "WeekOfYear", "Quarter", "DayOfYear",
    "IsWeekend", "IsMonthStart", "IsMonthEnd", "Month_sin", "Month_cos",
    "DayOfWeek_sin", "DayOfWeek_cos", "Demand_Lag_1", "Demand_Lag_2",
    "Demand_Lag_3", "Demand_Lag_7", "Demand_Lag_14", "Demand_Lag_30",
    "Demand_RollingMean_7", "Demand_RollingStd_7", "Demand_RollingMean_14",
    "Demand_RollingStd_14", "Demand_RollingMean_30", "Demand_RollingStd_30"
]

TARGET_COLUMN = "DailyDemand"

def prepare_product_data(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    product_df = df[df["StockCode"].astype(str) == str(stock_code)].copy()
    product_df["Date"] = pd.to_datetime(product_df["Date"])
    product_df = product_df.sort_values("Date")
    product_df = product_df.dropna(subset=[TARGET_COLUMN] + FEATURE_COLUMNS)
    return product_df

def train_evaluate_model(product_df: pd.DataFrame):
    split_index = max(int(len(product_df) * 0.8), 30)
    train_df = product_df.iloc[:split_index].copy()
    test_df = product_df.iloc[split_index:].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    comparison_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "ActualDemand": y_test.values,
        "PredictedDemand": preds
    })

    feature_importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    })

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return {
        "model": model,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "comparison_df": comparison_df,
        "feature_importance": feature_importance,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }

def build_future_forecast(product_df: pd.DataFrame, model, horizon: int = 7):
    history = product_df.copy().sort_values("Date").reset_index(drop=True)
    rows = []

    for _ in range(horizon):
        last = history.iloc[-1].copy()
        next_date = pd.to_datetime(last["Date"]) + pd.Timedelta(days=1)

        new_row = last.copy()
        new_row["Date"] = next_date
        new_row["Year"] = next_date.year
        new_row["Month"] = next_date.month
        new_row["Day"] = next_date.day
        new_row["DayOfWeek"] = next_date.dayofweek
        new_row["WeekOfYear"] = int(next_date.isocalendar().week)
        new_row["Quarter"] = next_date.quarter
        new_row["DayOfYear"] = next_date.dayofyear
        new_row["IsWeekend"] = 1 if next_date.dayofweek >= 5 else 0
        new_row["IsMonthStart"] = 1 if next_date.day == 1 else 0
        month_end = (next_date + pd.offsets.MonthEnd(0)).day
        new_row["IsMonthEnd"] = 1 if next_date.day == month_end else 0

        import math
        new_row["Month_sin"] = math.sin(2 * math.pi * next_date.month / 12)
        new_row["Month_cos"] = math.cos(2 * math.pi * next_date.month / 12)
        new_row["DayOfWeek_sin"] = math.sin(2 * math.pi * next_date.dayofweek / 7)
        new_row["DayOfWeek_cos"] = math.cos(2 * math.pi * next_date.dayofweek / 7)

        demand_series = history["DailyDemand"].tolist()
        new_row["Demand_Lag_1"] = demand_series[-1]
        new_row["Demand_Lag_2"] = demand_series[-2]
        new_row["Demand_Lag_3"] = demand_series[-3]
        new_row["Demand_Lag_7"] = demand_series[-7]
        new_row["Demand_Lag_14"] = demand_series[-14]
        new_row["Demand_Lag_30"] = demand_series[-30]
        new_row["Demand_RollingMean_7"] = pd.Series(demand_series[-7:]).mean()
        new_row["Demand_RollingStd_7"] = pd.Series(demand_series[-7:]).std()
        new_row["Demand_RollingMean_14"] = pd.Series(demand_series[-14:]).mean()
        new_row["Demand_RollingStd_14"] = pd.Series(demand_series[-14:]).std()
        new_row["Demand_RollingMean_30"] = pd.Series(demand_series[-30:]).mean()
        new_row["Demand_RollingStd_30"] = pd.Series(demand_series[-30:]).std()

        X_next = pd.DataFrame([new_row[FEATURE_COLUMNS]])
        pred = float(model.predict(X_next)[0])
        new_row["DailyDemand"] = max(pred, 0.0)

        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        rows.append({"Date": next_date.date(), "PredictedDemand": round(pred, 3)})

    future_forecast = pd.DataFrame(rows)
    return history, future_forecast
