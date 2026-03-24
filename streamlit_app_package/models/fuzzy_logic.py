def fuzzy_reorder(current_stock, avg_forecast, lead_time, safety_stock, reorder_point):
    demand_pressure = (avg_forecast * lead_time) / max(current_stock, 1)
    stock_gap = reorder_point - current_stock

    if current_stock <= reorder_point * 0.5:
        membership_high = 1.0
    elif current_stock <= reorder_point:
        membership_high = 0.8
    elif current_stock <= reorder_point + safety_stock:
        membership_high = 0.5
    else:
        membership_high = 0.1

    if demand_pressure >= 1.5:
        membership_demand = 1.0
    elif demand_pressure >= 1.0:
        membership_demand = 0.7
    elif demand_pressure >= 0.6:
        membership_demand = 0.4
    else:
        membership_demand = 0.1

    if stock_gap > safety_stock:
        membership_gap = 1.0
    elif stock_gap > 0:
        membership_gap = 0.7
    elif stock_gap > -safety_stock:
        membership_gap = 0.4
    else:
        membership_gap = 0.1

    score = (membership_high * 0.45) + (membership_demand * 0.35) + (membership_gap * 0.20)

    if score >= 0.75:
        decision = "HIGH PRIORITY: Reorder immediately to avoid stockout."
    elif score >= 0.50:
        decision = "MEDIUM PRIORITY: Reorder soon and monitor stock daily."
    else:
        decision = "LOW PRIORITY: Stock is currently sufficient."

    return decision, score
