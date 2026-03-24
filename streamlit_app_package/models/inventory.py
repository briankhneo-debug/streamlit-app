from statistics import NormalDist

def calculate_safety_stock(demand_std: float, lead_time: int, service_level: float = 0.95) -> float:
    z_score = NormalDist().inv_cdf(service_level)
    return max(z_score * demand_std * (lead_time ** 0.5), 0.0)

def calculate_reorder_point(avg_daily_demand: float, lead_time: int, safety_stock: float) -> float:
    return max((avg_daily_demand * lead_time) + safety_stock, 0.0)

def classify_stock_status(current_stock: float, reorder_point: float, safety_stock: float) -> str:
    if current_stock < reorder_point * 0.75:
        return "Critical"
    if current_stock < reorder_point:
        return "Low"
    if current_stock <= reorder_point + (2 * safety_stock):
        return "Balanced"
    return "Overstock"
