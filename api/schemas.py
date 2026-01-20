from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    avg_order_value: float
    invoice_count: int
    total_quantity: int
    tenure_days: int

class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    threshold: float

class CLVResponse(BaseModel):
    raw_clv: float
    churn_probability: float
    final_clv: float
