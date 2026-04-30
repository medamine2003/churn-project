from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Churn Prediction API")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

VALID_MODELS = ["random_forest", "logistic", "xgboost", "mlp"]

class ClientData(BaseModel):
    model_name:             str   = "random_forest"
    gender:                 int
    age:                    float
    country:                int
    city:                   int
    customer_segment:       int
    tenure_months:          float
    signup_channel:         int
    contract_type:          int
    monthly_logins:         float
    weekly_active_days:     float
    avg_session_time:       float
    features_used:          float
    usage_growth_rate:      float
    last_login_days_ago:    float
    monthly_fee:            float
    total_revenue:          float
    payment_method:         int
    payment_failures:       int
    discount_applied:       float
    price_increase_last_3m: int
    support_tickets:        int
    avg_resolution_time:    float
    complaint_type:         int
    csat_score:             float
    escalations:            int
    email_open_rate:        float
    marketing_click_rate:   float
    nps_score:              float
    survey_response:        int
    referral_count:         int

@app.get("/health")
def health():
    return {"status": "ok", "models_available": VALID_MODELS}

@app.post("/predict")
def predict(data: ClientData):
    if data.model_name not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Modèle invalide. Choisir parmi : {VALID_MODELS}")

    try:
        model = joblib.load(os.path.join(MODELS_DIR, f"{data.model_name}.pkl"))

        input_data = data.dict()
        input_data.pop("model_name")

        df        = pd.DataFrame([input_data])
        df_scaled = scaler.transform(df)
        proba     = model.predict_proba(df_scaled)[0][1]
        churn     = int(proba > 0.5)

        return {
            "model_used":         data.model_name,
            "churn":              churn,
            "churn_probability":  round(float(proba), 4),
            "risk_level":         "Haut" if proba > 0.7 else "Moyen" if proba > 0.4 else "Faible"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))