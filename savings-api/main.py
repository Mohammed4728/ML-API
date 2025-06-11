# main.py
import os
import math
import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Load trained artifacts ─────────────────────────────────────────────────
model = joblib.load("savings_predictor_forest.joblib")
encoder = joblib.load("encoder.joblib")
scaler = joblib.load("scaler.joblib")
feature_order = joblib.load("feature_order.joblib")
numerical_features = joblib.load("numerical_features.joblib")
categorical_features = joblib.load("categorical_features.joblib")

# ─── Define request/response schemas ──────────────────────────────────────────
class PlanRequest(BaseModel):
    age: int
    dependents: int
    occupation: str
    city_tier: str
    goalAmount: float
    income: float
    rent: float
    loanPayment: float
    insurance: float
    groceries: float
    transport: float
    eatingOut: float
    education: float
    entertainment: float
    utilities: float
    healthcare: float
    otherMoney: float

class PlanResponse(BaseModel):
    groceriesSavings: float
    transportSavings: float
    eatingOutSavings: float
    entertainmentSavings: float
    utilitiesSavings: float
    healthcareSavings: float
    educationSavings: float
    otherMoneySavings: float
    endDate: int

# ─── FastAPI app setup ────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Core prediction logic ────────────────────────────────────────────────────
def predict_plan(p: PlanRequest) -> PlanResponse:
    # Assemble inputs
    d = {
        "Income": p.income,
        "Age": p.age,
        "Dependents": p.dependents,
        "Rent": p.rent,
        "Loan_Repayment": p.loanPayment,
        "Insurance": p.insurance,
        "Groceries": p.groceries,
        "Transport": p.transport,
        "Eating_Out": p.eatingOut,
        "Entertainment": p.entertainment,
        "Utilities": p.utilities,
        "Healthcare": p.healthcare,
        "Education": p.education,
        "Miscellaneous": p.otherMoney,
        "Desired_Savings": p.goalAmount,
        "Occupation": p.occupation,
        "City_Tier": p.city_tier
    }
    # Derived
    total_fixed = sum(d[k] for k in ["Rent","Loan_Repayment","Insurance"])
    total_var   = sum(d[k] for k in [
        "Groceries","Transport","Eating_Out","Entertainment",
        "Utilities","Healthcare","Education","Miscellaneous"
    ])
    d["Disposable_Income"] = d["Income"] - (total_fixed + total_var)
    d["Desired_Savings_Percentage"] = (
        (d["Desired_Savings"] / d["Income"]) * 100
        if d["Income"] else 0.0
    )

    # DataFrame & preprocessing
    df = pd.DataFrame([d])
    num_scaled  = scaler.transform(df[numerical_features])
    cat_encoded = encoder.transform(df[categorical_features])
    X           = np.concatenate([num_scaled, cat_encoded], axis=1)
    X           = pd.DataFrame(X, columns=feature_order)

    # Predict
    preds = model.predict(X.to_numpy())[0]

    # Zero-adjust
    zero_map = {
        0: p.groceries,   1: p.transport,   2: p.eatingOut,
        3: p.entertainment, 4: p.utilities, 5: p.healthcare,
        6: p.education,   7: p.otherMoney
    }
    for idx, spent in zero_map.items():
        if spent == 0:
            preds[idx] = 0.0

    total_sav = preds.sum()
    months    = math.ceil(p.goalAmount / total_sav) if total_sav else 0

    return PlanResponse(
        groceriesSavings=float(preds[0]),
        transportSavings=float(preds[1]),
        eatingOutSavings=float(preds[2]),
        entertainmentSavings=float(preds[3]),
        utilitiesSavings=float(preds[4]),
        healthcareSavings=float(preds[5]),
        educationSavings=float(preds[6]),
        otherMoneySavings=float(preds[7]),
        endDate=months
    )

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/receive-data")
async def receive_data(req: Request):
    payload = await req.json()
    try:
        plan = PlanRequest(**payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    app.state.latest_plan = plan
    return {"status": "success", "message": "Data received"}

@app.get("/send-data", response_model=PlanResponse)
async def send_data():
    plan = getattr(app.state, "latest_plan", None)
    if not plan:
        raise HTTPException(status_code=404, detail="No data received")
    return predict_plan(plan)

# ─── Uvicorn runner for local/Procfile use ────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
