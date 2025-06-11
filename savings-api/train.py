# train.py
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ─── 1) Load raw data ──────────────────────────────────────────────────────────
df = pd.read_csv("data.csv")

# ─── 2) Define features & targets ─────────────────────────────────────────────
variable_expenses = [
    'Groceries', 'Transport', 'Eating_Out', 'Entertainment',
    'Utilities', 'Healthcare', 'Education', 'Miscellaneous'
]
target_columns = [f"Potential_Savings_{cat}" for cat in variable_expenses]

# ─── 3) Derived features ───────────────────────────────────────────────────────
df['Disposable_Income'] = (
    df['Income'] - df[['Rent','Loan_Repayment','Insurance'] + variable_expenses].sum(axis=1)
)
df['Desired_Savings_Percentage'] = (df['Desired_Savings'] / df['Income']) * 100

# ─── 4) Build X, y ─────────────────────────────────────────────────────────────
numerical_features = [
    'Income','Age','Dependents','Rent','Loan_Repayment','Insurance',
    'Disposable_Income','Desired_Savings','Desired_Savings_Percentage'
] + variable_expenses
categorical_features = ['Occupation','City_Tier']

# One-hot encode categorical
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_cat = pd.DataFrame(
    encoder.fit_transform(df[categorical_features]),
    columns=encoder.get_feature_names_out(categorical_features),
    index=df.index
)
X_num = df[numerical_features]
X = pd.concat([X_num, X_cat], axis=1)
y = df[target_columns]

# Save feature order
joblib.dump(X.columns.tolist(), "feature_order.joblib")

# Scale numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Train/test split & model fitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ─── 5) Dump all artifacts ────────────────────────────────────────────────────
joblib.dump(model, "savings_predictor_forest.joblib")
joblib.dump(encoder, "encoder.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(numerical_features, "numerical_features.joblib")
joblib.dump(categorical_features, "categorical_features.joblib")

print("✅ Training complete; artifacts saved.")
