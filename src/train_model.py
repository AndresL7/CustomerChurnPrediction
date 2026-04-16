import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocess
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle imbalance
ratio = (y_train == 0).sum() / (y_train == 1).sum()

# Train model
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=ratio,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/xgb_model.pkl")

print("Model trained and saved!")