import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed_data"
MODELS = ROOT / "models"
DATA = ROOT / "data"
METRICS = ROOT / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

X_test = pd.read_csv(PROC / "X_test_scaled.csv")
y_test = pd.read_csv(PROC / "y_test.csv").values.ravel()

with open(MODELS / "gbr_model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

pred = X_test.copy()
pred["silica_concentrate_true"] = y_test
pred["silica_concentrate_pred"] = y_pred
pred.to_csv(DATA / "prediction.csv", index=False)

scores = {
    "mse": float(mean_squared_error(y_test, y_pred)),
    "r2": float(r2_score(y_test, y_pred)),
}
with open(METRICS / "scores.json", "w") as f:
    json.dump(scores, f, indent=4)

print("Evaluation done ->", scores)
