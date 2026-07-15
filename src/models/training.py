import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed_data"
MODELS = ROOT / "models"

X_train = pd.read_csv(PROC / "X_train_scaled.csv")
y_train = pd.read_csv(PROC / "y_train.csv").values.ravel()

with open(MODELS / "best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

model = GradientBoostingRegressor(random_state=42, **best_params)
model.fit(X_train, y_train)

with open(MODELS / "gbr_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved ->", best_params)
