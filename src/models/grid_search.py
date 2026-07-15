import pandas as pd
import pickle
import yaml
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed_data"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

params = yaml.safe_load(open(ROOT / "params.yaml"))["gridsearch"]

X_train = pd.read_csv(PROC / "X_train_scaled.csv")
y_train = pd.read_csv(PROC / "y_train.csv").values.ravel()

grid = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=params["param_grid"],
    cv=params["cv"],
    scoring="r2",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

with open(MODELS / "best_params.pkl", "wb") as f:
    pickle.dump(grid.best_params_, f)

print("Best params ->", grid.best_params_)
