import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed_data"

X_train = pd.read_csv(PROC / "X_train.csv")
X_test = pd.read_csv(PROC / "X_test.csv")

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

X_train_scaled.to_csv(PROC / "X_train_scaled.csv", index=False)
X_test_scaled.to_csv(PROC / "X_test_scaled.csv", index=False)

print("Normalization done")
