import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
params = yaml.safe_load(open(ROOT / "params.yaml"))["split"]

RAW = ROOT / "data" / "raw_data" / "raw.csv"
OUT = ROOT / "data" / "processed_data"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)

# Drop a leading non-feature column if present (date / index)
for col in ["date", "Unnamed: 0"]:
    if col in df.columns:
        df = df.drop(columns=[col])

target = params["target"]
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["test_size"],
    random_state=params["random_state"],
)

X_train.to_csv(OUT / "X_train.csv", index=False)
X_test.to_csv(OUT / "X_test.csv", index=False)
y_train.to_csv(OUT / "y_train.csv", index=False)
y_test.to_csv(OUT / "y_test.csv", index=False)

print("Split done ->", X_train.shape, X_test.shape)
