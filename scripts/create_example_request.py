import json
import joblib
import pandas as pd

pipe = joblib.load("models/model.joblib")
features = list(pipe.feature_names_in_)

X_val = pd.read_csv("data/processed/X_val.csv")
val_result = pd.read_csv("data/processed/val_result.csv")

race_id = val_result["race_id"].iloc[0]
mask = val_result["race_id"] == race_id

X_race = X_val.loc[mask].reset_index(drop=True)

runners = []

for i, (_, row) in enumerate(X_race.iterrows()):
    runners.append({
        "runner_id": f"runner_{i + 1}",
        "features": {
            col: None if pd.isna(row[col]) else float(row[col])
            for col in features
        }
    })

request = {
    "race_id": str(race_id),
    "runners": runners
}

with open("scripts/example_request.json", "w") as f:
    json.dump(request, f, indent=2)

print("Example request saved!")