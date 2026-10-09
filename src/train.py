from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
# data/processed/
#     ↓
# load training data
#     ↓
# build pipeline
#     ↓
# pipe.fit()
#     ↓
# validation prediction
#     ↓
# race-level normalisation
#     ↓
# evaluate vs market
#     ↓
# save trained pipeline
#         ↓
# models/model.joblib

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

def load_processed_data():

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

    X_val = pd.read_csv(DATA_DIR / "X_val.csv")
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze("columns")

    val_result = pd.read_csv(DATA_DIR / "val_result.csv")

    return X_train, y_train, X_val, y_val, val_result

def build_pipeline():

    pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_iter=100,
                random_state=42
            )
        )
    ])

    return pipe

def normalise_by_race(df, probability_col):

    race_sum = (
        df.groupby("race_id")[probability_col]
        .transform("sum")
    )

    return df[probability_col] / race_sum

def evaluate_model(pipe, X_val, y_val, val_result):

    val_result = val_result.copy()

    # Model probabilities
    val_result["model_prob"] = (
        pipe.predict_proba(X_val)[:, 1]
    )

    val_result["normalised_model_prob"] = (
        normalise_by_race(
            val_result,
            "model_prob"
        )
    )

    # Market-implied probabilities
    val_result["market_prob"] = (
        1 / val_result["win_odds"]
    )

    val_result["normalised_market_prob"] = (
        normalise_by_race(
            val_result,
            "market_prob"
        )
    )

    model_logloss = log_loss(
        y_val,
        val_result["normalised_model_prob"]
    )

    market_logloss = log_loss(
        y_val,
        val_result["normalised_market_prob"]
    )

    return model_logloss, market_logloss, val_result
def save_model(pipe):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "model.joblib"

    joblib.dump(pipe, model_path)

    print(f"Model saved to: {model_path}")
def main():

    print("Loading processed data...")
    X_train, y_train, X_val, y_val, val_result = (
        load_processed_data()
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    print("Building model...")
    pipe = build_pipeline()

    print("Training model...")
    pipe.fit(X_train, y_train)

    print("Evaluating model...")
    model_logloss, market_logloss, val_result = (
        evaluate_model(
            pipe,
            X_val,
            y_val,
            val_result
        )
    )

    print(f"Model log loss:  {model_logloss:.4f}")
    print(f"Market log loss: {market_logloss:.4f}")

    print("Saving model...")
    save_model(pipe)


if __name__ == "__main__":
    main()