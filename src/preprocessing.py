import numpy as np
import pandas as pd
from pathlib import Path


TARGET = "win_flag"

VAL_CUTOFF_DATE = pd.Timestamp("2023-08-01")

BASE_NUMERIC_FEATURES = [
    "added_money",
    "age",
    "distance_yards",
    "num_places",
    "num_runners",
    "overweight",
    "place_odds",
    "stall",
    "weight_pounds",
    "win_odds",
]

HISTORICAL_FEATURES = [
    "jockey_prev_races",
    "jockey_prev_wins",
    "jockey_win_rate",
    "trainer_prev_races",
    "trainer_prev_wins",
    "trainer_win_rate",
]

def load_data(path):
    df = pd.read_csv(path)
    return df

def split_data(df,VAL_CUTOFF_DATE = np.datetime64('2023-08-01').astype('datetime64[ns]')):
    train_df = df[df['date'].astype('datetime64[ns]') < VAL_CUTOFF_DATE]
    val_df = df[df['date'].astype('datetime64[ns]') >= VAL_CUTOFF_DATE]
    return train_df, val_df


def add_all_historical_win_features(df):
    # add win flag as historical features
    df = (
        df.sort_values(
            ["date", "scheduled_time", "race_id"]
        )
        .copy()
    )

    # Expanding prior from completed races only. All runners in the same
    # race receive the same prior, so the current result cannot leak in.
    race_history = (
        df.groupby("race_id", sort=False)["win_flag"]
        .agg(total_wins="sum", total_races="size")
    )
    race_history["wins_so_far"] = (
        race_history["total_wins"].cumsum().shift(1)
    )
    race_history["races_so_far"] = (
        race_history["total_races"].cumsum().shift(1)
    )
    race_history["global_win_rate"] = (
        race_history["wins_so_far"]
        / race_history["races_so_far"]
    )
    global_win_rate = df["race_id"].map(
        race_history["global_win_rate"]
    )

    entities = {
        "horse_id": "horse",
        "jockey_id": "jockey",
        "trainer_id": "trainer"
    }

    for id_col, prefix in entities.items():

        df[f"{prefix}_prev_races"] = (
            df.groupby(id_col).cumcount()
        )

        df[f"{prefix}_prev_wins"] = (
            df.groupby(id_col)["win_flag"].cumsum()
            - df["win_flag"]
        )

        df[f"{prefix}_win_rate"] = (
            df[f"{prefix}_prev_wins"]
            / df[f"{prefix}_prev_races"].replace(0, np.nan)
        )

        df[f"{prefix}_win_rate"] = (
            df[f"{prefix}_win_rate"]
            .fillna(global_win_rate)
        )

    return df


def construct_features(with_hist=True):
    features = BASE_NUMERIC_FEATURES.copy()

    if with_hist:
        features.extend(HISTORICAL_FEATURES)

    return features


def prepare_training_data(df, with_hist=True):
    df = df.copy()

    if with_hist:
        df = add_all_historical_win_features(df)

    train_df, val_df = split_data(df)
    features = construct_features(with_hist)

    X_train = train_df[features].copy()
    y_train = train_df[TARGET].copy()

    X_val = val_df[features].copy()
    y_val = val_df[TARGET].copy()

    # Keep only what market evaluation needs.
    val_result = val_df[["race_id", "win_odds"]].copy()

    return X_train, y_train, X_val, y_val, val_result

def save_processed_data(
    X_train,
    y_train,
    X_val,
    y_val,
    val_result,
    output_dir,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)

    X_val.to_csv(output_dir / "X_val.csv", index=False)
    y_val.to_csv(output_dir / "y_val.csv", index=False)

    val_result.to_csv(
        output_dir / "val_result.csv",
        index=False,
    )

def main():

    input_path = "data/raw/train.csv"
    output_dir = "data/processed"

    print("Loading raw data...")
    df = load_data(input_path)

    print(f"Raw data shape: {df.shape}")

    print("Preparing training data...")
    X_train, y_train, X_val, y_val, val_result = (
        prepare_training_data(df)
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    print("Saving processed data...")
    save_processed_data(
        X_train,
        y_train,
        X_val,
        y_val,
        val_result,
        output_dir,
    )

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()