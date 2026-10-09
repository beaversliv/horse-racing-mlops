import numpy as np
import pandas as pd

from src.train import build_pipeline
from src.train import normalise_by_race

def test_pipeline_can_fit_and_predict():
    '''Test that the pipeline can fit and predict on a small dataset.'''
    X = pd.DataFrame({
        "age": [3, 4, 5, 6, 7, 8],
        "win_odds": [2.0, 3.5, 5.0, 8.0, 4.0, 10.0],
        "weight_pounds": [120, 125, 130, 128, 126, 132],
    })

    y = pd.Series([1, 0, 0, 0, 1, 0])

    pipe = build_pipeline()

    pipe.fit(X, y)

    probabilities = pipe.predict_proba(X)[:, 1]

    assert len(probabilities) == len(X)

def test_predictions_are_probabilities():

    X = pd.DataFrame({
        "age": [3, 4, 5, 6, 7, 8],
        "win_odds": [2.0, 3.5, 5.0, 8.0, 4.0, 10.0],
        "weight_pounds": [120, 125, 130, 128, 126, 132],
    })

    y = pd.Series([1, 0, 0, 0, 1, 0])

    pipe = build_pipeline()
    pipe.fit(X, y)

    probabilities = pipe.predict_proba(X)[:, 1]

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

def test_race_probabilities_sum_to_one():

    df = pd.DataFrame({
        "race_id": [1, 1, 1, 2, 2],
        "model_prob": [0.2, 0.3, 0.4, 0.2, 0.6],
    })

    df["normalised_prob"] = normalise_by_race(
        df,
        "model_prob"
    )

    race_sums = (
        df.groupby("race_id")["normalised_prob"]
        .sum()
    )

    assert np.allclose(race_sums.values, 1.0)