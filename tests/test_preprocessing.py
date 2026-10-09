import pandas as pd

from src.preprocessing import add_all_historical_win_features


def test_historical_features_do_not_use_current_result():

    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
        ]),
        "scheduled_time": ["12:00", "12:00", "12:00"],
        "race_id": [1, 2, 3],

        "horse_id": [10, 10, 10],
        "jockey_id": [20, 20, 20],
        "trainer_id": [30, 30, 30],

        "win_flag": [1, 0, 1],
    })

    result = add_all_historical_win_features(df)

    assert result.iloc[0]["horse_prev_races"] == 0
    assert result.iloc[1]["horse_prev_races"] == 1
    assert result.iloc[2]["horse_prev_races"] == 2

    assert result.iloc[0]["horse_prev_wins"] == 0
    assert result.iloc[1]["horse_prev_wins"] == 1
    assert result.iloc[2]["horse_prev_wins"] == 1