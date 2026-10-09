# Horse Racing Win-Probability Modelling

## Project overview

This project develops machine-learning models to estimate the probability that each horse will win a race. The data are time-ordered and tabular: each row represents one runner in one race, and multiple rows belonging to the same race form a group of mutually competing outcomes.

The main aims were to:

1. establish interpretable and non-linear modelling baselines;
2. measure the predictive value of historical horse, jockey and trainer performance;
3. compare model probabilities with market-implied probabilities; and
4. investigate whether residual learning could capture signals not already reflected in the betting market.

## Prediction target and evaluation

The target is `win_flag`:

- `1` if the runner won the race;
- `0` otherwise.

Models output a win probability for every runner. Performance is evaluated using binary log loss:

```text
log loss = -mean(y * log(p) + (1 - y) * log(1 - p))
```

Lower log loss indicates better probability estimates. This metric penalises confident incorrect predictions more heavily than uncertain predictions and therefore assesses both discrimination and probability calibration.

Because the observations are chronological, the supplied time-based split is used rather than a random train-validation split. This more closely represents the real task of training on past races and predicting future races.

## Features

### Race and runner features

The numeric inputs included:

- prize and race information: `added_money`, `distance_yards`, `num_places`, `num_runners`;
- runner information: `age`, `overweight`, `stall`, `weight_pounds`;
- market information: `place_odds`, `win_odds`.

Numeric features were standardised for logistic regression. XGBoost does not require feature scaling, but the same cleaned feature set was used to keep comparisons consistent.

### Historical performance features

The current race's `win_flag` must not be supplied directly to a model because the result is unknown before the race. However, outcomes from earlier races can be used to describe past performance.

For each runner, jockey and trainer, the following features were constructed:

- number of previous races;
- number of previous wins;
- historical win rate.

For an entity \(e\) before race \(t\):

```text
previous win rate(e, t) = previous wins(e, t) / previous races(e, t)
```

The implementation sorts rows chronologically and subtracts the current `win_flag` from the cumulative win count, preventing the current row's result from being included in its own features. Adding these historical outcome features reduced validation log loss by approximately **0.02**.

## Models

### Logistic regression

Logistic regression was used as an interpretable baseline. It estimates a linear relationship between the input features and the log-odds of winning.

### XGBoost

XGBoost was used to capture non-linear relationships and interactions among race characteristics, historical performance and market odds. It produced the strongest individual machine-learning result.

### XGBoost ensemble

Multiple XGBoost predictions were averaged to test whether ensembling improved generalisation. In this experiment, the ensemble did not outperform the best individual XGBoost model, possibly because the component models were too similar for averaging to provide useful variance reduction.

## Market benchmark

Decimal win odds were converted to implied probabilities. When required, raw implied probabilities can be normalised within each race to remove the bookmaker overround:

```text
raw market probability = 1 / decimal odds

normalised market probability_i =
    raw market probability_i / sum(raw market probabilities in the race)
```

The market probability provides a strong benchmark because the odds aggregate substantial public and bookmaker information.

## Residual learning

The direct models came close to, but did not improve upon, the market benchmark. A residual-learning experiment was therefore used to treat the market prediction as the starting point rather than replacing it.

Conceptually, the correction is:

```text
final probability = market probability + alpha * predicted residual
```

where XGBoost learns a systematic residual signal from the non-market and historical features, and `alpha` controls how much of the learned correction is added to the market probability. The adjusted predictions must be clipped to a valid probability range and, where appropriate, renormalised within each race.

This experiment tests whether horse, jockey, trainer and race information contains incremental predictive value not already captured by market odds. It should not be interpreted as beating the market unless the corrected probabilities achieve a log loss below the market benchmark on unseen, chronologically later races.

## Results

| Method | Validation log loss |
| --- | ---: |
| Market-implied probability | **0.2868** |
| XGBoost | 0.2871-0.2878* |
| XGBoost ensemble | 0.2883 |
| Logistic regression | 0.2930 |

\*Two recorded XGBoost runs produced 0.2871 and 0.2878. The exact configuration and random seed should be recorded alongside each result before selecting the final reported score.

The market-implied probability achieved the lowest log loss. The best recorded XGBoost result of 0.2871 was close to the benchmark, while the ensemble slightly worsened performance. These results illustrate both the strength of the betting market and the importance of calibration when predicting probabilities.

## Key findings

- Historical performance features provided meaningful predictive value, reducing log loss by about 0.02.
- XGBoost outperformed logistic regression, indicating that non-linear effects and feature interactions were useful.
- Market odds remained the strongest standalone predictor in the recorded experiments.
- Averaging similar XGBoost models did not automatically improve probability quality.
- Residual learning offered a principled way to test for incremental signal relative to the market.

## Limitations and improvements

### Preventing temporal leakage

All outcome-derived features must use information available strictly before the race being predicted. Two implementation details require particular care:

1. A fallback global win rate calculated from the complete dataframe would include future outcomes. It should instead be estimated from the training period or calculated as a shifted expanding prior.
2. If one trainer has multiple runners in the same race, row-wise cumulative calculations could allow a later row to use the result of another runner from that race. Historical statistics should therefore be updated between races, not between rows within the same race.

### Probability calibration

Calibration plots, calibration error and post-hoc methods such as isotonic regression could complement log loss and help diagnose whether a model is systematically overconfident or underconfident.

### Race-level probability constraints

Since each race has one winner, runner probabilities should form a coherent race-level distribution and sum to one. Future work could compare independent binary models with grouped alternatives such as softmax or ranking-based models.

### Reproducibility and engineering

Further development should separate feature generation, training and evaluation into tested modules; define configuration files for feature sets and model parameters; fix random seeds; log every experiment; and add unit tests that explicitly check for temporal leakage.

## Technology

- Python
- pandas and NumPy
- scikit-learn
- XGBoost

## Summary

This project demonstrates an end-to-end approach to probabilistic modelling on time-dependent tabular data. It combines chronological validation, historical feature engineering, interpretable and tree-based models, market benchmarking, ensembling and residual correction. Although the market remained marginally stronger, the experiments quantified the contribution of historical information and identified clear directions for improving model calibration, validation and production readiness.
