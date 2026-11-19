# Results

Every model is scored on the same test split (10% of games, chosen by game id) of the Kaggle High-Level 3v3 dataset, 1 million test frames from about 4,000 games, unless noted.

## Who scores next (blue, orange, nobody)

2026-11-19. Log loss and Brier: lower is better.

| Model | Inputs | Log loss | Brier | Accuracy |
|---|---|---|---|---|
| No model (overall rates) | none | 1.006 | 0.613 | 42.8% |
| Ball-position rule | which tenth of the field the ball is in | 1.004 | 0.611 | 45.5% |
| LightGBM | clock and score only | 0.828 | 0.539 | 49.5% |
| LightGBM | all 133 | 0.817 | 0.531 | 52.5% |

The field adds little because most frames are far from the next goal. Picking the scoring team is 98.7% right with the goal under 2 s away, 84.5% at 2 to 5 s, and about 51% beyond 20 s.

## When the next goal comes

2026-11-19. Value target: +0.5^(t/5) for a blue goal in t seconds, negative for orange. "Blue within W" AUC: how well the model ranks frames where blue scores within W seconds above all others.

| Model | Train rows | Value MSE (no model 0.0448) | Value corr | AUC 2 s | AUC 5 s | AUC 10 s | AUC 20 s |
|---|---|---|---|---|---|---|---|
| LightGBM value regression | 4M | 0.0348 | 0.475 | 0.971 | 0.864 | 0.717 | 0.632 |
| LightGBM survival, 2 s slices to 30 s | 2M | 0.0360 | 0.438 | 0.976 | 0.867 | 0.721 | 0.637 |

The survival model stopped after 39 rounds and is underconfident at low probabilities (it says 3.1% where 4.5% happens), so it is undertrained. Top inputs are now ball position and velocity, not the clock.
