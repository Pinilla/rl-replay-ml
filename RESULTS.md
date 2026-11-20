# Results

All numbers below are from the retrain on 2026-11-19 after overtime was marked correctly (it changed them by under 0.001).

Every model is scored on the same test split (10% of games, chosen by game id) of the Kaggle High-Level 3v3 dataset, 1 million test frames from about 4,000 games, unless noted.

## Who scores next (blue, orange, nobody)

2026-11-19. Log loss and Brier: lower is better.

| Model | Inputs | Log loss | Brier | Accuracy |
|---|---|---|---|---|
| No model (overall rates) | none | 1.006 | 0.613 | 42.8% |
| Ball-position rule | which tenth of the field the ball is in | 1.004 | 0.611 | 45.5% |
| LightGBM | clock and score only | 0.828 | 0.539 | 49.5% |
| LightGBM | all 134 | 0.817 | 0.531 | 52.5% |

The field adds little because most frames are far from the next goal. Picking the scoring team is 98.7% right with the goal under 2 s away, 84.5% at 2 to 5 s, and about 51% beyond 20 s.

## When the next goal comes

2026-11-19. Value target: +0.5^(t/5) for a blue goal in t seconds, negative for orange. "Blue within W" AUC: how well the model ranks frames where blue scores within W seconds above all others.

| Model | Train rows | Value MSE (no model 0.0448) | Value corr | AUC 2 s | AUC 5 s | AUC 10 s | AUC 20 s |
|---|---|---|---|---|---|---|---|
| LightGBM value regression | 4M | 0.0348 | 0.475 | 0.971 | 0.864 | 0.717 | 0.632 |
| LightGBM survival, 2 s slices to 30 s | 2M | 0.0360 | 0.438 | 0.976 | 0.867 | 0.721 | 0.637 |
| Neural net (MLP 3 x 256, 174,111 weights), survival | 23.4M | 0.0311 | 0.551 | 0.991 | 0.906 | 0.754 | 0.657 |
| Sequence net (GRU 2 x 256 over last 4 s, 963,615 weights), survival | 23.4M | 0.0335 | 0.500 | 0.985 | 0.891 | 0.740 | 0.649 |

The survival model stopped after 39 rounds and is underconfident at low probabilities (it says 3.1% where 4.5% happens), so it is undertrained. Top inputs are now ball position and velocity, not the clock.

The neural net trained on the desktop RTX 3090 in 5.6 minutes: 30 epochs of about 6 s each, plus 104 s to load the data. Its test log loss over the 31 survival classes is 1.600 (validation 1.605, best at epoch 29 of 30). The LightGBM survival model didn't record a log loss, so compare on value MSE and AUC.

It is well calibrated. For "blue scores within 5 s" it says 3.2% where 5.0% happens, 13.9% where 16.0% happens, and 90.8% where 90.6% happens. LightGBM said 13.7% where 18.2% happened.

Two caveats. It saw 12 times as many training rows as the LightGBM survival model, so part of the gain is data, not the model type. The validation score was still improving slightly when the 30 epochs ran out.

The sequence net (2026-11-19, `models/seq`, 12 minutes on the 3090, about 59 s per epoch) did worse than the MLP. Both were scored on all 2.87M test rows. Its validation score was best after epoch 6 of 30 and then got worse, so early stopping ended the run at epoch 10 while the learning rate was still high. Test log loss was 1.621 for the sequence net and 1.600 for the MLP.

## How often the top guess gets the timing right

All 2.87M test rows. "No goal within 30 s" is the right answer for 67.0% of frames, so always guessing it is 67.0% right. The MLP's top pick is right 68.2% of the time and the sequence net's 67.8%. For frames where a goal does come within 30 s, the table shows how often the top goal slice has the right team and the right 2-second slice, or a slice within one of it.

| Actual time to goal | Share of frames | MLP exact | MLP within one | Sequence exact | Sequence within one |
|---|---|---|---|---|---|
| Under 2 s | 2.4% | 89.4% | 98.5% | 85.1% | 97.7% |
| 2 to 5 s | 3.8% | 55.4% | 84.3% | 50.5% | 80.2% |
| 5 to 10 s | 6.2% | 15.5% | 39.4% | 13.8% | 36.7% |
| 10 to 20 s | 11.1% | 1.5% | 4.5% | 2.0% | 5.1% |
| 20 to 30 s | 9.4% | 0.0% | 0.1% | 0.0% | 0.1% |
