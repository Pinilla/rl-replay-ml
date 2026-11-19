# rl-replay-ml

The owner's project for learning to train local models, using Rocket League replays. The owner is new to machine learning (has watched 3Blue1Brown's neural network series), so explain terms the first time they come up and say what each number means.

Two projects share the code:

1. **Next-goal model** (current work). For each moment of live play, predict when the next goal comes and who scores it. The target is a "survival" output: 31 classes, blue scoring in each 2-second slice up to 30 s, orange the same, or no goal within 30 s. The goal is a review tool that marks good and bad plays in the owner's replays.
2. **Rank predictor** (on hold). Predict a player's rank from their movement. Replays arrive 50 a day from ballchasing.com on server1.

## Where things run

- **server1** (the home server, 192.168.4.24, no GPU) downloads and parses replays, builds the training examples, and trains LightGBM baselines. It holds the master copy of the data and records results.
- **The desktop** (Windows, RTX 3090, WSL2 Ubuntu) trains neural nets. Setup and the training steps are in `DESKTOP.md`.

## Layout

| Path | Contents |
|---|---|
| `src/rl_replay/kaggle_download.py`, `download.py` | Replay downloads (Kaggle dataset, ballchasing) |
| `src/rl_replay/parse.py` | Replays to per-frame arrays, goal detection |
| `src/rl_replay/frames.py` | Training examples: 134 inputs per moment, labels, `mirror()` for orange's view |
| `src/rl_replay/baseline.py`, `horizon.py` | LightGBM baselines |
| `src/rl_replay/train_nn.py` | PyTorch survival model on one moment (MLP) |
| `src/rl_replay/train_seq.py` | PyTorch survival model on the last 4 s of play (GRU) |
| `RESULTS.md` | Every model's scores on the same test games. Add a row for each new model |
| `data/`, `models/` | Not in git |

## Rules

- Score every model on the test split (`split == 2`, chosen by game id) and compare with `RESULTS.md`. Never train on validation or test games.
- Use `uv run --extra gpu ...` on the desktop and `uv run --extra cpu ...` on server1 for anything that imports torch.
- Write in plain language: short sentences, no em dashes, sentence-case headings.
