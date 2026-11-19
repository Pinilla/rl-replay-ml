# rl-replay-ml

Machine learning on Rocket League replays, as a way to learn how to train local models. Two projects share the replay downloader and parser:

1. **Rank predictor.** Predict a player's rank from about a minute of their movement, then find which inputs the model uses to tell ranks apart. Replays come from the ballchasing.com API, 50 a day on a free account.
2. **Next-goal model.** Estimate, at every moment of a game, which team scores next. A swing in that probability marks a good or bad play, which feeds a review tool for your own replays. Training data is the [High-Level Rocket League Replay Dataset](https://www.kaggle.com/datasets/rolvarild/high-level-rocket-league-replay-dataset) on Kaggle.

Replays are parsed with [subtr-actor](https://github.com/rlrml/subtr-actor).

## Setup

```sh
uv sync
echo 'BALLCHASING_TOKEN=your-token' > .env
```

Replays and parsed data go in `data/`, which git ignores.

## Usage

```sh
uv run python -m rl_replay.download --max 50   # ballchasing replays, rotating through rank bands
uv run python -m rl_replay.parse               # replays to per-frame arrays in data/parsed/
```

## License

MIT. See [LICENSE](LICENSE).

## Training the neural net

See [DESKTOP.md](DESKTOP.md) for training on a Windows PC with an NVIDIA GPU through WSL2. Scores for every model are in [RESULTS.md](RESULTS.md).
