# rl-rank

Predict a Rocket League player's rank from about a minute of their movement in a replay, then find which inputs the model uses to tell ranks apart.

Replays come from the ballchasing.com API and are parsed with [subtr-actor](https://github.com/rlrml/subtr-actor).

## Setup

```sh
uv sync
echo 'BALLCHASING_TOKEN=your-token' > .env
```

Replays and parsed data go in `data/`, which git ignores.

## License

MIT. See [LICENSE](LICENSE).
