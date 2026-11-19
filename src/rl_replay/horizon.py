"""LightGBM baselines that care when the next goal comes.

Two targets, both built from each example's label and seconds to the next goal:

  value     One number per frame: +0.5 ** (t / half_life) if blue scores
            next in t seconds, the same but negative for orange, 0 if
            nobody scores. Regression. A play's impact is how far it moves
            this number.
  survival  Which team scores, and in which time slice: blue in 0-2 s,
            blue in 2-4 s, ... up to --max-secs, the same for orange, or
            "no goal by then". Multiclass. Any window can be read off
            afterward by adding slices.

Both are scored on the same test frames:
  value_mse    squared error against the true discounted value. The
               survival model's value is computed from its slice probabilities.
  auc_within   for each window W, how well the model ranks "blue scores
               within W seconds" above everything else (0.5 is a coin flip).

    uv run python -m rl_replay.horizon --target value
    uv run python -m rl_replay.horizon --target survival
"""

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from rl_replay.frames import mirror

WINDOWS = (2, 5, 10, 20)


def load(frames_dir: Path, split: int, max_rows: int, rng: np.random.Generator):
    shards = sorted(frames_dir.glob("shard-*.npz"))
    per = max_rows // len(shards) + 1
    Xs, ys, ts = [], [], []
    for s in shards:
        d = np.load(s)
        rows = np.flatnonzero(d["split"] == split)
        if len(rows) > per:
            rows = np.sort(rng.choice(rows, per, replace=False))
        Xs.append(d["X"][rows])
        ys.append(d["y"][rows])
        ts.append(d["secs_to_goal"][rows])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ts)


def value_target(y: np.ndarray, t: np.ndarray, half_life: float) -> np.ndarray:
    sign = np.where(y == 0, 1.0, np.where(y == 1, -1.0, 0.0))
    return sign * np.where(np.isfinite(t), 0.5 ** (np.nan_to_num(t, posinf=0) / half_life), 0.0)


def survival_class(y: np.ndarray, t: np.ndarray, bin_secs: float, n_bins: int) -> np.ndarray:
    """Class k * n_bins + b for team k scoring in slice b; 2 * n_bins for no goal in time."""
    b = np.floor(np.nan_to_num(t, posinf=1e9) / bin_secs).astype(np.int64)
    scored = (y < 2) & (b < n_bins)
    return np.where(scored, y.astype(np.int64) * n_bins + np.minimum(b, n_bins - 1), 2 * n_bins)


def survival_to_value(p: np.ndarray, bin_secs: float, n_bins: int, half_life: float) -> np.ndarray:
    mid = (np.arange(n_bins) + 0.5) * bin_secs
    w = 0.5 ** (mid / half_life)
    return p[:, :n_bins] @ w - p[:, n_bins:2 * n_bins] @ w


def survival_within(p: np.ndarray, bin_secs: float, n_bins: int, team: int, secs: float) -> np.ndarray:
    k = int(round(secs / bin_secs))
    return p[:, team * n_bins: team * n_bins + k].sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=["value", "survival"], required=True)
    ap.add_argument("--dir", type=Path, default=Path("data/kaggle"))
    ap.add_argument("--train-rows", type=int, default=4_000_000)
    ap.add_argument("--eval-rows", type=int, default=1_000_000)
    ap.add_argument("--half-life", type=float, default=5.0)
    ap.add_argument("--bin-secs", type=float, default=2.0)
    ap.add_argument("--max-secs", type=float, default=30.0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    out = Path("models") / args.target
    n_bins = int(args.max_secs / args.bin_secs)

    frames_dir = args.dir / "frames"
    columns = json.loads((frames_dir / "columns.json").read_text())
    jump_cols = [i for i, c in enumerate(columns) if c.endswith("_jump")]
    rng = np.random.default_rng(0)

    t0 = time.time()
    Xtr, ytr, ttr = load(frames_dir, 0, args.train_rows // 2, rng)
    Xm, ym = mirror(Xtr, ytr)
    Xtr, ytr, ttr = np.concatenate([Xtr, Xm]), np.concatenate([ytr, ym]), np.concatenate([ttr, ttr])
    del Xm, ym
    Xva, yva, tva = load(frames_dir, 1, args.eval_rows // 2, rng)
    Xte, yte, tte = load(frames_dir, 2, args.eval_rows, rng)
    print(f"loaded train {len(ytr):,}, val {len(yva):,}, test {len(yte):,} in {time.time() - t0:.0f}s", flush=True)

    base = {"learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 200,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 1.0, "num_threads": args.threads, "verbose": -1}
    if args.target == "value":
        params = {**base, "objective": "regression"}
        lab_tr = value_target(ytr, ttr, args.half_life)
        lab_va = value_target(yva, tva, args.half_life)
    else:
        params = {**base, "objective": "multiclass", "num_class": 2 * n_bins + 1, "learning_rate": 0.1}
        lab_tr = survival_class(ytr, ttr, args.bin_secs, n_bins)
        lab_va = survival_class(yva, tva, args.bin_secs, n_bins)

    train = lgb.Dataset(Xtr, lab_tr, feature_name=columns, categorical_feature=jump_cols)
    val = lgb.Dataset(Xva, lab_va, reference=train)
    t0 = time.time()
    model = lgb.train(params, train, num_boost_round=3000, valid_sets=[val],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
    minutes = (time.time() - t0) / 60
    print(f"trained {model.best_iteration} rounds in {minutes:.1f} min", flush=True)

    pred = model.predict(Xte, num_iteration=model.best_iteration)
    true_v = value_target(yte, tte, args.half_life)
    if args.target == "value":
        v = pred
        within = {w: v for w in WINDOWS}  # one score ranks every window
    else:
        v = survival_to_value(pred, args.bin_secs, n_bins, args.half_life)
        within = {w: survival_within(pred, args.bin_secs, n_bins, 0, w) for w in WINDOWS}

    results = {
        "target": args.target, "rounds": model.best_iteration, "train_minutes": round(minutes, 1),
        "value_mse": float(((v - true_v) ** 2).mean()),
        "value_mse_no_model": float((true_v ** 2).mean()),
        "value_corr": float(np.corrcoef(v, true_v)[0, 1]),
        "auc_within": {w: float(roc_auc_score((yte == 0) & (tte <= w), within[w])) for w in WINDOWS},
    }
    if args.target == "survival":
        # Calibration of "blue scores within 5 s".
        p5, hit5 = within[5], (yte == 0) & (tte <= 5)
        edges = [0, .02, .05, .1, .2, .4, .7, 1.01]
        results["calibration_blue_5s"] = [
            (f"{lo:.2f}-{hi:.2f}", int(m.sum()), round(float(p5[m].mean()), 3), round(float(hit5[m].mean()), 3))
            for lo, hi in zip(edges[:-1], edges[1:]) if (m := (p5 >= lo) & (p5 < hi)).sum() > 100]
    imp = model.feature_importance("gain")
    results["top_features"] = [(c, round(float(x), 4)) for c, x in
                               sorted(zip(columns, imp / imp.sum()), key=lambda t: -t[1])[:15]]

    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "model.txt"), num_iteration=model.best_iteration)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
