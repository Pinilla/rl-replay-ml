"""Gradient-boosting baseline for the next-goal model.

Trains LightGBM on a sample of training examples, each also used in
orange's view (mirror), stops on the validation games, and scores the
test games. For comparison it also scores the "no model" baseline that
always predicts the overall label rates.

Scores, all on test games, lower is better except AUC:
    log loss  how surprised the model is by what happened
    Brier     squared gap between the probabilities and the outcome
    AUC       for each label, how well the model ranks frames where it
              happened above frames where it didn't (0.5 is a coin flip)
The calibration table checks honesty: when the model says 70% blue,
blue should score next about 70% of the time.

    uv run python -m rl_replay.baseline --train-rows 4000000
"""

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from rl_replay.frames import mirror

LABELS = ["blue", "orange", "nobody"]


def load(frames_dir: Path, split: int, max_rows: int, rng: np.random.Generator):
    """Up to max_rows examples of one split, sampled evenly across shards."""
    shards = sorted(frames_dir.glob("shard-*.npz"))
    per = max_rows // len(shards) + 1
    Xs, ys, gs = [], [], []
    for s in shards:
        d = np.load(s)
        rows = np.flatnonzero(d["split"] == split)
        if len(rows) > per:
            rows = np.sort(rng.choice(rows, per, replace=False))
        Xs.append(d["X"][rows])
        ys.append(d["y"][rows])
        gs.append(d["game"][rows])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    onehot = np.eye(p.shape[1])[y]
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier": brier(y, p),
        "accuracy": float((p.argmax(1) == y).mean()),
        **{f"auc_{LABELS[k]}": float(roc_auc_score(y == k, p[:, k])) for k in range(3)},
    }


def calibration(y: np.ndarray, p: np.ndarray, k: int = 0, bins: int = 10) -> list[tuple]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p[:, k] >= lo) & (p[:, k] < hi)
        if m.sum() > 100:
            out.append((f"{lo:.1f}-{hi:.1f}", int(m.sum()), float(p[m, k].mean()), float((y[m] == k).mean())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("data/kaggle"))
    ap.add_argument("--train-rows", type=int, default=4_000_000)
    ap.add_argument("--eval-rows", type=int, default=1_000_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("models/baseline"))
    args = ap.parse_args()

    frames_dir = args.dir / "frames"
    columns = json.loads((frames_dir / "columns.json").read_text())
    rng = np.random.default_rng(0)

    t0 = time.time()
    Xtr, ytr, _ = load(frames_dir, 0, args.train_rows // 2, rng)
    Xm, ym = mirror(Xtr, ytr)
    Xtr, ytr = np.concatenate([Xtr, Xm]), np.concatenate([ytr, ym])
    del Xm, ym
    Xva, yva, _ = load(frames_dir, 1, args.eval_rows, rng)
    Xte, yte, gte = load(frames_dir, 2, args.eval_rows, rng)
    print(f"loaded train {len(ytr):,} (half mirrored), val {len(yva):,}, test {len(yte):,} "
          f"in {time.time() - t0:.0f}s", flush=True)

    jump_cols = [i for i, c in enumerate(columns) if c.endswith("_jump")]
    params = {
        "objective": "multiclass", "num_class": 3, "learning_rate": 0.05,
        "num_leaves": 255, "min_data_in_leaf": 200, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
        "num_threads": args.threads, "verbose": -1,
    }
    train = lgb.Dataset(Xtr, ytr, feature_name=columns, categorical_feature=jump_cols, free_raw_data=True)
    val = lgb.Dataset(Xva, yva, reference=train)
    t0 = time.time()
    model = lgb.train(params, train, num_boost_round=3000, valid_sets=[val],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    print(f"trained {model.best_iteration} rounds in {(time.time() - t0) / 60:.1f} min", flush=True)

    p = model.predict(Xte, num_iteration=model.best_iteration)
    prior = np.bincount(ytr, minlength=3) / len(ytr)
    results = {"model": scores(yte, p), "no_model": scores(yte, np.tile(prior, (len(yte), 1)))}
    # Symmetry check: orange's view of a test frame should give the mirrored answer.
    Xmt, _ = mirror(Xte, yte)
    pm = model.predict(Xmt, num_iteration=model.best_iteration)[:, [1, 0, 2]]
    results["mirror_gap"] = float(np.abs(pm - p).mean())
    results["calibration_blue"] = calibration(yte, p, 0)
    results["test_games"] = int(len(np.unique(gte)))

    imp = model.feature_importance("gain")
    top = sorted(zip(columns, imp / imp.sum()), key=lambda t: -t[1])[:25]
    results["top_features"] = [(c, round(float(v), 4)) for c, v in top]

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.out / "model.txt"), num_iteration=model.best_iteration)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: results[k] for k in ("model", "no_model", "mirror_gap", "test_games")}, indent=1))
    print("calibration (blue): bin, frames, predicted, actual")
    for row in results["calibration_blue"]:
        print("  ", row)
    print("top features by gain:")
    for c, v in results["top_features"]:
        print(f"   {c:28} {v:.4f}")


if __name__ == "__main__":
    main()
