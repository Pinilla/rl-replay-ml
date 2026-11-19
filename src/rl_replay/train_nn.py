"""Train the next-goal survival model as a neural net (PyTorch).

Same examples, labels and test games as the LightGBM survival baseline
(rl_replay.horizon), so the scores compare directly against RESULTS.md.

Model: 133 inputs -> 3 hidden layers of 256 (GELU) -> 31 outputs, softmax.
Outputs: blue scores in each 2 s slice up to 30 s (15), orange the same (15),
no goal within 30 s (1).

All training examples are loaded into GPU memory as float16 (about 8 GB for
the 3090's 24 GB). Each batch is turned into orange's view (mirror) for a
random half of its rows, then normalized with the training set's mean and
standard deviation, which are saved in the checkpoint.

    uv run --extra gpu python -m rl_replay.train_nn                 # full run on the GPU
    uv run --extra cpu python -m rl_replay.train_nn --shards 1 --epochs 1 --max-rows 200000   # quick CPU check

Writes models/<name>/model.pt, results.json and log.jsonl.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_replay.frames import FLIP, NC, NG
from rl_replay.horizon import WINDOWS, survival_class, survival_to_value, survival_within, value_target

BIN_SECS, N_BINS, HALF_LIFE = 2.0, 15, 5.0
N_OUT = 2 * N_BINS + 1


def load_split(frames_dir: Path, split: int, shards: int, max_rows: int, rng):
    """X as float16, survival class, label and seconds to goal for one split."""
    paths = sorted(frames_dir.glob("shard-*.npz"))[:shards or None]
    per = max_rows // len(paths) + 1 if max_rows else None
    Xs, ys, ts = [], [], []
    for p in paths:
        d = np.load(p)
        rows = np.flatnonzero(d["split"] == split)
        if per and len(rows) > per:
            rows = np.sort(rng.choice(rows, per, replace=False))
        Xs.append(d["X"][rows].astype(np.float16))
        ys.append(d["y"][rows])
        ts.append(d["secs_to_goal"][rows])
        print(f"  {p.name}: {len(rows):,} rows", flush=True)
    X, y, t = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ts)
    return X, y, t


class Net(nn.Module):
    def __init__(self, n_in: int, width: int, depth: int, dropout: float):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU(), nn.Dropout(dropout)]
            d = width
        layers.append(nn.Linear(d, N_OUT))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class Prep:
    """Mirror (orange's view) and normalize a batch on the device."""

    def __init__(self, mean, std, device):
        self.mean, self.std = mean.to(device), std.to(device)
        self.flip = torch.tensor(np.where(FLIP, -1.0, 1.0), dtype=torch.float32, device=device)
        n = NG + 6 * NC
        perm = list(range(NG)) + [NG + 3 * NC + i for i in range(3 * NC)] + [NG + i for i in range(3 * NC)]
        assert len(perm) == n
        self.perm = torch.tensor(perm, device=device)
        # Survival class under the mirror: blue slices <-> orange slices.
        cls = list(range(N_BINS, 2 * N_BINS)) + list(range(N_BINS)) + [2 * N_BINS]
        self.cls_swap = torch.tensor(cls, device=device)

    def __call__(self, x, c=None, mirror_mask=None):
        x = x.float()
        if mirror_mask is not None and mirror_mask.any():
            xm = (x[mirror_mask] * self.flip)[:, self.perm]
            x = x.clone()
            x[mirror_mask] = xm
            if c is not None:
                c = c.clone()
                c[mirror_mask] = self.cls_swap[c[mirror_mask]]
        x = (x - self.mean) / self.std
        return x, c


@torch.no_grad()
def predict(model, prep, X, device, batch=65536):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb, _ = prep(X[i:i + batch].to(device, non_blocking=True))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(F.softmax(model(xb).float(), dim=1).cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("data/kaggle"))
    ap.add_argument("--name", default="mlp")
    ap.add_argument("--shards", type=int, default=0, help="use only the first N shards (0 for all)")
    ap.add_argument("--max-rows", type=int, default=0, help="cap training rows (0 for all)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=4, help="stop after this many epochs without a better val score")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)
    frames_dir = args.dir / "frames"
    out = Path("models") / args.name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("loading train", flush=True)
    Xtr, ytr, ttr = load_split(frames_dir, 0, args.shards, args.max_rows, rng)
    print("loading val and test", flush=True)
    eval_rows = min(1_000_000, args.max_rows // 4 if args.max_rows else 1_000_000)
    Xva, yva, tva = load_split(frames_dir, 1, args.shards, eval_rows, rng)
    Xte, yte, tte = load_split(frames_dir, 2, args.shards, eval_rows, rng)
    print(f"train {len(ytr):,}, val {len(yva):,}, test {len(yte):,} rows, loaded in {time.time() - t0:.0f}s", flush=True)

    ctr = survival_class(ytr, ttr, BIN_SECS, N_BINS)
    cva = survival_class(yva, tva, BIN_SECS, N_BINS)

    # Normalization from a sample of training rows, both views, so blue and orange match.
    s = Xtr[rng.choice(len(Xtr), min(500_000, len(Xtr)), replace=False)].astype(np.float32)
    from rl_replay.frames import mirror
    sm, _ = mirror(s, np.zeros(len(s), np.int8))
    both = np.concatenate([s, sm])
    mean = torch.tensor(both.mean(0), dtype=torch.float32)
    std = torch.tensor(both.std(0) + 1e-3, dtype=torch.float32)
    prep = Prep(mean, std, device)

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ctr_t = torch.from_numpy(ctr).to(device)
    Xva_t, Xte_t = torch.from_numpy(Xva), torch.from_numpy(Xte)
    del Xtr, s, sm, both

    model = Net(Xtr_t.shape[1], args.width, args.depth, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {Xtr_t.shape[1]} -> {args.depth} x {args.width} -> {N_OUT}, {n_params:,} weights and biases", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(ctr_t) / args.batch)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)

    best, best_epoch, bad = float("inf"), -1, 0
    log = (out / "log.jsonl").open("w")
    for epoch in range(args.epochs):
        model.train()
        te = time.time()
        order = torch.randperm(len(ctr_t), device=device)
        run_loss = 0.0
        for i in range(steps_per_epoch):
            idx = order[i * args.batch:(i + 1) * args.batch]
            xb, cb = Xtr_t[idx], ctr_t[idx]
            mm = torch.rand(len(idx), device=device) < 0.5
            xb, cb = prep(xb, cb, mm)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(xb).float(), cb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() if i % 50 == 0 else 0.0
        train_secs = time.time() - te
        pva = predict(model, prep, Xva_t, device)
        val = float(-np.log(np.clip(pva[np.arange(len(cva)), cva], 1e-12, 1)).mean())
        rec = {"epoch": epoch + 1, "val_log_loss": round(val, 5), "train_secs": round(train_secs, 1),
               "rows_per_sec": round(len(ctr_t) / train_secs)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if val < best - 1e-5:
            best, best_epoch, bad = val, epoch + 1, 0
            torch.save({"state": model.state_dict(), "mean": mean, "std": std, "args": vars(args) | {"dir": str(args.dir)},
                        "n_in": Xtr_t.shape[1]}, out / "model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"no better val score for {args.patience} epochs, stopping", flush=True)
                break

    ck = torch.load(out / "model.pt", map_location=device)
    model.load_state_dict(ck["state"])
    p = predict(model, prep, Xte_t, device)
    true_v = value_target(yte, tte, HALF_LIFE)
    v = survival_to_value(p, BIN_SECS, N_BINS, HALF_LIFE)
    from sklearn.metrics import roc_auc_score
    cte = survival_class(yte, tte, BIN_SECS, N_BINS)
    p5, hit5 = survival_within(p, BIN_SECS, N_BINS, 0, 5), (yte == 0) & (tte <= 5)
    edges = [0, .02, .05, .1, .2, .4, .7, 1.01]
    results = {
        "target": "survival", "model": f"MLP {args.depth} x {args.width}", "params": n_params,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "train_rows": int(len(ctr_t)), "best_epoch": best_epoch, "val_log_loss": round(best, 5),
        "test_log_loss": float(-np.log(np.clip(p[np.arange(len(cte)), cte], 1e-12, 1)).mean()),
        "value_mse": float(((v - true_v) ** 2).mean()),
        "value_mse_no_model": float((true_v ** 2).mean()),
        "value_corr": float(np.corrcoef(v, true_v)[0, 1]),
        "auc_within": {w: float(roc_auc_score((yte == 0) & (tte <= w),
                                              survival_within(p, BIN_SECS, N_BINS, 0, w))) for w in WINDOWS},
        "calibration_blue_5s": [
            (f"{lo:.2f}-{hi:.2f}", int(m.sum()), round(float(p5[m].mean()), 3), round(float(hit5[m].mean()), 3))
            for lo, hi in zip(edges[:-1], edges[1:]) if (m := (p5 >= lo) & (p5 < hi)).sum() > 100],
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
