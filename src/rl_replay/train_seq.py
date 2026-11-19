"""Train the next-goal survival model on a short history of play (sequence model).

The MLP in rl_replay.train_nn sees one frozen moment. This model sees the last
--steps moments of the same stretch of live play, 0.5 s apart (8 steps is the
last 4 s), so it can tell who is speeding up, turning or rotating back.

The training examples are already stored in time order within each game, one
every 0.5 s, so a moment's history is just the rows before it. History never
crosses a goal or kickoff; missing steps at the start of play are filled with
the earliest moment and marked with a "real step" input of 0.

Model: each step's 134 inputs (plus the real-step flag) -> 256 -> a 2-layer GRU
(a recurrent layer that reads the steps in order and keeps a running summary)
-> 256 -> 31 outputs, the same survival classes and test games as the MLP.

At the end it scores this model and, if models/mlp exists, the MLP on every
test row, so the two compare on exactly the same moments.

    uv run --extra gpu python -m rl_replay.train_seq                 # full run on the GPU
    uv run --extra cpu python -m rl_replay.train_seq --shards 1 --max-rows 200000 --epochs 1   # quick CPU check

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

from rl_replay.frames import mirror
from rl_replay.horizon import WINDOWS, survival_class, survival_to_value, survival_within, value_target
from rl_replay.train_nn import BIN_SECS, HALF_LIFE, N_BINS, N_OUT, Net, Prep

STEP_FRAMES = 5  # rows are 5 replay frames (0.5 s) apart


def load_split(frames_dir: Path, split: int, shards: int, max_rows: int):
    """All rows of one split in time order: X (float16), y, seconds to goal, and
    for each row the index of the first row of its stretch of play."""
    paths = sorted(frames_dir.glob("shard-*.npz"))[:shards or None]
    Xs, ys, ts, starts, n = [], [], [], [], 0
    for p in paths:
        d = np.load(p)
        rows = np.flatnonzero(d["split"] == split)
        g, f = d["game"][rows], d["frame"][rows]
        new = np.ones(len(rows), bool)
        new[1:] = (g[1:] != g[:-1]) | (f[1:] - f[:-1] != STEP_FRAMES)
        start = np.maximum.accumulate(np.where(new, np.arange(len(rows)), 0)) + n
        Xs.append(d["X"][rows].astype(np.float16))
        ys.append(d["y"][rows])
        ts.append(d["secs_to_goal"][rows])
        starts.append(start)
        n += len(rows)
        print(f"  {p.name}: {len(rows):,} rows", flush=True)
        if max_rows and n >= max_rows:
            break
    X, y, t, s = (np.concatenate(a) for a in (Xs, ys, ts, starts))
    if max_rows:
        X, y, t, s = X[:max_rows], y[:max_rows], t[:max_rows], s[:max_rows]
    return X, y, t, s


class SeqNet(nn.Module):
    def __init__(self, n_in: int, width: int, layers: int):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_in + 1, width), nn.GELU())
        self.gru = nn.GRU(width, width, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, N_OUT))

    def forward(self, x):  # x: (batch, steps, n_in + 1), oldest step first
        e = self.embed(x)
        with torch.autocast(x.device.type, enabled=False):
            h, _ = self.gru(e.float())
        return self.head(torch.cat([h[:, -1], e[:, -1].float()], 1))


class Windows:
    """Builds (batch, steps, inputs) histories on the device from row indices."""

    def __init__(self, X, start, steps, prep):
        self.X, self.start, self.prep = X, start, prep
        self.back = torch.arange(steps - 1, -1, -1, device=X.device)  # steps-1 ... 0

    def __call__(self, idx, c=None, mirror_mask=None):
        h = idx[:, None] - self.back[None, :]
        st = self.start[idx][:, None]
        real = (h >= st).float()
        h = torch.maximum(h, st)
        b, k = h.shape
        x = self.X[h.reshape(-1)]
        mm = mirror_mask.repeat_interleave(k) if mirror_mask is not None else None
        x, _ = self.prep(x, None, mm)
        if c is not None and mirror_mask is not None:
            c = torch.where(mirror_mask, self.prep.cls_swap[c], c)
        x = torch.cat([x.reshape(b, k, -1), real[..., None]], 2)
        return x, c


@torch.no_grad()
def predict_seq(model, win, rows, device, batch=32768):
    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        xb, _ = win(rows[i:i + batch])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(F.softmax(model(xb).float(), dim=1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def predict_mlp(ck, X, device, batch=65536):
    a = ck["args"]
    m = Net(ck["n_in"], a["width"], a["depth"], a["dropout"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    prep = Prep(ck["mean"], ck["std"], device)
    out = []
    for i in range(0, len(X), batch):
        xb, _ = prep(X[i:i + batch].to(device))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(F.softmax(m(xb).float(), dim=1).cpu())
    return torch.cat(out).numpy()


def score(p, y, t):
    """Every number RESULTS.md reports, plus how often the top guess gets the timing right."""
    from sklearn.metrics import roc_auc_score
    c = survival_class(y, t, BIN_SECS, N_BINS)
    none = 2 * N_BINS
    true_v = value_target(y, t, HALF_LIFE)
    v = survival_to_value(p, BIN_SECS, N_BINS, HALF_LIFE)
    p5, hit5 = survival_within(p, BIN_SECS, N_BINS, 0, 5), (y == 0) & (t <= 5)
    edges = [0, .02, .05, .1, .2, .4, .7, 1.01]
    team = lambda k: np.minimum(k // N_BINS, 2)  # noqa: E731
    top_goal = p[:, :none].argmax(1)
    timing = {}
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 30)]:
        m = (c != none) & (t >= lo) & (t < hi)
        tg, cm = top_goal[m], c[m]
        timing[f"{lo}-{hi}s"] = {
            "share": round(float(m.mean()), 4),
            "exact_slice": round(float((tg == cm).mean()), 4),
            "within_one_slice": round(float(((team(tg) == team(cm)) & (np.abs(tg % N_BINS - cm % N_BINS) <= 1)).mean()), 4),
        }
    return {
        "test_rows": int(len(y)),
        "test_log_loss": float(-np.log(np.clip(p[np.arange(len(c)), c], 1e-12, 1)).mean()),
        "value_mse": float(((v - true_v) ** 2).mean()),
        "value_mse_no_model": float((true_v ** 2).mean()),
        "value_corr": float(np.corrcoef(v, true_v)[0, 1]),
        "auc_within": {w: float(roc_auc_score((y == 0) & (t <= w), survival_within(p, BIN_SECS, N_BINS, 0, w)))
                       for w in WINDOWS},
        "calibration_blue_5s": [
            (f"{lo:.2f}-{hi:.2f}", int(m.sum()), round(float(p5[m].mean()), 3), round(float(hit5[m].mean()), 3))
            for lo, hi in zip(edges[:-1], edges[1:]) if (m := (p5 >= lo) & (p5 < hi)).sum() > 100],
        "top_pick_right": float((p.argmax(1) == c).mean()),
        "always_no_goal_right": float((c == none).mean()),
        "timing_when_a_goal_comes": timing,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("data/kaggle"))
    ap.add_argument("--name", default="seq")
    ap.add_argument("--steps", type=int, default=8, help="moments of history, 0.5 s apart (8 = last 4 s)")
    ap.add_argument("--shards", type=int, default=0, help="use only the first N shards (0 for all)")
    ap.add_argument("--max-rows", type=int, default=0, help="cap training rows, taken in order (0 for all)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=4, help="stop after this many epochs without a better val score")
    ap.add_argument("--val-rows", type=int, default=1_000_000, help="val rows scored after each epoch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)
    frames_dir = args.dir / "frames"
    out = Path("models") / args.name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("loading train", flush=True)
    Xtr, ytr, ttr, str_ = load_split(frames_dir, 0, args.shards, args.max_rows)
    print("loading val and test", flush=True)
    cap = args.max_rows // 4 if args.max_rows else 0
    Xva, yva, tva, sva = load_split(frames_dir, 1, args.shards, cap)
    Xte, yte, tte, ste = load_split(frames_dir, 2, args.shards, cap)
    print(f"train {len(ytr):,}, val {len(yva):,}, test {len(yte):,} rows, loaded in {time.time() - t0:.0f}s", flush=True)

    ctr = survival_class(ytr, ttr, BIN_SECS, N_BINS)
    s = Xtr[rng.choice(len(Xtr), min(500_000, len(Xtr)), replace=False)].astype(np.float32)
    sm, _ = mirror(s, np.zeros(len(s), np.int8))
    both = np.concatenate([s, sm])
    mean = torch.tensor(both.mean(0), dtype=torch.float32)
    std = torch.tensor(both.std(0) + 1e-3, dtype=torch.float32)
    prep = Prep(mean, std, device)
    del s, sm, both

    to = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    win_tr = Windows(to(Xtr), to(str_), args.steps, prep)
    win_va = Windows(to(Xva), to(sva), args.steps, prep)
    ctr_t = to(ctr)
    n_in = Xtr.shape[1]
    del Xtr
    va_rows = np.sort(rng.choice(len(yva), min(args.val_rows, len(yva)), replace=False))
    cva = survival_class(yva[va_rows], tva[va_rows], BIN_SECS, N_BINS)
    va_rows_t = to(va_rows)

    model = SeqNet(n_in, args.width, args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.steps} steps x {n_in} -> {args.width} -> GRU {args.layers} x {args.width} -> {N_OUT}, "
          f"{n_params:,} weights and biases", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(ctr_t) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs,
                                                pct_start=0.05)

    best, best_epoch, bad = float("inf"), -1, 0
    log = (out / "log.jsonl").open("w")
    for epoch in range(args.epochs):
        model.train()
        te = time.time()
        order = torch.randperm(len(ctr_t), device=device)
        for i in range(steps_per_epoch):
            idx = order[i * args.batch:(i + 1) * args.batch]
            mm = torch.rand(len(idx), device=device) < 0.5
            xb, cb = win_tr(idx, ctr_t[idx], mm)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(xb).float(), cb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        train_secs = time.time() - te
        pva = predict_seq(model, win_va, va_rows_t, device)
        val = float(-np.log(np.clip(pva[np.arange(len(cva)), cva], 1e-12, 1)).mean())
        rec = {"epoch": epoch + 1, "val_log_loss": round(val, 5), "train_secs": round(train_secs, 1),
               "rows_per_sec": round(len(ctr_t) / train_secs)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if val < best - 1e-5:
            best, best_epoch, bad = val, epoch + 1, 0
            torch.save({"state": model.state_dict(), "mean": mean, "std": std, "n_in": n_in,
                        "args": vars(args) | {"dir": str(args.dir)}}, out / "model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"no better val score for {args.patience} epochs, stopping", flush=True)
                break

    del win_tr, ctr_t
    model.load_state_dict(torch.load(out / "model.pt", map_location=device)["state"])
    win_te = Windows(to(Xte), to(ste), args.steps, prep)
    p = predict_seq(model, win_te, torch.arange(len(yte), device=device), device)
    results = {"target": "survival", "model": f"GRU {args.layers} x {args.width}, {args.steps} steps of 0.5 s",
               "params": n_params, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
               "train_rows": int(len(ctr)), "best_epoch": best_epoch, "val_log_loss": round(best, 5)}
    results |= score(p, yte, tte)
    mlp = Path("models/mlp/model.pt")
    if mlp.exists():
        print("scoring models/mlp on the same test rows", flush=True)
        results["mlp_same_rows"] = score(predict_mlp(torch.load(mlp, map_location=device), torch.from_numpy(Xte), device),
                                         yte, tte)
    results["minutes"] = round((time.time() - t0) / 60, 1)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
