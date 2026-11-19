"""Sanity checks on parsed replays before building training data.

Reads a random sample of data/<dir>/parsed/*.npz and prints summary numbers:
game lengths, goals, kickoffs, team balance, value ranges, missing data, and
how the "who scores next" label would split.

    uv run python -m rl_replay.explore --dir data/kaggle --sample 3000
"""

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rl_replay.parse import N_GLOBAL, N_PLAYER

PLAYER_COLS = ["x", "y", "z", "rx", "ry", "rz", "vx", "vy", "vz", "avx", "avy", "avz", "boost", "jump"]


def summarize(path: Path) -> dict:
    d = np.load(path)
    f, goals = d["frames"], d["goals"]
    players = json.loads(str(d["players"]))
    n = len(f)
    ball = f[:, :12]
    t, left = f[:, 12], f[:, 13]
    cars = f[:, N_GLOBAL:].reshape(n, -1, N_PLAYER)

    kickoff = (ball[:, 0] == 0) & (ball[:, 1] == 0)
    kick_starts = int((kickoff & ~np.concatenate(([False], kickoff[:-1]))).sum())

    # Label: team that scores the next goal after each frame, 2 if nobody does.
    label = np.full(n, 2, dtype=np.int8)
    prev = 0
    for frame, team in goals:
        label[prev:frame] = team
        prev = frame
    # Frames after the last goal keep label 2 (time ran out).

    # A car whose row is all zeros has no data that frame (demolished, or not yet spawned).
    car_missing = (np.abs(cars[:, :, :12]).sum(axis=2) == 0)

    return {
        "frames": n,
        "seconds": float(t[-1] - t[0]),
        "overtime": bool((left == 0).mean() > 0.05 and len(goals) and goals[-1][0] > np.argmax(left == 0)),
        "goals": len(goals),
        "goals_blue": int((goals[:, 1] == 0).sum()) if len(goals) else 0,
        "kickoff_starts": kick_starts,
        "label_counts": np.bincount(label, minlength=3).tolist(),
        "goal_x": [float(ball[g, 0]) for g, _ in goals],
        "goal_z": [float(ball[g, 2]) for g, _ in goals],
        "ball_max": np.abs(ball[:, :3]).max(axis=0).tolist(),
        "car_max": np.abs(cars[:, :, :3]).max(axis=(0, 1)).tolist(),
        "car_speed_max": float(np.linalg.norm(cars[:, :, 6:9], axis=2).max()),
        "boost_max": float(cars[:, :, 12].max()),
        "boost_min": float(cars[:, :, 12].min()),
        "jump_values": sorted(set(np.unique(cars[:, :, 13]).round(3).tolist()))[:5],
        "car_missing_frac": float(car_missing.mean()),
        "nan": int(np.isnan(f).sum()),
        "n_players": len(players),
        "ranks": [p["rank_id"] for p in players if p["rank_tier"] > 0],
        "secs_left_start": float(left[0]),
        "time_steps": np.round(np.diff(t), 2).tolist()[:2000:50],
    }


def pct(values, qs=(5, 50, 95)) -> str:
    v = np.percentile(values, qs)
    return ", ".join(f"p{q} {x:.0f}" for q, x in zip(qs, v))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("data/kaggle"))
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    files = sorted((args.dir / "parsed").glob("*.npz"))
    random.seed(0)
    files = random.sample(files, min(args.sample, len(files)))
    with ProcessPoolExecutor(args.workers) as pool:
        S = list(pool.map(summarize, files, chunksize=32))

    print(f"{len(S)} games sampled of {len(list((args.dir / 'parsed').glob('*.npz')))}\n")
    print("Game length")
    print(f"  frames: {pct([s['frames'] for s in S])}")
    print(f"  replay seconds: {pct([s['seconds'] for s in S])}")
    print(f"  clock at first frame: {Counter(round(s['secs_left_start']) for s in S).most_common(3)}")
    steps = np.concatenate([s["time_steps"] for s in S])
    print(f"  seconds between frames: {pct(steps, (1, 50, 99))} (x100: {np.percentile(steps, [1, 50, 99]).round(3).tolist()})")
    print(f"  players per game: {Counter(s['n_players'] for s in S)}")

    print("\nGoals")
    g = [s["goals"] for s in S]
    print(f"  per game: mean {np.mean(g):.2f}, {pct(g)}, zero-goal games {sum(x == 0 for x in g)}")
    blue = sum(s["goals_blue"] for s in S)
    print(f"  blue share of goals: {blue / max(1, sum(g)):.3f}")
    off = [s["kickoff_starts"] - s["goals"] for s in S]
    print(f"  kickoffs minus goals (expect 1, or 0 when the game ends on a goal): {Counter(off).most_common(5)}")
    gx = np.concatenate([s["goal_x"] for s in S])
    gz = np.concatenate([s["goal_z"] for s in S])
    print(f"  ball x when entering goal: {pct(np.abs(gx))}; z: {pct(gz)}")

    print("\nNext-goal label over all frames")
    lc = np.sum([s["label_counts"] for s in S], axis=0)
    print(f"  blue {lc[0] / lc.sum():.3f}, orange {lc[1] / lc.sum():.3f}, nobody {lc[2] / lc.sum():.3f}")

    print("\nValue ranges (max absolute over the sample)")
    print(f"  ball x, y, z: {np.max([s['ball_max'] for s in S], axis=0).round(0).tolist()}")
    print(f"  car x, y, z: {np.max([s['car_max'] for s in S], axis=0).round(0).tolist()}")
    print(f"  car speed: {pct([s['car_speed_max'] for s in S], (50, 99, 100))}")
    print(f"  boost: min {min(s['boost_min'] for s in S):.1f}, max {max(s['boost_max'] for s in S):.1f}")
    print(f"  jump column values seen: {sorted(set(v for s in S for v in s['jump_values']))[:6]}")
    print(f"  NaN values: {sum(s['nan'] for s in S)}")
    print(f"  car rows with no data: {np.mean([s['car_missing_frac'] for s in S]):.4f} of car-frames")

    ranks = Counter(r for s in S for r in s["ranks"])
    print(f"\nRanked players seen: {sum(ranks.values())}, {ranks.most_common(8)}")


if __name__ == "__main__":
    main()
