"""Build next-goal training examples from parsed replays.

Each example is one moment of live play, seen from blue's side, with the
label "who scores the next goal": 0 blue, 1 orange, 2 nobody (time ran out).
mirror() turns any example into orange's view, so training can use both.

Live play runs from the first car movement after each kickoff reset (the
end of the countdown, so the kickoff approach counts) until the next goal or
reset. The goal celebration and the countdown are dropped. In a game that
ends at 0:00, play ends when the ball first touches the ground.

In overtime the replay's "seconds remaining" counts up from 0, so the
examples carry an overtime flag, seconds remaining 0, and seconds into
overtime. A live stretch is overtime if it starts with the clock at 0.

Cars are ordered blue then orange, and within a team by distance to the
ball, so slot 0 is always blue's closest player to the ball. A demolished car
has zeros and present = 0.

Writes data/<dir>/frames/shard-NNN.npz (X, y, game, frame, secs_to_goal, split)
and columns.json. The split comes from the game id: 80% train, 10% val, 10% test.

    uv run python -m rl_replay.frames --dir data/kaggle --every 5
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rl_replay.parse import N_GLOBAL, N_PLAYER

FPS = 10
BALL_REST_Z = (90.0, 96.0)  # ball radius is about 92.75
MOVE_SPEED = 10.0  # a car faster than this has left its spawn spot

GLOBAL_COLS = [
    "ball_x", "ball_y", "ball_z", "ball_vx", "ball_vy", "ball_vz",
    "ball_avx", "ball_avy", "ball_avz",
    "secs_left", "overtime", "overtime_secs", "score_diff", "secs_since_kickoff",
]
CAR_FIELDS = [
    "x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz", "ux", "uy", "uz",
    "avx", "avy", "avz", "boost", "jump", "present", "dist_ball", "speed",
]
SLOTS = [f"{team}{i}" for team in ("blue", "orange") for i in range(3)]
COLUMNS = GLOBAL_COLS + [f"{s}_{f}" for s in SLOTS for f in CAR_FIELDS]
NG, NC = len(GLOBAL_COLS), len(CAR_FIELDS)

# Columns that flip sign when the field turns 180 degrees around its center.
_FLIP_GLOBAL = {"ball_x", "ball_y", "ball_vx", "ball_vy", "ball_avx", "ball_avy", "score_diff"}
_FLIP_CAR = {"x", "y", "vx", "vy", "fx", "fy", "ux", "uy", "avx", "avy"}
FLIP = np.array([c in _FLIP_GLOBAL for c in GLOBAL_COLS]
                + [f in _FLIP_CAR for _ in SLOTS for f in CAR_FIELDS])


def mirror(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orange's view: turn the field 180 degrees, swap the teams and the label."""
    X = X.copy()
    X[:, FLIP] *= -1
    cars = X[:, NG:].reshape(len(X), 2, 3 * NC)
    X[:, NG:] = cars[:, ::-1].reshape(len(X), -1)
    y = np.where(y == 2, 2, 1 - y).astype(y.dtype)
    return X, y


def split_of(game_id: str) -> int:
    """0 train, 1 val, 2 test, fixed by the game id."""
    h = int(hashlib.sha1(game_id.encode()).hexdigest(), 16) % 10
    return 2 if h == 0 else 1 if h == 1 else 0


def directions(rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward and up unit vectors from (roll, pitch, yaw) in radians, ZYX order."""
    r, p, w = rot[..., 0], rot[..., 1], rot[..., 2]
    cr, sr, cp, sp, cw, sw = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(w), np.sin(w)
    fwd = np.stack([cp * cw, cp * sw, sp], -1)
    up = np.stack([cw * sp * cr + sw * sr, sw * sp * cr - cw * sr, cp * cr], -1)
    return fwd, up


def live_segments(ball: np.ndarray, cars: np.ndarray, present: np.ndarray,
                  goals: np.ndarray, secs_left: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) frame ranges of live play."""
    n = len(ball)
    at_center = ((ball[:, 0] == 0) & (ball[:, 1] == 0)
                 & (ball[:, 2] > BALL_REST_Z[0]) & (ball[:, 2] < BALL_REST_Z[1]))
    resets = np.flatnonzero(at_center & ~np.r_[False, at_center[:-1]])
    moving = ((np.linalg.norm(cars[:, :, 6:9], axis=2) > MOVE_SPEED) & present).any(axis=1)
    goal_frames = goals[:, 0] if len(goals) else np.array([], dtype=int)

    segs = []
    for i, r in enumerate(resets):
        if not moving[r:].any():
            break  # the replay ends before anyone moves
        go = r + np.argmax(moving[r:])
        nxt = [n]
        if i + 1 < len(resets):
            nxt.append(resets[i + 1])
        later_goals = goal_frames[goal_frames > go]
        if len(later_goals):
            nxt.append(later_goals[0])
        end = min(nxt)
        # Regulation ending at 0:00 with no goal: play stops when the ball lands.
        # Overtime also shows 0:00 but runs until a goal, so skip segments
        # that start with the clock already at zero.
        if secs_left[go] > 0 and (end == n or (i + 1 < len(resets) and end == resets[i + 1])):
            zero = np.flatnonzero((secs_left[go:end] == 0) & (ball[go:end, 2] < BALL_REST_Z[1]))
            if len(zero):
                end = go + zero[0]
        if end - go > FPS:  # skip slivers under a second
            segs.append((go, end))
    return segs


def game_examples(path: Path, every: int) -> dict | None:
    d = np.load(path)
    a, goals = d["frames"], d["goals"]
    players = json.loads(str(d["players"]))
    if len(players) != 6 or [p["team"] for p in players] != [0, 0, 0, 1, 1, 1]:
        return None
    if a.shape[1] != N_GLOBAL + 6 * N_PLAYER:
        return None
    n = len(a)
    ball, secs_left = a[:, :12], a[:, 13]
    cars = a[:, N_GLOBAL:].reshape(n, 6, N_PLAYER)
    present = np.abs(cars[:, :, :12]).sum(axis=2) > 0

    segs = live_segments(ball, cars, present, goals, secs_left)
    if not segs:
        return None

    # Pick every `every`-th live frame, with a per-game offset so frames don't
    # always line up with the kickoff.
    rng = np.random.default_rng(int(hashlib.sha1(path.stem.encode()).hexdigest()[:8], 16))
    idx, since, ot = [], [], []
    for go, end in segs:
        f = np.arange(go + rng.integers(every), end, every)
        idx.append(f)
        since.append((f - go) / FPS)
        ot.append(np.full(len(f), secs_left[go] == 0))
    idx, since, ot = np.concatenate(idx), np.concatenate(since), np.concatenate(ot)

    # Label and time to the next goal, from the full goal list.
    gf = goals[:, 0] if len(goals) else np.array([n])
    gt = goals[:, 1] if len(goals) else np.array([2])
    k = np.searchsorted(gf, idx, side="right")
    has_next = k < len(goals)
    y = np.where(has_next, gt[np.minimum(k, len(gt) - 1)], 2).astype(np.int8)
    secs_to_goal = np.where(has_next, (gf[np.minimum(k, len(gf) - 1)] - idx) / FPS, np.inf).astype(np.float32)
    # Score difference before each frame (blue minus orange).
    before = np.searchsorted(gf, idx, side="right") if len(goals) else np.zeros(len(idx), int)
    blue_cum = np.r_[0, np.cumsum(gt == 0)] if len(goals) else np.zeros(1)
    orange_cum = np.r_[0, np.cumsum(gt == 1)] if len(goals) else np.zeros(1)
    score_diff = blue_cum[before] - orange_cum[before]

    b, c, pr = ball[idx], cars[idx], present[idx]
    fwd, up = directions(c[:, :, 3:6])
    car = np.concatenate([
        c[:, :, 0:3], c[:, :, 6:9], fwd, up, c[:, :, 9:12],
        c[:, :, 12:13] / 2.55, c[:, :, 13:14], pr[:, :, None].astype(np.float32),
        np.linalg.norm(c[:, :, 0:3] - b[:, None, 0:3], axis=2, keepdims=True),
        np.linalg.norm(c[:, :, 6:9], axis=2, keepdims=True),
    ], axis=2)
    car[~pr] = 0
    # Within each team, order cars by distance to the ball; missing cars last.
    dist = np.where(pr, car[:, :, 18], np.inf)
    order = np.concatenate([np.argsort(dist[:, :3], axis=1), 3 + np.argsort(dist[:, 3:], axis=1)], axis=1)
    car = np.take_along_axis(car, order[:, :, None], axis=1)

    glob = np.column_stack([
        b[:, 0:3], b[:, 6:9], b[:, 9:12], np.where(ot, 0.0, secs_left[idx]), ot.astype(np.float32),
        np.where(ot, secs_left[idx], 0.0), score_diff, since,
    ])
    X = np.concatenate([glob, car.reshape(len(idx), -1)], axis=1).astype(np.float32)
    return {"X": X, "y": y, "frame": idx.astype(np.int32), "secs_to_goal": secs_to_goal}


def _work(args):
    path, every = args
    try:
        return path.stem, game_examples(path, every)
    except Exception as e:
        return path.stem, str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("data/kaggle"))
    ap.add_argument("--every", type=int, default=5, help="keep every Nth live frame (10 fps)")
    ap.add_argument("--shard-games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted((args.dir / "parsed").glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    out = args.dir / "frames"
    out.mkdir(exist_ok=True)
    (out / "columns.json").write_text(json.dumps(COLUMNS))
    games = []
    skipped = errors = 0
    for s in range(0, len(files), args.shard_games):
        batch = files[s : s + args.shard_games]
        parts = []
        with ProcessPoolExecutor(args.workers) as pool:
            for gid, res in pool.map(_work, [(f, args.every) for f in batch], chunksize=16):
                if res is None:
                    skipped += 1
                elif isinstance(res, str):
                    errors += 1
                    print(f"error {gid}: {res}", flush=True)
                else:
                    res["game"] = np.full(len(res["y"]), len(games), dtype=np.int32)
                    res["split"] = np.full(len(res["y"]), split_of(gid), dtype=np.int8)
                    games.append(gid)
                    parts.append(res)
        shard = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(out / f"shard-{s // args.shard_games:03d}.npz", **shard)
        print(f"shard {s // args.shard_games}: {len(batch)} games, {len(shard['y'])} examples", flush=True)
    (out / "games.json").write_text(json.dumps(games))
    print(f"{len(games)} games, {skipped} skipped (not 6 players or no live play), {errors} errors")


if __name__ == "__main__":
    main()
