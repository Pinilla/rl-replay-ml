"""Parse downloaded replays into per-frame arrays with player rank labels.

For each replay in data/index.jsonl, writes data/parsed/<id>.npz holding:

    frames   float32 (T, 14 + 14 * P) at 10 fps: 12 ball columns, replay time,
             seconds remaining, then 14 columns for each player
    players  JSON list, one entry per player in column order:
             name, team (0 blue, 1 orange), rank_tier (1 bronze I .. 22 SSL, or -1),
             rank_id ("platinum-2", or "")
    goals    int32 (G, 2): frame where the ball enters the net, and scoring team

Rerunning skips replays already parsed.

    uv run python -m rl_replay.parse
"""

import json
from pathlib import Path

import numpy as np
import subtr_actor

DATA = Path(__file__).resolve().parents[2] / "data"
FPS = 10.0
GLOBAL = ["BallRigidBody", "CurrentTime", "SecondsRemaining"]
PLAYER = ["PlayerRigidBody", "PlayerBoost", "PlayerAnyJump"]
N_GLOBAL, N_PLAYER = 14, 14

# Field geometry in Unreal units. The goal line is at |y| = 5120; the ball
# (radius about 93) counts as inside once its center is well past it.
GOAL_Y, POST_X, CROSSBAR_Z = 5150, 900, 650
# After a goal the ball stays in the net for about 3 seconds before the reset.
# A ball that pokes past the line and comes back out (a save) is shorter.
MIN_GOAL_FRAMES = 10


def find_goals(frames: np.ndarray) -> np.ndarray:
    """First frame of each stretch of at least MIN_GOAL_FRAMES with the ball inside a goal.

    The replay header's goal frames count network frames that don't map
    evenly onto replay time, so we find goals from the ball position instead.
    The ball entering the goal at y < 0 is a goal for orange (team 1),
    whoever touched it last, so own goals come out right.
    """
    x, y, z = frames[:, 0], frames[:, 1], frames[:, 2]
    inside = (np.abs(y) > GOAL_Y) & (np.abs(x) < POST_X) & (z < CROSSBAR_Z)
    edges = np.diff(np.concatenate(([0], inside.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    goals = [(s, 1 if y[s] < 0 else 0)
             for s, e in zip(starts, ends)
             if e - s >= MIN_GOAL_FRAMES or e == len(frames)]
    return np.array(goals, dtype=np.int32).reshape(-1, 2)


def ballchasing_ranks(entry: dict) -> dict[tuple[int, str], dict]:
    """Map (team, player name) to the rank ballchasing reported, if any."""
    ranks = {}
    for team, color in enumerate(("blue", "orange")):
        for p in entry.get(color, {}).get("players", []):
            if "rank" in p:
                ranks[(team, p["name"])] = p["rank"]
    return ranks


def parse_one(path: Path, entry: dict) -> dict:
    info, frames = subtr_actor.get_ndarray_with_info_from_replay_filepath(
        str(path), global_feature_adders=GLOBAL, player_feature_adders=PLAYER, fps=FPS)
    meta = info["replay_meta"]
    ranks = ballchasing_ranks(entry)

    # subtr-actor lays out player columns as team zero (blue), then team one.
    players = []
    for team, key in ((0, "team_zero"), (1, "team_one")):
        for p in meta[key]:
            rank = ranks.get((team, p["name"]))
            players.append({
                "name": p["name"],
                "team": team,
                "rank_tier": rank["tier"] if rank else -1,
                "rank_id": rank["id"] if rank else "",
            })

    expected = N_GLOBAL + N_PLAYER * len(players)
    if frames.shape[1] != expected:
        raise ValueError(f"{frames.shape[1]} columns, expected {expected}")

    headers = dict(meta["all_headers"])
    goals = find_goals(frames)
    # The header's per-goal team is the scorer's team, which is wrong for own
    # goals, so check against the final score instead.
    found = [int((goals[:, 1] == t).sum()) for t in (0, 1)]
    final = [headers.get("Team0Score", 0), headers.get("Team1Score", 0)]
    if found != final:
        raise ValueError(f"found score {found}, header says {final}")

    return {
        "frames": frames.astype(np.float32),
        "players": json.dumps(players),
        "goals": goals,
    }


def main() -> None:
    out_dir = DATA / "parsed"
    out_dir.mkdir(exist_ok=True)
    done = ok = failed = 0
    for line in (DATA / "index.jsonl").open():
        entry = json.loads(line)
        rid = entry["id"]
        out = out_dir / f"{rid}.npz"
        if out.exists():
            done += 1
            continue
        try:
            result = parse_one(DATA / "raw" / f"{rid}.replay", entry)
        except Exception as e:  # a few replays are corrupt or from odd game versions
            failed += 1
            print(f"skip {rid}: {e}", flush=True)
            continue
        tmp = out.with_name(out.stem + ".part.npz")
        np.savez_compressed(tmp, **result)
        tmp.replace(out)
        ok += 1
    print(f"parsed {ok}, already done {done}, failed {failed}")


if __name__ == "__main__":
    main()
