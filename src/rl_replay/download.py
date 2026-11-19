"""Download ranked 3v3 replays from ballchasing.com, rotating through rank bands.

Each replay goes to data/raw/<id>.replay, and its ballchasing listing entry
(which holds each player's rank) is appended to data/index.jsonl. Already
downloaded replays are skipped, so every run adds new ones.

Free accounts may download 50 replays uploaded by other people per UTC day.
The script takes one replay from each band in turn, so the bands stay
balanced however many replays a day allows. It stops when it reaches --max,
when every band has --per-band replays, or when ballchasing keeps refusing
downloads (the daily limit).

    uv run python -m rl_replay.download --max 50
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import requests

API = "https://ballchasing.com/api"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# Every replay in a band has all of its ranked players inside that band.
BANDS = {
    "bronze": ("bronze-1", "bronze-3"),
    "silver": ("silver-1", "silver-3"),
    "gold": ("gold-1", "gold-3"),
    "platinum": ("platinum-1", "platinum-3"),
    "diamond": ("diamond-1", "diamond-3"),
    "champion": ("champion-1", "champion-3"),
    "grand-champion": ("grand-champion-1", "grand-champion-3"),
    "supersonic-legend": ("supersonic-legend", "supersonic-legend"),
}

MIN_GAP = 1.0  # seconds between API calls; downloads allow 1 per second
MAX_BACKOFF = 64  # give up on a request after waiting this long


class RateLimited(Exception):
    pass


def load_token() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("BALLCHASING_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("BALLCHASING_TOKEN missing from .env")


class Client:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers["Authorization"] = token
        self.last_call = 0.0

    def get(self, url: str, **kwargs) -> requests.Response:
        # Back off 2s, 4s, ... 64s on "too many requests", then give up.
        # Short limits clear within that; the daily download limit doesn't.
        backoff = 2
        while True:
            wait = MIN_GAP - (time.monotonic() - self.last_call)
            if wait > 0:
                time.sleep(wait)
            self.last_call = time.monotonic()
            resp = self.session.get(url, timeout=60, **kwargs)
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp
            if backoff > MAX_BACKOFF:
                raise RateLimited(url)
            time.sleep(backoff)
            backoff *= 2


def list_band(client: Client, lo: str, hi: str, after: str | None):
    """Yield listing entries for one band, newest first, page by page."""
    params = {
        "playlist": "ranked-standard",
        "min-rank": lo,
        "max-rank": hi,
        "count": 200,
        "sort-by": "replay-date",
        "sort-dir": "desc",
    }
    if after:
        params["replay-date-after"] = after
    url, query = f"{API}/replays", params
    while url:
        page = client.get(url, params=query).json()
        yield from page.get("list", [])
        url, query = page.get("next"), None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max", type=int, default=50, help="replays to download this run")
    ap.add_argument("--per-band", type=int, default=500, help="stop filling a band at this many")
    ap.add_argument("--after", default="2025-06-01T00:00:00Z",
                    help="only replays played after this date (RFC 3339)")
    args = ap.parse_args()

    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    index_path = DATA / "index.jsonl"
    entries = [json.loads(l) for l in index_path.open()] if index_path.exists() else []
    indexed = {e["id"] for e in entries}
    counts = Counter(e["band"] for e in entries)

    client = Client(load_token())
    listings = {b: list_band(client, lo, hi, args.after) for b, (lo, hi) in BANDS.items()}
    open_bands = [b for b in BANDS if counts[b] < args.per_band]
    got = 0
    with index_path.open("a") as index:
        try:
            while open_bands and got < args.max:
                # Fill the band with the fewest replays first.
                band = min(open_bands, key=lambda b: counts[b])
                entry = next((e for e in listings[band] if e["id"] not in indexed), None)
                if entry is None:
                    print(f"{band}: no more replays to list", flush=True)
                    open_bands.remove(band)
                    continue
                rid = entry["id"]
                try:
                    data = client.get(f"{API}/replays/{rid}/file").content
                except requests.HTTPError as e:
                    print(f"skip {rid}: {e}", flush=True)
                    indexed.add(rid)
                    continue
                path = raw / f"{rid}.replay"
                tmp = path.with_suffix(".part")
                tmp.write_bytes(data)
                os.replace(tmp, path)
                entry["band"] = band
                index.write(json.dumps(entry) + "\n")
                index.flush()
                indexed.add(rid)
                counts[band] += 1
                got += 1
                if counts[band] >= args.per_band:
                    open_bands.remove(band)
        except RateLimited:
            print("ballchasing keeps refusing requests; daily limit reached", flush=True)

    print(f"downloaded {got}; totals: " + ", ".join(f"{b} {counts[b]}" for b in BANDS), flush=True)


if __name__ == "__main__":
    main()
