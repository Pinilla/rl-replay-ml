"""Download ranked 3v3 replays from ballchasing.com, balanced across rank bands.

Each replay goes to data/raw/<id>.replay, and its ballchasing listing entry
(which holds each player's rank) is appended to data/index.jsonl. Rerunning
skips replays already on disk, so an interrupted run picks up where it stopped.

    uv run python -m rl_rank.download --per-band 500
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests

API = "https://ballchasing.com/api"
DATA = Path(__file__).resolve().parents[2] / "data"

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

MIN_GAP = 0.6  # seconds between API calls


def load_token() -> str:
    env = Path(__file__).resolve().parents[2] / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("BALLCHASING_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("BALLCHASING_TOKEN missing from .env")


class Client:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers["Authorization"] = token
        self.last_call = 0.0

    def get(self, url: str, **kwargs) -> requests.Response:
        # Ballchasing sends no rate-limit headers and sometimes "Retry-After: 0",
        # so back off on our own: 2s, 4s, 8s, ... up to 5 minutes.
        backoff = 2
        while True:
            wait = MIN_GAP - (time.monotonic() - self.last_call)
            if wait > 0:
                time.sleep(wait)
            self.last_call = time.monotonic()
            resp = self.session.get(url, timeout=60, **kwargs)
            if resp.status_code == 429:
                print(f"  rate limited, waiting {backoff}s", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            resp.raise_for_status()
            return resp


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
    ap.add_argument("--per-band", type=int, default=500)
    ap.add_argument("--after", default="2025-06-01T00:00:00Z",
                    help="only replays played after this date (RFC 3339)")
    ap.add_argument("--bands", nargs="*", default=list(BANDS))
    args = ap.parse_args()

    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    index_path = DATA / "index.jsonl"
    indexed = set()
    if index_path.exists():
        indexed = {json.loads(l)["id"] for l in index_path.open()}

    client = Client(load_token())
    with index_path.open("a") as index:
        for band in args.bands:
            lo, hi = BANDS[band]
            have = sum(1 for l in index_path.open() if json.loads(l)["band"] == band)
            print(f"{band}: {have} already downloaded", flush=True)
            for entry in list_band(client, lo, hi, args.after):
                if have >= args.per_band:
                    break
                rid = entry["id"]
                if rid in indexed:
                    continue
                path = raw / f"{rid}.replay"
                try:
                    data = client.get(f"{API}/replays/{rid}/file").content
                except requests.HTTPError as e:
                    print(f"  skip {rid}: {e}", flush=True)
                    continue
                tmp = path.with_suffix(".part")
                tmp.write_bytes(data)
                os.replace(tmp, path)
                entry["band"] = band
                index.write(json.dumps(entry) + "\n")
                index.flush()
                indexed.add(rid)
                have += 1
                if have % 25 == 0:
                    print(f"  {band}: {have}/{args.per_band}", flush=True)
            print(f"{band}: done with {have}", flush=True)


if __name__ == "__main__":
    main()
