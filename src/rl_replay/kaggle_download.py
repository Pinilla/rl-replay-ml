"""Download the 3v3 games of the Kaggle High-Level Rocket League Replay Dataset.

https://www.kaggle.com/datasets/rolvarild/high-level-rocket-league-replay-dataset
holds about 42,000 GC-and-above ranked 3v3 games (and 1v1 and 2v2, which we skip).
Writes:

    data/kaggle/index.jsonl     one ballchasing listing entry per game, with ranks
    data/kaggle/raw/<id>.replay

Reruns skip files already on disk.

The official kaggle tool can't fetch single files from this dataset: it follows
the API's redirect to the file and then tries to read the file as JSON. So we
call the API ourselves and follow the redirect. The token is the one from
kaggle.com/settings, saved in ~/.kaggle/access_token.

    uv run python -m rl_replay.kaggle_download --workers 4
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API = "https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset"
OWNER, SLUG = "rolvarild", "high-level-rocket-league-replay-dataset"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "kaggle"


def session() -> requests.Session:
    token = (Path.home() / ".kaggle" / "access_token").read_text().strip()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def fetch(s: requests.Session, name: str, dest: Path, tries: int = 5) -> None:
    """Download one dataset file to dest, retrying with backoff."""
    for attempt in range(tries):
        try:
            r = s.post(API, json={"ownerSlug": OWNER, "datasetSlug": SLUG, "fileName": name},
                       allow_redirects=False, timeout=60)
            if r.status_code == 429:
                raise requests.HTTPError("429 too many requests")
            if r.status_code != 302:
                r.raise_for_status()
                raise requests.HTTPError(f"expected a redirect, got {r.status_code}")
            # The signed storage URL needs no Kaggle token.
            f = requests.get(r.headers["Location"], timeout=300)
            f.raise_for_status()
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(f.content)
            os.replace(tmp, dest)
            return
        except requests.RequestException as e:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
            print(f"  retry {name}: {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many games (0 for all)")
    args = ap.parse_args()

    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    s = session()

    meta = OUT / "metadata-3v3.jsonl"
    if not meta.exists():
        print("downloading metadata", flush=True)
        fetch(s, "high_level/3v3/metadata.json", meta)

    # index.jsonl keeps only the ballchasing entry, in the same shape the
    # ballchasing downloader writes, so the parser reads both the same way.
    index = OUT / "index.jsonl"
    if not index.exists():
        with meta.open() as src, index.open("w") as dst:
            for line in src:
                rec = json.loads(line)
                entry = rec["data"]
                entry["band"] = "high-level"
                entry["hl_score"] = rec.get("score")
                dst.write(json.dumps(entry) + "\n")

    ids = [json.loads(l)["id"] for l in index.open()]
    if args.limit:
        ids = ids[: args.limit]
    todo = [i for i in ids if not (raw / f"{i}.replay").exists()]
    print(f"{len(ids) - len(todo)} of {len(ids)} already downloaded", flush=True)

    done = failed = 0
    start = time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(fetch, s, f"high_level/3v3/{i}.replay", raw / f"{i}.replay"): i
                for i in todo}
        for job in as_completed(jobs):
            try:
                job.result()
                done += 1
            except Exception as e:
                failed += 1
                print(f"failed {jobs[job]}: {e}", flush=True)
            if (done + failed) % 500 == 0:
                rate = done / (time.monotonic() - start)
                left = (len(todo) - done - failed) / rate / 3600 if rate else 0
                print(f"  {done + failed}/{len(todo)}, {rate:.1f}/s, about {left:.1f} h left", flush=True)
    print(f"downloaded {done}, failed {failed}", flush=True)


if __name__ == "__main__":
    main()
