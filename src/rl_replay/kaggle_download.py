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

Fetching replays one by one hits Kaggle's rate limit (about 25 a minute, so
30 hours for the 3v3 set). Instead we ask Kaggle once for the whole dataset's
zip (147 GB), read its table of contents with HTTP range requests, and stream
only the stretch that holds the 3v3 replays (about 66 GB), decompressing each
file as it passes. The zip's signed link lasts 72 hours.

    uv run python -m rl_replay.kaggle_download
"""

import argparse
import json
import os
import struct
import time
import zlib
from pathlib import Path

import requests
from remotezip import RemoteZip

API = "https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset"
OWNER, SLUG = "rolvarild", "high-level-rocket-league-replay-dataset"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "kaggle"


def session() -> requests.Session:
    token = (Path.home() / ".kaggle" / "access_token").read_text().strip()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def signed_url(s: requests.Session, name: str | None = None) -> str:
    """Ask Kaggle for a signed storage URL for one file, or the whole dataset's zip."""
    body = {"ownerSlug": OWNER, "datasetSlug": SLUG}
    if name:
        body["fileName"] = name
    for attempt in range(8):
        r = s.post(API, json=body, allow_redirects=False, timeout=60)
        if r.status_code == 302:
            return r.headers["Location"]
        if r.status_code != 429:
            r.raise_for_status()
            raise requests.HTTPError(f"expected a redirect, got {r.status_code}")
        time.sleep(int(r.headers.get("Retry-After", 5)) * (attempt + 1))
    raise requests.HTTPError("Kaggle kept answering 429")


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


class Stream:
    """Read a byte range of a URL sequentially, reconnecting where it left off."""

    def __init__(self, url: str, start: int, end: int):
        self.url, self.pos, self.end = url, start, end
        self.buf, self.it = b"", None

    def _connect(self) -> None:
        r = requests.get(self.url, headers={"Range": f"bytes={self.pos + len(self.buf)}-{self.end - 1}"},
                         stream=True, timeout=60)
        if r.status_code != 206:
            raise requests.HTTPError(f"range request got {r.status_code}")
        self.it = r.iter_content(1 << 20)

    def read(self, n: int) -> bytes:
        while len(self.buf) < n:
            for attempt in range(6):
                try:
                    if self.it is None:
                        self._connect()
                    self.buf += next(self.it)
                    break
                except (requests.RequestException, StopIteration) as e:
                    if isinstance(e, StopIteration) and self.pos + len(self.buf) >= self.end:
                        raise EOFError("range ended early")
                    self.it = None
                    time.sleep(2 ** attempt)
                    print(f"  reconnecting: {e!r}", flush=True)
            else:
                raise RuntimeError("download kept failing")
        out, self.buf = self.buf[:n], self.buf[n:]
        self.pos += n
        return out

    def skip_to(self, offset: int) -> None:
        if offset < self.pos:
            raise ValueError("zip entries out of order")
        while self.pos < offset:
            self.read(min(offset - self.pos, 1 << 20))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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

    url = signed_url(s)
    entries = sorted(
        (i for i in RemoteZip(url).infolist()
         if i.filename.startswith("high_level/3v3/") and i.filename.endswith(".replay")),
        key=lambda i: i.header_offset)
    todo = [i for i in entries if not (raw / Path(i.filename).name).exists()]
    print(f"{len(entries) - len(todo)} of {len(entries)} already downloaded", flush=True)
    if not todo:
        return

    last = todo[-1]
    end = last.header_offset + 30 + len(last.filename.encode()) + 1024 + last.compress_size
    stream = Stream(url, todo[0].header_offset, end)
    start, done, got_bytes = time.monotonic(), 0, 0
    for info in todo:
        stream.skip_to(info.header_offset)
        header = stream.read(30)
        sig, *_, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
        if sig != 0x04034B50:
            raise ValueError(f"bad local header for {info.filename}")
        stream.read(name_len + extra_len)
        data = stream.read(info.compress_size)
        if info.compress_type == 8:
            data = zlib.decompress(data, -15)
        if zlib.crc32(data) != info.CRC:
            raise ValueError(f"CRC mismatch for {info.filename}")
        dest = raw / Path(info.filename).name
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        done += 1
        got_bytes += info.compress_size
        if done % 500 == 0:
            secs = time.monotonic() - start
            rate = got_bytes / secs
            left = sum(i.compress_size for i in todo[done:]) / rate / 3600
            print(f"  {done}/{len(todo)}, {rate / 1e6:.0f} MB/s, about {left:.1f} h left", flush=True)
    print(f"downloaded {done}", flush=True)


if __name__ == "__main__":
    main()
