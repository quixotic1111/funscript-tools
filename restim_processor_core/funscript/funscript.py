# credit @diglet48 https://github.com/diglet48/restim/blob/master/funscript/funscript.py
import numpy as np
import json
import time
import logging
import hashlib
import pathlib


logger = logging.getLogger('restim.funscript')

funscript_cache = {}

# path+mtime fast-path cache. Sits in front of the content-hash
# ``funscript_cache`` below: when a path has already been loaded AND
# its mtime hasn't changed since, we return a copy immediately, with
# zero I/O. The content-hash cache still needs a full file read to
# compute its SHA-1 key, which on a multi-MB funscript costs 10-50 ms
# per call — pure overhead when the file hasn't been touched between
# clicks (the common case while tuning one variant).
#
# Key: (str path, mtime_ns). Value: Funscript. Cleared with the same
# process lifetime as funscript_cache (module-level dicts; persistent-
# worker processes live for the whole app session).
_path_mtime_cache = {}


# orjson is 2-3x faster than stdlib json for the action-list heavy
# payloads this module writes (one funscript can be tens of thousands
# of actions). Prefer it when available, fall back cleanly when not.
#
# Output is written WITHOUT indentation — a long funscript spends
# ~50 % of its bytes on newlines + per-field spaces when pretty-
# printed. Players ignore whitespace, so compact JSON is a free
# size win. Re-enable indent=2 here if you need human-readable
# dumps for debugging; the data is identical either way.
try:
    import orjson as _orjson

    def _dump_funscript(js: dict, path) -> None:
        with open(path, 'wb') as f:
            f.write(_orjson.dumps(js))

except ImportError:  # pragma: no cover — fallback path

    def _dump_funscript(js: dict, path) -> None:
        with open(path, 'w') as f:
            # separators=(',', ':') strips the default ", " and ": "
            # padding so stdlib json matches orjson's compact output.
            json.dump(js, f, separators=(',', ':'))


def sha1_hash(path):
    sha1 = hashlib.sha1()
    with path.open('rb') as f:
        while True:
            data = f.read(2 ** 16)
            sha1.update(data)
            if not data:
                break
    return sha1.hexdigest()


def _mark_informative_points(at: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Return a boolean mask the same length as ``at``/``pos`` that
    keeps only points a linear-interp player actually needs.

    A point i in (0, N-1) is removable iff it sits exactly on the
    line between point i-1 and point i+1 — i.e. the integer equality

        (pos[i] - pos[i-1]) * (at[i+1] - at[i-1]) ==
        (pos[i+1] - pos[i-1]) * (at[i] - at[i-1])

    holds. Both sides are int64 products of bounded values (pos is
    0..100, at is milliseconds over a video of bounded length), so
    this is exact — no floating-point wobble. The check is cheap:
    two numpy broadcasts over the interior.

    Endpoints are always kept so the first and last timestamps of
    the original signal are preserved.

    This collapses:
      - flat runs (pos[i-1] == pos[i] == pos[i+1]) to just the first
        and last point of each run;
      - linear ramps (pos evenly stepping between neighbors) to their
        endpoints;
      - mixed flat+ramp sections similarly.
    """
    n = at.shape[0]
    if n <= 2:
        return np.ones(n, dtype=bool)
    # Work in int64 so the multiplications don't overflow for long
    # videos (hours × 1000 ms/s fits comfortably in int64).
    at64 = at.astype(np.int64, copy=False)
    pos64 = pos.astype(np.int64, copy=False)
    t0 = at64[:-2]
    t1 = at64[1:-1]
    t2 = at64[2:]
    p0 = pos64[:-2]
    p1 = pos64[1:-1]
    p2 = pos64[2:]
    lhs = (p1 - p0) * (t2 - t0)
    rhs = (p2 - p0) * (t1 - t0)
    collinear = (lhs == rhs)
    keep = np.ones(n, dtype=bool)
    keep[1:-1] = ~collinear
    return keep


class Funscript:
    def __init__(self, x, y, metadata=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.metadata = metadata if metadata is not None else {}

    @staticmethod
    def from_file(filename_or_path):
        start = time.time()

        if isinstance(filename_or_path, str):
            path = pathlib.Path(filename_or_path)
        else:
            path = filename_or_path

        # Fast path: if we've loaded this exact path with this exact
        # mtime before, skip the file read entirely. This is the
        # common case while iteratively tuning one variant — the
        # input file hasn't changed between clicks, so re-reading it
        # to compute SHA-1 is pure overhead.
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = None
        path_key = (str(path), mtime_ns) if mtime_ns is not None else None
        if path_key is not None:
            cached = _path_mtime_cache.get(path_key)
            if cached is not None:
                return cached.copy()

        hash = sha1_hash(path)
        if hash in funscript_cache:
            logger.info(f'imported {path} from cache')
            funscript = funscript_cache[hash]
            # Also populate the path+mtime cache so the next call
            # with the same (path, mtime) skips sha1_hash too.
            if path_key is not None:
                _path_mtime_cache[path_key] = funscript
            return funscript.copy()

        x = []
        y = []
        with path.open(encoding='utf-8') as f:
            js = json.load(f)
            for action in js['actions']:
                at = float(action['at']) / 1000
                pos = float(action['pos']) * 0.01
                x.append(at)
                y.append(pos)

            # Extract metadata if present
            metadata = {}
            for key in ['title', 'creator', 'description', 'url', 'tags', 'duration', 'metadata']:
                if key in js:
                    metadata[key] = js[key]

        end = time.time()
        logger.info(f'imported {path} in {end-start} seconds')
        funscript = Funscript(x, y, metadata)
        funscript_cache[hash] = funscript
        if path_key is not None:
            _path_mtime_cache[path_key] = funscript
        return funscript

    def save_to_path(self, path, simplify: bool = True):
        """Serialize this Funscript to ``path`` as JSON.

        When ``simplify`` is True (default), interior points that lie
        exactly on the line between their neighbors are dropped before
        writing. Funscript players do linear interpolation between
        adjacent (at, pos) pairs, so any point whose integer position
        equals the integer-exact linear interp of its neighbors is
        pure file-size overhead — removing it produces byte-identical
        playback. Typical reduction on 60 Hz outputs with long flat
        runs and linear ramps: 20-35 %.

        Pass ``simplify=False`` to write every sample as-is (useful
        for debugging or for tools that don't interpolate).
        """
        x = np.asarray(self.x)
        y = np.asarray(self.y)
        # Vectorized at/pos conversion: compute both integer arrays in
        # one numpy pass, then use .tolist() (a tight C loop) instead
        # of per-element int() calls. Drops save time from ~9 ms to
        # ~3 ms on a 400 k-sample funscript.
        at_arr = (x * 1000).astype(np.int64)
        pos_arr = (y * 100).astype(np.int64)

        if simplify and at_arr.size > 2:
            keep = _mark_informative_points(at_arr, pos_arr)
            at_arr = at_arr[keep]
            pos_arr = pos_arr[keep]

        at_ms = at_arr.tolist()
        pos_pct = pos_arr.tolist()
        actions = [{"at": a, "pos": p} for a, p in zip(at_ms, pos_pct)]
        js = {"actions": actions}

        # Add metadata if present
        if self.metadata:
            for key, value in self.metadata.items():
                js[key] = value

        _dump_funscript(js, path)

    def copy(self):
        return Funscript(self.x.copy(), self.y.copy(), self.metadata.copy() if self.metadata else {})