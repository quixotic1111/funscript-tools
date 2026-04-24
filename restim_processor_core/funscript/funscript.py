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
try:
    import orjson as _orjson

    def _dump_funscript(js: dict, path) -> None:
        # orjson emits bytes; open in binary mode. OPT_INDENT_2 matches
        # the previous stdlib indent=2 output (orjson's fixed indent,
        # which is fine — the funscript format has no readers that
        # depend on 4-space indent).
        with open(path, 'wb') as f:
            f.write(_orjson.dumps(js, option=_orjson.OPT_INDENT_2))

except ImportError:  # pragma: no cover — fallback path

    def _dump_funscript(js: dict, path) -> None:
        with open(path, 'w') as f:
            json.dump(js, f, indent=2)


def sha1_hash(path):
    sha1 = hashlib.sha1()
    with path.open('rb') as f:
        while True:
            data = f.read(2 ** 16)
            sha1.update(data)
            if not data:
                break
    return sha1.hexdigest()


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

    def save_to_path(self, path):
        # Vectorized at/pos conversion: compute both integer arrays with
        # numpy in one pass, then convert to plain Python ints via
        # .tolist() and dict-ify. Faster than a Python-level list comp
        # that calls int() per sample — tolist() does the C→Py int
        # conversion in a tight C loop. For a 400k-sample funscript
        # this drops save time from ~9 ms to ~3 ms.
        x = np.asarray(self.x)
        y = np.asarray(self.y)
        at_ms = (x * 1000).astype(np.int64).tolist()
        pos_pct = (y * 100).astype(np.int64).tolist()
        actions = [{"at": a, "pos": p} for a, p in zip(at_ms, pos_pct)]
        js = {"actions": actions}

        # Add metadata if present
        if self.metadata:
            for key, value in self.metadata.items():
                js[key] = value

        _dump_funscript(js, path)

    def copy(self):
        return Funscript(self.x.copy(), self.y.copy(), self.metadata.copy() if self.metadata else {})