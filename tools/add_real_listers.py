"""OFFLINE: add "lists_real" to maps/index.pkl.gz: per directory under the
repo, the tests that enumerated it with their own code (os.listdir, scandir,
glob, walk, pkgutil) rather than through the import system, whose directory
scans only resolve module names already being imported.

usage: python add_real_listers.py <index.pkl.gz> <realls_out_dir>
(traces from a full-suite run with an audit hook that skips importlib frames)"""
import glob
import gzip
import json
import os
import pickle
import sys


def main():
    path, d = sys.argv[1:3]
    idx = pickle.loads(gzip.decompress(open(path, "rb").read()))
    pos = {t: i for i, t in enumerate(idx["tests"])}
    acc: dict = {}
    every = (1 << len(idx["tests"])) - 1
    for f in glob.glob(os.path.join(d, "out_*.json")):
        for ctx, dirs in json.load(open(f)).items():
            for p in dirs:
                if ctx in pos:
                    acc[p] = acc.get(p, 0) | 1 << pos[ctx]
                elif ctx == "<collect>":
                    # pytest scanning for test files: a test module appearing
                    # or disappearing changes only its own tests (tests/ is a
                    # package, no basename clashes); conftest.py and pytest
                    # configuration are handled as configuration changes
                    continue
                else:  # between tests: anyone
                    acc[p] = every
    # partial tests: a whole-test bit also needs its base pseudo-test
    for p in idx.get("parts", {}).values():
        for k in acc:
            if acc[k] >> p["real"] & 1:
                acc[k] |= 1 << p["base"]
    sets = idx["sets"]
    out = {}
    for k, b in acc.items():
        out[k] = len(sets)
        sets.append(b)
    idx["lists_real"] = out
    with gzip.open(path, "wb", compresslevel=9) as fh:
        fh.write(pickle.dumps(idx, protocol=5))
    print({k: bin(b).count("1") for k, b in acc.items()}, file=sys.stderr)


if __name__ == "__main__":
    main()
