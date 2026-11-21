"""OFFLINE: add the impbase map (tools/impbase_pass.py) to the index.

Stored compactly: `impbase_cls` maps a class key to a bit, `impbase` maps an
import-time site (file, line, name) to the bitmask of classes it read from,
or None when unknown.

usage: python merge_impbase.py <index.pkl.gz> <impbase.pkl> <out index.pkl.gz>
"""
import gzip
import pickle
import sys

src, imp, dst = sys.argv[1:4]
idx = pickle.loads(gzip.decompress(open(src, "rb").read()))
raw = pickle.load(open(imp, "rb"))
names = sorted({c for v in raw.values() if v is not None for c in v})
bit = {c: i for i, c in enumerate(names)}
masks = {}
sites = {}
for k, v in raw.items():
    if v is None:
        sites[k] = None
        continue
    m = 0
    for c in v:
        m |= 1 << bit[c]
    sites[k] = masks.setdefault(m, m)
idx["impbase_cls"] = bit
idx["impbase"] = sites
data = gzip.compress(pickle.dumps(idx, protocol=pickle.HIGHEST_PROTOCOL), 9, mtime=0)
open(dst, "wb").write(data)
print(f"impbase merged: {len(sites)} sites, {len(names)} classes, {len(masks)} masks; {len(data)} bytes", file=sys.stderr)
