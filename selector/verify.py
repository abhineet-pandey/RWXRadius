"""Build-time check that the committed baseline index matches the checkout."""
import gzip
import hashlib
import os
import pickle
import sys

root = sys.argv[1]
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(here, "maps", "index.pkl.gz"), "rb") as fh:
    idx = pickle.loads(gzip.decompress(fh.read()))
bad = []
for rel, h in idx["src_hash"].items():
    p = os.path.join(root, rel)
    if not os.path.isfile(p) or hashlib.sha1(open(p, "rb").read()).hexdigest() != h:
        bad.append(rel)
print(f"verified {len(idx['src_hash'])} files, {len(bad)} mismatches", bad[:10])
sys.exit(1 if bad else 0)
