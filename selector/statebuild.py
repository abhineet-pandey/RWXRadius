"""BUILD TIME: fingerprint the pinned checkout's import-time state.

usage: python -m selector.statebuild <repo_root> <out.pkl>

Runs in the image, with the image's Python, so bytecode in function
fingerprints matches what the run-time child computes. The state is captured
under several hash seeds; labels whose fingerprint depends on the seed (class
body loop variables over sets, ...) are recorded as noisy and ignored when
diffing.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile

SEEDS = (0, 1, 2, 3, 4, 5)

_CHILD = r"""
import os, pickle, sys
root, out, here = sys.argv[1:4]
sys.path.insert(0, here)
sys.path.insert(0, root)
os.chdir(root)
os.environ["SKIP_INTEGRATION"] = "0"
# same environment as the replay child: it runs under unittest, and some
# module constants record that at import (expressions.core.UNITTEST)
import unittest, concurrent.futures, pickle as _p, importlib
# shift the heap: sets of objects hashed by address (classes) iterate in an
# address-dependent order, and sequences built from them must show up as noisy
_pad = [object() for _ in range(int(os.environ["BR_PAD"]))], bytearray(int(os.environ["BR_PAD"]) * 97)
from selector import statefp
mods = statefp.import_all()
pickle.dump((mods, statefp.snapshot(mods)), open(out, "wb"), protocol=5)
"""


def main():
    root, out = sys.argv[1:3]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    snaps = []
    with tempfile.TemporaryDirectory() as td:
        for s in SEEDS:
            p = os.path.join(td, f"s{s}.pkl")
            env = dict(os.environ, PYTHONHASHSEED=str(s), PYTHONDONTWRITEBYTECODE="1", BR_PAD=str(s * 7919 + 13))
            subprocess.run([sys.executable, "-c", _CHILD, root, p, here], check=True, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            snaps.append(pickle.load(open(p, "rb")))
    from . import statefp

    mods, base = snaps[0]
    noisy: set = set()  # labels
    noisy_keys: dict = {}  # label -> keys
    for m, s in snaps[1:]:
        if m != mods:
            raise SystemExit("module set depends on hash seed")
        keyed, other = statefp.diff(base, s)
        noisy.update(lab for lab, _, _ in other)
        for lab, (ks, _) in keyed.items():
            noisy_keys.setdefault(lab, set()).update(ks)
    pickle.dump({"mods": mods, "state": base, "noisy": noisy, "noisy_keys": noisy_keys},
                open(out, "wb"), protocol=5)
    # the modules that import at the baseline, for import probes
    import json

    json.dump(mods, open(os.path.join(os.path.dirname(out), "modules.json"), "w"))
    print(f"state: {len(mods)} modules, {len(base)} labels, noisy={sorted(noisy)} "
          f"noisy_keys={ {k: len(v) for k, v in noisy_keys.items()} }", file=sys.stderr)


if __name__ == "__main__":
    main()
