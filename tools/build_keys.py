"""OFFLINE: compact keyprobe traces (BRMAP_KEYS=1 runs of the pinned
checkout's own test suite) into maps/keys.pkl.gz.

usage: python build_keys.py <tests.json> <out.pkl.gz> <keys_dir> [<keys_dir>...]

Layout:
  tests    test ids, index order (same as maps/index.pkl.gz)
  sets     bitsets over test positions
  km       canonical label -> {key: set id}; "*" whole-dict readers, "#" size
           readers, "?" readers through keys with no stable form
  aliases  label -> canonical label of every tracked dict
  leaky    canonical labels also referenced from somewhere untracked
  unknown  set id of tests whose reads are not fully known (did not run
           cleanly under the probe, started processes, or were never seen)
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import pickle
import sys
from collections import defaultdict


def bits(idxs) -> int:
    if not idxs:
        return 0
    m = max(idxs)
    ba = bytearray(m // 8 + 1)
    for i in idxs:
        ba[i >> 3] |= 1 << (i & 7)
    return int.from_bytes(ba, "little")


def main():
    tests_json, out = sys.argv[1:3]
    dirs = sys.argv[3:]
    order = [t["id"] for t in json.load(open(tests_json))["tests"]]
    pos = {t: i for i, t in enumerate(order)}
    per_test = defaultdict(set)
    per_class_setup = defaultdict(set)
    collect = set()
    aliases: dict = {}
    leaky: set = set()
    bad: set = set()
    seen: set = set()
    runs = 0
    for d in dirs:
        for f in glob.glob(os.path.join(d, "keys_*.pkl")):
            x = pickle.load(open(f, "rb"))
            runs += 1
            for lab, canon in x["aliases"].items():
                if aliases.setdefault(lab, canon) != canon:
                    raise SystemExit(f"alias mismatch {lab}: {aliases[lab]} vs {canon}")
            leaky.update(x["leaky"])
            collect |= x["collect"]
            bad.update(x["subproc"])
            for nid, r in x["tests"].items():
                per_test[nid] |= r["call"]
                per_class_setup[nid.rsplit("::", 1)[0]] |= r["setup"]
            for nid, oc in x["outcomes"].items():
                seen.add(nid)
                if any(o not in ("passed", "skipped") for _, o in oc):
                    bad.add(nid)
    km = defaultdict(lambda: defaultdict(list))
    for nid in order:
        i = pos[nid]
        acc = per_test.get(nid, set()) | per_class_setup.get(nid.rsplit("::", 1)[0], set())
        for lab, k in acc:
            km[lab][k].append(i)
    everyone = list(range(len(order)))
    for lab, k in collect:
        km[lab][k] = everyone

    sets: list = []
    sid_of: dict = {}

    def sid(idxs) -> int:
        b = bits(sorted(set(idxs)))
        k = sid_of.get(b)
        if k is None:
            k = sid_of[b] = len(sets)
            sets.append(b)
        return k

    kmi = {lab: {k: sid(v) for k, v in ks.items()} for lab, ks in km.items()}
    unknown = [pos[t] for t in order if t not in seen or t in bad]
    data = {
        "version": 1,
        "tests": order,
        "sets": sets,
        "km": kmi,
        "aliases": aliases,
        "leaky": sorted(leaky),
        "unknown": sid(unknown),
    }
    raw = pickle.dumps(data, protocol=5)
    with gzip.open(out, "wb", compresslevel=9) as fh:
        fh.write(raw)
    print(f"runs={runs} tracked={len(set(aliases.values()))} read_labels={len(kmi)} "
          f"entries={sum(len(v) for v in kmi.values())} sets={len(sets)} unknown={len(unknown)} "
          f"leaky={len(leaky)} collect={len(collect)} raw={len(raw)} gz={os.path.getsize(out)}", file=sys.stderr)


if __name__ == "__main__":
    main()
