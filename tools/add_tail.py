"""OFFLINE: append execution-phase pseudo-tests T@tail (selector/eqfp.py) to
maps/index.pkl.gz and maps/keys.pkl.gz (after add_parts / add_real_listers).

usage: python add_tail.py <index> <keys> <out_index> <out_keys> <repo_root>
                          --lines <dir>... --keys <dir>...

Traces come from tests/test_br_tail.py (dev only): T__TestExecutor__<meth>::
test_tail ran T with its head phases (parsing, optimizing, planning,
transpiling to duckdb) attributed to "<nodeid>#head".

For each such test T, T@tail gets:
  * a line / dict key / receiver class of T when the tail trace has it, or when
    neither trace has it (import-time code, anything unseen: conservative);
    only events seen in the head alone are left out,
  * identifier readers recomputed from the refined line map,
  * every other map: wherever T is.
The runtime replaces T by a fingerprint check of its head outputs only when
the change does not reach T@tail.
"""
from __future__ import annotations

import glob
import gzip
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from add_parts import load, save  # noqa: E402

TAIL = ("tests/test_executor.py::TestExecutor::test_execute_tpcds",
        "tests/test_executor.py::TestExecutor::test_execute_tpch")


def trace_of(nodeid):
    try:
        _, cls, meth = nodeid.split("::")
    except ValueError:
        return None
    head = meth.endswith("#head")
    if not cls.startswith("T__") or meth.split("#")[0] != "test_tail":
        return None
    _, cname, tmeth = cls.split("__", 2)
    tid = f"tests/test_executor.py::{cname}::{tmeth}"
    return (tid, "head" if head else "tail") if tid in TAIL else None


def main():
    a = sys.argv[1:]
    idx_in, keys_in, idx_out, keys_out, root = a[:5]
    rest = a[5:]
    line_dirs = rest[rest.index("--lines") + 1: rest.index("--keys")]
    key_dirs = rest[rest.index("--keys") + 1:]
    idx = load(idx_in)
    km = load(keys_in)
    assert km["tests"] == idx["tests"]
    if idx.get("eq"):
        raise SystemExit("index already has tail pseudo-tests")
    tests = list(idx["tests"])
    pos = {t: i for i, t in enumerate(tests)}
    eq = {}
    for tid in TAIL:
        eq[tid] = {"real": pos[tid], "tail": len(tests)}
        tests.append(tid + "@tail")
    real_bit = {tid: 1 << p["real"] for tid, p in eq.items()}
    tail_bit = {tid: 1 << p["tail"] for tid, p in eq.items()}

    lines, recv, keys = {}, {}, {}
    ok = set()
    for d in line_dirs:
        for f in glob.glob(os.path.join(d, "map_*.pkl")):
            x = pickle.load(open(f, "rb"))
            for nid, v in x["tests"].items():
                tp = trace_of(nid)
                if tp is None:
                    continue
                s = lines.setdefault(tp, set())
                for fn, ls in list(v["lines"].items()) + list(v["setup"].items()):
                    s.update((fn, ln) for ln in ls)
                recv.setdefault(tp, set()).update(v["recv"])
            for nid, oc in x["outcomes"].items():
                tp = trace_of(nid)
                if tp and ("call", "passed") in {tuple(o) for o in oc}:
                    ok.add(("L",) + tp)
    for d in key_dirs:
        for f in glob.glob(os.path.join(d, "keys_*.pkl")):
            x = pickle.load(open(f, "rb"))
            for nid, r in x["tests"].items():
                tp = trace_of(nid)
                if tp is None:
                    continue
                keys.setdefault(tp, set()).update(r["call"] | r["setup"])
            if any(trace_of(n) for n in x["subproc"]):
                raise SystemExit("tail trace started a subprocess")
            for nid, oc in x["outcomes"].items():
                tp = trace_of(nid)
                if tp and ("call", "passed") in {tuple(o) for o in oc}:
                    ok.add(("K",) + tp)
    for tid in TAIL:
        if ("L", tid, "tail") not in ok or ("K", tid, "tail") not in ok:
            raise SystemExit(f"missing tail trace for {tid}")
        for tp in ((tid, "tail"), (tid, "head")):
            lines.setdefault(tp, set())
            recv.setdefault(tp, set())
            keys.setdefault(tp, set())

    def conservative(s):
        acc = s
        for tid in TAIL:
            if s & real_bit[tid]:
                acc |= tail_bit[tid]
        return acc

    def precise(s, ev, tables):
        acc = s
        for tid in TAIL:
            if s & real_bit[tid] and (ev in tables[(tid, "tail")] or ev not in tables[(tid, "head")]):
                acc |= tail_bit[tid]
        return acc

    sets = [conservative(s) for s in idx["sets"]]
    old_sets = idx["sets"]
    sid_of = {}
    for i, s in enumerate(sets):
        sid_of.setdefault(s, i)

    def sid(b):
        k = sid_of.get(b)
        if k is None:
            k = sid_of[b] = len(sets)
            sets.append(b)
        return k

    out = dict(idx)
    out["tests"] = tests
    out["lines"] = {fn: {ln: sid(precise(old_sets[k], (fn, ln), lines)) for ln, k in lt.items()}
                    for fn, lt in idx["lines"].items()}
    out["users"] = {c: sid(precise(old_sets[v], c, recv)) for c, v in idx["users"].items()}
    from build_index import scan_names

    name_lines, _, _ = scan_names(os.path.realpath(root), idx["imptime"])
    tails = 0
    for b in tail_bit.values():
        tails |= b
    names = {}
    for nm, files in name_lines.items():
        acc = 0
        for rel, lns in files.items():
            lt = out["lines"].get(rel)
            if not lt:
                continue
            for ln in lns:
                k = lt.get(ln)
                if k is not None:
                    acc |= sets[k]
        if acc:
            names[nm] = acc
    bad = [nm for nm in idx["names"] if names.get(nm, 0) & ~tails != old_sets[idx["names"][nm]]]
    if bad or names.keys() != idx["names"].keys():
        raise SystemExit(f"recomputed identifier map differs: {bad[:5]}")
    out["names"] = {nm: sid(b) for nm, b in names.items()}
    out["sets"] = sets
    out["eq"] = eq
    n1 = save(out, idx_out)

    ksets = km["sets"]
    knew = [conservative(s) for s in ksets]
    kid_of = {}
    for i, s in enumerate(knew):
        kid_of.setdefault(s, i)

    def ksid(b):
        k = kid_of.get(b)
        if k is None:
            k = kid_of[b] = len(knew)
            knew.append(b)
        return k

    kout = dict(km)
    kout["tests"] = tests
    kout["km"] = {lab: {k: ksid(precise(ksets[v], (lab, k), keys)) for k, v in ent.items()}
                  for lab, ent in km["km"].items()}
    kout["sets"] = knew
    n2 = save(kout, keys_out)
    for tid in TAIL:
        print(tid, {k: len(v[(tid, "tail")]) for k, v in (("lines", lines), ("keys", keys), ("recv", recv))},
              file=sys.stderr)
    print(f"index raw={n1} keys raw={n2}", file=sys.stderr)


if __name__ == "__main__":
    main()
