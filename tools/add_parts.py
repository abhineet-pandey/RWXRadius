"""OFFLINE: append the partial-replay pseudo-tests (selector/parts.py) to
maps/index.pkl.gz and maps/keys.pkl.gz.

usage: python add_parts.py <index.pkl.gz> <keys.pkl.gz> <out_index> <out_keys> <repo_root>
                          --lines <dir>... --keys <dir>...

The pseudo-tests were traced from tests/test_br_parts.py (dev only): class
P__<Class>__<method>, test_pbase (no items) and test_p<i> (item i alone).

For each partial test T:
  * line map: a line T executed is credited to T@base if the base trace ran
    it, to T@i if item i's trace ran it, and to T@base if neither did
    (conservative),
  * key map: the same per (dict, key) read,
  * class users: the same per receiver class (class fixtures count as base),
  * identifier readers: recomputed from the refined line map (same scan as
    build_index), checked to reproduce the old map on the real tests,
  * every other map (file readers, test files, incomplete): T@base wherever
    T appears, so any rule not refined per item keeps the whole test.
The real test keeps all its bits; the runtime replays T on the items whose
pseudo-tests are candidates only when T@base is not.
"""
from __future__ import annotations

import glob
import gzip
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from selector.parts import PARTS, pseudo_id  # noqa: E402


def load(p):
    with open(p, "rb") as fh:
        return pickle.loads(gzip.decompress(fh.read()))


def save(obj, p):
    raw = pickle.dumps(obj, protocol=5)
    with gzip.open(p, "wb", compresslevel=9) as fh:
        fh.write(raw)
    return len(raw)


def pseudo_of(nodeid):
    """tests/test_br_parts.py::P__TestExecutor__test_execute_tpcds::test_p017 -> (T, 17)"""
    try:
        _, cls, meth = nodeid.split("::")
    except ValueError:
        return None
    if not cls.startswith("P__"):
        return None
    _, cname, tmeth = cls.split("__", 2)
    for tid in PARTS:
        if tid.endswith(f"::{cname}::{tmeth}"):
            part = "base" if meth == "test_pbase" else int(meth[len("test_p"):])
            return tid, part
    return None


def main():
    a = sys.argv[1:]
    idx_in, keys_in, idx_out, keys_out, root = a[:5]
    rest = a[5:]
    line_dirs = rest[rest.index("--lines") + 1: rest.index("--keys")]
    key_dirs = rest[rest.index("--keys") + 1:]

    idx = load(idx_in)
    km = load(keys_in)
    assert km["tests"] == idx["tests"], "index and key map disagree"
    if any("@" in t for t in idx["tests"]):
        raise SystemExit("index already has pseudo-tests")
    tests = list(idx["tests"])
    pos = {t: i for i, t in enumerate(tests)}
    parts = {}
    for tid, (_, _, _, n) in PARTS.items():
        if tid not in pos:
            continue
        b = len(tests)
        tests.append(pseudo_id(tid, "base"))
        ps = []
        for i in range(n):
            ps.append(len(tests))
            tests.append(pseudo_id(tid, i))
        parts[tid] = {"real": pos[tid], "base": b, "parts": ps}
    real_bit = {tid: 1 << p["real"] for tid, p in parts.items()}

    # ---- traces
    recv_of: dict = {}  # (tid, part) -> receiver class names
    recv_setup: dict = {}  # tid -> receivers seen in class fixtures
    lines_of: dict = {}  # (tid, part) -> set((file, line))
    setup_of: dict = {}  # tid -> set((file, line)) of the pseudo class's fixtures
    bad = set()
    for d in line_dirs:
        for f in glob.glob(os.path.join(d, "map_*.pkl")):
            x = pickle.load(open(f, "rb"))
            for nid, v in x["tests"].items():
                pp = pseudo_of(nid)
                if pp is None:
                    continue
                s = lines_of.setdefault(pp, set())
                for fn, ls in v["lines"].items():
                    s.update((fn, ln) for ln in ls)
                su = setup_of.setdefault(pp[0], set())
                for fn, ls in v["setup"].items():
                    su.update((fn, ln) for ln in ls)
                if "recv_setup" not in v:
                    raise SystemExit("traces without recv_setup: re-trace with the current brmap_plugin")
                recv_of.setdefault(pp, set()).update(v["recv"])
                recv_setup.setdefault(pp[0], set()).update(v["recv_setup"])
            for nid, oc in x["outcomes"].items():
                pp = pseudo_of(nid)
                if pp and any(o not in ("passed", "skipped") for _, o in oc):
                    bad.add(pp)
    keys_of: dict = {}
    ksetup_of: dict = {}
    for d in key_dirs:
        for f in glob.glob(os.path.join(d, "keys_*.pkl")):
            x = pickle.load(open(f, "rb"))
            for nid, r in x["tests"].items():
                pp = pseudo_of(nid)
                if pp is None:
                    continue
                keys_of.setdefault(pp, set()).update(r["call"])
                ksetup_of.setdefault(pp[0], set()).update(r["setup"])
            for nid in x["subproc"]:
                pp = pseudo_of(nid)
                if pp:
                    bad.add(pp)
            for nid, oc in x["outcomes"].items():
                pp = pseudo_of(nid)
                if pp and any(o not in ("passed", "skipped") for _, o in oc):
                    bad.add(pp)
    # a pseudo-test that ran and passed but executed no traced line / read no
    # tracked dict has no record: that is an empty trace
    ran_l, ran_k = set(), set()
    for d in line_dirs:
        for f in glob.glob(os.path.join(d, "map_*.pkl")):
            for nid, oc in pickle.load(open(f, "rb"))["outcomes"].items():
                pp = pseudo_of(nid)
                if pp and any(w == "call" and o == "passed" for w, o in oc):
                    ran_l.add(pp)
    for d in key_dirs:
        for f in glob.glob(os.path.join(d, "keys_*.pkl")):
            for nid, oc in pickle.load(open(f, "rb"))["outcomes"].items():
                pp = pseudo_of(nid)
                if pp and any(w == "call" and o == "passed" for w, o in oc):
                    ran_k.add(pp)
    missing = []
    for tid, p in parts.items():
        for part in ["base"] + list(range(len(p["parts"]))):
            if (tid, part) not in lines_of and (tid, part) in ran_l:
                lines_of[(tid, part)] = set()
                recv_of[(tid, part)] = set()
            if (tid, part) not in keys_of and (tid, part) in ran_k:
                keys_of[(tid, part)] = set()
            if (tid, part) not in lines_of or (tid, part) not in keys_of:
                missing.append((tid, part))
    if missing or bad:
        raise SystemExit(f"incomplete pseudo traces: missing={missing[:5]} bad={sorted(bad)[:5]}")
    # class fixtures run for every item: they belong to the base
    for tid in parts:
        lines_of[(tid, "base")] |= setup_of.get(tid, set())
        keys_of[(tid, "base")] |= ksetup_of.get(tid, set())
        recv_of[(tid, "base")] |= recv_setup.get(tid, set())

    def base_bits(s):
        acc = 0
        for tid, p in parts.items():
            if s & real_bit[tid]:
                acc |= 1 << p["base"]
        return acc

    def precise(s, hit):
        """`hit(tid, part)`: whether that pseudo trace has the event."""
        acc = 0
        for tid, p in parts.items():
            if not s & real_bit[tid]:
                continue
            got = 0
            if hit(tid, "base"):
                got |= 1 << p["base"]
            for i, q in enumerate(p["parts"]):
                if hit(tid, i):
                    got |= 1 << q
            acc |= got or (1 << p["base"])
        return acc

    # ---- index
    sets = idx["sets"]
    new_sets: list = []
    sid_of: dict = {}

    def sid(b):
        k = sid_of.get(b)
        if k is None:
            k = sid_of[b] = len(new_sets)
            new_sets.append(b)
        return k

    conv_cache: dict = {}

    def conv(old_sid):
        r = conv_cache.get(old_sid)
        if r is None:
            s = sets[old_sid]
            r = conv_cache[old_sid] = sid(s | base_bits(s))
        return r

    out = dict(idx)
    out["tests"] = tests
    for key in ("file_users", "opens", "lists", "test_files"):
        out[key] = {k: conv(v) for k, v in idx[key].items()}
    out["users"] = {
        c: sid(sets[v] | precise(sets[v], lambda tid, part: c in recv_of[(tid, part)]))
        for c, v in idx["users"].items()
    }
    out["incomplete"] = conv(idx["incomplete"])
    new_lines = {}
    for fn, lt in idx["lines"].items():
        nl = {}
        for ln, k in lt.items():
            s = sets[k]
            nl[ln] = sid(s | precise(s, lambda tid, part: (fn, ln) in lines_of[(tid, part)]))
        new_lines[fn] = nl
    out["lines"] = new_lines
    # identifier readers from the refined line map
    from build_index import scan_names

    name_lines, _, _ = scan_names(os.path.realpath(root), idx["imptime"])
    real_mask = (1 << len(idx["tests"])) - 1
    names = {}
    mismatch = 0
    for nm, files in name_lines.items():
        acc = 0
        for rel, lns in files.items():
            lt = new_lines.get(rel)
            if not lt:
                continue
            for ln in lns:
                k = lt.get(ln)
                if k is not None:
                    acc |= new_sets[k]
        if acc:
            old_k = idx["names"].get(nm)
            if old_k is None or sets[old_k] != acc & real_mask:
                mismatch += 1
            names[nm] = sid(acc)
    if mismatch or names.keys() != idx["names"].keys():
        raise SystemExit(f"recomputed identifier map differs from the index: {mismatch} names, "
                         f"{len(names.keys() ^ idx['names'].keys())} keys")
    out["names"] = names
    out["sets"] = new_sets
    out["parts"] = parts
    out["local_durations"] = idx["local_durations"]
    n1 = save(out, idx_out)

    # ---- key map
    ksets = km["sets"]
    knew: list = []
    ksid_of: dict = {}

    def ksid(b):
        k = ksid_of.get(b)
        if k is None:
            k = ksid_of[b] = len(knew)
            knew.append(b)
        return k

    kout = dict(km)
    kout["tests"] = tests
    kout["km"] = {
        lab: {k: ksid(ksets[v] | precise(ksets[v], lambda tid, part: (lab, k) in keys_of[(tid, part)]))
              for k, v in ent.items()}
        for lab, ent in km["km"].items()
    }
    kout["unknown"] = ksid(ksets[km["unknown"]] | base_bits(ksets[km["unknown"]]))
    kout["sets"] = knew
    n2 = save(kout, keys_out)
    print(f"pseudo-tests {len(tests) - len(idx['tests'])} for {len(parts)} tests; "
          f"index sets {len(sets)}->{len(new_sets)} raw={n1}; key sets {len(ksets)}->{len(knew)} raw={n2}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
