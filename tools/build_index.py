"""OFFLINE: compact raw baseline traces (brmap_plugin + import_pass) into the
runtime index. Every input is derived from the unchanged pinned checkout.

usage: python build_index.py <repo_root> <tests.json> <out.pkl.gz> <import_pass.pkl> <map_dir> [<map_dir>...]
"""
from __future__ import annotations

import ast
import hashlib
import glob
import gzip
import json
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor


def bits(idxs) -> int:
    if not idxs:
        return 0
    m = max(idxs)
    ba = bytearray(m // 8 + 1)
    for i in idxs:
        ba[i >> 3] |= 1 << (i & 7)
    return int.from_bytes(ba, "little")


def collect_closure(args):
    root, mod = args
    code = (
        "import sys,os,json;sys.path.insert(0,%r);os.chdir(%r);os.environ['SKIP_INTEGRATION']='0'\n"
        "import importlib\n"
        "try:\n importlib.import_module(%r); ok=True\nexcept Exception as e:\n ok=False\n"
        "r=os.path.realpath(%r)+os.sep\n"
        "fs=sorted({os.path.realpath(m.__file__)[len(r):] for m in list(sys.modules.values()) "
        "if getattr(m,'__file__',None) and os.path.realpath(m.__file__).startswith(r)})\n"
        "print(json.dumps([ok,fs]))\n"
    ) % (root, root, mod, root)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ)
    try:
        return mod, json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        print("closure failed", mod, out.stderr[-2000:], file=sys.stderr)
        return mod, [False, []]


def main():
    root, tests_json, out, imp_pkl = sys.argv[1:5]
    map_dirs = sys.argv[5:]
    root = os.path.realpath(root)
    manifest = json.load(open(tests_json))["tests"]
    order = [t["id"] for t in manifest]
    pos = {t: i for i, t in enumerate(order)}

    # ---- merge raw traces (union over runs, e.g. several hash seeds)
    per_test_lines: dict = defaultdict(lambda: defaultdict(set))
    per_test_setup: dict = defaultdict(lambda: defaultdict(set))
    opens: dict = defaultdict(set)
    lists: dict = defaultdict(set)
    recv: dict = defaultdict(set)
    bad_tests: set = set()
    seen: set = set()
    durations: dict = {}
    for md in map_dirs:
        for f in glob.glob(os.path.join(md, "map_*.pkl")):
            d = pickle.load(open(f, "rb"))
            for nid, v in d["tests"].items():
                seen.add(nid)
                for fn, ls in v["lines"].items():
                    per_test_lines[nid][fn].update(ls)
                for fn, ls in v["setup"].items():
                    per_test_setup[nid][fn].update(ls)
                for ev, p in v["opens"]:
                    if ev in ("open", "subprocess"):
                        opens[p].add(nid)
                    else:
                        lists[p].add(nid)
                recv[nid].update(v["recv"])
            for nid, oc in d["outcomes"].items():
                if any(o not in ("passed", "skipped") for _, o in oc):
                    bad_tests.add(nid)
            for nid, du in d.get("durations", {}).items():
                durations[nid] = max(durations.get(nid, 0), du)

    # class-scoped setup/teardown is attributed to every test of the class
    cls_setup: dict = defaultdict(lambda: defaultdict(set))
    for nid, files in per_test_setup.items():
        c = nid.rsplit("::", 1)[0]
        for fn, ls in files.items():
            cls_setup[c][fn].update(ls)

    line_tests: dict = defaultdict(lambda: defaultdict(list))
    file_tests: dict = defaultdict(set)
    for nid in order:
        i = pos[nid]
        c = nid.rsplit("::", 1)[0]
        files = defaultdict(set)
        for fn, ls in per_test_lines.get(nid, {}).items():
            files[fn] |= ls
        for fn, ls in cls_setup.get(c, {}).items():
            files[fn] |= ls
        for fn, ls in files.items():
            file_tests[fn].add(i)
            lt = line_tests[fn]
            for ln in ls:
                lt[ln].append(i)

    sets: list = []
    set_ids: dict = {}

    def sid(idxs) -> int:
        b = bits(sorted(set(idxs)))
        k = set_ids.get(b)
        if k is None:
            k = set_ids[b] = len(sets)
            sets.append(b)
        return k

    lines_idx = {fn: {ln: sid(ts) for ln, ts in lt.items()} for fn, lt in line_tests.items()}
    file_users = {fn: sid(ts) for fn, ts in file_tests.items()}
    opens_idx = {p: sid([pos[t] for t in ts if t in pos]) for p, ts in opens.items()}
    lists_idx = {p: sid([pos[t] for t in ts if t in pos]) for p, ts in lists.items()}
    users_raw: dict = defaultdict(list)
    for nid, names in recv.items():
        if nid in pos:
            for n in names:
                users_raw[n].append(pos[nid])
    users = {n: sid(ts) for n, ts in users_raw.items()}
    unknown = [pos[t] for t in order if t not in seen]
    incomplete = sid([pos[t] for t in bad_tests if t in pos] + unknown)

    test_files: dict = defaultdict(list)
    for t in manifest:
        test_files[t["file"]].append(pos[t["id"]])
    test_files_idx = {f: sid(ts) for f, ts in test_files.items()}

    # ---- import-time lines
    imp = pickle.load(open(imp_pkl, "rb"))
    imptime = {fn: frozenset(ls) for fn, ls in imp["lines"].items()}

    # ---- identifier readers (runtime lines) and import-time identifier sites
    name_lines: dict = defaultdict(lambda: defaultdict(set))
    imp_name_sites: dict = defaultdict(set)
    py_files = []
    for sub in ("sqlglot", "tests"):
        for dp, dn, fns in os.walk(os.path.join(root, sub)):
            dn[:] = [x for x in dn if x not in ("__pycache__", "fixtures")]
            for f in fns:
                if f.endswith(".py"):
                    py_files.append(os.path.relpath(os.path.join(dp, f), root))
    for rel in sorted(py_files):
        try:
            tree = ast.parse(open(os.path.join(root, rel), "rb").read())
        except SyntaxError:
            continue
        it = imptime.get(rel, frozenset())
        stack = [(tree, False)]
        while stack:
            node, in_func = stack.pop()
            keys = ()
            lo = hi = None
            base = None
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Store):
                keys, lo, hi = (node.id,), node.lineno, node.end_lineno
            elif isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Store):
                v = node.value
                base = v.id if isinstance(v, ast.Name) else v.attr if isinstance(v, ast.Attribute) else "?"
                keys, lo, hi = (node.attr, f"{base}.{node.attr}"), node.end_lineno, node.end_lineno
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.isidentifier():
                keys, lo, hi = (node.value,), node.lineno, node.end_lineno
                base = "?"
            elif isinstance(node, ast.alias):
                keys = (node.name.split(".")[-1], (node.asname or node.name).split(".")[0])
                lo, hi = node.lineno, node.end_lineno
                base = "?"
            elif isinstance(node, ast.Call):
                fn = node.func
                fname = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
                dyn = False
                if fname in ("getattr", "setattr", "hasattr", "delattr") and len(node.args) >= 2:
                    dyn = not isinstance(node.args[1], ast.Constant)
                elif fname in ("vars", "dir", "getmembers", "getattr_static"):
                    dyn = True
                if dyn:
                    for ln in range(node.lineno, node.end_lineno + 1):
                        name_lines["<dynamic>"][rel].add(ln)
            if isinstance(node, ast.Attribute) and node.attr == "__dict__":
                for ln in range(node.lineno, node.end_lineno + 1):
                    name_lines["<dynamic>"][rel].add(ln)
            for nm in keys:
                for ln in range(lo, hi + 1):
                    name_lines[nm][rel].add(ln)
                    if ln in it and "." not in nm:
                        imp_name_sites[nm].add((rel, ln, base, in_func))
            for f in node._fields:
                v = getattr(node, f, None)
                child_func = in_func or (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and f == "body"
                ) or (isinstance(node, ast.Lambda) and f == "body")
                if isinstance(v, list):
                    for x in v:
                        if isinstance(x, ast.AST):
                            stack.append((x, child_func))
                elif isinstance(v, ast.AST):
                    stack.append((v, child_func))
    names_idx = {}
    for nm, files in name_lines.items():
        acc = 0
        for rel, lns in files.items():
            lt = lines_idx.get(rel)
            if not lt:
                continue
            for ln in lns:
                k = lt.get(ln)
                if k is not None:
                    acc |= sets[k]
        if acc:
            k = set_ids.get(acc)
            if k is None:
                k = set_ids[acc] = len(sets)
                sets.append(acc)
            names_idx[nm] = k
    imp_name_sites = {k: sorted(v, key=repr) for k, v in imp_name_sites.items()}

    # ---- collection-time import closure of every test module
    test_mods = sorted({t["file"][:-3].replace("/", ".") for t in manifest})
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as ex:
        closures = dict(ex.map(collect_closure, [(root, m) for m in test_mods]))
    _, core = collect_closure((root, "sqlglot"))
    collect_importers: dict = defaultdict(set)
    collect_ok = {}
    for m, (ok, fs) in closures.items():
        tf = m.replace(".", "/") + ".py"
        collect_ok[tf] = ok
        for f in fs:
            collect_importers[f].add(tf)
    collect_importers = {k: sorted(v) for k, v in collect_importers.items()}

    # ---- comparison value profiles (optional): site -> observed signatures
    cmp = {}
    for pth in filter(None, os.environ.get("BR_CMP_PROFILES", "").split(",")):
        for k, sigs in pickle.load(open(pth, "rb")).items():
            cmp.setdefault(k[:5], set()).update(sigs)
    cmp = {k: frozenset(v) for k, v in cmp.items()}

    index = {
        "version": 1,
        "cmp": cmp,
        "tests": order,
        "sets": sets,
        "lines": lines_idx,
        "file_users": file_users,
        "opens": opens_idx,
        "lists": lists_idx,
        "users": users,
        "incomplete": incomplete,
        "test_files": test_files_idx,
        "imptime": imptime,
        "names": names_idx,
        "imp_name_sites": imp_name_sites,
        "collect_importers": collect_importers,
        "collect_ok": collect_ok,
        "core": sorted(core[1]),
        "src_hash": {rel: hashlib.sha1(open(os.path.join(root, rel), "rb").read()).hexdigest() for rel in py_files},
        "local_durations": {pos[k]: v for k, v in durations.items() if k in pos},
    }
    raw = pickle.dumps(index, protocol=5)
    with gzip.open(out, "wb", compresslevel=9) as fh:
        fh.write(raw)
    print(
        f"tests={len(order)} seen={len(seen & set(order))} sets={len(sets)} files={len(lines_idx)} "
        f"users={len(users)} names={len(names_idx)} bad={len(bad_tests)} raw={len(raw)} "
        f"gz={os.path.getsize(out)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
