"""OFFLINE: per-test baseline tracing, every test in a fresh forked process.

A zygote preloads third-party modules (never sqlglot or the tests), then forks
one child per test. The child traces (sys.monitoring) while it imports sqlglot
and the test module and runs the single unittest case, class fixtures
included. Lazily built caches, class setup and import-time code are therefore
attributed to every test that depends on them, not only to the first test of a
process.

usage: python isomap.py <repo_root> <tests.json> <out_dir> [jobs]
Writes <out_dir>/map_iso.pkl in the same format as brmap_plugin.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

root = os.path.realpath(sys.argv[1])
tests_json = sys.argv[2]
out_dir = sys.argv[3]
jobs = int(sys.argv[4]) if len(sys.argv) > 4 else (os.cpu_count() or 4)
CMP_MODE = len(sys.argv) > 5 and sys.argv[5] == "cmp"
ROOT = root + os.sep
os.makedirs(out_dir, exist_ok=True)
part_dir = os.path.join(out_dir, "parts")
os.makedirs(part_dir, exist_ok=True)

os.environ["SKIP_INTEGRATION"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ---- zygote preloads (third-party / stdlib only)
import concurrent.futures  # noqa
import concurrent.futures.process  # noqa
import csv  # noqa
import gzip  # noqa
import importlib  # noqa
import multiprocessing  # noqa
import unittest  # noqa
import unittest.mock  # noqa

for m in ("numpy", "pandas", "pandas.testing", "duckdb"):
    try:
        __import__(m)
    except Exception as e:  # noqa
        print("preload failed", m, e, file=sys.stderr)


# class-creation hooks run for every class at import; their receivers say
# nothing about which classes a test uses
CREATION_HOOKS = {"__init_subclass__", "__set_name__", "__prepare__", "__class_getitem__"}


def child(nodeid: str, out_path: str):
    mon = sys.monitoring
    TOOL = 3
    lines: dict = {}
    recv: set = set()
    opens: set = set()
    code_keep: dict = {}
    recv_code: dict = {}
    seen_runtime: set = set()

    def rel_of(code):
        try:
            return code_keep[code]
        except KeyError:
            fn = code.co_filename
            r = fn[len(ROOT):] if fn.startswith(ROOT) else None
            code_keep[code] = r
            return r

    def line_cb(code, line):
        r = rel_of(code)
        if r is not None:
            s = lines.get(r)
            if s is None:
                s = lines[r] = set()
            s.add(line)
        return mon.DISABLE

    def start_cb(code, offset):
        try:
            info = recv_code[code]
        except KeyError:
            info = None
            if (
                rel_of(code) is not None
                and code.co_argcount
                and code.co_varnames[0] in ("self", "cls")
                and code.co_name not in CREATION_HOOKS
            ):
                info = code.co_varnames[0]
            recv_code[code] = info
        if info is None:
            return mon.DISABLE
        f = sys._getframe(1)
        try:
            obj = f.f_locals[info]
        except Exception:  # noqa
            return None
        if info == "self" and isinstance(obj, type) and code.co_name in ("__init__", "__new__"):
            return None
        t = obj if info == "cls" and isinstance(obj, type) else type(obj)
        key = (code, t)
        if key in seen_runtime:
            return None
        # receivers seen while a module is being imported (registries built
        # in class bodies) are the same for every importer: ignore them
        g = f.f_back
        while g is not None:
            if g.f_code.co_filename.startswith("<frozen importlib"):
                return None
            g = g.f_back
        seen_runtime.add(key)
        recv.add(t)
        return None

    def audit(event, args):
        if event in ("subprocess.Popen", "os.posix_spawn", "os.exec", "os.system"):
            opens.add(("subprocess", "*"))
            return
        if event == "open":
            path = args[0]
        elif event in ("os.listdir", "os.scandir", "glob.glob", "os.walk"):
            path = args[0] if args else "."
        else:
            return
        if path is None or isinstance(path, (bytes, int)):
            return
        path = os.fspath(path)
        ab = os.path.normpath(path if os.path.isabs(path) else os.path.join(os.getcwd(), path))
        if not ab.startswith(ROOT):
            return
        rel = ab[len(ROOT):]
        if rel.endswith(".pyc") or "__pycache__" in rel:
            return
        if rel.endswith(".py") and event == "open":
            f = sys._getframe(1)
            while f is not None:
                if f.f_code.co_filename.startswith("<frozen importlib"):
                    return
                f = f.f_back
        opens.add((event, rel))

    # process pools run in-process with the same pickle round-trip
    import pickle as _pk

    class SyncExecutor(concurrent.futures.Executor):
        def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **k):
            if initializer is not None:
                initializer(*_pk.loads(_pk.dumps(tuple(initargs))))

        def submit(self, fn, /, *args, **kwargs):
            fut = concurrent.futures.Future()
            try:
                fn, args, kwargs = _pk.loads(_pk.dumps((fn, args, kwargs)))
                fut.set_result(_pk.loads(_pk.dumps(fn(*args, **kwargs))))
            except BaseException as e:  # noqa
                fut.set_exception(e)
            return fut

        def map(self, fn, *iterables, timeout=None, chunksize=1):
            return [self.submit(fn, *a).result() for a in zip(*iterables)]

        def shutdown(self, wait=True, *, cancel_futures=False):
            pass

    concurrent.futures.ProcessPoolExecutor = SyncExecutor
    concurrent.futures.process.ProcessPoolExecutor = SyncExecutor
    try:
        import pandas.testing as pt

        orig = pt.assert_frame_equal

        def afe(*a, **k):
            if any(type(x).__name__ == "FakeFrame" for x in a):
                return None
            return orig(*a, **k)

        pt.assert_frame_equal = afe
    except Exception:  # noqa
        pass

    os.chdir(root)
    sys.path.insert(0, root)
    sys.dont_write_bytecode = True
    if CMP_MODE:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import cmpprobe

        cmpprobe.install(root)
    else:
        mon.use_tool_id(TOOL, "iso")
        mon.register_callback(TOOL, mon.events.LINE, line_cb)
        mon.register_callback(TOOL, mon.events.PY_START, start_cb)
        mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)
        sys.addaudithook(audit)
    t0 = time.process_time()
    outcome = "failed"
    try:
        f, c, m = nodeid.split("::")
        mod = importlib.import_module(f[:-3].replace("/", "."))
        tc = getattr(mod, c)(m)
        res = unittest.TestResult()
        unittest.TestSuite([tc]).run(res)
        bad = res.errors or res.failures or res.unexpectedSuccesses
        outcome = "failed" if bad else "passed"
        if bad:
            msg = (res.errors + res.failures)[0][1][-600:] if (res.errors or res.failures) else "unexpected success"
            print(f"FAIL {nodeid}: {msg}", file=sys.stderr)
    except BaseException as e:  # noqa
        print(f"ERROR {nodeid}: {type(e).__name__}: {e}", file=sys.stderr)
    if CMP_MODE:
        cmpprobe.dump(out_path)
        return
    mon.set_events(TOOL, 0)
    cpu = time.process_time() - t0

    def tname(t):
        return f"{t.__module__}.{t.__qualname__}"

    rec = {
        "lines": {k: sorted(v) for k, v in lines.items()},
        "setup": {},
        "opens": sorted(opens),
        "recv": sorted({tname(k) for t in recv for k in getattr(t, "__mro__", (t,))}),
    }
    with open(out_path, "wb") as fh:
        pickle.dump({"nodeid": nodeid, "rec": rec, "outcome": outcome, "cpu": cpu}, fh)


def main():
    tests = json.load(open(tests_json))["tests"]
    ids = [t["id"] for t in tests]
    # heaviest first for better packing
    secs = {t["id"]: t["seconds"] for t in tests}
    order = sorted(range(len(ids)), key=lambda i: -secs[ids[i]])
    todo = [i for i in order if not os.path.exists(os.path.join(part_dir, f"{i}.pkl"))]
    running = {}
    t0 = time.time()
    done = 0
    while todo or running:
        while todo and len(running) < jobs:
            i = todo.pop(0)
            pid = os.fork()
            if pid == 0:
                try:
                    child(ids[i], os.path.join(part_dir, f"{i}.pkl"))
                finally:
                    os._exit(0)
            running[pid] = (i, time.time())
        pid, status = os.wait()
        if pid in running:
            running.pop(pid)
            done += 1
            if done % 100 == 0:
                print(f"{done} done, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    if CMP_MODE:
        prof = {}
        for i, nid in enumerate(ids):
            p = os.path.join(part_dir, f"{i}.pkl")
            if not os.path.exists(p):
                print("MISSING", nid, file=sys.stderr)
                continue
            for k, sigs in pickle.load(open(p, "rb")).items():
                prof.setdefault(k, set()).update(sigs)
        with open(os.path.join(out_dir, "cmp_profile.pkl"), "wb") as fh:
            pickle.dump(prof, fh)
        print(f"cmp profile: {len(prof)} sites in {time.time() - t0:.0f}s", file=sys.stderr)
        return
    data = {}
    outcomes = {}
    durations = {}
    for i, nid in enumerate(ids):
        p = os.path.join(part_dir, f"{i}.pkl")
        if not os.path.exists(p):
            print("MISSING", nid, file=sys.stderr)
            continue
        d = pickle.load(open(p, "rb"))
        data[nid] = d["rec"]
        outcomes[nid] = [("call", d["outcome"])]
        durations[nid] = d["cpu"]
    with open(os.path.join(out_dir, "map_iso.pkl"), "wb") as fh:
        pickle.dump({"tests": data, "collect": {}, "outcomes": outcomes, "durations": durations}, fh)
    print(f"wrote {len(data)} tests in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
