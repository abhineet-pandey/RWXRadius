"""Replay candidate tests against a patched copy of the pinned checkout.

The parent process (zygote) pre-imports heavy third-party modules once, then
forks one child per commit. A child builds a symlink overlay of the checkout in
/tmp with the patched files written for real, imports sqlglot and the test
modules fresh from it and runs the requested unittest cases. A test counts as
passing only when it ran to completion with no failure, error, failing subtest
or unexpected success; anything else (timeouts, crashes, unrun) is failing.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import time

TMP = os.environ.get("BR_TMP", "/tmp/br")


def _uses_duckdb(root: str, files) -> bool:
    """Whether any of these test files (as patched) mentions duckdb."""
    for f in files:
        try:
            with open(os.path.join(root, f), "rb") as fh:
                if b"duckdb" in fh.read():
                    return True
        except OSError:
            return True  # unknown: do not risk an address-space kill
    return False


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- overlay
def build_overlay(root: str, changed: dict, dest: str):
    real_dirs = {""}
    for p in changed:
        parts = p.split("/")
        for i in range(1, len(parts)):
            real_dirs.add("/".join(parts[:i]))

    def mirror(rel: str):
        src = os.path.join(root, rel) if rel else root
        dst = os.path.join(dest, rel) if rel else dest
        os.makedirs(dst, exist_ok=True)
        names = set(os.listdir(src)) if os.path.isdir(src) else set()
        for p in list(changed) + list(real_dirs):
            if p and os.path.dirname(p) == rel:
                names.add(os.path.basename(p))
        for name in sorted(names):
            r = f"{rel}/{name}" if rel else name
            d = os.path.join(dst, name)
            if r in changed:
                txt = changed[r]
                if txt is None:
                    continue
                with open(d, "wb") as fh:
                    fh.write(txt.encode("utf-8", "surrogateescape"))
                os.utime(d, (0, 0))  # never matches a cached .pyc timestamp
            elif r in real_dirs:
                mirror(r)
            else:
                os.symlink(os.path.join(src, name), d)

    mirror("")


# ---------------------------------------------------------------- pools
def _install_sync_pools():
    import concurrent.futures
    import concurrent.futures.process
    import pickle

    class SyncExecutor(concurrent.futures.Executor):
        """In-process stand-in for a process pool. Calls and results still go
        through pickle, as they would between processes."""

        def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **k):
            if initializer is not None:
                initializer(*pickle.loads(pickle.dumps(tuple(initargs))))

        def submit(self, fn, /, *args, **kwargs):
            fut = concurrent.futures.Future()
            try:
                fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
                fut.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
            except Exception as e:  # noqa
                fut.set_exception(e)
            return fut

        def map(self, fn, *iterables, timeout=None, chunksize=1):
            return [self.submit(fn, *a).result() for a in zip(*iterables)]

        def shutdown(self, wait=True, *, cancel_futures=False):
            pass

    concurrent.futures.ProcessPoolExecutor = SyncExecutor
    concurrent.futures.process.ProcessPoolExecutor = SyncExecutor


class _Timeout(BaseException):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


class _StopAfterFail(list):
    """Test list of a suite that, once the commit has a failing test, skips the
    cases not worth replaying any more. Skipped cases never pass, so they
    stay selected."""

    def __init__(self, cases, skip):
        super().__init__(cases)
        self._skip = skip

    def __iter__(self):
        for tc in list.__iter__(self):
            if not self._skip(tc):
                yield tc


# ---------------------------------------------------------------- child
def _child(job: dict, wfd: int):
    import importlib
    import unittest

    out = {"probe": {}, "fail": [], "pass": [], "t": {}}
    t0 = time.process_time()

    def emit():
        data = json.dumps(out).encode()
        view = memoryview(data)
        while view:
            n = os.write(wfd, view)
            view = view[n:]
        os.close(wfd)

    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        sys.stdout = sys.stderr = open(os.devnull, "w")
        ov = job["overlay"]
        os.chdir(ov)
        sys.path.insert(0, ov)
        sys.dont_write_bytecode = True
        os.environ["SKIP_INTEGRATION"] = "0"
        _install_sync_pools()
        signal.signal(signal.SIGALRM, _on_alarm)
        try:
            # a runaway mutant should kill this child, not the container
            import resource

            vm = 0
            with open("/proc/self/status") as fh:
                for ln in fh:
                    if ln.startswith("VmSize:"):
                        vm = int(ln.split()[1]) * 1024
            resource.setrlimit(resource.RLIMIT_AS, (vm + job.get("mem_headroom", 1536 << 20),) * 2)
        except Exception:  # noqa
            pass

        def probe_imports():
            # Changed modules must import; tests already imported them in their
            # natural order, this catches the ones nothing imported (or failed).
            for m in job["probe"]:
                if m in sys.modules:
                    continue
                try:
                    signal.setitimer(signal.ITIMER_REAL, 60)
                    importlib.import_module(m)
                except BaseException as e:  # noqa
                    out["probe"][m] = f"{type(e).__name__}: {e}"[:300]
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)

        def import_everything():
            # A definition-time change can break importing *other* modules
            # (one that builds objects from the changed class while importing).
            # Every module that imports at the baseline must still import.
            mods = job.get("import_all") or []
            if not mods:
                return
            from . import statefp

            try:
                signal.setitimer(signal.ITIMER_REAL, 60)
                ok = set(statefp.import_all(only=[m for m in mods if m not in sys.modules]))
            except BaseException:  # noqa
                ok = set()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            bad = [m for m in mods if m not in sys.modules and m not in ok]
            if bad:
                out["import_fail"] = bad

        sd = job.get("sd")
        if sd is not None:
            # Which import-time state did the patch change? Import every
            # module the baseline snapshot covers from the overlay, fingerprint
            # it and let the parent's planner pick the tests from the diff.
            res = None
            try:
                from . import statefp

                signal.setitimer(signal.ITIMER_REAL, 60)
                ok = statefp.import_all(only=sd["mods"])
                if len(ok) == len(sd["mods"]):
                    res = statefp.diff(sd["state"], statefp.snapshot(ok))
                else:
                    out["import_fail"] = sorted(set(sd["mods"]) - set(ok))
            except BaseException:  # noqa
                res = None
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            tests, rep, blind, note = sd["plan"](res)
            job["tests"] = tests
            out["sd"] = {"rep": rep, "blind": blind, "note": note}

        failed = set()
        passed = set()
        # equivalence check (selector/eqfp.py): items whose head outputs are as
        # at the baseline pass; only the others run for real
        eq_pass, eq_keep = set(), {}
        for item in job["tests"]:
            if len(item) > 5 and item[5] and job.get("eq_base") and item[0] in job["eq_base"]:
                nid = item[0]
                te = time.process_time()
                try:
                    from . import eqfp

                    signal.setitimer(signal.ITIMER_REAL, 120)
                    d = eqfp.differing(nid, job["eq_base"][nid], None if item[3] is None else set(item[3]),
                                       (job.get("eq_cost") or {}).get(nid))
                except BaseException:  # noqa
                    d = None
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                out.setdefault("eq", {})[nid] = {"diff": d if d is None or d == "over" else len(d),
                                                  "cpu": round(time.process_time() - te, 3)}
                if d is None:
                    continue
                if d == "over":
                    # running the changed items costs more than keeping the
                    # test: it stays selected unrun
                    failed.add(nid)
                    continue
                if not d:
                    eq_pass.add(nid)
                    continue
                eq_keep[nid] = d
        out["pre_cpu"] = round(time.process_time() - t0, 3)
        # group: file -> class -> [(nodeid, method, timeout)]
        groups: dict = {}
        keep_of = {}
        # tests still worth replaying once the commit is known to fail (their
        # replay is cheap against the schedule time they remove)
        worth = set()
        for item in job["tests"]:
            nid, timeout = item[0], item[1]
            if nid in failed:
                continue
            if nid in eq_pass:
                passed.add(nid)
                continue
            if len(item) > 3 and item[3] is not None:
                keep_of[nid] = item[3]
            if nid in eq_keep:
                keep_of[nid] = eq_keep[nid]
            if len(item) > 4 and item[4]:
                worth.add(nid)
            f, c, m = nid.split("::")
            groups.setdefault(f, {}).setdefault(c, []).append((nid, m, timeout))

        class Result(unittest.TestResult):
            def __init__(self, timeouts):
                super().__init__()
                self.timeouts = timeouts
                self.cur = None

            def _nid(self, test):
                return self.ids.get(id(test))

            def startTest(self, test):
                super().startTest(test)
                self._t0 = time.process_time()
                signal.setitimer(signal.ITIMER_REAL, self.timeouts.get(id(test), 30))

            def stopTest(self, test):
                if job.get("measure"):
                    nid = self._nid(test)
                    if nid:
                        out.setdefault("tcpu", {})[nid] = round(time.process_time() - self._t0, 4)
                signal.setitimer(signal.ITIMER_REAL, 120)
                super().stopTest(test)

            def _fail(self, test):
                nid = self._nid(test)
                if nid is None:
                    # class/module fixture error holder: the whole class fails
                    failed.update(self.ids.values())
                else:
                    failed.add(nid)

            def _why(self, test, err):
                if os.environ.get("BR_DEBUG_FAIL"):
                    import traceback

                    out.setdefault("why", {})[str(self._nid(test))] = "".join(
                        traceback.format_exception(*err))[-700:]

            def addError(self, test, err):
                self._why(test, err)
                self._fail(test)

            def addFailure(self, test, err):
                self._why(test, err)
                self._fail(test)

            def addSubTest(self, test, subtest, err):
                if err is not None:
                    self._why(test, err)
                    self._fail(test)

            def addUnexpectedSuccess(self, test):
                self._fail(test)

            def addSuccess(self, test):
                nid = self._nid(test)
                if nid:
                    passed.add(nid)

            def addSkip(self, test, reason):
                nid = self._nid(test)
                if nid:
                    passed.add(nid)

            def addExpectedFailure(self, test, err):
                nid = self._nid(test)
                if nid:
                    passed.add(nid)

        remaining_ms = sum(int(x[2]) for x in job["tests"]) if job["tests"] and len(job["tests"][0]) > 2 else 0
        stop_ms = job.get("stop_after_fail_ms", 0)
        dur_of = {x[0]: (x[2] if len(x) > 2 else 0) for x in job["tests"]}
        def worth_left(classes):
            return any(nid in worth for cl in classes.values() for nid, _, _ in cl)

        for f, classes in groups.items():
            if failed and remaining_ms <= stop_ms and not worth_left(classes):
                continue
            modname = f[:-3].replace("/", ".")
            tf = time.process_time()
            try:
                signal.setitimer(signal.ITIMER_REAL, 120)
                mod = importlib.import_module(modname)
            except BaseException as e:  # noqa
                for cl in classes.values():
                    failed.update(nid for nid, _, _ in cl)
                out.setdefault("modfail", {})[modname] = f"{type(e).__name__}: {e}"[:300]
                continue
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            out.setdefault("mcpu", {})[f] = round(time.process_time() - tf, 4)
            for cname, items in classes.items():
                if failed and remaining_ms <= stop_ms and not any(nid in worth for nid, _, _ in items):
                    continue
                cls = getattr(mod, cname, None)
                cases = []
                ids = {}
                touts = {}
                for nid, meth, timeout in items:
                    try:
                        tc = cls(meth)
                        if nid in keep_of:
                            # long fixture-driven test: only the items the change reaches
                            from .parts import restrict

                            restrict(tc, nid, keep_of[nid])
                    except BaseException:  # noqa
                        failed.add(nid)
                        continue
                    cases.append(tc)
                    ids[id(tc)] = nid
                    touts[id(tc)] = timeout
                if not cases:
                    continue
                res = Result(touts)
                res.ids = ids
                try:
                    signal.setitimer(signal.ITIMER_REAL, 120)
                    tc0 = time.process_time()
                    suite = unittest.TestSuite()
                    suite._cleanup = False
                    suite._tests = _StopAfterFail(cases, lambda tc: (
                        failed and remaining_ms <= stop_ms and ids.get(id(tc)) not in worth))
                    suite.run(res)
                    if job.get("measure"):
                        out.setdefault("ccpu", {})[f"{f}::{cname}"] = round(time.process_time() - tc0, 4)
                    for nid, _, _ in items:
                        remaining_ms -= dur_of.get(nid, 0)
                except BaseException:  # noqa
                    failed.update(ids.values())
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            out["t"][f] = round(time.process_time() - tf, 3)
        probe_imports()
        import_everything()
        for item in job["tests"]:
            nid = item[0]
            if nid in failed or nid not in passed:
                out["fail"].append(nid)
            else:
                out["pass"].append(nid)
    except BaseException as e:  # noqa
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    out["cpu"] = round(time.process_time() - t0, 3)
    emit()


# ---------------------------------------------------------------- parent
class Replayer:
    def __init__(self, repo: str, parallel: int = 2):
        self.repo = repo
        self.parallel = parallel
        self.running: dict = {}  # pid -> (job, rfd, start, buf)
        self.results: dict = {}
        os.makedirs(TMP, exist_ok=True)

    def preload(self, heavy: bool):
        import unittest  # noqa
        import concurrent.futures  # noqa
        import pickle  # noqa
        import importlib  # noqa

        if heavy:
            for m in ("numpy", "pandas", "pandas.testing", "duckdb"):
                try:
                    __import__(m)
                except Exception as e:  # noqa
                    log("preload failed", m, e)

    def start(self, key: str, overlay: dict, probe: list, tests: list, deadline: float, stop_after_fail_ms: int = 0,
              sd: dict | None = None, import_all: list | None = None, eq_base: dict | None = None,
              eq_cost: dict | None = None):
        dest = os.path.join(TMP, key)
        shutil.rmtree(dest, ignore_errors=True)
        build_overlay(self.repo, overlay, dest)
        job = {"overlay": dest, "probe": probe, "tests": tests, "stop_after_fail_ms": stop_after_fail_ms,
               "sd": sd, "import_all": import_all, "measure": bool(os.environ.get("BR_MEASURE")),
               "eq_base": eq_base, "eq_cost": eq_cost}
        if _uses_duckdb(dest, {str(t[0]).split("::")[0] for t in tests}):
            # duckdb reserves several GB of address space when it connects: an
            # address-space cap kills the child (measured: 3 GB dies, 4 GB ok),
            # losing every result of the commit. The container's memory limit
            # still bounds a runaway mutant; the kernel then kills this child,
            # whose candidates are all kept.
            job["mem_headroom"] = 16384 << 20
        if sd is not None:
            job["mem_headroom"] = 16384 << 20  # the child picks its tests later
        if os.environ.get("BR_MEM_HEADROOM_MB"):
            job["mem_headroom"] = int(os.environ["BR_MEM_HEADROOM_MB"]) << 20
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(r)
                _child(job, w)
            finally:
                os._exit(0)
        os.close(w)
        os.set_blocking(r, False)
        self.running[pid] = {"key": key, "fd": r, "buf": bytearray(), "deadline": deadline, "dest": dest}

    def poll(self, block: bool = True):
        """Collect finished children. Returns list of keys finished."""
        import select

        done = []
        while self.running:
            fds = {v["fd"]: pid for pid, v in self.running.items()}
            now = time.monotonic()
            timeout = max(0.0, min(v["deadline"] for v in self.running.values()) - now)
            rl, _, _ = select.select(list(fds), [], [], timeout if block else 0)
            for fd in rl:
                pid = fds[fd]
                st = self.running[pid]
                try:
                    chunk = os.read(fd, 1 << 16)
                except BlockingIOError:
                    continue
                if chunk:
                    st["buf"] += chunk
                    continue
                os.close(fd)
                os.waitpid(pid, 0)
                del self.running[pid]
                try:
                    self.results[st["key"]] = json.loads(st["buf"].decode())
                except Exception:  # noqa
                    self.results[st["key"]] = {"error": "child died"}
                shutil.rmtree(st["dest"], ignore_errors=True)
                done.append(st["key"])
            now = time.monotonic()
            for pid, st in list(self.running.items()):
                if now >= st["deadline"]:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    os.waitpid(pid, 0)
                    os.close(st["fd"])
                    del self.running[pid]
                    self.results[st["key"]] = {"error": "deadline"}
                    shutil.rmtree(st["dest"], ignore_errors=True)
                    done.append(st["key"])
            if done or not block:
                break
        return done
