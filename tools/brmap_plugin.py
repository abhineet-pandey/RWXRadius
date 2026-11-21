"""pytest plugin used OFFLINE to build baseline dependency maps of the pinned sqlglot.

Per test it records:
  * executed lines (sys.monitoring LINE events) of files under the repo,
    split into setup-phase and the rest (class/module fixtures are later
    attributed to every test sharing them),
  * repository files opened / directories listed (audit hook),
  * receiver classes (type(self) / cls) of every sqlglot function call.

Enabled with `-p brmap_plugin` and env BRMAP_OUT=<dir>.
"""
from __future__ import annotations

import concurrent.futures
import concurrent.futures.process
import os
import pickle
import sys
import threading

import pytest

ROOT = os.path.realpath(os.environ.get("BRMAP_ROOT", os.getcwd())) + os.sep
OUT = os.environ.get("BRMAP_OUT")
RECV = os.environ.get("BRMAP_RECV", "1") == "1"
CMP = os.environ.get("BRMAP_CMP") == "1"  # comparison value profiles instead of tracing
KEYS = os.environ.get("BRMAP_KEYS") == "1"  # dict-key read map instead of tracing

mon = sys.monitoring
TOOL = 3  # free tool id (0..5); 1 is coverage, 2 profiler, 5 optimizer

_state = {"ctx": None, "phase": "collect"}
DATA: dict = {}
COLLECT_LINES: dict = {}
_code_keep: dict = {}  # code -> relpath or None


def _rel(code):
    try:
        return _code_keep[code]
    except KeyError:
        fn = code.co_filename
        r = None
        if fn.startswith(ROOT):
            r = fn[len(ROOT):]
        _code_keep[code] = r
        return r


def _bucket(kind):
    ctx = _state["ctx"]
    if ctx is None:
        return None
    rec = DATA.get(ctx)
    if rec is None:
        rec = DATA[ctx] = {"lines": {}, "setup": {}, "opens": set(), "recv": set()}
    if kind == "lines":
        return rec["setup"] if _state["phase"] == "setup" else rec["lines"]
    return rec[kind]


def _line_cb(code, line):
    rel = _rel(code)
    if rel is None:
        return mon.DISABLE
    ctx = _state["ctx"]
    if ctx is None:
        COLLECT_LINES.setdefault(rel, set()).add(line)
    else:
        b = _bucket("lines")
        s = b.get(rel)
        if s is None:
            s = b[rel] = set()
        s.add(line)
    return mon.DISABLE


_sqlglot_prefix = "sqlglot" + os.sep
_recv_code: dict = {}


def _start_cb(code, offset):
    try:
        info = _recv_code[code]
    except KeyError:
        rel = _rel(code)
        info = None
        if rel is not None and code.co_argcount and code.co_varnames[0] in ("self", "cls"):
            info = code.co_varnames[0]
        _recv_code[code] = info
    if info is None:
        return mon.DISABLE
    if _state["ctx"] is None:
        return None
    f = sys._getframe(1)
    try:
        obj = f.f_locals[info]
    except Exception:
        return None
    t = obj if info == "cls" and isinstance(obj, type) else type(obj)
    _bucket("recv").add(t)
    return None


def _audit(event, args):
    if event in ("subprocess.Popen", "os.posix_spawn", "os.exec", "os.system", "os.fork", "os.forkpty"):
        if _state["ctx"] is not None:
            _bucket("opens").add(("subprocess", "*"))
        return
    if event == "open":
        path = args[0]
        if isinstance(path, (bytes, int)) or path is None:
            return
        path = os.fspath(path)
    elif event in ("os.listdir", "os.scandir", "glob.glob", "os.walk"):
        path = args[0] if args else "."
        if path is None or isinstance(path, (bytes, int)):
            return
        path = os.fspath(path)
    else:
        return
    ab = os.path.realpath(path) if not os.path.isabs(path) else os.path.normpath(path)
    if not ab.startswith(ROOT):
        return
    rel = ab[len(ROOT):]
    if rel.endswith((".pyc",)) or "__pycache__" in rel:
        return
    if _state["ctx"] is None:
        return
    # skip source reads by the import system
    if rel.endswith(".py"):
        f = sys._getframe(1)
        while f is not None:
            if f.f_code.co_filename.startswith("<frozen importlib"):
                return
            f = f.f_back
    _bucket("opens").add((event, rel))


class _SyncExecutor(concurrent.futures.Executor):
    """In-process stand-in for ProcessPoolExecutor while mapping, so work done
    in pool tasks is attributed to the test that submitted it."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **k):
        self._lock = threading.Lock()
        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(tuple(initargs))))

    def submit(self, fn, /, *args, **kwargs):
        # Same serialization path as a real process pool: pickle the call and
        # the result (sqlglot expressions pickle through serde).
        fut = concurrent.futures.Future()
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            fut.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except BaseException as e:  # noqa
            fut.set_exception(e)
        return fut

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return [self.submit(fn, *a).result() for a in zip(*iterables)]

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass


def pytest_configure(config):
    if not OUT:
        return
    concurrent.futures.ProcessPoolExecutor = _SyncExecutor
    concurrent.futures.process.ProcessPoolExecutor = _SyncExecutor
    try:
        import pandas.testing as pt

        orig = pt.assert_frame_equal

        def afe(*a, **k):
            if any(type(x).__name__ == "FakeFrame" for x in a):
                return None
            return orig(*a, **k)

        pt.assert_frame_equal = afe
    except Exception:
        pass
    if CMP:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import cmpprobe

        cmpprobe.install(ROOT)
        return
    if KEYS:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import keyprobe

        keyprobe.install(_state)
        sys.addaudithook(_audit)
        return
    mon.use_tool_id(TOOL, "brmap")
    mon.register_callback(TOOL, mon.events.LINE, _line_cb)
    ev = mon.events.LINE
    if RECV:
        mon.register_callback(TOOL, mon.events.PY_START, _start_cb)
        ev |= mon.events.PY_START
    mon.set_events(TOOL, ev)
    sys.addaudithook(_audit)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    if not OUT or CMP:
        yield
        return
    _state["ctx"] = item.nodeid
    if KEYS:
        _bucket("opens")
    _state["phase"] = "call"
    mon.restart_events()
    try:
        yield
    finally:
        _state["ctx"] = None


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_setup(item):
    _state["phase"] = "setup"
    try:
        yield
    finally:
        _state["phase"] = "call"


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    _state["phase"] = "setup"  # class/module teardown is shared too
    try:
        yield
    finally:
        _state["phase"] = "call"


OUTCOMES: dict = {}
DUR: dict = {}


def pytest_runtest_logreport(report):
    if not OUT:
        return
    o = OUTCOMES.setdefault(report.nodeid, set())
    o.add((report.when, report.outcome))
    DUR[report.nodeid] = DUR.get(report.nodeid, 0.0) + report.duration


def pytest_sessionfinish(session):
    if not OUT:
        return
    os.makedirs(OUT, exist_ok=True)
    wid = os.environ.get("PYTEST_XDIST_WORKER", "main")
    if KEYS:
        import keyprobe

        keyprobe.dump(os.path.join(OUT, f"keys_{wid}.pkl"), DATA, OUTCOMES)
        return
    if CMP:
        import cmpprobe

        cmpprobe.dump(os.path.join(OUT, f"cmp_{wid}.pkl"))
        return
    mon.set_events(TOOL, 0)

    def tname(t):
        return f"{t.__module__}.{t.__qualname__}"

    data = {}
    for k, v in DATA.items():
        data[k] = {
            "lines": {f: sorted(s) for f, s in v["lines"].items()},
            "setup": {f: sorted(s) for f, s in v["setup"].items()},
            "opens": sorted(v["opens"]),
            "recv": sorted({tname(k) for t in v["recv"] for k in getattr(t, "__mro__", (t,))}),
        }
    with open(os.path.join(OUT, f"map_{wid}.pkl"), "wb") as fh:
        pickle.dump(
            {
                "tests": data,
                "collect": {f: sorted(s) for f, s in COLLECT_LINES.items()},
                "outcomes": {k: sorted(v) for k, v in OUTCOMES.items()},
                "durations": DUR,
            },
            fh,
        )
