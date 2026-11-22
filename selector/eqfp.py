"""Equivalence check for the TPC executor tests (test_execute_tpcds, 21.5 s).

Per fixture item the test parses the query, sends it through a process pool,
optimizes and plans it (the head), runs the plan in the Python executor and
compares the rows with duckdb's answer to the query transpiled to duckdb (the
tail). Every head output the tail consumes is fingerprinted: the parsed tree
as the pool hands it over, the optimized tree, the Plan, the duckdb SQL. When
a change does not reach the tail's code or data (pseudo-test T@tail, see
tools/add_tail.py) and leaves an item's fingerprint as at the baseline, the
tail runs exactly as at the baseline for that item and passes. Items whose
fingerprint differs are run for real.

The baseline fingerprints are taken in the image (python -m selector.eqfp)
with the same Python the run uses.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import types

EQ_TESTS = {
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": ("optimizer/tpc-ds/tpc-ds.sql", "TPCDS_SCHEMA"),
    "tests/test_executor.py::TestExecutor::test_execute_tpch": ("optimizer/tpc-h/tpc-h.sql", "TPCH_SCHEMA"),
}

_SKIP_SLOTS = {"parent", "_hash"}


_FIELDS: dict = {}


def _slots_of(t):
    f = _FIELDS.get(t)
    if f is None:
        f = []
        for c in reversed(t.__mro__):
            sl = c.__dict__.get("__slots__", ()) or ()
            if isinstance(sl, str):
                sl = (sl,)
            f.extend(x for x in sl if x not in _SKIP_SLOTS and x not in f)
        f = _FIELDS[t] = tuple(f)
    return f


_ATOMS = (bool, int, str, bytes, type(None))
_CALLABLES = (type, types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.BuiltinMethodType)
_EDGES = ("dependencies", "dependents")


class _Canon:
    """Deterministic text of an object graph: containers in order (sets
    sorted), objects as class + slots + __dict__ (no parent pointers or
    cached hashes), planner steps referenced by their content key."""

    def __init__(self, step_keys=None):
        self.out = []
        self.busy = set()
        self.step_keys = step_keys if step_keys is not None else {}

    def emit(self, x):
        o = self.out
        t = type(x)
        if t in _ATOMS:
            o.append(f"{t.__name__}:{x!r}")
            return
        if t is float:
            o.append(f"f:{x!r}")
            return
        if t is list or t is tuple:
            o.append("[" if t is list else "(")
            for v in x:
                self.emit(v)
            o.append("]")
            return
        if t is dict:
            o.append("{")
            for k, v in x.items():
                self.emit(k)
                o.append("=")
                self.emit(v)
            o.append("}")
            return
        if t is set or t is frozenset:
            o.append("<")
            o.extend(sorted(_one(v, self.step_keys) for v in x))
            o.append(">")
            return
        if isinstance(x, _CALLABLES):
            o.append(f"T:{getattr(x, '__module__', '')}.{getattr(x, '__qualname__', '?')}")
            return
        i = id(x)
        if i in self.busy:
            o.append(f"cycle:{t.__qualname__}")
            return
        self.busy.add(i)
        try:
            o.append(f"<{t.__module__}.{t.__qualname__}")
            n = 0
            for k in _slots_of(t):
                try:
                    v = getattr(x, k)
                except AttributeError:
                    continue
                n += 1
                o.append(k)
                self.emit(v)
            d = getattr(x, "__dict__", None)
            if d:
                for k, v in d.items():
                    if k in _SKIP_SLOTS:
                        continue
                    n += 1
                    o.append(k)
                    if k in _EDGES and (type(v) is set or type(v) is list):
                        # planner steps refer to each other: by content key
                        o.append("<" + ",".join(sorted(_step_key(s, self.step_keys) for s in v)) + ">")
                    else:
                        self.emit(v)
            if not n and t.__repr__ is not object.__repr__:
                o.append(repr(x))
            o.append(">")
        finally:
            self.busy.discard(i)


def _one(x, step_keys=None) -> str:
    c = _Canon(step_keys)
    c.emit(x)
    return "".join(c.out)


def _step_key(step, memo) -> str:
    """A planner step's content without its edges, plus its dependencies'
    keys (the dependency graph is acyclic)."""
    k = memo.get(id(step))
    if k is not None:
        return k
    memo[id(step)] = "pending"
    c = _Canon(memo)
    c.out.append(type(step).__qualname__)
    for a, v in vars(step).items():
        if a in _EDGES:
            continue
        c.out.append(a)
        c.emit(v)
    deps = sorted(_step_key(s, memo) for s in getattr(step, "dependencies", ()))
    c.out.append("deps<" + ",".join(deps) + ">")
    k = memo[id(step)] = hashlib.sha1("".join(c.out).encode("utf-8", "surrogateescape")).hexdigest()
    return k


def _h(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", "surrogateescape"))
        h.update(b"\0")
    return h.hexdigest()


def _plan_canon(plan) -> str:
    memo: dict = {}
    seen = []
    stack = [plan.root]
    ids = set()
    while stack:
        s = stack.pop()
        if id(s) in ids:
            continue
        ids.add(id(s))
        seen.append(s)
        stack.extend(getattr(s, "dependencies", ()))
        stack.extend(getattr(s, "dependents", ()))
    steps = sorted(_step_key(s, memo) + ":up<" + ",".join(
        sorted(_step_key(d, memo) for d in getattr(s, "dependents", ()))) + ">" for s in seen)
    rest = {k: v for k, v in vars(plan).items() if k not in ("root", "_dag")}
    return "plan[" + _step_key(plan.root, memo) + "|" + "|".join(steps) + "|" + _one(rest, memo) + "]"


def _item_fp(ex, parse_one, transpile, schema_src, meta, sql) -> str:
    parts = [repr(sorted(meta.items(), key=str)) if isinstance(meta, dict) else repr(meta)]
    try:
        e = parse_one(sql)
        e = pickle.loads(pickle.dumps(e))
        parts.append(_one(e))
        if meta.get("execute"):
            schema = ex.ensure_schema(pickle.loads(pickle.dumps(schema_src)), dialect=None)
            o = ex.optimize(e, schema, leave_tables_isolated=True, dialect=None)
            p = ex.Plan(o)
            parts.append(_one(o))
            parts.append(_plan_canon(p))
            parts.append(transpile(sql, write="duckdb")[0])
    except BaseException as err:  # noqa
        parts.append(f"raised {type(err).__name__}: {err}"[:500])
    return _h(*parts)


def _items(test_id):
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    import sqlglot.executor as ex
    import tests.helpers as th
    from sqlglot import parse_one, transpile

    fixture, schema_name = EQ_TESTS[test_id]
    items = list(th.load_sql_fixture_pairs(fixture))
    schema_src = pickle.loads(pickle.dumps(getattr(th, schema_name)))
    return items, lambda meta, sql: _item_fp(ex, parse_one, transpile, schema_src, meta, sql)


def fingerprints(test_id: str, which=None) -> list:
    """Item fingerprints of an EQ test with the sqlglot / tests modules this
    process imports. `which`: item indices (default all)."""
    items, fp = _items(test_id)
    return [None if which is not None and i not in which else fp(meta, sql)
            for i, (meta, sql, _) in enumerate(items)]


def differing(test_id: str, base: list, which=None, cost=None):
    """Item indices to run for real (fingerprint not as at the baseline).
    None: the item list changed. "over": running the changed items would cost
    more than `cost` = (base ms, per-item ms, limit ms) allows (checked as
    items are fingerprinted, executable items first)."""
    items, fp = _items(test_id)
    if len(items) != len(base):
        return None
    idx = range(len(items)) if which is None else sorted(which)
    idx = sorted(idx, key=lambda i: (not items[i][0].get("execute"), i))
    out = []
    run = cost[0] if cost else 0
    for i in idx:
        meta, sql, _ = items[i]
        if fp(meta, sql) != base[i]:
            out.append(i)
            if cost:
                its = cost[1]
                run += max(0, (its[i] if i < len(its) else 0) - cost[0])
                if run > cost[2]:
                    return "over"
    return sorted(out)


def build(repo: str, out: str):
    os.chdir(repo)
    sys.path.insert(0, repo)
    os.environ["SKIP_INTEGRATION"] = "0"
    res = {}
    for tid in EQ_TESTS:
        a = fingerprints(tid)
        b = fingerprints(tid)
        res[tid] = [x if x == y else "unstable" for x, y in zip(a, b)]
    with open(out, "w") as fh:
        json.dump(res, fh)
    print({k: (len(v), v.count("unstable")) for k, v in res.items()}, file=sys.stderr)


if __name__ == "__main__":
    build(os.path.realpath(sys.argv[1]), sys.argv[2])
