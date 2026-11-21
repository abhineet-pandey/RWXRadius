"""OFFLINE (map build, BRMAP_KEYS=1): which dict entries each test reads.

After importing every sqlglot module, each plain dict held by a class
attribute or module global of sqlglot is replaced with a recording copy
(`TDict`). Reads record (container label, key); anything that sees the whole
dict (iteration, copies, views, equality, repr) records "*"; size checks
record "#". Phases follow brmap_plugin: setup/teardown reads are attributed to
every test of the class, reads outside any test (collection) to every test.

A dict that something else still references after the swap (an instance or
closure built at import time) is "leaky": reads through that reference are not
seen, so the run-time selector never narrows on it.
"""
from __future__ import annotations

import gc
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selector import statefp  # noqa: E402

_st: dict = {}
REC: dict = {}  # ctx -> {"call": set, "setup": set}
COLLECT: set = set()
LABELS: list = []  # lid -> canonical label
ALIASES: dict = {}  # label -> canonical label
LEAKY: set = set()
TRACKED: set = set()


_mute = [0]


def _rec(lid, key):
    if _mute[0]:
        return
    ctx = _st["ctx"]
    if ctx is None:
        b = COLLECT
    else:
        r = REC.get(ctx)
        if r is None:
            r = REC[ctx] = {"call": set(), "setup": set()}
        b = r["setup"] if _st["phase"] == "setup" else r["call"]
    try:
        b.add((lid, key))
    except TypeError:
        b.add((lid, "?"))


_get = dict.__getitem__


class TDict(dict):
    __slots__ = ("_lid",)

    def __getitem__(self, k):
        _rec(self._lid, k)
        return _get(self, k)

    def get(self, k, default=None):
        _rec(self._lid, k)
        return dict.get(self, k, default)

    def __contains__(self, k):
        _rec(self._lid, k)
        return dict.__contains__(self, k)

    def setdefault(self, k, default=None):
        _rec(self._lid, k)
        _rec(self._lid, "#")
        return dict.setdefault(self, k, default)

    def pop(self, k, *a):
        _rec(self._lid, k)
        _rec(self._lid, "#")
        return dict.pop(self, k, *a)

    def __setitem__(self, k, v):
        _rec(self._lid, k)
        _rec(self._lid, "#")
        dict.__setitem__(self, k, v)

    def __delitem__(self, k):
        _rec(self._lid, k)
        _rec(self._lid, "#")
        dict.__delitem__(self, k)

    def __len__(self):
        _rec(self._lid, "#")
        return dict.__len__(self)

    def _all(self):
        _rec(self._lid, "*")

    def __iter__(self):
        self._all()
        return dict.__iter__(self)

    def __reversed__(self):
        self._all()
        return dict.__reversed__(self)

    def keys(self):
        self._all()
        return dict.keys(self)

    def values(self):
        self._all()
        return dict.values(self)

    def items(self):
        self._all()
        return dict.items(self)

    def copy(self):
        self._all()
        return dict.copy(self)

    def __or__(self, o):
        self._all()
        return dict.__or__(dict.copy(self), o)

    def __ror__(self, o):
        self._all()
        r = dict(o)
        dict.update(r, dict.items(self))
        return r

    def __ior__(self, o):
        self._all()
        dict.update(self, o)
        return self

    def update(self, *a, **k):
        self._all()
        dict.update(self, *a, **k)

    def popitem(self):
        self._all()
        return dict.popitem(self)

    def clear(self):
        self._all()
        dict.clear(self)

    def __eq__(self, o):
        self._all()
        return dict.__eq__(self, o)

    def __ne__(self, o):
        self._all()
        return dict.__ne__(self, o)

    __hash__ = None

    def __repr__(self):
        self._all()
        return dict.__repr__(self)

    def __reduce_ex__(self, proto):
        self._all()
        return (dict, (dict(dict.items(self)),))

    def __reduce__(self):
        return self.__reduce_ex__(2)


class DispatchDict(dict):
    """The generator's per-class dispatch table (generator._DISPATCH_CACHE),
    built once per worker from TRANSFORMS and *_sql methods and then shared by
    every later test. A lookup is recorded as the read of that key of the
    TRANSFORMS dict it was built from (hit or miss: adding a key changes the
    handler), and of all of EXPR_CLASSES (which maps methods to classes)."""

    __slots__ = ("_tlid", "_elid")

    def _r(self, k):
        if self._tlid is not None:
            _rec(self._tlid, k)
        else:
            _rec(self._elid, "*")
        if self._elid is not None:
            _rec(self._elid, "*")

    def get(self, k, default=None):
        self._r(k)
        return dict.get(self, k, default)

    def __getitem__(self, k):
        self._r(k)
        return _get(self, k)

    def __contains__(self, k):
        self._r(k)
        return dict.__contains__(self, k)


def _wrap_dispatch():
    import sqlglot.generator as g
    from sqlglot import expressions as exp

    orig = g._build_dispatch
    e = getattr(exp, "EXPR_CLASSES", None)
    elid = getattr(e, "_lid", None)

    def build(cls):
        _mute[0] += 1
        try:
            d = orig(cls)
        finally:
            _mute[0] -= 1
        dd = DispatchDict(d)
        tr = cls.TRANSFORMS
        dd._tlid = getattr(tr, "_lid", None)
        dd._elid = elid
        if dd._tlid is None or elid is None:
            raise RuntimeError(f"untracked TRANSFORMS/EXPR_CLASSES for {cls}")
        return dd

    g._build_dispatch = build
    if g._DISPATCH_CACHE:
        raise RuntimeError("dispatch cache filled before the probe")


def install(state):
    _st.update(ctx=None, phase="collect")
    globals()["_st"] = state
    mods = statefp.import_all()
    by_id: dict = {}
    origs: dict = {}
    for label, owner, name, v in statefp.containers(mods):
        if type(v) is not dict or name.startswith("__") or not statefp.keyable(v):
            continue
        t = by_id.get(id(v))
        if t is None:
            t = TDict(v)
            t._lid = len(LABELS)
            LABELS.append(label)
            by_id[id(v)] = t
            origs[id(v)] = v
        ALIASES[label] = LABELS[t._lid]
        setattr(owner, name, t)
    TRACKED.update(LABELS)
    _wrap_dispatch()
    # anything still holding an original dict reads it unseen
    del v, t
    gc.collect()
    skip = {id(origs), id(by_id)}
    for obj in gc.get_objects():
        if id(obj) in skip:
            continue
        for r in gc.get_referents(obj):
            i = id(r)
            if i in origs and origs[i] is r:
                LEAKY.add(LABELS[by_id[i]._lid])
    print(f"keyprobe: tracked {len(LABELS)} dicts, {len(LEAKY)} leaky", file=sys.stderr)


def dump(path, data, outcomes):
    def conv(s):
        out = set()
        for lid, k in s:
            if isinstance(k, str) and k in ("*", "#", "?"):
                out.add((LABELS[lid], k))
                continue
            sk = statefp.stable_key(k)
            out.add((LABELS[lid], "?" if sk is None else sk))
        return out

    tests = {nid: {"call": conv(r["call"]), "setup": conv(r["setup"])} for nid, r in REC.items()}
    subproc = sorted(nid for nid, v in data.items() if any(e == "subprocess" for e, _ in v.get("opens", ())))
    with open(path, "wb") as fh:
        pickle.dump({
            "labels": list(LABELS), "aliases": dict(ALIASES), "leaky": sorted(LEAKY),
            "tests": tests, "collect": conv(COLLECT), "subproc": subproc,
            "outcomes": {k: sorted(v) for k, v in outcomes.items()},
        }, fh)
