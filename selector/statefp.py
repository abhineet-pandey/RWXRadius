"""Fingerprints of sqlglot's import-time state (class attributes and module
globals), keyed per entry for plain dicts.

The same walker labels containers for three users:
  * tools/keymap_plugin.py (offline): which dict keys each test reads,
  * tools/build_statefp.py (offline): the pinned checkout's fingerprints,
  * the replay child (run time): the patched checkout's fingerprints, diffed
    against the baseline to find exactly which entries a patch changed —
    including state derived from them while importing (merged dicts, tries,
    registries built by metaclasses).

Fingerprints ignore line numbers, so code that only moved compares equal.
"""
from __future__ import annotations

import enum
import hashlib
import importlib
import logging
import pkgutil
import sys
import types

OPAQUE = "\x00opaque"
MAX_DEPTH = 40

_PRIM = (str, int, float, bool, bytes, complex, type(None), type(Ellipsis))


def stable_key(k):
    """A process-independent text form of a dict key, or None."""
    t = type(k)
    if t is str:
        return "s" + k
    if t in (int, bool, float, type(None)):
        return f"{t.__name__}:{k!r}"
    if isinstance(k, enum.Enum):
        return f"e:{type(k).__module__}.{type(k).__qualname__}.{k.name}"
    if isinstance(k, type):
        return f"t:{k.__module__}.{k.__qualname__}"
    if t is tuple:
        parts = [stable_key(x) for x in k]
        if any(p is None for p in parts):
            return None
        return "(" + "\x1f".join(parts) + ")"
    if t is frozenset:
        parts = [stable_key(x) for x in k]
        if any(p is None for p in parts):
            return None
        return "{" + "\x1f".join(sorted(parts)) + "}"
    return None


class Fingerprinter:
    def __init__(self):
        self.memo: dict = {}
        self.keep: list = []  # keep memoised objects alive (ids stay unique)
        self.opaque = False

    def fp(self, v, depth=0) -> str:
        t = type(v)
        if t in _PRIM:
            return f"{t.__name__}:{v!r}"
        i = id(v)
        r = self.memo.get(i)
        if r is not None:
            return r
        if depth > MAX_DEPTH:
            self.opaque = True
            return OPAQUE
        self.memo[i] = "\x00cycle"
        self.keep.append(v)
        r = self._fp(v, t, depth + 1)
        if len(r) > 64:
            r = "h" + hashlib.blake2b(r.encode("utf-8", "surrogatepass"), digest_size=12).hexdigest()
        self.memo[i] = r
        return r

    def _fp(self, v, t, d) -> str:
        f = self.fp
        if isinstance(v, enum.Enum):
            return f"e:{t.__module__}.{t.__qualname__}.{v.name}={f(v.value, d)}"
        if isinstance(v, type):
            return f"t:{v.__module__}.{v.__qualname__}"
        if isinstance(v, dict):
            items = []
            for k, x in v.items():
                sk = stable_key(k)
                items.append((sk if sk is not None else "?" + f(k, d), f(x, d)))
            items.sort()
            return "d{" + ",".join(f"{a}:{b}" for a, b in items) + "}"
        if isinstance(v, (list, tuple)):
            return f"{t.__name__}[" + ",".join(f(x, d) for x in v) + "]"
        if isinstance(v, (set, frozenset)):
            return f"{t.__name__}{{" + ",".join(sorted(f(x, d) for x in v)) + "}"
        if isinstance(v, types.FunctionType):
            cl = v.__closure__ or ()
            cells = []
            for c in cl:
                try:
                    cells.append(f(c.cell_contents, d))
                except ValueError:
                    cells.append("<empty>")
            return (
                f"fn:{v.__name__}:{self.code(v.__code__)}:{f(v.__defaults__, d)}:{f(v.__kwdefaults__, d)}:"
                f"[{','.join(cells)}]:{f(getattr(v, '__wrapped__', None), d)}"
            )
        if isinstance(v, types.CodeType):
            return self.code(v)
        if isinstance(v, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            s = getattr(v, "__self__", None)
            own = "" if s is None or isinstance(s, types.ModuleType) else f(s, d)
            return f"bf:{getattr(v, '__module__', '')}.{v.__qualname__}:{own}"
        if isinstance(v, types.MethodType):
            return f"m:{f(v.__func__, d)}:{f(v.__self__, d)}"
        if isinstance(v, (staticmethod, classmethod)):
            return f"{t.__name__}:{f(v.__func__, d)}"
        if isinstance(v, property):
            return f"prop:{f(v.fget, d)}:{f(v.fset, d)}:{f(v.fdel, d)}"
        if isinstance(v, types.ModuleType):
            return f"mod:{v.__name__}"
        if isinstance(v, (logging.Logger, logging.PlaceHolder)):
            # caches and handlers change while running; never patched data
            return f"logger:{getattr(v, 'name', '')}"
        if t.__name__ == "partial" and hasattr(v, "func"):
            return f"partial:{f(v.func, d)}:{f(v.args, d)}:{f(v.keywords, d)}"
        if t.__name__ in ("Pattern",) and hasattr(v, "pattern"):
            return f"re:{v.pattern!r}:{v.flags}"
        if isinstance(v, (types.MappingProxyType,)):
            return f(dict(v), d)
        # generic object: type plus instance state
        parts = [f"o:{t.__module__}.{t.__qualname__}"]
        st = getattr(v, "__dict__", None)
        has_state = False
        if isinstance(st, dict):
            parts.append(f(st, d))
            has_state = True
        for k in _slots(t):
            has_state = True
            try:
                parts.append(f"{k}={f(object.__getattribute__(v, k), d)}")
            except AttributeError:
                parts.append(f"{k}=<unset>")
        if not has_state:
            try:
                r = repr(v)
            except Exception:  # noqa
                r = " at 0x"
            if " at 0x" in r or "object at" in r:
                if callable(v) and hasattr(v, "__qualname__"):
                    parts.append(getattr(v, "__module__", "") + "." + v.__qualname__)
                else:
                    self.opaque = True
                    parts.append(OPAQUE)
            else:
                parts.append(r)
        return "|".join(parts)

    def code(self, c) -> str:
        i = id(c)
        r = self.memo.get(i)
        if r is not None:
            return r
        consts = ",".join(self.code(x) if isinstance(x, types.CodeType) else self.fp(x) for x in c.co_consts)
        s = (
            f"{c.co_name}|{c.co_code.hex()}|{consts}|{c.co_names}|{c.co_varnames}|{c.co_freevars}|"
            f"{c.co_cellvars}|{c.co_argcount}|{c.co_posonlyargcount}|{c.co_kwonlyargcount}|{c.co_flags}"
        )
        r = "c" + hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=12).hexdigest()
        self.memo[i] = r
        self.keep.append(c)
        return r


_slot_cache: dict = {}


def _slots(t):
    r = _slot_cache.get(t)
    if r is None:
        names = []
        for k in getattr(t, "__mro__", ()):
            s = k.__dict__.get("__slots__", ())
            if isinstance(s, str):
                s = (s,)
            for n in s:
                if n not in ("__dict__", "__weakref__") and n not in names:
                    names.append(n)
        r = _slot_cache[t] = tuple(names)
    return r


def import_all(prefix="sqlglot", only=None) -> list:
    """Import every module of the package (or the given list); returns the
    names that imported."""
    if only is None:
        pkg = importlib.import_module(prefix)
        only = [prefix] + sorted(m.name for m in pkgutil.walk_packages(pkg.__path__, prefix + "."))
    ok = []
    for m in only:
        try:
            importlib.import_module(m)
            ok.append(m)
        except BaseException:  # noqa
            pass
    return ok


def containers(modules):
    """Yield (label, owner, name, value) for every module global and
    class attribute (nested classes included) defined in these modules."""
    for mn in sorted(modules):
        m = sys.modules.get(mn)
        if m is None:
            continue
        md = vars(m)
        for name in sorted(md):
            if name.startswith("__") and name.endswith("__"):
                continue
            v = md[name]
            yield f"{mn}:{name}", m, name, v
            if isinstance(v, type) and v.__module__ == mn and v.__qualname__ == name:
                stack = [v]
                while stack:
                    c = stack.pop()
                    cd = c.__dict__
                    for a in sorted(cd):
                        if a in ("__dict__", "__weakref__", "__doc__", "__module__", "__qualname__",
                                 "__annotations__", "__annotate__", "__static_attributes__",
                                 "__firstlineno__", "__orig_bases__", "__parameters__"):
                            continue
                        x = cd[a]
                        yield f"{mn}:{c.__qualname__}.{a}", c, a, x
                        if isinstance(x, type) and x.__module__ == mn and x.__qualname__ == f"{c.__qualname__}.{a}":
                            stack.append(x)


def keyable(v) -> bool:
    if not isinstance(v, dict):
        return False
    for k in v:
        if stable_key(k) is None:
            return False
    return True


def snapshot(modules) -> dict:
    """label -> ("D", {key: fp}) for keyable dicts, ("F", fp) for code (functions,
    classes, descriptors, modules), ("V", fp) for other data.
    Values fingerprinted under an opaque marker are flagged with a "!" kind
    ("D!", "V!"): they cannot be proven unchanged."""
    fpr = Fingerprinter()
    out = {}
    for label, _, _, v in containers(modules):
        if keyable(v):
            ent = {}
            bad = False
            for k, x in dict.items(v):
                fpr.opaque = False
                ent[stable_key(k)] = fpr.fp(x)
                bad |= fpr.opaque
            out[label] = ("D!" if bad else "D", ent)
        else:
            fpr.opaque = False
            r = fpr.fp(v)
            k = "F" if isinstance(v, _CODEISH) else "V"
            out[label] = (k + "!" if fpr.opaque else k, r)
    return out


_CODEISH = (type, types.FunctionType, types.MethodType, staticmethod, classmethod, property,
            types.BuiltinFunctionType, types.ModuleType)


def diff(base: dict, new: dict):
    """-> (keyed, other): keyed maps label -> (set of changed keys, whether the
    key set changed); other lists (label, base kind, new kind) of labels that
    changed in any other way (kind None: label absent)."""
    keyed: dict = {}
    other = []
    for label in base.keys() | new.keys():
        b = base.get(label)
        n = new.get(label)
        if b is None or n is None:
            other.append((label, b and b[0], n and n[0]))
            continue
        bk, bv = b
        nk, nv = n
        if bk[0] == "D" and nk[0] == "D":
            ch = set()
            for k in bv.keys() | nv.keys():
                x, y = bv.get(k), nv.get(k)
                if x != y or (x is not None and OPAQUE in x):
                    ch.add(k)
            if ch:
                keyed[label] = (ch, bv.keys() != nv.keys())
            continue
        if bk != nk or bv != nv or bk.endswith("!"):
            other.append((label, bk, nk))
    return keyed, sorted(other)
