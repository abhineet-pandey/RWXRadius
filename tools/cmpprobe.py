"""OFFLINE: comparison value profiles of the pinned checkout.

An import hook rewrites every single-operator comparison inside sqlglot
function / lambda bodies into a call that computes the original result and
also records how the alternative operators would have evaluated on the same
operands. Alternatives are only evaluated when both operands use side-effect
free builtin/identity equality; otherwise the evaluation is recorded as
unknown. The profile says, per comparison site, which operator swaps are
indistinguishable on every evaluation the test suite performs.
"""
from __future__ import annotations

import ast
import builtins
import enum
import importlib.abc
import importlib.machinery
import importlib.util
import operator
import os
import pickle
import sys

OPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Is: operator.is_, ast.IsNot: operator.is_not,
}
SITES: list = []  # site id -> key
PROFILE: dict = {}  # site id -> set of signatures
_SAFE_EQ = {object.__eq__, str.__eq__, int.__eq__, float.__eq__, bytes.__eq__, bool.__eq__}
_PRIM = (type(None), bool, int, float, str, bytes)


def _safe(x, depth=0) -> bool:
    t = type(x)
    if t in _PRIM:
        return True
    if t is tuple or t is frozenset:
        return depth < 2 and len(x) <= 16 and all(_safe(e, depth + 1) for e in x)
    eq = getattr(t, "__eq__", None)
    if eq in _SAFE_EQ and getattr(t, "__ne__", None) in (object.__ne__, str.__ne__, int.__ne__, float.__ne__,
                                                          bytes.__ne__, bool.__ne__):
        # identity-equality objects and builtin-mixin enums; ordering only for builtins
        if isinstance(x, enum.Enum) and not isinstance(x, (str, int)):
            return getattr(t, "__eq__") is object.__eq__ or getattr(t, "__eq__") is enum.Enum.__eq__
        return True
    return False


def _r(f, a, b):
    try:
        return bool(f(a, b))
    except Exception:  # noqa
        return "E"


def _probe(site, op, a, b):
    res = op(a, b)
    try:
        if _safe(a) and _safe(b):
            sig = (a is b, _r(operator.eq, a, b), _r(operator.lt, a, b), _r(operator.le, a, b),
                   _r(operator.gt, a, b), _r(operator.ge, a, b))
        else:
            sig = ("U",)
        s = PROFILE.get(site)
        if s is None:
            PROFILE[site] = {sig}
        elif sig not in s:
            s.add(sig)
    except Exception:  # noqa
        PROFILE.setdefault(site, set()).add(("U",))
    return res


class _Rewriter(ast.NodeTransformer):
    def __init__(self, rel):
        self.rel = rel
        self.depth = 0  # inside function / lambda bodies

    def _body(self, node, fields):
        for f in node._fields:
            v = getattr(node, f)
            inside = f in fields
            if inside:
                self.depth += 1
            if isinstance(v, list):
                setattr(node, f, [self.visit(x) if isinstance(x, ast.AST) else x for x in v])
            elif isinstance(v, ast.AST):
                setattr(node, f, self.visit(v))
            if inside:
                self.depth -= 1
        return node

    def visit_FunctionDef(self, node):
        return self._body(node, ("body",))

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        return self._body(node, ("body",))

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.depth == 0 or len(node.ops) != 1 or type(node.ops[0]) not in OPS:
            return node
        # deterministic site key (same in every process), embedded as a constant
        sid = (self.rel, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset,
               type(node.ops[0]).__name__)
        call = ast.Call(
            func=ast.Name("_br_cmp_probe_", ast.Load()),
            args=[ast.Constant(sid), ast.Constant(None), node.left, node.comparators[0]],
            keywords=[],
        )
        # the operator is passed by name lookup on the operator module at call time
        call.args[1] = ast.Attribute(ast.Name("_br_cmp_ops_", ast.Load()), OPS[type(node.ops[0])].__name__,
                                     ast.Load())
        return ast.copy_location(call, node)


class _Loader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, path, rel):
        super().__init__(fullname, path)
        self.rel = rel

    def source_to_code(self, data, path, *, _optimize=-1):
        tree = ast.parse(data, path)
        tree = _Rewriter(self.rel).visit(tree)
        ast.fix_missing_locations(tree)
        return compile(tree, path, "exec", dont_inherit=True)

    def get_code(self, fullname):
        # never use the checkout's bytecode; cache the rewritten code instead
        source = self.get_data(self.path)
        import hashlib
        import marshal

        cache_dir = os.environ.get("BR_CMP_CACHE", "/tmp/cmpcache")
        key = hashlib.sha1(source + self.rel.encode()).hexdigest()
        cp = os.path.join(cache_dir, key + ".marshal")
        try:
            with open(cp, "rb") as fh:
                return marshal.loads(fh.read())
        except OSError:
            pass
        code = self.source_to_code(source, self.path)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = cp + f".{os.getpid()}"
            with open(tmp, "wb") as fh:
                fh.write(marshal.dumps(code))
            os.replace(tmp, cp)
        except OSError:
            pass
        return code


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.root = root

    def find_spec(self, fullname, path, target=None):
        if not (fullname == "sqlglot" or fullname.startswith("sqlglot.")):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            return spec
        origin = os.path.realpath(spec.origin)
        if not origin.startswith(self.root):
            return spec
        rel = origin[len(self.root):]
        spec.loader = _Loader(fullname, spec.origin, rel)
        return spec


def install(root: str):
    root = os.path.realpath(root) + os.sep
    builtins._br_cmp_probe_ = _probe
    builtins._br_cmp_ops_ = operator
    sys.meta_path.insert(0, _Finder(root))


def dump(path: str):
    out = {}
    for sid, sigs in PROFILE.items():
        out[sid] = sigs
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
