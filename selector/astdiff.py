"""Structural AST diff between baseline and patched Python source.

Produces change *sites* located in the OLD tree:
  Site(node=<old node>, path=[ancestors...])            replaced/removed node
  Site(node=None, parent=<old node>, field, index, path) insertion into a list

Nodes lying entirely outside the patch's changed lines have identical source
text on both sides and are compared by that text (cheap); nodes touching
changed lines are compared structurally, field by field.
"""
from __future__ import annotations

import ast
import bisect
import difflib
from dataclasses import dataclass, field


@dataclass
class Site:
    node: ast.AST | None
    path: list  # ancestors from Module (exclusive of node)
    parent: ast.AST | None = None
    fieldname: str | None = None
    index: int | None = None
    new_nodes: list = field(default_factory=list)  # new-tree nodes involved


def _dump(node) -> str:
    return ast.dump(node, annotate_fields=False, include_attributes=False)


class _Side:
    def __init__(self, src_lines, changed):
        self.lines = src_lines
        self.changed = sorted(changed)
        self.cache: dict = {}

    def touches(self, lo, hi) -> bool:
        i = bisect.bisect_left(self.changed, lo)
        return i < len(self.changed) and self.changed[i] <= hi

    def key(self, node):
        k = id(node)
        v = self.cache.get(k)
        if v is None:
            lo = getattr(node, "lineno", None)
            hi = getattr(node, "end_lineno", None)
            if lo is not None and hi is not None:
                if self.touches(lo, hi):
                    v = None  # unknown: compare structurally
                else:
                    v = ("U", type(node).__name__, node.col_offset, node.end_col_offset,
                         tuple(self.lines[lo - 1:hi]))
            else:
                v = ("D", _dump(node))
            self.cache[k] = v
        return v


class _Differ:
    def __init__(self, old: _Side, new: _Side):
        self.o = old
        self.n = new
        self.sites: list[Site] = []
        self._eq: dict = {}

    def eq(self, a, b) -> bool:
        if isinstance(a, ast.AST) and isinstance(b, ast.AST):
            if type(a) is not type(b):
                return False
            ka, kb = self.o.key(a), self.n.key(b)
            if ka is not None and kb is not None:
                return ka == kb
            k = (id(a), id(b))
            r = self._eq.get(k)
            if r is None:
                r = self._eq[k] = self._struct_eq(a, b)
            return r
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(self.eq(x, y) for x, y in zip(a, b))
        return type(a) is type(b) and a == b

    def _struct_eq(self, a, b) -> bool:
        for f in a._fields:
            x = getattr(a, f, None)
            y = getattr(b, f, None)
            if isinstance(x, (ast.AST, list)) or isinstance(y, (ast.AST, list)):
                if (x is None) != (y is None):
                    return False
                if x is not None and not self.eq(x, y):
                    return False
            elif type(x) is not type(y) or x != y:
                return False
        return True

    def site(self, node, path, new=None):
        # Report at the nearest node carrying source positions.
        p = list(path)
        while node is not None and not hasattr(node, "lineno") and p:
            node = p.pop()
        self.sites.append(Site(node=node, path=p, new_nodes=[new] if new is not None else []))

    def diff(self, o, n, path):
        if type(o) is not type(n):
            self.site(o, path, n)
            return
        if (
            isinstance(o, ast.Compare)
            and len(o.ops) == 1
            and len(n.ops) == 1
            and type(o.ops[0]) is not type(n.ops[0])
            and self.eq(o.left, n.left)
            and self.eq(o.comparators, n.comparators)
        ):
            # only the operator changed: keep both nodes for the analysis
            self.sites.append(Site(node=o, path=list(path), new_nodes=[n]))
            return
        sub = path + [o]
        for f in o._fields:
            a = getattr(o, f, None)
            b = getattr(n, f, None)
            if isinstance(a, list) or isinstance(b, list):
                a = a or []
                b = b or []
                if not self.eq(a, b):
                    self.diff_list(a, b, o, f, sub)
            elif isinstance(a, ast.AST) or isinstance(b, ast.AST):
                if a is None or b is None:
                    self.site(o, path, n)
                elif not self.eq(a, b):
                    self.diff(a, b, sub)
            else:
                if a != b or type(a) is not type(b):
                    self.site(o, path, n)
                    return

    def diff_list(self, a, b, parent, fname, sub):
        if len(a) == len(b):
            for x, y in zip(a, b):
                if not self.eq(x, y):
                    if isinstance(x, ast.AST) and isinstance(y, ast.AST):
                        self.diff(x, y, sub)
                    else:
                        self.site(parent, sub[:-1], None)
            return
        # An element added/removed in a non-statement list: the parent changes.
        if not all(isinstance(x, ast.stmt) for x in a + b):
            self.site(parent, sub[:-1])
            return
        ka = [self.o.key(x) or ("C", i) for i, x in enumerate(a)]
        kb = [self.n.key(x) or ("C", -1 - i) for i, x in enumerate(b)]
        # statements touching the change: equal if structurally equal
        for i, x in enumerate(a):
            if ka[i][0] == "C":
                for j, y in enumerate(b):
                    if kb[j][0] == "C" and type(x) is type(y) and self.eq(x, y):
                        ka[i] = kb[j] = ("E", i)
                        break
        sm = difflib.SequenceMatcher(None, ka, kb, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace" and (i2 - i1) == (j2 - j1):
                for x, y in zip(a[i1:i2], b[j1:j2]):
                    self.diff(x, y, sub)
                continue
            for x in a[i1:i2]:
                self.sites.append(Site(node=x, path=list(sub)))
            if j2 > j1:
                self.sites.append(
                    Site(node=None, path=list(sub), parent=parent, fieldname=fname, index=i1,
                         new_nodes=b[j1:j2])
                )


def diff_trees(old: ast.AST, new: ast.AST, old_lines=None, new_lines=None, old_changed=(), new_changed=()) -> list[Site]:
    if old_lines is None:
        # no region information: every node is compared structurally
        old_lines, new_lines = [], []
        old_changed = new_changed = (0,)
        o = _Side(old_lines, ())
        n = _Side(new_lines, ())
        o.touches = n.touches = lambda lo, hi: True  # type: ignore
    else:
        o = _Side(old_lines, old_changed)
        n = _Side(new_lines, new_changed)
    df = _Differ(o, n)
    df.diff(old, new, [])
    seen = set()
    out = []
    for st in df.sites:
        k = (id(st.node), id(st.parent), st.fieldname, st.index)
        if k not in seen:
            seen.add(k)
            out.append(st)
    return out


def same_ast(old: ast.AST, new: ast.AST) -> bool:
    return _dump(old) == _dump(new)
