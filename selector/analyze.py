"""Static impact analysis: patch -> candidate tests (bitset over index order).

Soundness model (deterministic execution): a test can only diverge from the
baseline if it executes changed code, reads a changed file, or depends on a
definition-time binding that changed. Runtime code uses per-test statement
coverage; definition-time code (module / class bodies, decorators, defaults,
code run at import) is mapped through class-user and identifier-reader maps
and widened conservatively when the consumer is not tracked.
"""
from __future__ import annotations

import ast
import gzip
import os
import pickle
import sys
from dataclasses import dataclass, field

from . import astdiff, patchparse

FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)
STMTLIKE = (ast.stmt, ast.excepthandler, ast.match_case)


def log(*a):
    print(*a, file=sys.stderr)


class Index:
    def __init__(self, path: str):
        with open(path, "rb") as fh:
            d = pickle.loads(gzip.decompress(fh.read()))
        self.__dict__.update(d)
        self.n = len(self.tests)
        self.ALL = (1 << self.n) - 1
        self.collect_importers = {k: set(v) for k, v in self.collect_importers.items()}
        self.core = set(self.core)
        self.subprocess_tests = self.sets[self.opens["*"]] if "*" in self.opens else 0

    def s(self, sid):
        return self.sets[sid] if sid is not None else 0

    def lines_tests(self, file: str, lines) -> int:
        lt = self.lines.get(file)
        if not lt:
            return 0
        acc = 0
        sets = self.sets
        for ln in lines:
            k = lt.get(ln)
            if k is not None:
                acc |= sets[k]
        return acc

    def file_users_of(self, file: str) -> int:
        return self.s(self.file_users.get(file))

    def users_of(self, key: str) -> int:
        return self.s(self.users.get(key))

    def names_of(self, name: str) -> int:
        return self.s(self.names.get(name))

    def names_of_all(self, names) -> int:
        acc = 0
        for n in names:
            acc |= self.names_of(n)
        return acc

    def opens_of(self, path: str) -> int:
        return self.s(self.opens.get(path))

    def lists_of(self, path: str) -> int:
        return self.s(self.lists.get(path))

    def tests_in_file(self, test_file: str) -> int:
        return self.s(self.test_files.get(test_file))

    def importers(self, file: str) -> int:
        """Tests whose module imports `file` at collection, plus every test that
        ran any of its code (lazy importers)."""
        acc = self.file_users_of(file)
        for tf in self.collect_importers.get(file, ()):
            acc |= self.tests_in_file(tf)
        if file.startswith("tests/") and file in self.test_files:
            acc |= self.tests_in_file(file)
        return acc


def module_name(rel: str) -> str | None:
    if not rel.endswith(".py"):
        return None
    m = rel[:-3].replace("/", ".")
    if m.endswith(".__init__"):
        m = m[: -len(".__init__")]
    return m


@dataclass
class Analysis:
    cand: int = 0  # tests that may fail; to be replayed or included
    forced: int = 0  # tests to include without replay
    replayable: bool = True
    overlay: dict = field(default_factory=dict)  # rel path -> new text | None
    probe_modules: list = field(default_factory=list)
    on_import_fail: dict = field(default_factory=dict)  # module -> bitset
    notes: list = field(default_factory=list)
    code_changed: bool = False  # some Python change with semantic effect
    needs_probe: bool = False  # a change that runs at import time could break importing


def _op_result(op, sig):
    """Value of `a <op> b` from a profile signature
    (is, ==, <, <=, >, >=); None when unknown or raising."""
    i = {ast.Is: 0, ast.IsNot: 0, ast.Eq: 1, ast.NotEq: 1, ast.Lt: 2, ast.LtE: 3, ast.Gt: 4, ast.GtE: 5}.get(op)
    if i is None:
        return None
    v = sig[i]
    if v not in (True, False):
        return None
    return (not v) if op in (ast.IsNot, ast.NotEq) else v


def _span(node) -> range:
    return range(node.lineno, (node.end_lineno or node.lineno) + 1)


def _body_lines(fn) -> range:
    """Lines a call of `fn` executes. The `def` line, decorators and default
    values run when the definition executes (import time for module and class
    level functions), so every importer covers them; any call runs at least
    one instruction of the body. Falls back to the whole definition when the
    body shares the `def` line or has no statement that surely gets a line."""
    body = getattr(fn, "body", None) or []
    real = [
        s for s in body
        if not isinstance(s, ast.Pass) and not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]
    if not real:
        return _span(fn)
    # last line of the signature (defaults / annotations can span lines)
    sig_end = fn.lineno
    for n in ast.walk(fn.args):
        sig_end = max(sig_end, getattr(n, "end_lineno", None) or 0)
    if fn.returns is not None:
        sig_end = max(sig_end, fn.returns.end_lineno or 0)
    start = real[0].lineno
    if start <= sig_end:
        return _span(fn)
    return range(start, (fn.end_lineno or start) + 1)


def _header_span(stmt) -> range:
    """Lines of a compound statement's header (before its first body stmt)."""
    first_body = None
    for f in ("body", "orelse", "handlers", "finalbody", "cases"):
        v = getattr(stmt, f, None)
        if isinstance(v, list):
            for x in v:
                if hasattr(x, "lineno"):
                    first_body = x.lineno if first_body is None else min(first_body, x.lineno)
    if first_body is None or first_body <= stmt.lineno:
        return _span(stmt)
    return range(stmt.lineno, first_body)


def _in_field(parent, child, fname) -> bool:
    v = getattr(parent, fname, None)
    if isinstance(v, list):
        return any(x is child for x in v)
    return v is child


def _child_field(parent, child):
    for f in parent._fields:
        v = getattr(parent, f, None)
        if v is child or (isinstance(v, list) and any(x is child for x in v)):
            return f
    return None


def _targets(stmt) -> set:
    names = set()
    if isinstance(stmt, (ast.Assign,)):
        tg = stmt.targets
    elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        tg = [stmt.target]
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in stmt.names}
    else:
        tg = []
    for t in tg:
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
    return names


class PyAnalyzer:
    def __init__(self, idx: Index, rel: str, old_tree, an: Analysis):
        self.idx = idx
        self.rel = rel
        self.old = old_tree
        self.an = an
        self.mod = module_name(rel) or rel
        self.imptime = idx.imptime.get(rel, frozenset())
        self.is_test = rel.startswith("tests/")

    # -- helpers
    def add(self, bits, why):
        if bits:
            self.an.cand |= bits
        self.an.notes.append(f"{self.rel}: {why} -> {bin(bits).count('1')}")

    def module_scope(self) -> int:
        acc = self.idx.importers(self.rel)
        if self.rel in self.idx.core:
            acc |= self.idx.subprocess_tests
        return acc

    def name_scope(self, names, exclude_lines=()) -> int:
        """Runtime readers of identifiers plus importers of import-time code
        reading them outside the changed statement."""
        return self.idx.names_of_all(names) | self.consumers(names, set(exclude_lines), None)

    def class_key(self, chain) -> str:
        return self.mod + "." + ".".join(c.name for c in chain)

    # -- site handling
    def is_docstring(self, site) -> bool:
        node = site.node
        owner = site.path[-1] if site.path else None
        if isinstance(node, ast.Constant) and len(site.path) >= 2 and isinstance(owner, ast.Expr):
            node, owner = owner, site.path[-2]
        for nn in site.new_nodes:
            if isinstance(nn, ast.Expr):
                nn = nn.value
            if not (isinstance(nn, ast.Constant) and isinstance(nn.value, str)):
                return False
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and isinstance(owner, (ast.Module, ast.ClassDef) + FUNC)
            and owner.body
            and owner.body[0] is node
        )

    def handle(self, site: astdiff.Site):
        path = site.path
        if site.node is not None and self.is_docstring(site):
            # Only __doc__ changes. At the pinned commit the sole reader of
            # __doc__ is Expr.__init_subclass__ (expressions/core.py), which
            # writes the value straight back (`or ""`); it is executed at import
            # by every test, so the name index would select the whole suite.
            # Readers of other `__doc__` sites, getdoc() and `doc` still count.
            # Text readers of the file are covered by the file-open map, and the
            # import probe still guards against the module failing to import.
            readers = self.idx.names_of_all({"getdoc", "doc"})
            if self.idx.imp_name_sites.get("__doc__", ()) != [("sqlglot/expressions/core.py", 112, "?", True)]:
                readers |= self.idx.names_of("__doc__")
            self.an.needs_probe = True
            self.add(readers, "docstring")
            return
        nodes = path + ([site.node] if site.node is not None else [])
        # innermost runtime scope (function / lambda body) containing the site
        func_i = None
        for i, a in enumerate(path):
            child = nodes[i + 1] if i + 1 < len(nodes) else None
            if isinstance(a, FUNC + (ast.Lambda,)):
                if child is None:
                    if site.node is None and site.fieldname == "body":
                        func_i = i
                elif (isinstance(a, ast.Lambda) and a.body is child) or (
                    isinstance(a, FUNC) and _in_field(a, child, "body")
                ):
                    func_i = i
        if func_i is not None:
            self.runtime_site(site, nodes, func_i)
        else:
            self.deftime_site(site, nodes)

    def equivalent_compare(self, site) -> bool:
        """An operator swap at a comparison whose baseline value profile shows
        both operators agreeing on every evaluation cannot change execution."""
        o = site.node
        if not (isinstance(o, ast.Compare) and site.new_nodes and isinstance(site.new_nodes[0], ast.Compare)):
            return False
        prof = getattr(self.idx, "cmp", {}).get((self.rel, o.lineno, o.col_offset, o.end_lineno, o.end_col_offset))
        if not prof:
            return False
        a, b = type(o.ops[0]), type(site.new_nodes[0].ops[0])
        for sig in prof:
            if len(sig) != 6:
                return False
            ra, rb = _op_result(a, sig), _op_result(b, sig)
            if ra is None or rb is None or ra != rb:
                return False
        return True

    def runtime_site(self, site, nodes, func_i):
        func = nodes[func_i]
        if self.equivalent_compare(site):
            self.add(0, f"compare {site.node.lineno}: {type(site.node.ops[0]).__name__}->"
                        f"{type(site.new_nodes[0].ops[0]).__name__} equivalent on all baseline evaluations")
            return
        if site.node is None:
            lst = getattr(site.parent, site.fieldname)
            i = site.index
            lines = set()
            if i > 0:
                lines.update(_span(lst[i - 1]))
            if i < len(lst):
                lines.update(_span(lst[i]))
            if not lst:
                lines.update(_span(site.parent))
            why = f"insert@{site.parent.__class__.__name__}.{site.fieldname}"
            if any(isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal))
                   for nn in site.new_nodes for n in ast.walk(nn)):
                lines.update(_span(func))
        elif isinstance(site.node, ast.expr) and hasattr(site.node, "lineno"):
            # An expression only affects executions that evaluate it, and any
            # evaluation runs an instruction located within its own span.
            lines = set(_span(site.node))
            why = f"expr {min(lines)}-{max(lines)}"
        else:
            stmt = None
            for n in reversed(nodes[func_i + 1:]):
                if isinstance(n, STMTLIKE):
                    stmt = n
                    break
            if stmt is None:  # inside a lambda expression body
                lines = set(_span(func.body if isinstance(func, ast.Lambda) else func))
            else:
                k = next(j for j in range(len(nodes)) if nodes[j] is stmt)
                child = nodes[k + 1] if k + 1 < len(nodes) else None
                if child is not None and isinstance(stmt, ast.stmt) and _child_field(stmt, child) in (
                    "body", "orelse", "handlers", "finalbody", "cases"
                ):
                    lines = set(_span(stmt))
                elif child is None:
                    lines = set(_span(stmt))
                else:
                    lines = set(_header_span(stmt))
            # docstring of a function: definition-time attribute, use callers
            if isinstance(func, FUNC) and func.body and site.node is func.body[0] and isinstance(
                site.node, ast.Expr) and isinstance(getattr(site.node, "value", None), ast.Constant):
                lines.update(_span(func))
            why = f"stmt {min(lines)}-{max(lines)}"
        bits = self.idx.lines_tests(self.rel, lines)
        if self.imptime.intersection(lines):
            # code executed while importing (metaclass hooks, factories in class
            # bodies): its products are used by every user of the module
            bits |= self.module_scope() | self.idx.lines_tests(self.rel, _span(func))
            why += " (import-time)"
            self.an.needs_probe = True
        self.add(bits, why)

    def deftime_site(self, site, nodes):
        self.an.needs_probe = True
        # New statements inserted directly into a module / class body.
        if site.node is None and isinstance(site.parent, (ast.Module, ast.ClassDef)) and site.fieldname == "body":
            chain = [x for x in nodes if isinstance(x, ast.ClassDef)]
            names = set()
            for nn in site.new_nodes:
                names |= _targets(nn)
            self.def_names(chain, names, None, site.new_nodes)
            return
        # Innermost module/class body statement on the path.
        top = None
        k_top = None
        chain = []
        for i, a in enumerate(nodes):
            if isinstance(a, (ast.Module, ast.ClassDef)) and i + 1 < len(nodes) and _in_field(a, nodes[i + 1], "body"):
                top = nodes[i + 1]
                k_top = i + 1
                chain = [x for x in nodes[: i + 1] if isinstance(x, ast.ClassDef)]
        if top is None:
            self.add(self.module_scope(), "deftime (unlocated)")
            return
        whole = k_top == len(nodes) - 1 and site.node is top
        if isinstance(top, FUNC) and not whole:
            # decorators / defaults / signature of a def: affects its callers.
            # Every call that completed at the baseline ran the body; the def
            # line and default values themselves run at import for everyone.
            body = self.idx.lines_tests(self.rel, _body_lines(top))
            child = nodes[k_top + 1] if k_top + 1 < len(nodes) else None
            fld = _child_field(top, child) if child is not None else None
            if fld in ("args", "returns") and not top.decorator_list:
                # Parameters / defaults / annotations of an undecorated def:
                # only calls observe them (parameter-name introspection in the
                # pinned sources is limited to calls that follow it).
                self.add(body, f"def signature {top.name} (callers)")
                return
            self.add(body, f"def header {top.name}")
            if chain:
                self.add(self.idx.users_of(self.class_key(chain)) & self.idx.names_of(top.name),
                         f"method {top.name} readers")
            else:
                self.add(self.name_scope({top.name}, _span(top)), f"func {top.name} readers")
            return
        if isinstance(top, ast.ClassDef) and not whole:
            # bases / keywords / decorators of a class
            key = self.class_key(chain + [top])
            self.add(self.idx.users_of(key) | self.name_scope({top.name}, _span(top)), f"class header {top.name}")
            if not self.idx.users.get(key):
                self.add(self.module_scope(), "class unseen -> module scope")
            return
        if (
            whole
            and isinstance(top, FUNC)
            and chain
            and not self.is_test
            and not top.decorator_list
            and site.new_nodes
            and all(isinstance(n, FUNC) and not n.decorator_list for n in site.new_nodes)
            and self.idx.users.get(self.class_key(chain))
        ):
            self.method_replaced(chain, top, site.new_nodes)
            return
        new_names = set()
        for nn in site.new_nodes or ():
            new_names |= _targets(nn)
        self.def_names(chain, _targets(top) | new_names, top, [])

    def method_replaced(self, chain, top, new_nodes):
        """An undecorated method replaced by undecorated method(s), e.g. renamed.
        Calls that resolved to the old method ran its body. Names the class no
        longer defines resolve elsewhere only for those callers (attribute
        dispatch in the pinned sources always calls what it finds). Names it
        newly defines can capture lookups that never reached the old body
        (`dir()`-based dispatch), so they keep the class-user/reader scope."""
        key = self.class_key(chain)
        kname = chain[-1].name
        excl = set(_span(top))
        self.add(self.idx.lines_tests(self.rel, _body_lines(top)), f"method {top.name} replaced (callers)")
        old = {top.name}
        new = set()
        for n in new_nodes:
            new |= _targets(n)
        # a new name may also shadow/override an existing definition in the
        # class: every name other than the old one counts as added
        added = new - old
        removed = old - new
        users = self.idx.users_of(key)
        if added:
            readers = self.idx.names_of_all(added)
            # class-level dir() scans in the pinned sources (generator and
            # simplify dispatch tables) only pick public `*_sql` names; other
            # dynamic reads target module members and instance dicts
            if any(a.endswith("_sql") and not a.startswith("_") for a in added):
                readers |= self.idx.names_of("<dynamic>")
            self.add(users & readers | self.idx.names_of_all({f"{kname}.{a}" for a in added}),
                     f"method names added {sorted(added)[:3]}")
        if removed or added:
            self.add(self.consumers(removed | added, excl, kname, key), "method import-time consumers")
        if removed:
            self.add(self.idx.names_of_all({f"{kname}.{a}" for a in removed}), "qualified readers of removed")

    def def_names(self, chain, names, top, new_nodes):
        excl = set(_span(top)) if top is not None else set()
        if self.is_test:
            bits = self.module_scope()
            self.add(bits, f"test deftime {sorted(names)[:3]}")
            return
        if top is not None and isinstance(top, FUNC):
            self.add(self.idx.lines_tests(self.rel, _body_lines(top)), f"def {top.name} replaced")
        if top is not None and isinstance(top, ast.ClassDef):
            key = self.class_key(chain + [top])
            self.add(self.idx.users_of(key), f"class {top.name} replaced")
        if chain:
            key = self.class_key(chain)
            if not self.idx.users.get(key):
                self.add(self.module_scope(), f"class {key} unseen -> module scope")
                return
            users = self.idx.users_of(key)
            kname = chain[-1].name
            qualified = self.idx.names_of_all({f"{kname}.{a}" for a in names})
            assign = (ast.Assign, ast.AnnAssign, ast.AugAssign)
            assign_only = isinstance(top, assign) if top is not None else bool(new_nodes) and all(
                isinstance(n, assign) for n in new_nodes)
            if names and assign_only and not self.import_time_loads(names, excl, key):
                # Nothing derives from the attribute at import time: only tests
                # using the class that read it by name can see it. Dynamic
                # attribute reads in the pinned sources (getattr with computed
                # names, dir(), vars(), __dict__) only ever target *_sql
                # methods, module members, instance dicts and token fields, never
                # UPPER_CASE class constants; other names also keep them.
                readers = self.idx.names_of_all(names)
                if not all(n.isupper() for n in names):
                    readers |= self.idx.names_of("<dynamic>")
                narrowed = users & readers
                self.add(narrowed | qualified, f"class attr {kname}.{sorted(names)[:3]} (narrowed from {bin(users).count('1')})")
            else:
                self.add(users | qualified, f"class attr {kname}.{sorted(names)[:3]}")
                # other definitions copying it by class name at import time
                self.add(self.consumers(names, excl, kname, key), "attr import-time consumers")
            if not names:
                self.add(self.module_scope(), "class stmt w/o names")
        else:
            if not names or (top is not None and not isinstance(top, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Import, ast.ImportFrom) + FUNC + (ast.ClassDef,))):
                self.add(self.module_scope(), "module stmt")
                return
            self.add(self.idx.names_of_all(names), f"module names {sorted(names)[:3]}")
            self.add(self.consumers(names, excl, None), "module names import-time consumers")
            if top is not None and isinstance(top, (ast.Import, ast.ImportFrom)):
                self.add(self.idx.file_users_of(self.rel), "import change")

    def site_relevant(self, f, ln, nm, base, okey) -> bool:
        """Whether an import-time read of `nm` can have resolved through the
        class `okey`. The impbase map (built on the pinned checkout) records
        the MRO of every object such a read used while importing; a read that
        never saw `okey` in it cannot have observed `okey.nm`. Unknown sites,
        bare names and classes the tracer never named stay relevant."""
        if okey is None or base is None or base == "?":
            return True
        ib = getattr(self.idx, "impbase", None)
        if not ib or not self.idx.users.get(okey):
            return True
        seen = ib.get((f, ln, nm))
        if seen is None:
            return True
        b = self.idx.impbase_cls.get(okey)
        return b is not None and bool(seen >> b & 1)

    def import_time_loads(self, names, excl, okey=None) -> bool:
        for nm in names:
            for f, ln, base, in_func in self.idx.imp_name_sites.get(nm, ()):
                if f == self.rel and ln in excl:
                    continue
                if not self.site_relevant(f, ln, nm, base, okey):
                    continue
                return True
        return False

    def consumers(self, names, excl, owner, okey=None) -> int:
        """Import-time code that reads one of `names` (outside the changed
        statement) builds derived definitions: every importer of that file."""
        acc = 0
        for nm in names:
            for f, ln, base, in_func in self.idx.imp_name_sites.get(nm, ()):
                if f == self.rel and ln in excl:
                    continue
                if not self.site_relevant(f, ln, nm, base, okey):
                    continue
                if owner is not None:
                    # attribute of a class: only references through that class
                    # name (or its bare name inside its own module) derive from it
                    if base != owner and not (base is None and f == self.rel):
                        continue
                    if in_func and f == self.rel:
                        continue  # metaclass/hook processing of the class itself
                acc |= self.idx.importers(f)
        return acc


def analyze(patch_text: str, idx: Index, repo_root: str) -> Analysis:
    an = Analysis()
    try:
        fps = patchparse.parse(patch_text)
    except Exception as e:  # noqa
        an.forced = idx.ALL
        an.replayable = False
        an.notes.append(f"unparseable patch: {e}")
        return an
    if not fps:
        an.notes.append("empty patch")
        return an
    for fp in fps:
        try:
            _analyze_file(fp, idx, repo_root, an)
        except Exception as e:  # noqa
            an.cand |= idx.ALL
            an.notes.append(f"{fp.path}: analysis error {type(e).__name__}: {e}")
    return an


def _read(repo_root, rel):
    p = os.path.join(repo_root, rel)
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as fh:
        return fh.read().decode("utf-8", errors="surrogateescape")


def _analyze_file(fp: patchparse.FilePatch, idx: Index, root: str, an: Analysis):
    rels = {p for p in (fp.old_path, fp.new_path) if p}
    for rel in rels:
        if ".." in rel.split("/") or rel.startswith("/"):
            raise ValueError("unsafe path")
    if fp.binary:
        an.replayable = False
        for rel in rels:
            an.forced |= idx.opens_of(rel) | _listers(idx, rel)
            if rel.endswith(".py") or _config_breaks_pytest(rel, ""):
                an.forced |= idx.ALL
        an.notes.append(f"binary {fp.path}")
        return
    old_rel = fp.old_path
    old_text = _read(root, old_rel) if old_rel else None
    if old_rel and old_text is None:
        # patch against a file we do not have
        an.cand |= idx.ALL
        an.notes.append(f"missing baseline {old_rel}")
        an.replayable = False
        return
    new_text = patchparse.apply(old_text, fp) if fp.hunks or fp.is_deleted else old_text
    if fp.rename and fp.old_path and fp.new_path and fp.old_path != fp.new_path:
        an.overlay[fp.old_path] = None
    if fp.new_path:
        an.overlay[fp.new_path] = new_text
    elif fp.old_path:
        an.overlay[fp.old_path] = None

    for rel in rels:
        if _config_breaks_pytest(rel, new_text):
            # pytest configuration / collection hooks: not modelled by replay
            an.forced |= idx.ALL
            an.notes.append(f"config {rel}")

    # Files read as data (fixtures, sources read as text) and directory scans
    for rel in rels:
        an.cand |= idx.opens_of(rel)
        if fp.is_new or fp.is_deleted or fp.rename:
            an.cand |= _listers(idx, rel)

    py = [r for r in rels if r.endswith(".py") and (r.startswith("sqlglot/") or r.startswith("tests/"))]
    if not py:
        an.notes.append(f"data {fp.path} -> {bin(an.cand).count('1')}")
        return
    rel = fp.old_path or fp.new_path
    mod = module_name(fp.new_path or fp.old_path)
    # import failure of this module fails every importer
    fail_bits = 0
    for r in rels:
        fail_bits |= idx.importers(r)
        if r in idx.core or (r.startswith("sqlglot/") and r.endswith("__init__.py")):
            fail_bits |= idx.ALL if r in idx.core else 0
    if fp.is_deleted or fp.rename:
        an.cand |= fail_bits
        an.notes.append(f"deleted/renamed module {rel}")
        if fp.rename and fp.new_path:
            an.probe_modules.append(module_name(fp.new_path))
        return
    if fp.is_new:
        # a new module only matters if something imports/discovers it; the
        # importers are changed files themselves (analysed separately) or
        # directory scans (above).
        an.probe_modules.append(mod)
        an.on_import_fail[mod] = 0
        an.notes.append(f"new module {fp.new_path}")
        return
    an.probe_modules.append(mod)
    an.on_import_fail[mod] = fail_bits

    try:
        old_tree = ast.parse(old_text.encode("utf-8", "surrogateescape"))
    except SyntaxError:
        an.cand |= idx.ALL
        return
    try:
        new_bytes = new_text.encode("utf-8", "surrogateescape")
        new_tree = ast.parse(new_bytes)
    except (SyntaxError, ValueError) as e:
        an.forced |= fail_bits
        an.notes.append(f"{rel}: does not parse ({type(e).__name__}) -> importers {bin(fail_bits).count('1')}")
        return
    oc, nc = patchparse.changed_line_sets(fp)
    sites = astdiff.diff_trees(old_tree, new_tree, old_text.split("\n"), new_text.split("\n"), oc, nc)
    if not sites:
        an.notes.append(f"{rel}: AST-identical")
        return
    try:
        compile(new_tree, rel, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as e:
        an.forced |= fail_bits
        an.notes.append(f"{rel}: does not compile ({type(e).__name__}) -> importers {bin(fail_bits).count('1')}")
        return
    an.code_changed = True
    pa = PyAnalyzer(idx, rel, old_tree, an)
    for site in sites:
        pa.handle(site)


def _config_breaks_pytest(rel: str, new_text) -> bool:
    base = os.path.basename(rel)
    if base in ("conftest.py", "pytest.ini", "tox.ini", ".pytest.ini"):
        return True
    if rel == "pyproject.toml":
        if new_text is None:
            return False
        try:
            import tomllib

            data = tomllib.loads(new_text)
        except Exception:  # noqa
            return True
        return "pytest" in data.get("tool", {})
    if rel == "setup.cfg":
        if new_text is None:
            return False
        import configparser

        cp = configparser.ConfigParser()
        try:
            cp.read_string(new_text)
        except Exception:  # noqa
            return True
        return cp.has_section("tool:pytest")
    return False


def _listers(idx: Index, rel: str) -> int:
    acc = 0
    parts = rel.split("/")
    for i in range(len(parts)):
        acc |= idx.lists_of("/".join(parts[:i]) or ".")
    return acc
