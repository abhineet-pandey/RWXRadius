"""OFFLINE: for every attribute read that executes while importing (the
import-time identifier sites of the index), record which classes the object it
was read from belongs to (whole MRO, as `module.QualName`).

A change to `K.NAME` can only alter what import-time code computed if one of
these reads resolved `NAME` through `K`, i.e. `K` is in the recorded MRO set.
Sites whose base cannot be evaluated reliably (comprehension / loop targets
bound on the same statement, evaluation errors) are recorded as unknown (None)
and keep the conservative treatment.

Runs the same imports as import_pass.py, on the unchanged pinned checkout.
usage: python impbase_pass.py <repo_root> <index.pkl.gz> <out.pkl>
"""
import ast
import gzip
import importlib
import os
import pickle
import pkgutil
import sys

root = os.path.realpath(sys.argv[1]) + os.sep
index_path, out = sys.argv[2], sys.argv[3]
idx = pickle.loads(gzip.decompress(open(index_path, "rb").read()))
sys.path.insert(0, root[:-1])
os.chdir(root)

# (rel, line) -> [(attr, compiled base expr | None)]
wanted: dict = {}
site_keys = set()
for nm, sites in idx["imp_name_sites"].items():
    for rel, ln, base, in_func in sites:
        if base is None or base == "?":
            continue
        site_keys.add((rel, ln, nm))

trees = {}
for rel, ln, nm in sorted(site_keys):
    if rel not in trees:
        trees[rel] = ast.parse(open(os.path.join(root, rel), "rb").read())
by_file: dict = {}
for rel, ln, nm in site_keys:
    by_file.setdefault(rel, set()).add((ln, nm))


def _bound_in(stmt_like, name) -> bool:
    for n in ast.walk(stmt_like):
        tgt = None
        if isinstance(n, ast.comprehension):
            tgt = n.target
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            tgt = n.target
        elif isinstance(n, ast.NamedExpr):
            tgt = n.target
        if tgt is not None and any(isinstance(x, ast.Name) and x.id == name for x in ast.walk(tgt)):
            return True
    return False


for rel, want in by_file.items():
    tree = trees[rel]
    # innermost statement for every node
    parent_stmt = {}
    stack = [(tree, None)]
    while stack:
        node, st = stack.pop()
        if isinstance(node, ast.stmt):
            st = node
        parent_stmt[id(node)] = st
        for ch in ast.iter_child_nodes(node):
            stack.append((ch, st))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Store)):
            continue
        key = (node.end_lineno, node.attr)
        if key not in want:
            continue
        v = node.value
        chain = v
        while isinstance(chain, ast.Attribute):
            chain = chain.value
        code = None
        if isinstance(chain, ast.Name):
            st = parent_stmt.get(id(node))
            # Line events of a loop on the same statement fire before the
            # target is rebound: the value seen would be stale.
            if st is None or not _bound_in(st, chain.id) or isinstance(st, (ast.FunctionDef, ast.ClassDef)):
                try:
                    code = compile(ast.Expression(body=v), rel, "eval")
                except Exception:  # noqa
                    code = None
        wanted.setdefault((rel, node.end_lineno), []).append((node.attr, code))

RESULT: dict = {}  # (rel, line, attr) -> set of class keys | None (unknown)
SEEN_LINES = set()


def _keys(obj):
    cls = obj if isinstance(obj, type) else type(obj)
    out = set()
    for c in getattr(cls, "__mro__", (cls,)):
        out.add(f"{getattr(c, '__module__', '?')}.{getattr(c, '__qualname__', '?')}")
    return out


mon = sys.monitoring
TOOL = 4


def cb(code, line):
    fn = code.co_filename
    if not fn.startswith(root):
        return mon.DISABLE
    rel = fn[len(root):]
    items = wanted.get((rel, line))
    if not items:
        return mon.DISABLE
    SEEN_LINES.add((rel, line))
    frame = sys._getframe(1)
    if frame.f_code is not code:
        for attr, _ in items:
            RESULT[(rel, line, attr)] = None
        return None
    for attr, expr in items:
        k = (rel, line, attr)
        if k in RESULT and RESULT[k] is None:
            continue
        if expr is None:
            RESULT[k] = None
            continue
        try:
            obj = eval(expr, frame.f_globals, frame.f_locals)
        except BaseException:  # noqa  (unbound yet, property raising, ...)
            RESULT[k] = None
            continue
        RESULT.setdefault(k, set()).update(_keys(obj))
    return None


mon.use_tool_id(TOOL, "impbase")
mon.register_callback(TOOL, mon.events.LINE, cb)
mon.set_events(TOOL, mon.events.LINE)

os.environ.setdefault("SKIP_INTEGRATION", "0")

import sqlglot  # noqa

for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
    if info.name == "sqlglot.__main__":
        continue
    try:
        importlib.import_module(info.name)
    except Exception as e:  # noqa
        print("import fail", info.name, e, file=sys.stderr)

from sqlglot.dialects.dialect import Dialect, Dialects  # noqa

for d in Dialects:
    try:
        dia = Dialect.get_or_raise(d.value)
        dia.tokenizer_class, dia.parser_class, dia.generator_class  # noqa
    except Exception as e:  # noqa
        print("dialect fail", d, e, file=sys.stderr)

test_mods = []
for dp, dn, fn in os.walk(os.path.join(root, "tests")):
    dn[:] = [x for x in dn if x not in ("fixtures", "__pycache__")]
    for f in fn:
        if f.endswith(".py"):
            rel = os.path.relpath(os.path.join(dp, f), root)
            test_mods.append(rel[:-3].replace(os.sep, "."))
for m in sorted(test_mods):
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa
        print("test import fail", m, e, file=sys.stderr)

mon.set_events(TOOL, 0)

# every wanted site that is import-time per the index but was never observed
# here stays unknown
final = {}
for rel, ln, nm in site_keys:
    k = (rel, ln, nm)
    if k in RESULT:
        final[k] = frozenset(RESULT[k]) if RESULT[k] is not None else None
    else:
        final[k] = None
with open(out, "wb") as fh:
    pickle.dump(final, fh)
known = sum(1 for v in final.values() if v is not None)
print(f"impbase: {len(final)} sites, {known} resolved, {len(final) - known} unknown", file=sys.stderr)
