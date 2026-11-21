"""OFFLINE: record which lines of the pinned checkout execute at import time
(module bodies, class bodies, metaclass hooks, factory calls in class dicts)
and which modules each test module imports when it is collected.

usage: python import_pass.py <repo_root> <out.pkl>
"""
import importlib
import os
import pickle
import pkgutil
import sys

root = os.path.realpath(sys.argv[1]) + os.sep
out = sys.argv[2]
sys.path.insert(0, root[:-1])
os.chdir(root)

mon = sys.monitoring
TOOL = 3
LINES: dict = {}


def cb(code, line):
    fn = code.co_filename
    if fn.startswith(root):
        LINES.setdefault(fn[len(root):], set()).add(line)
    return mon.DISABLE


mon.use_tool_id(TOOL, "imp")
mon.register_callback(TOOL, mon.events.LINE, cb)
mon.set_events(TOOL, mon.events.LINE)

os.environ.setdefault("SKIP_INTEGRATION", "0")

import sqlglot  # noqa

mods = []
for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
    if info.name == "sqlglot.__main__":
        continue
    try:
        importlib.import_module(info.name)
        mods.append(info.name)
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

files = {}
for name, m in list(sys.modules.items()):
    f = getattr(m, "__file__", None)
    if f and os.path.realpath(f).startswith(root):
        files[name] = os.path.realpath(f)[len(root):]

with open(out, "wb") as fh:
    pickle.dump({"lines": {k: sorted(v) for k, v in LINES.items()}, "modules": files}, fh)
print("import pass:", len(LINES), "files", sum(len(v) for v in LINES.values()), "lines", file=sys.stderr)
