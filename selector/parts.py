"""Partial replay of long fixture-driven tests.

A few tests run one independent check per item of a TPC-H / TPC-DS fixture
file and dominate the cost of any commit that selects them (test_execute_tpcds
alone is 21.5 s). The baseline maps trace each such test once per item (alone)
and once with no items ("base"), as pseudo-tests next to the real ones. When a
change reaches the test only through some items, the replay child runs the
real test method restricted to those items: the test's own code, assertions
and fixtures, with the item list filtered. Items the change does not reach run
exactly as at the baseline, so if the selected items pass, the test passes.

The same `restrict` is used to trace the pseudo-tests offline and to replay at
run time, so both see identical item lists.
"""
from __future__ import annotations

import sys

# real test id -> (how its items are reached, name, fixture file, item count)
#   attr:   a class attribute filled by setUpClass from the fixture file
#   loader: a module-level loader function the test body calls with the file
PARTS = {
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": ("attr", "tpcds_sqls", "optimizer/tpc-ds/tpc-ds.sql", 99),
    "tests/test_executor.py::TestExecutor::test_optimized_tpch": ("attr", "tpch_sqls", "optimizer/tpc-h/tpc-h.sql", 22),
    "tests/test_optimizer.py::TestOptimizer::test_tpcds": ("loader", "load_sql_fixture_pairs", "optimizer/tpc-ds/tpc-ds.sql", 99),
    "tests/test_optimizer.py::TestOptimizer::test_tpch": ("loader", "load_sql_fixture_pairs", "optimizer/tpc-h/tpc-h.sql", 22),
}


def pseudo_id(test_id: str, part) -> str:
    """part: item index, or "base" (no items)."""
    return f"{test_id}@{part}"


def restrict(tc, test_id: str, keep) -> None:
    """Make unittest case `tc` run its test method on the items in `keep`
    (indices into the fixture's item list) only."""
    kind, name, fixture, _ = PARTS[test_id]
    meth_name = tc._testMethodName
    orig = getattr(tc, meth_name)
    keep = frozenset(keep)

    if kind == "attr":
        def run():
            cls = type(tc)
            # the attribute lives on the class that ran setUpClass
            owner = next(k for k in cls.__mro__ if name in k.__dict__)
            items = owner.__dict__[name]
            setattr(owner, name, [x for i, x in enumerate(items) if i in keep])
            try:
                return orig()
            finally:
                setattr(owner, name, items)
    else:
        def run():
            mod = sys.modules[orig.__func__.__module__]
            loader = getattr(mod, name)

            def filtered(filename, *a, **k):
                it = loader(filename, *a, **k)
                if filename != fixture:
                    return it
                return (x for i, x in enumerate(it) if i in keep)

            setattr(mod, name, filtered)
            try:
                return orig()
            finally:
                setattr(mod, name, loader)

    setattr(tc, meth_name, run)
