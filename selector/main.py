"""Blast Radius selector: which sqlglot tests can each patch break?

1. Static impact analysis against baseline coverage/definition maps gives a
   conservative candidate set per commit.
2. Cheap candidates are replayed against the patched checkout; only the ones
   that actually fail are kept. Expensive candidates are kept without replay
   when replaying them would cost more CPU than it could save.
3. The kept tests are packed onto up to 8 workers minimising
   busiest-worker time + 2 s per worker.
"""
from __future__ import annotations

import json
import os
import sys
import time

from . import analyze as an_mod
from .schedule import cost_ms, optimize

DATA = os.environ.get("BR_DATA", "/data")
REPO = os.environ.get("BR_REPO", "/opt/sqlglot")
INDEX = os.environ.get("BR_INDEX", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps", "index.pkl.gz"))
NO_REPLAY = os.environ.get("BR_NO_REPLAY") == "1"
REPLAY_DATA_ONLY = os.environ.get("BR_REPLAY_DATA_ONLY") == "1"
LABEL_OUT = os.environ.get("BR_LABEL_OUT")  # local-only: replay everything, dump failing sets

# CPU model (server seconds). Replaying a test costs about its recorded time;
# a commit's child costs a fixed amount plus per test-module import.
CPU_RATIO = float(os.environ.get("BR_CPU_RATIO", "1.15"))
# maps/cpu.json was measured on a machine ~1.27x faster than the one the
# durations were recorded on (median over single-process tests): scale up
CPU_FACTOR = float(os.environ.get("BR_CPU_FACTOR", "1.3"))
# measured (fork + fresh sqlglot import from the overlay): ~0.28 s; importing a
# dialect test module also imports its dialect (~0.15 s); test_optimizer
# imports and sets up much more (~1 s for a handful of cheap tests)
CHILD_BASE_MS = int(os.environ.get("BR_CHILD_BASE_MS", "150"))
MODULE_IMPORT_MS = {"tests/test_executor.py": 800, "tests/test_optimizer.py": 650,
                    "tests/test_parser.py": 50, "tests/test_expressions.py": 20}
DEFAULT_MODULE_MS = 150
CLASS_SETUP_MS = {"tests/test_executor.py::TestExecutor": 1200}  # measured: import+setUpClass ~1.75 s
HEAVY_MODULES = ("tests/test_executor.py", "tests/test_optimizer.py")
PER_COMMIT_WALL_S = 6.0  # 180 s / 30 commits (practice), 270 s / 45 (final)
# Once one replayed test fails the commit needs a worker anyway. Including the
# remaining candidates blind costs at most their summed duration (less when
# they spread over workers), replaying them costs at least that in CPU: stop.
STOP_AFTER_FAIL_MS = int(os.environ.get("BR_STOP_AFTER_FAIL_MS", str(10 ** 9)))
# Replay only pays off when every replayed candidate passes; a failure means
# the CPU is spent and a worker is needed anyway. Prior probability that the
# replayed candidates of a commit all pass (half the sample's code commits with
# candidates break something).
P_PASS = float(os.environ.get("BR_P_PASS", "0.5"))
AFTER_FAIL_RATIO = float(os.environ.get("BR_AFTER_FAIL_RATIO", "0.4"))
FILE_CPU_RATIO = {"tests/test_optimizer.py": 2.0, "tests/test_executor.py": 2.0, "tests/test_generator.py": 0.85}
# State diff (see statefp): importing every sqlglot module from the overlay
# and fingerprinting it costs about this much CPU on top of a replay child.
SD_CPU_MS = int(os.environ.get("BR_SD_CPU_MS", "700"))
NO_SD = os.environ.get("BR_NO_SD") == "1"
SD_MIN_GAIN_MS = int(os.environ.get("BR_SD_MIN_GAIN_MS", "600"))
MAPS = os.path.dirname(INDEX)
KEYS = os.environ.get("BR_KEYS", os.path.join(MAPS, "keys.pkl.gz"))
STATE = os.environ.get("BR_STATE", os.path.join(MAPS, "state.pkl"))
CPU_MAP = os.environ.get("BR_CPU_MAP", os.path.join(MAPS, "cpu.json"))
MODULES = os.environ.get("BR_MODULES", os.path.join(MAPS, "modules.json"))
IMPORT_ALL_MS = 300


class KeyMap:
    """Baseline dict-entry readers (maps/keys.pkl.gz) and import-time state
    fingerprints (maps/state.pkl, built in the image). Loaded on first use."""

    def __init__(self, idx_tests):
        import gzip
        import pickle

        with open(KEYS, "rb") as fh:
            d = pickle.loads(gzip.decompress(fh.read()))
        if d["tests"] != idx_tests:
            raise ValueError("key map built for other tests")
        self.__dict__.update(d)
        self.leaky = set(self.leaky)
        with open(STATE, "rb") as fh:
            st = pickle.load(fh)
        self.mods = st["mods"]
        self.mod_set = set(self.mods)
        self.state = st["state"]
        self.noisy = st["noisy"]
        self.noisy_keys = st["noisy_keys"]

    def readers(self, res, changed_mods, idx=None):
        """Tests that read a changed entry, or None when the diff is not
        fully explained. Changed dict entries: their recorded readers. Other
        changed data attributes of classes: tests using the class that execute
        a line naming the attribute (the static rule for class constants)."""
        if res is None:
            return None, "no diff"
        keyed, other = res
        other = [x for x in other if not (x[0] in self.noisy and x[0].split(":")[0] not in changed_mods)]
        extra = 0
        for lab, bk, nk in other:
            mod, _, q = lab.partition(":")
            cls, _, attr = q.rpartition(".")
            if idx is not None and not cls and bk in ("V", None) and nk in ("V", None):
                # module-level data: the code that names it (copies made by
                # other modules while importing are their own changed labels)
                extra |= idx.names_of(attr)
                if not attr.isupper():
                    extra |= idx.names_of("<dynamic>")
                continue
            if (idx is None or not cls or bk not in ("V", None) or nk not in ("V", None)
                    or not idx.users.get(f"{mod}.{cls}")):
                return None, f"other state changed: {[x[0] for x in other][:4]}"
            rd = idx.names_of(attr)
            if not attr.isupper():
                rd |= idx.names_of("<dynamic>")
            extra |= (idx.users_of(f"{mod}.{cls}") & rd) | idx.names_of(f"{cls.rsplit('.', 1)[-1]}.{attr}")
        acc = self.sets[self.unknown]
        sets = self.sets
        acc_idx = extra
        for lab, (keys, keyset) in keyed.items():
            canon = self.aliases.get(lab)
            if canon is None or canon in self.leaky:
                return None, f"untracked {lab}"
            nk = self.noisy_keys.get(lab)
            if nk and lab.split(":")[0] not in changed_mods:
                keys = keys - nk
            e = self.km.get(canon, {})
            for k in keys:
                if k in e:
                    acc |= sets[e[k]]
            for k in ("*", "?") + (("#",) if keyset else ()):
                if k in e:
                    acc |= sets[e[k]]
        return (acc, acc_idx), (f"{sum(len(k) for k, _ in keyed.values())} entries of {len(keyed)} dicts"
                                + (f", {len(other)} attrs" if other else ""))


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def bits_to_list(b: int) -> list:
    out = []
    i = 0
    while b:
        if b & 1:
            out.append(i)
        b >>= 1
        i += 1
    return out


def bits_iter(b: int):
    # fast enough for 1.6k bits
    s = bin(b)[2:][::-1]
    return [i for i, c in enumerate(s) if c == "1"]


def _load_cpu_map():
    try:
        with open(CPU_MAP) as fh:
            return json.load(fh)
    except Exception as e:  # noqa
        log("cpu map unavailable:", e)
        return {"tests": {}, "classes": {}, "modules": {}}


_CPU = None


def _cpu():
    global _CPU
    if _CPU is None:
        _CPU = _load_cpu_map()
    return _CPU


def test_cpu_ms(n, tests_by_num, dur) -> float:
    """Replay CPU of one test: measured on the pinned checkout in this
    harness (maps/cpu.json) scaled to the scoring machine, else the
    recorded duration."""
    ms = _cpu()["tests"].get(tests_by_num[n]["id"])
    if ms is None:
        return dur[n] * CPU_RATIO
    return ms * CPU_FACTOR + 2


def file_cpu_ms(f) -> float:
    ms = _cpu()["modules"].get(f)
    return DEFAULT_MODULE_MS if ms is None else ms * CPU_FACTOR


def class_cpu_ms(c) -> float:
    return _cpu()["classes"].get(c, 0) * CPU_FACTOR


def partial_of(bits: int, idx, num_of_id) -> dict:
    """Long fixture-driven tests (selector/parts.py) the candidate bits reach
    only through some of their items: test number -> item indices to replay."""
    out = {}
    for tid, p in (getattr(idx, "parts", None) or {}).items():
        if not bits >> p["real"] & 1 or bits >> p["base"] & 1:
            continue
        keep = [i for i, q in enumerate(p["parts"]) if bits >> q & 1]
        n = num_of_id.get(tid)
        if keep and n is not None:
            out[n] = keep
    return out


def worth_after_fail(n, partial, tests_by_num, dur) -> bool:
    """Once a commit has a failing test it needs a worker anyway: including a
    candidate blind then costs at most its duration, so only replays far
    cheaper than that remain worth their CPU."""
    c = (partial_cpu_ms(n, partial[n], tests_by_num, dur) if partial and n in partial
         else test_cpu_ms(n, tests_by_num, dur))
    return c < AFTER_FAIL_RATIO * dur[n]


def whole_items(bits: int, idx) -> int:
    """`bits` with every item of a partial test whose base it has: a set that
    holds T@base stands for all of T."""
    for p in (getattr(idx, "parts", None) or {}).values():
        if bits >> p["base"] & 1:
            for q in p["parts"]:
                bits |= 1 << q
    return bits


def partial_cpu_ms(n, keep, tests_by_num, dur) -> float:
    from .parts import pseudo_id

    tid = tests_by_num[n]["id"]
    t = _cpu()["tests"]
    base = t.get(pseudo_id(tid, "base"))
    items = [t.get(pseudo_id(tid, i)) for i in keep]
    if base is None or any(x is None for x in items):
        return test_cpu_ms(n, tests_by_num, dur)
    return (base + sum(max(0, x - base) for x in items)) * CPU_FACTOR + 2


def replay_cpu_ms(nums, tests_by_num, dur, base_ms: int = CHILD_BASE_MS, partial=None) -> int:
    files = set()
    classes = set()
    total = base_ms
    for n in nums:
        t = tests_by_num[n]
        files.add(t["file"])
        classes.add(t["id"].rsplit("::", 1)[0])
        total += (partial_cpu_ms(n, partial[n], tests_by_num, dur) if partial and n in partial
                  else test_cpu_ms(n, tests_by_num, dur))
    for f in files:
        total += file_cpu_ms(f)
    for c in classes:
        total += class_cpu_ms(c)
    return int(total)


def sched_cost(nums, dur) -> int:
    if not nums:
        return 0
    return cost_ms(optimize(nums, dur, swap_trials=64), dur)


def lb_cost(total: int, longest: int, count: int) -> int:
    if count == 0:
        return 0
    return min(max(longest, -(-total // k)) + 2000 * k for k in range(1, min(8, count) + 1))


def plan_replay(cand: list, forced: list, dur, tests_by_num, base_ms: int = CHILD_BASE_MS, partial=None):
    """Split candidates into (replay, blind). Replay the cheapest prefix of the
    candidates (by duration) maximising (cost of selecting everything) -
    (cost of the rest) - (replay CPU). Uses the packing lower bound while
    searching, the real scheduler for the final check."""
    forced_set = set(forced)
    allsel = sorted(set(cand) | forced_set)
    base_cost = sched_cost(allsel, dur)
    # cheapest replay per second of schedule first (a test whose replay costs
    # far less CPU than its recorded duration comes early, however long)
    partial = partial or {}
    tcpu = {n: (partial_cpu_ms(n, partial[n], tests_by_num, dur) if n in partial
                else test_cpu_ms(n, tests_by_num, dur)) for n in set(cand) - forced_set}
    order = sorted(tcpu, key=lambda n: ((tcpu[n] + 20) / (dur[n] + 20), dur[n], n))
    if not order:
        return [], allsel, base_cost, 0
    f_total = sum(dur[n] for n in forced_set)
    f_max = max((dur[n] for n in forced_set), default=0)
    suffix_total = [0] * (len(order) + 1)
    suffix_max = [0] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        suffix_total[i] = suffix_total[i + 1] + dur[order[i]]
        suffix_max[i] = max(suffix_max[i + 1], dur[order[i]])
    files, classes = set(), set()
    cpu = base_ms
    best_net, best_i = 0, 0
    for i, n in enumerate(order):
        t = tests_by_num[n]
        if t["file"] not in files:
            files.add(t["file"])
            cpu += file_cpu_ms(t["file"])
        c = t["id"].rsplit("::", 1)[0]
        if c not in classes:
            classes.add(c)
            cpu += class_cpu_ms(c)
        cpu += tcpu[n]
        rest_n = len(order) - i - 1 + len(forced_set)
        rest_max = max(f_max, suffix_max[i + 1])
        net = P_PASS * (base_cost - lb_cost(f_total + suffix_total[i + 1], rest_max, rest_n)) - cpu
        if net > best_net:
            best_net, best_i = net, i + 1
    if best_i == 0:
        return [], allsel, base_cost, 0
    rep = order[:best_i]
    blind = sorted(forced_set | set(order[best_i:]))
    net = P_PASS * (base_cost - sched_cost(blind, dur)) - replay_cpu_ms(rep, tests_by_num, dur, base_ms, partial)
    if net <= 0:
        return [], allsel, base_cost, 0
    # tests that fit into the blind schedule's slack are free: no replay needed
    if blind:
        w = optimize(blind, dur, {n: 1.0 / (1 + dur[n]) for n in rep}, swap_trials=64)
        free = {n for x in w for n in x} - set(blind)
        if free:
            rep = [n for n in rep if n not in free]
            blind = sorted(set(blind) | free)
    return rep, blind, base_cost, net


def make_sd_plan(cid, pl, km, changed_mods, to_nums, dur, tests_by_num, idx, km_nums, num_of_id):
    """Planner run inside the replay child once the state diff is known.
    Falls back to the commit's static plan when the diff is not explained."""
    rest, forced = pl["sd"]["rest"], pl["forced"]
    static_keyed = set(to_nums(pl["an"].keyed)) | set(pl["sd"]["unknown"])

    def job(nums, partial=None):
        partial = partial or {}
        return [(tests_by_num[n]["id"], max(10.0, dur[n] / 1000.0 * 12 + 5), dur[n], partial.get(n),
                 worth_after_fail(n, partial, tests_by_num, dur))
                for n in sorted(nums, key=lambda n: (dur[n], n))]

    def plan(res):
        bits, note = km.readers(res, changed_mods, idx)
        if bits is None:
            return (job(pl["replay"], pl.get("partial")), list(pl["replay"]), list(pl["blind"]),
                    "fallback: " + note)
        kbits, ibits = bits
        # both the diff's readers and the static keyed candidates are supersets
        narrowed = (set(km_nums(kbits)) | set(to_nums(ibits))) & static_keyed
        cand = (rest | narrowed) - forced
        partial = partial_of(pl["an"].other | ((kbits | ibits) & whole_items(pl["an"].keyed, idx)), idx, num_of_id)
        # the child already paid for its start and sqlglot import
        rep, blind, _, _ = plan_replay(sorted(cand), sorted(forced), dur, tests_by_num, base_ms=0,
                                       partial=partial)
        return job(rep, partial), list(rep), list(blind), f"{note} -> {len(cand)} cand"

    return plan


def write_out(out_fd: int, lines: list):
    data = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    view = memoryview(data)
    while view:
        n = os.write(out_fd, view)
        view = view[n:]


def fallback_lines() -> list:
    """Every test for every commit: valid output whatever went wrong."""
    tests = json.load(open(os.path.join(DATA, "tests.json")))["tests"]
    dur = {t["number"]: int(round(t["seconds"] * 1000)) for t in tests}
    workers = optimize(sorted(dur), dur, swap_trials=64)
    out = []
    for f in sorted(os.listdir(os.path.join(DATA, "commits"))):
        if f.endswith(".patch"):
            out.append(json.dumps({"commit": f[:-6], "workers": workers}, separators=(",", ":")))
    return out


def entry():
    # Keep a private handle on the real stdout and point fd 1 at stderr, so
    # nothing else (libraries, tests, child processes) can write answers.
    out_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    pid = os.getpid()
    try:
        lines = main()
    except BaseException as e:  # noqa
        if os.getpid() != pid:
            os._exit(1)
        import traceback

        traceback.print_exc()
        log(f"selector failed ({type(e).__name__}); falling back to all tests")
        lines = fallback_lines()
    write_out(out_fd, lines)
    os.close(out_fd)


def main():
    t0 = time.monotonic()
    manifest = json.load(open(os.path.join(DATA, "tests.json")))
    tests = manifest["tests"]
    tests_by_num = {t["number"]: t for t in tests}
    dur = {t["number"]: int(round(t["seconds"] * 1000)) for t in tests}
    patches = sorted(
        os.path.join(DATA, "commits", f) for f in os.listdir(os.path.join(DATA, "commits")) if f.endswith(".patch")
    )
    n_commits = len(patches)
    all_nums = sorted(dur)

    try:
        idx = an_mod.Index(INDEX)
        num_of_id = {t["id"]: t["number"] for t in tests}
        idx_to_num = [num_of_id.get(nid) for nid in idx.tests]
        known = {n for n in idx_to_num if n is not None}
        unknown = [n for n in all_nums if n not in known]
    except Exception as e:  # noqa
        log("index load failed", e)
        idx = None
        unknown = all_nums
    log(f"loaded index in {time.monotonic() - t0:.2f}s; {len(unknown)} unmapped tests")

    def to_nums(b: int) -> list:
        return [idx_to_num[i] for i in bits_iter(b) if idx_to_num[i] is not None]

    plans = {}
    for p in patches:
        cid = os.path.basename(p)[:-6]
        try:
            text = open(p, "rb").read().decode("utf-8", errors="surrogateescape")
        except Exception as e:  # noqa
            text = ""
            log(cid, "unreadable", e)
        if idx is None:
            plans[cid] = {"blind": all_nums, "replay": [], "an": None}
            continue
        try:
            a = an_mod.analyze(text, idx, REPO)
        except Exception as e:  # noqa
            log(cid, "analysis crashed", e)
            a = an_mod.Analysis(forced=idx.ALL, replayable=False)
        cand = set(to_nums(a.cand))
        forced = set(to_nums(a.forced))
        if unknown and (a.cand or a.forced):
            cand |= set(unknown)
        cand -= forced
        if not a.code_changed and REPLAY_DATA_ONLY is False:
            # only data files changed: their direct readers nearly always
            # fail, so replaying them is wasted CPU
            forced |= cand
            cand = set()
        if not a.replayable or NO_REPLAY:
            forced |= cand
            cand = set()
        if LABEL_OUT and a.replayable:
            # local labelling mode: replay every candidate, keep nothing blind
            cand |= forced
            forced = set()
            rep, blind, base_cost, net = sorted(cand), [], 0, 1
        else:
            partial = partial_of(a.cand, idx, num_of_id) if cand else {}
            partial = {n: k for n, k in partial.items() if n in cand}
            rep, blind, base_cost, net = plan_replay(sorted(cand), sorted(forced), dur, tests_by_num,
                                                     partial=partial)
        plans[cid] = {"replay": rep, "blind": blind, "an": a, "base": base_cost, "net": net,
                      "ncand": len(cand), "nforced": len(forced), "cand": cand, "forced": forced,
                      "partial": {n: k for n, k in partial.items() if n in rep} if not LABEL_OUT else {}}
        if a.keyed and a.replayable and a.code_changed and not NO_SD and not LABEL_OUT:
            # what the keyed part costs at most: the plan without it
            rest = (set(to_nums(a.other)) | (set(unknown) if a.other else set())) - forced
            r_rep, r_blind, _, r_net = plan_replay(sorted(rest), sorted(forced), dur, tests_by_num)
            # expected cost of the static plan if its replays pass, against the
            # best case of the diff (keyed candidates all dropped) plus its CPU
            static = sched_cost(blind, dur) + (replay_cpu_ms(rep, tests_by_num, dur, partial=partial) if rep else 0)
            narrow = sched_cost(r_blind, dur) + replay_cpu_ms(r_rep, tests_by_num, dur) + SD_CPU_MS
            gain = static - narrow
            if gain > SD_MIN_GAIN_MS:
                plans[cid]["sd"] = {"rest": rest, "gain": gain, "unknown": unknown}
        log(f"{cid}: cand={len(cand)} forced={len(forced)} base={base_cost/1000:.2f}s "
            f"replay={len(rep)} blind={len(blind)} est_net={net/1000:.2f}s"
            + (f" sd_gain={plans[cid]['sd']['gain']/1000:.2f}s" if "sd" in plans[cid] else "")
            + " | " + "; ".join(a.notes)[:600])
    log(f"analysis done at {time.monotonic() - t0:.2f}s")

    # ---- state diffs: keyed candidates narrowed in the replay child
    km = None
    if any("sd" in pl for pl in plans.values()):
        try:
            km = KeyMap(idx.tests)
        except Exception as e:  # noqa
            log("key map unavailable:", e)
    for cid, pl in plans.items():
        if "sd" not in pl:
            continue
        a = pl["an"]
        changed_mods = {an_mod.module_name(r) for r in a.overlay if r.endswith(".py")}
        if km is None or not a.keyed_modules <= km.mod_set or not (changed_mods - {None}) <= km.mod_set:
            del pl["sd"]
            continue
        # the assigned names must be tracked dicts, or the diff cannot be explained
        untracked = [x for x in a.keyed_labels if x is None or x not in km.aliases or km.aliases[x] in km.leaky]
        if untracked:
            log(f"{cid}: no state diff, untracked {untracked[:3]}")
            del pl["sd"]
            continue
        pl["sd"]["plan"] = make_sd_plan(cid, pl, km, changed_mods, to_nums, dur, tests_by_num, idx, to_nums, num_of_id)
        pl["net"] = max(pl["net"], pl["sd"]["gain"])
        pl["sd_tag"] = True

    # ---- replay
    to_run = [cid for cid in sorted(plans) if plans[cid]["replay"] or "sd" in plans[cid]]
    # Commits not replayed whose change runs at import time: a child that only
    # imports the changed modules, so an import-time failure is not missed.
    probe_only = [
        cid for cid in sorted(plans)
        if not plans[cid]["replay"] and "sd" not in plans[cid]
        and plans[cid]["an"] is not None and plans[cid]["an"].needs_probe
        and plans[cid]["an"].replayable and any(plans[cid]["an"].probe_modules)
    ]
    to_run += probe_only
    budget_end = t0 + PER_COMMIT_WALL_S * 0.72 * max(n_commits, 1)
    results = {}
    if to_run:
        from .replay import Replayer

        rp = Replayer(REPO, parallel=2)
        base_mods = None
        try:
            with open(MODULES) as fh:
                base_mods = json.load(fh)
        except Exception as e:  # noqa
            log("module list unavailable:", e)
        heavy = any(
            any(tests_by_num[n]["file"] in HEAVY_MODULES for n in plans[c]["replay"]) for c in to_run
        )
        rp.preload(heavy)
        log(f"preload done at {time.monotonic() - t0:.2f}s (heavy={heavy})")
        # most valuable first
        queue = sorted(to_run, key=lambda c: -plans[c]["net"])
        while queue or rp.running:
            while queue and len(rp.running) < rp.parallel:
                cid = queue.pop(0)
                pl = plans[cid]
                now = time.monotonic()
                est = replay_cpu_ms(pl["replay"], tests_by_num, dur, partial=pl.get("partial")) / 1000.0
                if "sd" in pl:
                    est += SD_CPU_MS / 1000.0
                elif pl["an"].needs_probe:
                    est += IMPORT_ALL_MS / 1000.0
                if now + est * 1.5 > budget_end and not LABEL_OUT:
                    log(f"{cid}: skip replay (budget)")
                    results[cid] = {"error": "budget"}
                    continue
                a = pl["an"]
                # cheapest first, so a breaking change is usually caught early
                job_tests = [
                    (tests_by_num[n]["id"], max(10.0, dur[n] / 1000.0 * 12 + 5), dur[n],
                     pl.get("partial", {}).get(n), worth_after_fail(n, pl.get("partial"), tests_by_num, dur))
                    for n in sorted(pl["replay"], key=lambda n: (dur[n], n))
                ]
                if pl.get("partial"):
                    log(f"{cid}: partial replay "
                        + ", ".join(f"{tests_by_num[n]['id'].rsplit('::', 1)[1]}[{len(k)}]"
                                    for n, k in pl["partial"].items()))
                deadline = now + 30 + est * 6
                try:
                    rp.start(cid, a.overlay, [m for m in a.probe_modules if m], job_tests, deadline,
                             stop_after_fail_ms=0 if LABEL_OUT else STOP_AFTER_FAIL_MS,
                             sd=({"mods": km.mods, "state": km.state, "plan": pl["sd"]["plan"]}
                                 if "sd" in pl else None),
                             import_all=base_mods if a.needs_probe else None)
                except Exception as e:  # noqa
                    log(f"{cid}: replay start failed {e}")
                    results[cid] = {"error": str(e)}
            if rp.running:
                rp.poll(block=True)
        results.update(rp.results)
        log(f"replay done at {time.monotonic() - t0:.2f}s")

    # ---- final selection
    id_to_num = {t["id"]: t["number"] for t in tests}
    total = 0
    out_lines = []
    labels = {}
    for cid in sorted(plans):
        pl = plans[cid]
        sel = set(pl["blind"])
        rep = pl["replay"]
        if cid in probe_only:
            r = results.get(cid)
            a = pl["an"]
            if r is None or "error" in r or r.get("probe"):
                log(f"{cid}: probe failed/unknown {r} -> importers")
                for m in a.probe_modules:
                    fb = a.on_import_fail.get(m)
                    sel |= set(to_nums(idx.ALL if fb is None else fb))
        elif rep or "sd" in pl:
            r = results.get(cid)
            if r is not None and "error" not in r and "sd" in r:
                # the child planned from the state diff
                log(f"{cid}: state diff: {r['sd']['note']}")
                sel = set(r["sd"]["blind"])
                rep = r["sd"]["rep"]
            if r is None or "error" in r:
                sel |= set(rep)
                if r is not None:
                    log(f"{cid}: replay error {r.get('error')}")
            else:
                if r.get("probe"):
                    log(f"{cid}: import failure {r['probe']}")
                    a = pl["an"]
                    for m in r["probe"]:
                        fb = a.on_import_fail.get(m)
                        if fb is None:
                            fb = idx.ALL
                        sel |= set(to_nums(fb))
                    sel |= set(rep)
                else:
                    failed = {id_to_num[x] for x in r["fail"] if x in id_to_num}
                    sel |= failed
                    ran = {id_to_num[x] for x in r["fail"] + r["pass"] if x in id_to_num}
                    sel |= set(rep) - ran
                    log(f"{cid}: replayed {len(rep)} in {r.get('cpu')}s cpu -> {len(failed)} failing {r.get('t')}")
        r = results.get(cid)
        if r is not None and r.get("import_fail"):
            # modules that imported at the baseline and no longer do: every
            # test importing them (or running their code) fails
            log(f"{cid}: modules fail to import {r['import_fail'][:5]}")
            # sqlglot modules are imported lazily by name, so their importers
            # are not fully mapped: any test can fail (see analyze.fail_bits)
            sel |= set(to_nums(idx.ALL))
        sel = sorted(sel)
        if LABEL_OUT:
            labels[cid] = {"failing": sel, "kind": "replay" if rep else "blind"}
        workers = optimize(sel, dur, swap_trials=256) if sel else []
        c = cost_ms(workers, dur)
        total += c
        out_lines.append(json.dumps({"commit": cid, "workers": workers}, separators=(",", ":")))
    if LABEL_OUT:
        json.dump({"commits": labels}, open(LABEL_OUT, "w"))
    log(f"done in {time.monotonic() - t0:.2f}s; simulated cost {total/1000:.1f}s")
    return out_lines


if __name__ == "__main__":
    entry()
