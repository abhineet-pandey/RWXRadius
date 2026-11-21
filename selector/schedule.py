"""Deterministic scored-cost scheduling and no-marginal-cost safety fill.

Only caller-supplied test IDs and baseline durations are used. No failure data.
Optional candidates must already satisfy the caller's safety policy for exclusion.
"""
from __future__ import annotations
from heapq import heapify, heappop, heappush
from typing import Iterable, Mapping

OVERHEAD = 2000


def cost_ms(workers, durations):
    loads = [sum(durations[i] for i in w) for w in workers if w]
    return max(loads) + OVERHEAD * len(loads) if loads else 0


def _lpt(ids, d, k):
    workers = [[] for _ in range(k)]
    heap = [(0, 0, j) for j in range(k)]
    heapify(heap)
    for i in ids:
        load, n, j = heappop(heap)
        workers[j].append(i)
        heappush(heap, (load + d[i], n+1, j))
    return workers


def _improve(workers, d, floor, trials):
    """Bounded strict-improvement swaps/moves, never increase largest load."""
    loads = [sum(d[i] for i in w) for w in workers]
    k = len(workers)
    while trials > 0 and max(loads) > floor:
        hi = max(range(k), key=lambda j: loads[j])
        peak = loads[hi]
        other = sorted((j for j in range(k) if j != hi), key=lambda j: loads[j])
        improved = False
        for lo in other:
            if trials <= 0: break
            for ai, a in enumerate(workers[hi]):
                if trials <= 0: break
                da = d[a]
                trials -= 1
                # Never remove the only test from a worker in this fixed-k search.
                rest = max((loads[j] for j in range(k) if j not in (hi,lo)), default=0)
                if len(workers[hi]) > 1 and max(rest, peak-da, loads[lo]+da) < peak:
                    workers[lo].append(workers[hi].pop(ai))
                    loads[hi] -= da; loads[lo] += da; improved = True; break
                for bi, b in enumerate(workers[lo]):
                    if trials <= 0: break
                    trials -= 1
                    diff = da-d[b]
                    if diff > 0 and max(rest, peak-diff, loads[lo]+diff) < peak:
                        workers[hi][ai], workers[lo][bi] = b, a
                        loads[hi] -= diff; loads[lo] += diff; improved = True; break
                if improved: break
            if improved: break
        if not improved: break
    return workers


def optimize(required: Iterable[int], durations: Mapping[int,int], optional_priority: Mapping[int,float] | None = None, *, swap_trials: int = 256):
    """Minimize cost for required tests, then fill slack with optional tests.

    Priority is an ordering signal, not a calibrated probability. Empty required
    set stays empty; risk decisions that justify opening workers belong upstream.
    Returns list of nonempty workers. Uses integer ms; at most eight workers.
    """
    ids = sorted(set(required), key=lambda i: (-durations[i], i))
    if not ids: return []
    if any(durations[i] < 0 for i in ids): raise ValueError('negative duration')
    optional = optional_priority or {}
    required_set = set(ids)
    opts = sorted((i for i in optional if i not in required_set), key=lambda i: (-optional[i], durations[i], i)) if optional else []
    total = sum(durations[i] for i in ids)
    longest = durations[ids[0]]
    best = [ids[:]]; best_cost = total + OVERHEAD
    schedules = [(best, best_cost)]
    for k in range(2, min(8,len(ids))+1):
        floor = max(longest,(total+k-1)//k)
        if floor + OVERHEAD*k > best_cost: continue
        w = _lpt(ids,durations,k)
        if swap_trials and max(sum(durations[i] for i in x) for x in w) > floor:
            _improve(w,durations,floor,swap_trials)
        c = cost_ms(w,durations)
        if c < best_cost: best_cost = c
        schedules.append((w,c))
    # Among cost ties, retain the schedule with most optional priority in free capacity.
    winner = None; winner_key = None
    for w,c in schedules:
        if c != best_cost: continue
        w = [x[:] for x in w]
        peak = max(sum(durations[i] for i in x) for x in w)
        slack = [peak-sum(durations[i] for i in x) for x in w]
        gained = 0.0; count = 0
        for i in opts:
            t = durations[i]
            slots = [j for j,s in enumerate(slack) if s >= t]
            if slots:
                j = min(slots,key=lambda j:(slack[j]-t,j))
                w[j].append(i); slack[j] -= t; gained += optional[i]; count += 1
        key = (gained,count,-len(w))
        if winner_key is None or key > winner_key: winner,winner_key = w,key
    assert winner is not None and cost_ms(winner,durations) == best_cost
    return winner

