"""Non-learning comparators: wirelength router and a thermal-aware Dijkstra."""
import heapq
import numpy as np
import params as P
import physics as ph


def dijkstra(geo, cell_cost, levels_allowed, width_change_cost=0.0):
    """Shortest path over (row, col, level). cell_cost(level, r, c) -> cost of entering."""
    n = geo.n
    legal = {lv: np.array([[geo.legal((r, c), lv) for c in range(n)] for r in range(n)])
             for lv in levels_allowed}
    best, prev = {}, {}
    pq = []
    for lv in levels_allowed:
        if legal[lv][geo.start]:
            s = (*geo.start, lv)
            best[s] = 0.0
            heapq.heappush(pq, (0.0, s))
    goal = None
    while pq:
        c, s = heapq.heappop(pq)
        if c > best.get(s, 1e18):
            continue
        r, cc, lv = s
        if (r, cc) == geo.target:
            goal = s
            break
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, c2 = r + dr, cc + dc
            if not (0 <= rr < n and 0 <= c2 < n):
                continue
            for lv2 in levels_allowed:
                if not legal[lv2][rr, c2]:
                    continue
                nc = c + cell_cost(lv2, rr, c2) + (width_change_cost if lv2 != lv else 0)
                t = (rr, c2, lv2)
                if nc < best.get(t, 1e18):
                    best[t], prev[t] = nc, s
                    heapq.heappush(pq, (nc, t))
    if goal is None:
        return None
    seq = [goal]
    while seq[-1] in prev:
        seq.append(prev[seq[-1]])
    seq.reverse()
    return [(a, b) for a, b, _ in seq], [c for _, _, c in seq]


def wirelength_router(geo, level):
    """Conventional auto-router: fixed width, shortest legal path."""
    return dijkstra(geo, lambda lv, r, c: 1.0, [level])


def evaluate(path, levels, geo):
    L = ph.loop_inductance_nh(path, levels, geo)
    T, ts = ph.solve_temperature(path, levels, geo)
    tr = ph.turn_off_transient(L)
    return dict(L_nH=L, Tmax_C=float(T.max()), Tsink_C=float(ts), Vpk_V=float(tr[2]),
                length_mm=(len(path) - 1) * geo.cell_m * 1e3)
