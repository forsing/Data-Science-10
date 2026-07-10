"""Grupa 10 — grafovi zavisnosti (Loto 7/39)."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4648_k55.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def presence_matrix(draws: np.ndarray) -> np.ndarray:
    x = np.zeros((len(draws), FRONT_N), dtype=float)
    for i, draw in enumerate(draws):
        for n in draw.tolist():
            x[i, n - 1] = 1.0
    return x


def cooccurrence_graph(draws: np.ndarray) -> np.ndarray:
    """Težinska matrica co-occurrence (ceo CSV)."""
    w = np.zeros((FRONT_N, FRONT_N), dtype=float)
    for draw in draws:
        nums = draw.tolist()
        for i, a in enumerate(nums):
            for b in nums[i + 1 :]:
                w[a - 1, b - 1] += 1.0
                w[b - 1, a - 1] += 1.0
    return w


def correlation_graph(draws: np.ndarray) -> np.ndarray:
    x = presence_matrix(draws)
    c = np.corrcoef(x, rowvar=False)
    np.fill_diagonal(c, 0.0)
    return np.nan_to_num(c, nan=0.0)


def mi_graph(draws: np.ndarray) -> np.ndarray:
    x = presence_matrix(draws)
    mi = np.zeros((FRONT_N, FRONT_N), dtype=float)
    for i in range(FRONT_N):
        for j in range(i + 1, FRONT_N):
            a, b = x[:, i], x[:, j]
            val = 0.0
            for aa in (0.0, 1.0):
                for bb in (0.0, 1.0):
                    pxy = np.mean((a == aa) & (b == bb))
                    px = np.mean(a == aa)
                    py = np.mean(b == bb)
                    if pxy > 0 and px > 0 and py > 0:
                        val += pxy * np.log2(pxy / (px * py))
            mi[i, j] = mi[j, i] = val
    return mi


def top_edges(mat: np.ndarray, top_k: int = 15, abs_w: bool = False) -> list[tuple]:
    edges = []
    for i in range(FRONT_N):
        for j in range(i + 1, FRONT_N):
            v = float(mat[i, j])
            score = abs(v) if abs_w else v
            edges.append((i + 1, j + 1, v, score))
    edges.sort(key=lambda t: (-t[3], t[0], t[1]))
    return [(a, b, v) for a, b, v, _ in edges[:top_k]]


def pagerank(w: np.ndarray, damping: float = 0.85, n_iter: int = 60) -> list[tuple]:
    """PageRank na težinskom grafu."""
    a = np.abs(w).copy()
    np.fill_diagonal(a, 0.0)
    row = a.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    p = a / row
    n = FRONT_N
    r = np.ones(n) / n
    teleport = np.ones(n) / n
    for _ in range(n_iter):
        r = damping * (p.T @ r) + (1 - damping) * teleport
    ranked = sorted(((i + 1, float(r[i])) for i in range(n)), key=lambda t: (-t[1], t[0]))
    return ranked


def degree_centrality(w: np.ndarray) -> list[tuple]:
    a = np.abs(w)
    np.fill_diagonal(a, 0.0)
    deg = a.sum(axis=1)
    return sorted(((i + 1, float(deg[i])) for i in range(FRONT_N)), key=lambda t: (-t[1], t[0]))


def eigenvector_centrality(w: np.ndarray) -> list[tuple]:
    a = np.abs(w)
    np.fill_diagonal(a, 0.0)
    # power iteration
    v = np.ones(FRONT_N)
    for _ in range(50):
        v = a @ v
        nrm = np.linalg.norm(v) + 1e-12
        v = v / nrm
    return sorted(((i + 1, float(v[i])) for i in range(FRONT_N)), key=lambda t: (-t[1], t[0]))


def betweenness_approx(w: np.ndarray, samples: int = 39) -> list[tuple]:
    """
    Aproksimacija betweenness: BFS na top-edge unweighted grafu,
    uzorak izvora (seed=39).
    """
    # threshold: top 15% |w|
    flat = np.abs(w)[np.triu_indices(FRONT_N, 1)]
    thr = float(np.quantile(flat, 0.85)) if len(flat) else 0.0
    adj = [[] for _ in range(FRONT_N)]
    for i in range(FRONT_N):
        for j in range(i + 1, FRONT_N):
            if abs(w[i, j]) >= thr:
                adj[i].append(j)
                adj[j].append(i)

    bet = np.zeros(FRONT_N)
    rng = np.random.default_rng(SEED)
    sources = rng.choice(FRONT_N, size=min(samples, FRONT_N), replace=False)
    for s in sources:
        # Brandes-lite
        stack = []
        pred = [[] for _ in range(FRONT_N)]
        sigma = np.zeros(FRONT_N)
        dist = -np.ones(FRONT_N)
        sigma[s] = 1.0
        dist[s] = 0
        q = [s]
        while q:
            v = q.pop(0)
            stack.append(v)
            for nei in adj[v]:
                if dist[nei] < 0:
                    dist[nei] = dist[v] + 1
                    q.append(nei)
                if dist[nei] == dist[v] + 1:
                    sigma[nei] += sigma[v]
                    pred[nei].append(v)
        delta = np.zeros(FRONT_N)
        while stack:
            w_ = stack.pop()
            for v in pred[w_]:
                if sigma[w_] > 0:
                    delta[v] += (sigma[v] / sigma[w_]) * (1.0 + delta[w_])
            if w_ != s:
                bet[w_] += delta[w_]
    return sorted(((i + 1, float(bet[i])) for i in range(FRONT_N)), key=lambda t: (-t[1], t[0]))


def louvain_communities(w: np.ndarray, n_passes: int = 8) -> dict:
    """
    Louvain-lite (modularity maximization) na |w|.
    """
    a = np.abs(w).copy()
    np.fill_diagonal(a, 0.0)
    m2 = a.sum()
    if m2 <= 0:
        return {"communities": {i: [i + 1] for i in range(FRONT_N)}, "modularity": 0.0}
    k = a.sum(axis=1)
    comm = np.arange(FRONT_N)  # node → community

    def modularity() -> float:
        q = 0.0
        for i in range(FRONT_N):
            for j in range(FRONT_N):
                if comm[i] == comm[j]:
                    q += a[i, j] - (k[i] * k[j] / m2)
        return float(q / m2)

    rng = np.random.default_rng(SEED)
    for _ in range(n_passes):
        improved = False
        order = rng.permutation(FRONT_N)
        for i in order:
            # candidate communities = neighbors'
            neigh_c = {comm[j] for j in range(FRONT_N) if a[i, j] > 0}
            neigh_c.add(comm[i])
            best_c, best_dq = comm[i], 0.0
            # remove i from its community contribution approx via full Q delta
            old_c = comm[i]
            for c in neigh_c:
                comm[i] = c
                # local gain: sum edges to c minus expected
                gain = 0.0
                for j in range(FRONT_N):
                    if comm[j] == c and j != i:
                        gain += a[i, j] - (k[i] * k[j] / m2)
                if gain > best_dq:
                    best_dq, best_c = gain, c
            comm[i] = best_c if best_dq > 0 else old_c
            if comm[i] != old_c:
                improved = True
        if not improved:
            break

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(FRONT_N):
        groups[int(comm[i])].append(i + 1)
    # renumber
    communities = {idx: groups[k] for idx, k in enumerate(sorted(groups))}
    return {"communities": communities, "modularity": modularity(), "labels": comm}


def triangle_motifs(w: np.ndarray, top_k: int = 10) -> list[tuple]:
    """Top trouglovi po min težini ivice (motif proxy)."""
    thr_edges = top_edges(w, top_k=80, abs_w=False)
    # build adj of strong positive co-occ
    strong = set()
    for a, b, v in thr_edges:
        if v > 0:
            strong.add((min(a, b), max(a, b)))
    tris = []
    nodes = list(range(1, FRONT_N + 1))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            eij = (nodes[i], nodes[j])
            if eij not in strong:
                continue
            for k in range(j + 1, len(nodes)):
                eik = (nodes[i], nodes[k])
                ejk = (nodes[j], nodes[k])
                if eik in strong and ejk in strong:
                    wi = w[nodes[i] - 1, nodes[j] - 1]
                    wj = w[nodes[i] - 1, nodes[k] - 1]
                    wk = w[nodes[j] - 1, nodes[k] - 1]
                    tris.append((nodes[i], nodes[j], nodes[k], float(min(wi, wj, wk))))
    tris.sort(key=lambda t: (-t[3], t[0], t[1], t[2]))
    return tris[:top_k]


def notears_proxy(draws: np.ndarray, top_k: int = 15) -> list[tuple]:
    """
    NOTEARS/LiNGAM-lite proxy: usmerene ivice iz lag-1 linearne regresije
    presence_i(t) → presence_j(t+1); top |coef|.
    """
    x = presence_matrix(draws)
    a, b = x[:-1], x[1:]
    # ridge-ish: coef_ij = cov / var
    edges = []
    for i in range(FRONT_N):
        xi = a[:, i]
        var = float(np.var(xi)) + 1e-12
        for j in range(FRONT_N):
            if i == j:
                continue
            coef = float(np.cov(xi, b[:, j])[0, 1] / var)
            edges.append((i + 1, j + 1, coef))
    edges.sort(key=lambda t: (-abs(t[2]), t[0], t[1]))
    return edges[:top_k]


def learn_next_rule(draws: np.ndarray) -> dict:
    """
    Pravilo next iz grupe 10:
    skor(y) = veza (co-occ / |corr| / MI) sa last draw
            + PageRank + community sa last.
    """
    co = cooccurrence_graph(draws)
    corr = correlation_graph(draws)
    mi = mi_graph(draws)
    pr = {n: s for n, s in pagerank(co)}
    max_pr = max(pr.values()) if pr else 1.0
    lou = louvain_communities(co)
    labels = lou["labels"]
    last = [int(v) for v in draws[-1].tolist()]
    last_comms = {int(labels[n - 1]) for n in last}

    max_co = float(co.max()) if co.max() > 0 else 1.0
    max_mi = float(mi.max()) if mi.max() > 0 else 1.0

    freq = Counter(draws.reshape(-1).tolist())
    max_f = max(freq.values()) if freq else 1

    number_score = {}
    for y in range(1, FRONT_N + 1):
        s = 0.0
        for x0 in last:
            if x0 == y:
                continue
            s += co[x0 - 1, y - 1] / max_co
            s += abs(corr[x0 - 1, y - 1])
            s += mi[x0 - 1, y - 1] / max_mi
        s += 0.5 * (pr[y] / max_pr)
        if int(labels[y - 1]) in last_comms:
            s += 0.25
        number_score[y] = s + 0.15 * (freq.get(y, 0) / max_f)

    return {
        "number_score": number_score,
        "last_draw": last,
        "target_sum": float(draws.sum(axis=1).mean()),
        "modularity": lou["modularity"],
        "n_communities": len(lou["communities"]),
    }


def _combo_fit(combo: list[int], rule: dict) -> float:
    score = sum(rule["number_score"][x] for x in combo)
    score -= 0.015 * abs(sum(combo) - rule["target_sum"])
    return score


def predict_next_from_rule(draws: np.ndarray, rule: dict | None = None) -> list[int]:
    if rule is None:
        rule = learn_next_rule(draws)
    ranked = sorted(rule["number_score"], key=lambda n: (-rule["number_score"][n], n))
    best = None
    best_fit = -1e18
    for start in range(0, min(20, FRONT_N - FRONT_SELECT + 1)):
        base = sorted(ranked[start : start + FRONT_SELECT])
        for repl in ranked[:28]:
            cand = sorted(set(base[1:] + [repl]))
            if len(cand) != FRONT_SELECT:
                continue
            fit = _combo_fit(cand, rule)
            if fit > best_fit:
                best_fit = fit
                best = cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_grupa10(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | 7/39 | grupa10")
    print()

    co = cooccurrence_graph(draws)
    corr = correlation_graph(draws)
    mi = mi_graph(draws)

    print("=== co-occurrence top edges ===")
    print(top_edges(co, abs_w=False))
    print()

    print("=== |correlation| top edges ===")
    print(top_edges(corr, abs_w=True))
    print()

    print("=== MI top edges ===")
    print(top_edges(mi, abs_w=False))
    print()

    print("=== PageRank (co-occ) top15 ===")
    print(pagerank(co)[:15])
    print()

    print("=== eigenvector centrality top15 ===")
    print(eigenvector_centrality(co)[:15])
    print()

    print("=== betweenness approx top15 ===")
    print(betweenness_approx(co)[:15])
    print()

    print("=== Louvain communities ===")
    lou = louvain_communities(co)
    print({"modularity": lou["modularity"], "communities": lou["communities"]})
    print()

    print("=== triangle motifs top ===")
    print(triangle_motifs(co))
    print()

    print("=== NOTEARS/LiNGAM-lite lag1 top ===")
    print(notears_proxy(draws))
    print()

    print("=== pravilo → next (grupa 10) ===")
    rule = learn_next_rule(draws)
    combo = predict_next_from_rule(draws, rule)
    print(
        "rule:",
        {
            "last_draw": rule["last_draw"],
            "target_sum": round(rule["target_sum"], 2),
            "modularity": round(rule["modularity"], 4),
            "n_communities": rule["n_communities"],
        },
    )
    print("next:", combo)


if __name__ == "__main__":
    run_grupa10()


"""
10. Grafovi zavisnosti
co-occurrence graph, correlation graph, MI graph, Bayesian network, Markov network,
PC algorithm, FCI, GES, NOTEARS, LiNGAM, ICA-LiNGAM, Louvain/Leiden communities,
modularity, PageRank, betweenness/closeness/eigenvector centrality,
motif/subgraph mining, graphlet
"""



"""
CSV: loto7_4648_k55.csv
Kola: 4648 | seed=39 | 7/39 | grupa10

=== co-occurrence top edges ===
[(8, 28, 172.0), (23, 35, 171.0), (8, 23, 166.0), (8, 34, 163.0), (8, 37, 162.0), (23, 31, 162.0), (4, 29, 161.0), (6, 23, 161.0), (26, 34, 161.0), (26, 38, 161.0), (7, 26, 158.0), (8, 33, 158.0), (11, 33, 158.0), (19, 34, 158.0), (22, 23, 158.0)]

=== |correlation| top edges ===
[(12, 22, -0.06051897871926165), (21, 33, -0.058493677002582206), (8, 39, -0.05785027147713106), (18, 27, -0.05730895511690402), (3, 39, -0.056840949682804176), (4, 30, -0.05628505472506303), (9, 39, -0.05583804783930336), (2, 26, -0.05569647257770474), (32, 34, -0.05536784762625945), (24, 37, -0.05531370380468911), (10, 11, -0.05484719134550472), (14, 18, -0.05482289198663594), (1, 36, -0.05402052235721309), (1, 27, -0.05342993716899333), (6, 24, -0.05331070285036583)]

=== MI top edges ===
[(12, 22, 0.002818828996207964), (21, 33, 0.0026236435452093674), (8, 39, 0.002549942892603024), (18, 27, 0.0025278101434903708), (3, 39, 0.002472771775508441), (4, 30, 0.0024371449489862546), (9, 39, 0.0023811463852808287), (2, 26, 0.002368728607009575), (32, 34, 0.0023337994412381893), (24, 37, 0.0023336579207556725), (14, 18, 0.002302260648427484), (10, 11, 0.002292632962064242), (1, 36, 0.002242284054845471), (1, 27, 0.002191131834661913), (6, 24, 0.0021691805021419753)]

=== PageRank (co-occ) top15 ===
[(8, 0.027758021143184028), (23, 0.027573959987205836), (34, 0.026735905736479017), (26, 0.026603369628111183), (37, 0.02639478965833249), (11, 0.02636725361283199), (32, 0.02631779689796565), (33, 0.026209232032289254), (29, 0.026133038922894165), (22, 0.02613002432937056), (39, 0.026056006001874105), (7, 0.02602634691189113), (10, 0.025999986133430558), (9, 0.02594977340124878), (35, 0.02592186181561462)]

=== eigenvector centrality top15 ===
[(8, 0.17510394045055036), (23, 0.17382505053366945), (34, 0.16781461939016692), (26, 0.16695115024224616), (37, 0.16539574816497893), (11, 0.16527384244434348), (32, 0.16475124400734267), (33, 0.1641865218176708), (22, 0.1636501068878632), (29, 0.16350245740753194), (39, 0.16286386434963063), (7, 0.1628452212169767), (10, 0.16266142164453348), (9, 0.1621769731988333), (38, 0.16211249299366617)]

=== betweenness approx top15 ===
[(8, 200.21883671883666), (39, 161.3612637362637), (23, 155.88652458652462), (37, 104.54502997002997), (33, 96.768315018315), (29, 76.69967532467531), (34, 59.7686813186813), (32, 55.00093795093795), (10, 52.28956043956045), (11, 42.7219807969808), (2, 35.29339826839827), (6, 33.63757631257632), (31, 32.38423520923521), (26, 28.66111111111111), (25, 24.550022200022195)]

=== Louvain communities ===
{'modularity': -0.007449882401703276, 'communities': {0: [3, 5, 8, 11, 15, 16, 20, 28, 33], 1: [6, 10, 12, 17, 21, 23, 25, 27, 30, 32, 35, 36, 39], 2: [1, 2, 4, 7, 9, 13, 14, 18, 19, 22, 24, 26, 29, 31, 34, 37, 38]}}

=== triangle motifs top ===
[(8, 11, 33, 156.0), (7, 23, 26, 155.0), (8, 10, 34, 155.0), (8, 11, 26, 154.0), (8, 23, 26, 154.0), (8, 26, 33, 154.0), (8, 26, 34, 154.0), (11, 26, 33, 154.0), (8, 11, 16, 153.0), (8, 22, 23, 153.0)]

=== NOTEARS/LiNGAM-lite lag1 top ===
[(38, 11, 0.04956010795816682), (39, 36, 0.04615224211335024), (36, 1, 0.04365352029420791), (8, 2, -0.04279908230658676), (15, 7, 0.042603730800462154), (32, 12, -0.042305884207715305), (27, 5, -0.04157222797802411), (13, 10, 0.04156303134986629), (21, 32, -0.04126852729398193), (1, 22, -0.039116286636356905), (27, 39, 0.03857757942869686), (37, 15, -0.038344945170567274), (38, 22, -0.03777149778374312), (33, 29, -0.03742173802288898), (6, 23, 0.037347259272009374)]

=== pravilo → next (grupa 10) ===
rule: {'last_draw': [3, 7, 12, 13, 18, 24, 29], 'target_sum': 140.43, 'modularity': -0.0074, 'n_communities': 3}
next: [8, 11, 22, 23, 30, 33, 39]
"""
