"""
Mission 05 — Part 1: Systems Architect
HNSW Multi-Layer Vector Proximity Index (pure Python + NumPy)
USSE Noosphere • Subsystem 05: Dual-Path Associative Memory Core • SEC-05-M01

Reference: Malkov & Yashunin (2018), "Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs", IEEE TPAMI.

What this file delivers (brief, Part 1):
1. Cosine similarity core   : vectors are L2-normalised once on insert, so
                              Sim(u, v) = u.v / (|u||v|) becomes one matrix product.
2. Layer assignment         : l = floor(-ln(U(0,1)) * mL), mL = 1 / ln(M).
3. Greedy skip-graph search : search_layer(entry_point, query, ef, layer) — greedy
                              hops on the sparse top layers (ef = 1), then a
                              best-first beam of width ef on layer 0.
4. Latency benchmark        : HNSW vs brute-force flat search on N = 25,000
                              768-dimensional embeddings, with recall@10.

Plain-language picture:
  Think of motorways, main roads and side streets. The top layer has very few
  "cities" joined by long motorways; each layer down adds more towns and shorter
  roads. A query drives on the motorway to roughly the right region, takes the
  exit, and only at the bottom walks the side streets. It looks at a few
  thousand vectors instead of all 25,000.

Run:  python 01_HNSW_Vector_Index.py           (full benchmark, ~ a few minutes on a laptop)
      python 01_HNSW_Vector_Index.py --quick   (N = 5,000, for a fast check)
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# 1. Cosine similarity core
# ============================================================================
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale every row to unit length (a single vector also works)."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised Sim(u, v) = u.v / (||u|| ||v||).

    u: (d,) or (n, d), v: (d,) or (m, d)  ->  scalar, (n,), (m,) or (n, m).
    """
    return l2_normalize(u) @ l2_normalize(v).T


# ============================================================================
# 2-3. The HNSW index
# ============================================================================
class HNSWIndex:
    """Hierarchical Navigable Small World graph for cosine similarity.

    Internally we work with distance = 1 - cosine similarity (smaller = closer),
    because the paper's algorithms are written for distances.

    Parameters
    ----------
    dim             : vector dimension (768 here)
    M               : links per node on layers >= 1 (layer 0 allows 2*M, as in the paper)
    ef_construction : beam width while inserting (bigger = better graph, slower build)
    capacity        : number of vectors we pre-allocate room for
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 100,
                 capacity: int = 1024, seed: int = 0) -> None:
        if M < 2:
            raise ValueError("M must be >= 2")
        self.dim = dim
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = max(ef_construction, M)
        self.mL = 1.0 / math.log(M)                # brief: mL = 1 / ln(M)
        self.rng = np.random.default_rng(seed)

        self.data = np.zeros((capacity, dim), dtype=np.float32)   # unit vectors
        self.count = 0
        self.levels: List[int] = []                 # top layer of every node
        self.graph: List[Dict[int, List[int]]] = []  # graph[layer][node] -> neighbour ids
        self.entry_point: Optional[int] = None
        self.max_level = -1
        self.dist_evals = 0                         # how many distances we computed

    # ------------------------------------------------------------------ utils
    def __len__(self) -> int:
        return self.count

    def _grow(self) -> None:
        new = np.zeros((self.data.shape[0] * 2, self.dim), dtype=np.float32)
        new[: self.count] = self.data[: self.count]
        self.data = new

    def _dist(self, q: np.ndarray, ids: Sequence[int]) -> np.ndarray:
        """Distances 1 - cos(q, x_i) for many ids in ONE matrix product."""
        self.dist_evals += len(ids)
        return 1.0 - self.data[np.asarray(ids, dtype=np.int64)] @ q

    # ------------------------------------------------- 2. layer assignment
    def random_level(self) -> int:
        """l = floor(-ln(U(0,1)) * mL).

        With mL = 1/ln(M), P(l >= k) = M^(-k): each layer holds about 1/M of the
        nodes of the layer below, which gives the O(log N) number of layers.
        """
        u = 1.0 - self.rng.random()                 # in (0, 1], so ln(u) is finite
        return int(math.floor(-math.log(u) * self.mL))

    # ------------------------------------------------- 3. greedy layer search
    def search_layer(self, entry_point: Union[int, Iterable[int]], query: np.ndarray,
                     ef: int, layer: int) -> List[Tuple[float, int]]:
        """Best-first search on ONE layer (Algorithm 2 of the paper).

        Keeps a candidate queue (closest first) and a result set of the ef closest
        nodes seen so far (as a max-heap). Stops when the closest unexplored
        candidate is further away than the worst of the current ef results:
        nothing closer can be reached from there.

        With ef = 1 this is the plain greedy hop used on the upper layers.
        Returns a list of (distance, node_id), closest first.
        """
        eps = [entry_point] if isinstance(entry_point, (int, np.integer)) else list(entry_point)
        eps = [int(e) for e in eps]
        d0 = self._dist(query, eps)
        visited = set(eps)
        candidates = [(float(d), e) for d, e in zip(d0, eps)]          # min-heap
        heapq.heapify(candidates)
        results = [(-float(d), e) for d, e in zip(d0, eps)]            # max-heap via -d
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)

        links = self.graph[layer]
        while candidates:
            d_c, c = heapq.heappop(candidates)
            if d_c > -results[0][0]:                 # closest candidate worse than worst result
                break
            fresh = [n for n in links.get(c, ()) if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            dists = self._dist(query, fresh)         # vectorised: all neighbours at once
            worst = -results[0][0]
            for d, n in zip(dists.tolist(), fresh):
                if len(results) < ef or d < worst:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(results, (-d, n))
                    if len(results) > ef:
                        heapq.heappop(results)
                    worst = -results[0][0]
        return sorted((-nd, n) for nd, n in results)

    # ------------------------------------------------- neighbour selection
    def _select_neighbors(self, candidates: List[Tuple[float, int]], m: int) -> List[int]:
        """Diversity heuristic (Algorithm 4 of the paper).

        Take candidates from closest to furthest; keep one only if it is closer
        to the base node than to every neighbour already kept. This keeps a few
        "long" links into other clusters instead of m links into the same
        cluster, which is what makes clustered data (real embeddings) searchable.
        If the heuristic keeps fewer than m, fill up with the closest skipped ones.
        """
        if len(candidates) <= m:
            return [n for _, n in candidates]
        ids = [n for _, n in candidates]
        vecs = self.data[ids]
        pair = 1.0 - vecs @ vecs.T                   # distances among the candidates
        self.dist_evals += len(ids) * len(ids)
        kept: List[int] = []                         # positions in `ids`
        skipped: List[int] = []
        for i, (d_q, _) in enumerate(candidates):
            if len(kept) >= m:
                break
            if all(pair[i, j] > d_q for j in kept):
                kept.append(i)
            else:
                skipped.append(i)
        for i in skipped:                            # keepPrunedConnections
            if len(kept) >= m:
                break
            kept.append(i)
        return [ids[i] for i in kept]

    def _shrink(self, node: int, layer: int) -> None:
        """A neighbour list got too long: re-select the best m_max links."""
        m_max = self.M_max0 if layer == 0 else self.M
        nbrs = self.graph[layer][node]
        if len(nbrs) <= m_max:
            return
        d = self._dist(self.data[node], nbrs)
        cand = sorted(zip(d.tolist(), nbrs))
        self.graph[layer][node] = self._select_neighbors(cand, m_max)

    # ------------------------------------------------- insert (Algorithm 1)
    def add(self, vector: np.ndarray, level: Optional[int] = None) -> int:
        if self.count == self.data.shape[0]:
            self._grow()
        q = l2_normalize(vector).reshape(-1)
        nid = self.count
        self.data[nid] = q
        self.count += 1
        lvl = self.random_level() if level is None else level
        self.levels.append(lvl)
        while len(self.graph) <= lvl:
            self.graph.append({})
        for l in range(lvl + 1):
            self.graph[l][nid] = []

        if self.entry_point is None:                 # very first node
            self.entry_point, self.max_level = nid, lvl
            return nid

        # Phase 1: greedy descent (ef = 1) through layers above the new node's level
        ep = self.entry_point
        for l in range(self.max_level, lvl, -1):
            ep = self.search_layer(ep, q, ef=1, layer=l)[0][1]

        # Phase 2: on every layer the node lives on, find ef_construction
        # candidates, link to the best M of them, and link back.
        eps: List[int] = [ep]
        for l in range(min(lvl, self.max_level), -1, -1):
            found = self.search_layer(eps, q, ef=self.ef_construction, layer=l)
            neigh = self._select_neighbors(found, self.M)
            self.graph[l][nid] = list(neigh)
            for n in neigh:
                self.graph[l][n].append(nid)
                self._shrink(n, l)
            eps = [n for _, n in found]

        if lvl > self.max_level:
            self.entry_point, self.max_level = nid, lvl
        return nid

    def add_items(self, vectors: np.ndarray, verbose: bool = False) -> None:
        vectors = l2_normalize(vectors)
        need = self.count + len(vectors)
        while self.data.shape[0] < need:
            self._grow()
        t0 = time.perf_counter()
        for i, v in enumerate(vectors):
            self.add(v)
            if verbose and (i + 1) % 5000 == 0:
                print(f"    inserted {i + 1:>6,}/{len(vectors):,}  ({time.perf_counter() - t0:5.1f} s)")

    # ------------------------------------------------- query (Algorithm 5)
    def search(self, query: np.ndarray, k: int = 10, ef: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ids, cosine similarities) of the approximate k nearest neighbours."""
        if self.entry_point is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        q = l2_normalize(query).reshape(-1)
        ep = self.entry_point
        for l in range(self.max_level, 0, -1):       # motorway: ef = 1 greedy hops
            ep = self.search_layer(ep, q, ef=1, layer=l)[0][1]
        found = self.search_layer(ep, q, ef=max(ef, k), layer=0)[:k]
        ids = np.array([n for _, n in found], dtype=np.int64)
        sims = np.array([1.0 - d for d, _ in found], dtype=np.float32)
        return ids, sims

    # ------------------------------------------------- inspection
    def layer_sizes(self) -> List[int]:
        return [len(g) for g in self.graph]


# ============================================================================
# Brute-force baseline (flat index)
# ============================================================================
class FlatIndex:
    """Exact search: compare the query with EVERY stored vector (O(N d))."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.data = l2_normalize(vectors)

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        sims = self.data @ l2_normalize(query).reshape(-1)
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return top, sims[top]


# ============================================================================
# Synthetic "embedding-like" data
# ============================================================================
def make_embeddings(n: int, dim: int = 768, n_topics: int = 200, intrinsic: int = 32,
                    seed: int = 0) -> np.ndarray:
    """768-d vectors that behave like sentence embeddings, not like pure noise.

    Real embeddings are clustered by topic and live close to a low-dimensional
    surface. We copy that: pick n_topics topic centres in a 32-dim latent space,
    add per-document variation, map to 768-d with a fixed random linear map,
    then add a little isotropic noise. (Pure Gaussian noise in 768-d is a known
    worst case for every ANN method and says nothing about RAG archives.)
    """
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((intrinsic, dim)).astype(np.float32) / math.sqrt(intrinsic)
    centres = rng.standard_normal((n_topics, intrinsic)).astype(np.float32) * 2.0
    topic = rng.integers(0, n_topics, size=n)
    latent = centres[topic] + rng.standard_normal((n, intrinsic)).astype(np.float32)
    x = latent @ proj + 0.15 * rng.standard_normal((n, dim)).astype(np.float32)
    return l2_normalize(x)


# ============================================================================
# Unit tests
# ============================================================================
def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def test_cosine_core() -> None:
    rng = np.random.default_rng(1)
    u, v = rng.standard_normal((5, 768)), rng.standard_normal((7, 768))
    loop = np.array([[a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) for b in v] for a in u])
    assert np.allclose(cosine_similarity(u, v), loop, atol=1e-5)
    assert abs(float(cosine_similarity(u[0], 3.7 * u[0])) - 1.0) < 1e-5       # scale-invariant
    assert abs(float(cosine_similarity(u[0], -u[0])) + 1.0) < 1e-5
    _ok("cosine core == explicit loop (5x7 matrix), scale invariant, sim(u,-u) = -1")


def test_layer_distribution(M: int = 16, n: int = 200_000) -> None:
    idx = HNSWIndex(dim=4, M=M, seed=7)
    lv = np.array([idx.random_level() for _ in range(n)])
    rows = []
    for k in (1, 2, 3):
        emp, theory = float((lv >= k).mean()), M ** (-k)
        rows.append(f"P(l>={k}) {emp:.5f} vs {theory:.5f}")
        assert abs(emp - theory) < 4 * math.sqrt(theory / n) + 1e-4, rows[-1]
    _ok(f"layer law with mL = 1/ln({M}): " + " | ".join(rows))


def test_graph_invariants_and_exact_small() -> None:
    X = make_embeddings(2000, dim=64, n_topics=20, intrinsic=8, seed=3)
    idx = HNSWIndex(dim=64, M=8, ef_construction=64, capacity=2000, seed=3)
    idx.add_items(X)
    for l, g in enumerate(idx.graph):
        cap = idx.M_max0 if l == 0 else idx.M
        assert all(len(nb) <= cap for nb in g.values()), f"degree cap broken on layer {l}"
        assert all(n in g for nbs in g.values() for n in nbs), "link to a node missing from its layer"
    # every node reachable from the entry point on layer 0
    seen, stack = {idx.entry_point}, [idx.entry_point]
    while stack:
        for n in idx.graph[0][stack.pop()]:
            if n not in seen:
                seen.add(n)
                stack.append(n)
    assert len(seen) == len(idx), f"layer 0 not connected: {len(seen)}/{len(idx)}"
    flat = FlatIndex(X)
    Q = make_embeddings(100, dim=64, n_topics=20, intrinsic=8, seed=4)
    hit = sum(len(set(idx.search(q, 10, ef=200)[0]) & set(flat.search(q, 10)[0])) for q in Q)
    rec = hit / (10 * len(Q))
    assert rec >= 0.98, f"recall@10 only {rec:.3f}"
    _ok(f"graph invariants (degree caps, valid links, layer 0 connected 2000/2000), "
        f"recall@10 = {rec:.3f} at ef = 200")


def test_search_layer_greedy() -> None:
    # a 1-D chain 0-1-2-...-9 on layer 0: greedy search from node 0 must walk to the target
    idx = HNSWIndex(dim=2, M=2, capacity=10)
    ang = np.linspace(0, math.pi / 2, 10)
    for a in ang:
        idx.add(np.array([math.cos(a), math.sin(a)]), level=0)
    idx.graph[0] = {i: [j for j in (i - 1, i + 1) if 0 <= j < 10] for i in range(10)}
    q = l2_normalize(np.array([math.cos(ang[8]), math.sin(ang[8])]))
    best = idx.search_layer(0, q, ef=1, layer=0)[0][1]
    assert best == 8, best
    _ok("search_layer(entry_point=0, ef=1) hops along a chain to the true nearest node (8)")


# ============================================================================
# 4. Latency benchmark
# ============================================================================
def recall_at_k(pred: Sequence[np.ndarray], truth: Sequence[np.ndarray], k: int) -> float:
    return sum(len(set(p[:k]) & set(t[:k])) for p, t in zip(pred, truth)) / (k * len(truth))


def time_queries(fn, queries: np.ndarray, repeat: int = 1) -> Tuple[float, list]:
    out = [fn(q) for q in queries[:5]]                # warm-up
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = [fn(q) for q in queries]
    return (time.perf_counter() - t0) / (repeat * len(queries)) * 1e3, out   # ms / query


def run_benchmark(N: int = 25_000, dim: int = 768, n_queries: int = 200, M: int = 16,
                  ef_construction: int = 100, k: int = 10, seed: int = 0,
                  make_plot: bool = True) -> Dict[str, object]:
    print(f"\n=== Benchmark: N = {N:,} vectors, d = {dim}, {n_queries} queries, k = {k} ===")
    X = make_embeddings(N, dim, seed=seed)
    Q = make_embeddings(n_queries, dim, seed=seed + 99)   # same topics, unseen documents

    flat = FlatIndex(X)
    t_flat, flat_out = time_queries(lambda q: flat.search(q, k), Q, repeat=3)
    truth = [ids for ids, _ in flat_out]

    idx = HNSWIndex(dim, M=M, ef_construction=ef_construction, capacity=N, seed=seed)
    print(f"  building HNSW (M = {M}, ef_construction = {ef_construction}) ...")
    t0 = time.perf_counter()
    idx.add_items(X, verbose=True)
    build_s = time.perf_counter() - t0
    sizes = idx.layer_sizes()
    print(f"  built in {build_s:.1f} s | layers: {len(sizes)} | nodes per layer: "
          + ", ".join(f"L{l}={s:,}" for l, s in enumerate(sizes)))

    rows = []
    for ef in (10, 20, 50, 100, 150, 200):
        idx.dist_evals = 0
        t, out = time_queries(lambda q: idx.search(q, k, ef=ef), Q)
        evals = idx.dist_evals / (len(Q) + 5)
        rows.append(dict(ef=ef, ms=t, recall=recall_at_k([o[0] for o in out], truth, k),
                         evals=evals, frac=evals / N))

    print("\n  Method                      ms/query   recall@10   distances/query   share of N")
    print("  " + "-" * 84)
    print(f"  {'Flat brute force (NumPy)':26s} {t_flat:9.3f}       1.000   {N:15,.0f}       100.0%")
    for r in rows:
        print(f"  {'HNSW ef = ' + str(r['ef']):26s} {r['ms']:9.3f}   {r['recall']:9.3f}   "
              f"{r['evals']:15,.0f}   {100 * r['frac']:9.1f}%")
    print("  " + "-" * 84)
    return dict(N=N, flat_ms=t_flat, rows=rows, build_s=build_s, sizes=sizes, index=idx, X=X)


def scaling_study(Ns: Sequence[int], dim: int = 768, target: float = 0.95, n_queries: int = 100,
                  seed: int = 0, prebuilt: Optional[Dict[int, Tuple[HNSWIndex, np.ndarray]]] = None
                  ) -> List[Dict[str, float]]:
    """How cost grows with N at EQUAL accuracy.

    For each N we raise ef until recall@10 >= target, then count the distances
    HNSW needed. Flat search always needs N. (Comparing at a fixed ef would be
    unfair: the same ef gives lower recall on a bigger graph.)
    """
    print(f"\n=== Scaling study: work per query at recall@10 >= {target:.2f} ===")
    print(f"  {'N':>7}  {'ef used':>7}  {'recall':>6}  {'HNSW dists':>10}  {'flat dists':>10}  "
          f"{'HNSW ms':>7}  {'flat ms':>7}")
    out = []
    for n in Ns:
        if prebuilt and n in prebuilt:
            idx, X = prebuilt[n]
        else:
            X = make_embeddings(n, dim, seed=seed)
            idx = HNSWIndex(dim, M=16, ef_construction=100, capacity=n, seed=seed)
            idx.add_items(X)
        Q = make_embeddings(n_queries, dim, seed=seed + 99)
        flat = FlatIndex(X)
        t_f, fo = time_queries(lambda q: flat.search(q, 10), Q, repeat=3)
        truth = [o[0] for o in fo]
        for ef in (10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600):
            idx.dist_evals = 0
            t_h, ho = time_queries(lambda q: idx.search(q, 10, ef=ef), Q)
            rec = recall_at_k([o[0] for o in ho], truth, 10)
            if rec >= target:
                break
        ev = idx.dist_evals / (n_queries + 5)
        out.append(dict(N=n, ef=ef, recall=rec, evals=ev, hnsw_ms=t_h, flat_ms=t_f))
        print(f"  {n:>7,}  {ef:>7}  {rec:6.3f}  {ev:10,.0f}  {n:10,}  {t_h:7.3f}  {t_f:7.3f}")
    return out


def plot_results(bench: Dict[str, object], scale: List[Dict[str, float]], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed: plot skipped)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    rows = bench["rows"]
    ax[0].plot([r["recall"] for r in rows], [r["ms"] for r in rows], "o-", label="HNSW (ef = 10 … 200)")
    for r in rows:
        ax[0].annotate(f"ef={r['ef']}", (r["recall"], r["ms"]), textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax[0].axhline(bench["flat_ms"], color="grey", ls="--", label="flat brute force (recall 1.0)")
    ax[0].set_xlabel("recall@10")
    ax[0].set_ylabel("ms per query")
    ax[0].set_title(f"Speed vs accuracy, N = {bench['N']:,}, d = 768")
    ax[0].legend()
    Ns = [s["N"] for s in scale]
    ax[1].loglog(Ns, Ns, "--", color="grey", label="flat: N distances (linear)")
    ax[1].loglog(Ns, [s["evals"] for s in scale], "o-", label="HNSW: distances per query (recall@10 >= 0.95)")
    ax[1].set_xlabel("N (vectors in index)")
    ax[1].set_ylabel("distance computations per query")
    ax[1].set_title("Work per query as the archive grows")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\n[SUCCESS] Plot saved to: {path}")


# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="N = 5,000 instead of 25,000")
    args = ap.parse_args()

    print("=" * 92)
    print("USSE NOOSPHERE • SUBSYSTEM 05: DUAL-PATH ASSOCIATIVE MEMORY CORE")
    print("Role Deliverable: Part 1 — Systems Architect (HNSW Vector Proximity Index)")
    print("=" * 92)
    print("\n=== Unit tests ===")
    test_cosine_core()
    test_layer_distribution()
    test_search_layer_greedy()
    test_graph_invariants_and_exact_small()

    N = 5_000 if args.quick else 25_000
    bench = run_benchmark(N=N)
    Ns = [1_000, 2_500, 5_000] if args.quick else [1_000, 3_000, 10_000, 25_000]
    scale = scaling_study(Ns, prebuilt={N: (bench["index"], bench["X"])})

    a, b = scale[0], scale[-1]
    best = max((r for r in bench["rows"] if r["recall"] >= 0.95),
               key=lambda r: -r["ms"], default=bench["rows"][-1])
    print("\nReading:")
    print(f"* At equal accuracy (recall@10 >= 0.95), N grew {b['N'] / a['N']:.0f}x but HNSW's work per query grew "
          f"only {b['evals'] / a['evals']:.1f}x")
    print(f"  ({a['evals']:,.0f} -> {b['evals']:,.0f} distances). Flat search grew {b['N'] / a['N']:.0f}x. "
          "That is the sub-linear, ~log N behaviour.")
    print(f"* At N = {N:,}: HNSW ef = {best['ef']} answers in {best['ms']:.2f} ms with recall@10 = "
          f"{best['recall']:.3f}, after looking at {100 * best['frac']:.1f}% of")
    print(f"  the archive; flat search takes {bench['flat_ms']:.2f} ms ({bench['flat_ms'] / best['ms']:.1f}x slower). "
          "Small ef (10-50) is sub-millisecond but misses neighbours.")
    print("* Honest caveat: flat search is ONE NumPy/BLAS matrix product, HNSW hops in a Python loop. The")
    print("  gap widens with N (flat is linear, HNSW is not); C++ libraries (hnswlib, FAISS) run the same")
    print("  algorithm far faster than this file (sub-millisecond at much higher recall).")
    print("* ef is the speed/accuracy dial: raise it until recall is good enough, then stop.")
    print("* Data note: synthetic topic-clustered 768-d vectors (embedding-like). Real sentence embeddings")
    print("  are usually easier for HNSW (lower intrinsic dimension), so recall here is a cautious estimate.")
    plot_results(bench, scale, os.path.join(HERE, "hnsw_benchmark.png"))
    print("[SUCCESS] Subsystem 05 HNSW Vector Proximity Index validated.")
    print("=" * 92)


if __name__ == "__main__":
    main()
