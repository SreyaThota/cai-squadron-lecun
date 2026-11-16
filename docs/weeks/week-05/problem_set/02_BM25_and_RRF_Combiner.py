"""
02_BM25_and_RRF_Combiner.py
Mission 5 - Protocol Officer deliverable
BM25 lexical retrieval + Reciprocal Rank Fusion (RRF, k = 60)

Pure Python / NumPy. The dense retriever here is a small self-contained
stand-in (concept lexicon + hashed character trigrams, cosine similarity,
brute-force). In the squad notebook, swap `DenseRetriever.search` for the
HNSW index from 01_HNSW_Vector_Index.py - the RRF code does not change.
"""

import math
import re
from collections import Counter

import numpy as np

# ----------------------------------------------------------------------------
# Tokenizer: keeps serial codes such as "QX-7731-B" as ONE token
# ----------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


# ----------------------------------------------------------------------------
# 1. BM25 Scoring Engine (Okapi BM25, k1 = 1.5, b = 0.75)
# ----------------------------------------------------------------------------
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in corpus]
        self.N = len(self.docs)
        self.doc_len = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = self.doc_len.mean()
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        # Robertson-Sparck Jones IDF with +1 so it never goes negative
        self.idf = {
            t: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()
        }

    def score(self, query, idx):
        s = 0.0
        tf = self.tf[idx]
        norm = self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
        for q in tokenize(query):
            f = tf.get(q, 0)
            if f == 0:
                continue
            s += self.idf[q] * f * (self.k1 + 1) / (f + norm)
        return s

    def search(self, query, top_k=10):
        scores = np.array([self.score(query, i) for i in range(self.N)])
        order = np.argsort(-scores, kind="stable")
        return [(int(i), float(scores[i])) for i in order[:top_k] if scores[i] > 0]


# ----------------------------------------------------------------------------
# Dense retriever stand-in (cosine similarity on unit vectors)
# ----------------------------------------------------------------------------
CONCEPTS = {
    "propulsion": "engine thruster propulsion drive motor ion xenon thrust move moves "
                  "moving travel accelerate accelerates speed push propel",
    "navigation": "navigation navigate coordinates astrogation star flight jump hyperspace "
                  "route course position locate where",
    "shielding": "shield shielding hull radiation plating armor protect protection cosmic "
                 "particles impacts micrometeorite defend",
    "food": "meals galley food eat water recyclers drink",
    "ai": "aura artificial intelligence ai computer brain",
    "logging": "telemetry logged logs log archive record records data",
}
CONCEPT_NAMES = list(CONCEPTS)
WORD2CONCEPT = {w: i for i, (_, ws) in enumerate(CONCEPTS.items()) for w in ws.split()}
HASH_DIM = 256


def embed(text):
    v = np.zeros(len(CONCEPT_NAMES) + HASH_DIM)
    toks = tokenize(text)
    for t in toks:
        if t in WORD2CONCEPT:
            v[WORD2CONCEPT[t]] += 3.0  # semantic signal dominates
        padded = f"#{t}#"
        for i in range(len(padded) - 2):
            h = hash_trigram(padded[i:i + 3])
            v[len(CONCEPT_NAMES) + h] += 0.3  # weak surface signal
    n = np.linalg.norm(v)
    return v / n if n else v


def hash_trigram(s):
    # deterministic (Python's hash() is salted per process)
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % 1_000_003
    return h % HASH_DIM


class DenseRetriever:
    def __init__(self, corpus):
        self.matrix = np.vstack([embed(d) for d in corpus])

    def search(self, query, top_k=10):
        sims = self.matrix @ embed(query)  # cosine, vectors are unit-norm
        order = np.argsort(-sims, kind="stable")
        return [(int(i), float(sims[i])) for i in order[:top_k]]


# ----------------------------------------------------------------------------
# 2. Reciprocal Rank Fusion:  RRF(d) = sum_m 1 / (k + rank_m(d)),  k = 60
# ----------------------------------------------------------------------------
def reciprocal_rank_fusion(rankings, k=60):
    """rankings: list of ranked lists of doc ids (best first, rank starts at 1)."""
    fused = Counter()
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


class HybridRetriever:
    def __init__(self, corpus, k=60):
        self.sparse = BM25(corpus)
        self.dense = DenseRetriever(corpus)
        self.k = k

    def search(self, query, top_k=3, depth=10):
        sparse_rank = [i for i, _ in self.sparse.search(query, depth)]
        dense_rank = [i for i, _ in self.dense.search(query, depth)]
        fused = reciprocal_rank_fusion([dense_rank, sparse_rank], self.k)
        return fused[:top_k], dense_rank, sparse_rank


# ----------------------------------------------------------------------------
# 3. Evaluation benchmark
# ----------------------------------------------------------------------------
CORPUS = [
    "The ion thruster provides propulsion for deep-space travel using xenon gas.",
    "Part number QX-7731-B is the coolant pump valve for reactor loop three.",
    "The navigation computer computes flight coordinates using star tracker data.",
    "Serial code ZT-4410-A identifies the synchrotron radiation shield module.",
    "The crew galley stores freeze-dried meals and water recyclers.",
    "Hull plating withstands micrometeorite impacts and cosmic radiation.",
    "Telemetry from the engine is logged every second to the archive.",
    "Part number QX-7731-C is the coolant pump gasket for reactor loop three.",
    "AURA is the ship's artificial intelligence managing life support.",
    "Astrogation logs record jump coordinates and hyperspace transitions.",
]

# (query, kind, set of relevant doc ids, method expected to dominate)
QUERIES = [
    ("What makes the ship accelerate through space?", "conceptual", {0, 6}, "dense"),
    ("How is the vessel protected from cosmic particles?", "conceptual", {5, 3}, "dense"),
    ("QX-7731-B", "exact code", {1}, "sparse"),
    ("ZT-4410-A", "exact code", {3}, "sparse"),
]


def main():
    hybrid = HybridRetriever(CORPUS, k=60)
    ok = 0
    for query, kind, relevant, _ in QUERIES:
        fused, dense_rank, sparse_rank = hybrid.search(query, top_k=3)
        fused_ids = [d for d, _ in fused]
        hit = bool(relevant & set(fused_ids[:2]))
        ok += hit
        print(f"\nQuery [{kind}]: {query}")
        print(f"  dense  top3: {dense_rank[:3]}")
        print(f"  sparse top3: {sparse_rank[:3]}")
        print("  hybrid top3: " + ", ".join(f"doc{d} ({s:.4f})" for d, s in fused))
        print(f"  relevant {sorted(relevant)} in hybrid top-2: {'PASS' if hit else 'FAIL'}")
    print(f"\nHybrid RRF passed {ok}/{len(QUERIES)} queries "
          f"(conceptual -> dense path, exact serial codes -> BM25 path).")
    assert ok == len(QUERIES)


if __name__ == "__main__":
    main()
