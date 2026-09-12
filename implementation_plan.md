# Vector Database from Scratch — Implementation Plan

Build a complete vector search engine using **only numpy**, with exact and approximate indices, measurable speed-accuracy tradeoff, and a working demo.

## Project Structure

```
c:\Users\HP\Desktop\itgeeks\
├── README.md                    # Project overview, setup, usage
├── requirements.txt             # numpy, matplotlib, sentence-transformers (data prep only)
├── SEED.txt                     # Reproducibility seed
│
├── core/                        # The vector database (numpy only)
│   ├── __init__.py
│   ├── distances.py             # L2 & cosine distance functions
│   ├── brute_force_index.py     # Exact index — ground truth
│   └── lsh_index.py             # LSH approximate index
│
├── data/                        # Data generation & loading
│   ├── __init__.py
│   ├── synthetic.py             # Clustered synthetic vectors (50k+)
│   └── text_corpus.py           # Real text corpus + embedding (5k texts)
│
├── eval/                        # Evaluation & benchmarking
│   ├── __init__.py
│   ├── ground_truth.py          # Compute exact k-NN for query set
│   ├── benchmark.py             # Run benchmarks at multiple settings
│   └── plot.py                  # Plot speed-accuracy curve
│
├── run_eval.py                  # Main evaluation script (single command)
├── demo.py                      # Interactive terminal demo
└── results/                     # Output: plots, metrics JSON
    ├── accuracy_vs_speed.png
    └── metrics.json
```

---

## Algorithm Deep-Dive (For Your Video)

### 1. Distance Functions — [`distances.py`]

Two metrics, both implemented with numpy broadcasting:

**Euclidean (L2) Distance:**
$$d(a, b) = \sqrt{\sum_{i=1}^{d} (a_i - b_i)^2}$$

```python
def l2_distance(query, vectors):
    # query: (d,), vectors: (n, d) → returns (n,)
    return np.sqrt(np.sum((vectors - query) ** 2, axis=1))
```

**Cosine Distance:**
$$d_{cos}(a, b) = 1 - \frac{a \cdot b}{\|a\| \cdot \|b\|}$$

```python
def cosine_distance(query, vectors):
    dot = vectors @ query
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    return 1 - dot / norms
```

> [!NOTE]
> We use **cosine distance** as the primary metric because it matches the LSH hash function we're building (random hyperplane LSH is a cosine similarity LSH). The brute force index supports both.

---

### 2. Exact Brute Force Index — [`brute_force_index.py`]

**What it does**: Computes distance from the query to **every single vector** in the database. Returns the true K nearest neighbors. This is the ground truth — everything else is scored against it.

**How it works**:
```
search(query, k=10):
    1. distances = cosine_distance(query, all_vectors)     # O(n * d)
    2. top_k_indices = argpartition(distances, k)[:k]      # O(n)
    3. sort those k indices by distance                    # O(k log k)
    4. return sorted indices + distances
```

**Complexity**: O(n × d) per query — linear in database size. Slow but provably correct.

**Operations**:
| Operation | How | Complexity |
|-----------|-----|------------|
| **Insert** | Append vector to numpy array, assign incremental ID | O(1) amortized |
| **Search** | Compute all distances, return top-k | O(n × d) |
| **Delete** | Mark ID as deleted in a set; skip during search | O(1) delete, slight overhead on search |

> [!IMPORTANT]
> This index is never used in production. It exists solely to **prove** that the approximate index is correct. Every recall number we report is measured against this ground truth.

---

### 3. LSH (Locality-Sensitive Hashing) Index — [`lsh_index.py`]

This is the core of the project. Here's how it works from first principles:

#### 3a. The Core Idea

**Problem**: Brute force checks all 50,000 vectors. Can we check fewer and still find the right neighbors?

**Insight**: If two vectors point in similar directions, a **random hyperplane** through the origin is unlikely to separate them.

```
         ↗ v1
        /
-------/-------- hyperplane
      /
     ↗ v2

v1 and v2 are on the same side → hash to same value
```

**Probability of same hash**: For two vectors with angle θ between them:
$$P[\text{same hash}] = 1 - \frac{\theta}{\pi}$$

Small angle (similar vectors) → high probability of same hash.

#### 3b. Building a Hash Function

A single hash function:
```python
def hash_one(vector, random_plane):
    return 1 if np.dot(vector, random_plane) >= 0 else 0
```

One bit isn't useful enough. So we use `num_hashes` (call it `K`) random planes to create a **K-bit signature**:

```python
def compute_hash(vector, planes):
    # planes: (K, d) — K random unit vectors of dimension d
    projections = planes @ vector        # (K,) dot products
    bits = (projections >= 0).astype(int)  # (K,) binary
    # Convert to integer hash key
    return tuple(bits)  # or int from binary
```

This creates **2^K buckets**. Similar vectors → same bucket with high probability.

#### 3c. The Problem with One Table

One hash table with K hash functions gives good precision but **misses neighbors** (false negatives). Two truly similar vectors might get different hashes just by bad luck with one of the K planes.

#### 3d. The Solution: Multiple Tables

Use **L independent hash tables**, each with its own set of K random planes.

```
Table 1:  planes_1 → hash → bucket → {id_3, id_7, id_42, ...}
Table 2:  planes_2 → hash → bucket → {id_7, id_15, id_42, ...}
Table 3:  planes_3 → hash → bucket → {id_7, id_42, id_88, ...}
...
Table L:  planes_L → hash → bucket → {id_42, id_91, ...}
```

**Query**: Hash into each table, collect **union** of all candidate IDs, compute exact distance only on candidates.

#### 3e. The Knob — `num_tables` (L)

This is the speed-accuracy tradeoff parameter:

| `num_tables` | Candidates examined | Recall | Speed |
|--------------|-------------------|--------|-------|
| L = 1 | Few (~1% of data) | Low (~40%) | Very fast |
| L = 5 | Moderate (~5%) | Medium (~70%) | Fast |
| L = 15 | Many (~15%) | High (~90%) | Moderate |
| L = 30 | Most (~30%) | Very high (~97%) | Slower |
| L = ∞ | All vectors | 100% (exact) | = Brute force |

> [!IMPORTANT]
> **This is the curve we plot.** The x-axis is Recall@10, the y-axis is Queries Per Second. Each point on the curve is a different value of `num_tables`. The curve bows — you get most of the accuracy gain from the first few tables, then diminishing returns.

#### 3f. Search Algorithm (Pseudocode)

```
LSH_SEARCH(query, k):
    candidates = empty set

    for each table t in 1..L:
        hash_key = compute_hash(query, planes[t])
        candidates = candidates ∪ table[t][hash_key]

    # Remove deleted IDs
    candidates = candidates - deleted_set

    # Exact reranking on candidates only
    candidate_vectors = get_vectors(candidates)
    distances = cosine_distance(query, candidate_vectors)
    top_k = argpartition(distances, k)[:k]
    return sorted candidates by distance
```

**Complexity**: O(L × K × d) for hashing + O(|candidates| × d) for reranking.  
When |candidates| << n, this is much faster than brute force.

#### 3g. Insert

```
INSERT(id, vector):
    store vector in data array at position id
    for each table t in 1..L:
        hash_key = compute_hash(vector, planes[t])
        table[t][hash_key].add(id)
        hash_signatures[id][t] = hash_key    # remember for deletion
```

#### 3h. Delete

```
DELETE(id):
    for each table t in 1..L:
        hash_key = hash_signatures[id][t]
        table[t][hash_key].remove(id)
    del hash_signatures[id]
    add id to deleted_set
    # Note: we don't remove from data array (fragmentation)
    # Vectors are skipped during search via deleted_set check
```

> [!NOTE]
> **Why deletion is manageable with LSH**: Each vector's hash signatures are stored at insert time. To delete, we just reverse the insert — remove from each table's bucket using the stored signature. This is O(L), not expensive. The tricky part is the data array: we either leave a "hole" (fragmentation) or compact the array (expensive). We go with holes + a deleted set, which is honest and efficient.

---

### 4. Data Generation

#### 4a. Clustered Synthetic Vectors — [`synthetic.py`]

```python
def generate_clustered(n=50000, d=128, num_clusters=50, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(num_clusters, d)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)  # unit sphere

    labels = rng.randint(0, num_clusters, n)
    noise = rng.randn(n, d) * 0.1
    vectors = centers[labels] + noise
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # normalize
    return vectors
```

**Why clustered, not random?** Uniform random vectors in high dimensions are all roughly equidistant (curse of dimensionality). Real embeddings form **clusters** — sentences about sports are near each other, sentences about cooking are near each other. Clustered data is harder for approximate search and more realistic.

#### 4b. Real Text Corpus — [`text_corpus.py`]

- Use a public dataset of ~5,000 short texts (quotes, news headlines, or movie descriptions)
- Embed with `sentence-transformers` (MiniLM, 384-dim) — this is **data prep**, not part of the index
- Save embeddings as `.npy` for fast loading
- The demo uses these: type a sentence → find the most similar sentences

---

### 5. Evaluation — [`eval/`]

#### 5a. Ground Truth — [`ground_truth.py`]

```python
def compute_ground_truth(data_vectors, query_vectors, k=10):
    """For each query, compute exact k-NN using brute force."""
    results = []
    for q in query_vectors:
        dists = cosine_distance(q, data_vectors)
        top_k = np.argpartition(dists, k)[:k]
        top_k = top_k[np.argsort(dists[top_k])]
        results.append(top_k)
    return np.array(results)  # (500, k)
```

#### 5b. Metrics

**Recall@K**: The fraction of true top-K neighbors found by the approximate index.
$$\text{Recall@K} = \frac{|\text{approx top-K} \cap \text{true top-K}|}{K}$$

Averaged over all 500 queries.

**QPS (Queries Per Second)**: 
$$\text{QPS} = \frac{500}{\text{total search time in seconds}}$$

**Candidate Ratio**: 
$$\text{Candidate Ratio} = \frac{\text{avg candidates examined}}{N}$$

#### 5c. Benchmark Script — [`benchmark.py`]

```
for num_tables in [1, 2, 3, 5, 8, 12, 16, 20, 25, 30]:
    index = LSHIndex(num_tables=num_tables, num_hashes=10, dim=128)
    index.bulk_insert(data_vectors)

    start = time.time()
    results = [index.search(q, k=10) for q in query_vectors]
    elapsed = time.time() - start

    recall = compute_recall(results, ground_truth, k=10)
    qps = 500 / elapsed

    record(num_tables, recall, qps)
```

#### 5d. Plot — [`plot.py`]

Generate `results/accuracy_vs_speed.png`:

```
            Speed vs Accuracy Tradeoff
    QPS ▲
   5000 │  •  L=1
        │
   2000 │     • L=3
        │
    800 │        • L=8
        │
    300 │            • L=16
        │
    100 │               • L=25
        │                  • L=30
        └──────────────────────────► Recall@10
          0.3  0.5  0.7  0.85  0.95
```

Two plots side by side:
1. **Recall@10 vs QPS** (the primary tradeoff curve)
2. **Recall@10 vs num_tables** (shows the knob directly)

---

### 6. Interactive Demo — [`demo.py`]

Terminal-based interactive demo:

```
═══════════════════════════════════════════
   Vector Search Engine — From Scratch
═══════════════════════════════════════════

[1] Search by text (real corpus, 5k texts)
[2] Run benchmark (50k synthetic vectors)
[3] Show speed-accuracy curve
[4] Test insert / search / delete
[5] Exit

> 1
Enter your text: "machine learning for beginners"

Top 5 Matches (LSH, num_tables=10):
──────────────────────────────────────────
 #1 [0.12] "Introduction to machine learning concepts"
 #2 [0.18] "Deep learning basics for newcomers"
 #3 [0.21] "Getting started with neural networks"
 #4 [0.25] "AI fundamentals course overview"
 #5 [0.29] "Statistical learning theory simplified"

 Candidates examined: 847 / 5000 (16.9%)
 Search time: 2.3ms
──────────────────────────────────────────
```

For the **insert/search/delete demo** (option 4):
1. Show initial search results for a query
2. Insert a new vector that should be the top result
3. Search again — new vector appears as #1
4. Delete it
5. Search again — it's gone, results revert

---

## Verification Plan

### Automated Tests (via `run_eval.py`)

```bash
python run_eval.py
```

This single command will:
1. Generate 50,000 clustered vectors + 500 queries
2. Compute ground truth (exact k-NN for all 500 queries)
3. Build LSH indices at 10+ different `num_tables` settings
4. Measure Recall@10 and QPS at each setting
5. Print a results table
6. Save `results/accuracy_vs_speed.png`
7. Save `results/metrics.json`

**Expected output**:
```
╔════════════╦════════════╦═════════╦══════════════╗
║ num_tables ║ Recall@10  ║   QPS   ║ Candidates%  ║
╠════════════╬════════════╬═════════╬══════════════╣
║     1      ║   0.38     ║  4200   ║    1.2%      ║
║     3      ║   0.61     ║  2800   ║    3.5%      ║
║     5      ║   0.73     ║  1900   ║    5.8%      ║
║     8      ║   0.83     ║  1200   ║    9.1%      ║
║    12      ║   0.90     ║   750   ║   13.5%      ║
║    16      ║   0.94     ║   480   ║   17.8%      ║
║    20      ║   0.96     ║   340   ║   21.4%      ║
║    25      ║   0.98     ║   230   ║   26.0%      ║
║    30      ║   0.99     ║   170   ║   30.5%      ║
╚════════════╩════════════╩═════════╩══════════════╝

Plot saved to: results/accuracy_vs_speed.png
Metrics saved to: results/metrics.json
```

### Manual Verification
- Run the interactive demo to visually confirm search results make sense
- Test insert/delete operations in demo mode 4
- Verify the plot shows the expected concave tradeoff curve

---

## Video Script Outline (for your explainer)

> [!TIP]
> Structure your video as: **Problem → Exact Solution → Why It's Slow → The LSH Trick → The Tradeoff → Demo**

1. **The Problem** (30s): "We have 50,000 vectors. Given a query, find the 10 closest. Brute force checks all 50k — too slow."

2. **Brute Force as Ground Truth** (30s): "We compute distance to every vector. It's O(n). Slow, but provably correct. This is our baseline."

3. **The LSH Insight** (1 min): "A random hyperplane splits space in two. Similar vectors land on the same side. Stack K planes → a hash signature. Similar vectors → same bucket."

4. **Multiple Tables** (30s): "One table misses neighbors. L tables give L chances. Union the candidates, rerank exactly."

5. **The Knob** (30s): "More tables = more candidates = higher recall but slower. This is the tradeoff. Here's the curve." [Show the plot]

6. **Deletion** (20s): "We stored each vector's hash signatures at insert time. Delete = reverse the insert across all L tables. O(L)."

7. **Live Demo** (1–2 min): Show text search, show insert/delete, show the benchmark output.

---

## Timeline (6 hours from now → deadline ~20:30 IST)

| Time | Block | Task |
|------|-------|------|
| 14:35 – 15:00 | **Core** | `distances.py` + `brute_force_index.py` |
| 15:00 – 16:30 | **Core** | `lsh_index.py` (insert, search, delete) |
| 16:30 – 17:15 | **Data** | `synthetic.py` + `text_corpus.py` |
| 17:15 – 18:00 | **Eval** | `ground_truth.py` + `benchmark.py` + `plot.py` |
| 18:00 – 18:30 | **Eval** | `run_eval.py` — wire everything together |
| 18:30 – 19:00 | **Demo** | `demo.py` interactive terminal |
| 19:00 – 19:30 | **Polish** | README, testing, edge cases, debugging |
| 19:30 – 20:00 | **Video** | Record demo + explanation |
| 20:00 – 20:30 | **Ship** | Push to GitHub, upload video |

---

## Open Questions

> [!IMPORTANT]
> **Embedding model for real text**: For the 5,000-text demo, we need to embed text into vectors. `sentence-transformers` (MiniLM) is the simplest option. This is purely for data preparation — the index itself is numpy-only. Is this acceptable, or would you prefer to pre-compute embeddings separately?

> [!IMPORTANT]
> **Text corpus source**: What 5,000 texts should we use? Options:
> - **Quotes dataset** (Kaggle) — short, diverse, easy to understand in a demo
> - **News headlines** — topical, good for showing semantic similarity
> - **Movie plot summaries** — longer, richer semantic content
> - Or we can generate/curate our own

> [!IMPORTANT]
> **Second approximate index**: The problem says "an approximate index you wrote yourself, of whatever design you can defend." LSH is the plan. Should we also implement IVF (Inverted File Index with k-means) as a bonus to show a second approach, or focus all effort on making LSH excellent?

> [!IMPORTANT]
> **GitHub repo**: Do you already have a GitHub repo created for this, or should I set one up?
