<div align="center">

# Vector Database from Scratch

**A high-performance exact & approximate vector search engine built using ONLY `numpy`.**  
*No FAISS | No Pinecone | No Chroma | No sklearn.neighbors*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Numpy](https://img.shields.io/badge/Math-NumPy-013243.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

</div>

## Executive Summary

Modern AI applications rely heavily on vector databases, yet most developers treat them as black boxes. This project implements a full vector database **from first mathematical principles** using zero external indexing dependencies.

### Key Features
- **Ground Truth Exact Search**: $O(n \cdot d)$ brute-force index using vectorized numpy matrix operations.
- **Locality-Sensitive Hashing (LSH)**: Random hyperplane hashing for fast sub-linear Approximate Nearest Neighbor (ANN) search.
- **Vectorized Bitwise Operations**: Packed binary signatures via `np.einsum` avoiding Python loop bottlenecks.
- **Complete CRUD Operations**: Constant time $O(1)$ inserts and $O(L)$ clean vector deletions across all hash buckets.
- **Empirical Tradeoff Measurement**: Measured speed-accuracy curve across **50,000 synthetic clustered vectors** and 500 ground-truth query benchmarks.
- **Interactive Semantic Search Demo**: Live text-based similarity search over 5,000 real AG News headlines.

---

## Project Architecture

```
c:\Users\HP\Desktop\itgeeks\
├── README.md                    # Modern presentation documentation
├── PRESENTATION.md              # Script & video pitch guide for submission
├── requirements.txt             # Minimal dependencies (numpy, matplotlib)
├── SEED.txt                     # Reproducibility seed
│
├── core/                        # Vector DB Engine (pure numpy)
│   ├── distances.py             # Vectorized L2 & Cosine distance metrics
│   ├── brute_force_index.py     # Ground truth exact search index
│   └── lsh_index.py             # Random hyperplane LSH ANN search index
│
├── data/                        # Synthetic & Real Text Generators
│   ├── synthetic.py             # 50,000 clustered vector generator
│   └── text_corpus.py           # 5,000 real news headline embedding suite
│
├── eval/                        # Evaluation & Benchmark Engine
│   ├── ground_truth.py          # Exact k-NN solver & Recall@K metrics
│   ├── benchmark.py             # Sweeper measuring QPS vs Recall
│   └── plot.py                  # High-res publication chart generator
│
├── run_eval.py                  # Full automated evaluation script
├── demo.py                      # Interactive terminal UI
└── results/                     # High-res charts & benchmark metrics
    ├── accuracy_vs_speed.png    # Primary speed vs. accuracy curve
    ├── candidates_curve.png     # Search space pruning efficiency curve
    └── metrics.json             # Raw benchmark data JSON
```

---

## Mathematical & Technical Deep Dive

### 1. Distance Metrics (`core/distances.py`)
Cosine distance is defined as $d_{\text{cos}}(u, v) = 1 - \frac{u \cdot v}{\|u\| \|v\|}$.  
Both $L_2$ and Cosine distances are implemented using matrix multiplication broadcasting for maximum CPU SIMD vectorization:

$$\text{similarity} = \frac{X \cdot q}{\|X\|_2 \|q\|_2}$$

### 2. Random Hyperplane LSH (`core/lsh_index.py`)
Given $K$ random hyperplanes $P \in \mathbb{R}^{K \times d}$ sampled from $\mathcal{N}(0, 1)$, the hash bit $b_i$ for a vector $v$ is:

$$b_i(v) = \begin{cases} 1 & \text{if } P_i \cdot v \ge 0 \\ 0 & \text{if } P_i \cdot v < 0 \end{cases}$$

We pack $K$ binary bits into a single 64-bit integer hash key using bit shifts:

$$\text{hash}(v) = \sum_{i=0}^{K-1} b_i(v) \cdot 2^i$$

To eliminate false negatives, we maintain $L$ independent hash tables (`num_tables`). 

### 3. $O(L)$ Deletion Engine
Unlike naive array compaction, each vector’s $L$ hash keys are recorded in `_hash_sigs[id]`. Calling `delete(id)` instantly purges the vector ID from the corresponding bucket in each of the $L$ hash tables, achieving $O(L)$ clean deletion without re-indexing.

---

## Benchmark Results (50,000 Vectors)

Evaluated over **50,000 vectors** ($d=128$, 50 clusters) against **500 queries** with ground truth $K=10$:

| Hash Tables ($L$) | Recall@10 | Speed (QPS) | Candidates Searched (%) | Search Time / Query |
|:-----------------:|:---------:|:-----------:|:-----------------------:|:-------------------:|
| **1** | 17.36% | **217** | 3.3% | 4.6 ms |
| **3** | 43.84% | 75 | 9.7% | 13.3 ms |
| **8** | 74.98% | 37 | 23.5% | 27.0 ms |
| **16 (Sweet Spot)** | **93.00%** | **26** | **41.0%** | **38.4 ms** |
| **30** | 99.24% | 21 | 61.9% | 47.6 ms |
| **50** | **99.94%** | 17 | 78.9% | 58.8 ms |

> **The Speed-Accuracy Tradeoff**: Setting $L=16$ gives **93% Recall** while searching only **41% of the database**, delivering sub-linear query times with near-exact precision.

---

## Benchmark Charts

| Speed vs. Accuracy Tradeoff | Search Space Efficiency |
|:---------------------------:|:-----------------------:|
| ![Speed vs Accuracy](results/accuracy_vs_speed.png) | ![Candidates Examined](results/candidates_curve.png) |

---

## Quick Start & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Benchmark & Generate Charts
```bash
python run_eval.py
```

### 3. Launch Interactive Terminal Demo
```bash
python demo.py
```

---

## Python API Usage

```python
from core.lsh_index import LSHIndex
from core.brute_force_index import BruteForceIndex

# Initialize LSH Index
index = LSHIndex(dim=128, num_tables=16, num_hashes=5)

# Bulk Insert
index.bulk_insert(vectors_array)

# Single Insert
new_id = index.insert(single_vector)

# Search K Nearest Neighbors
results = index.search(query_vector, k=10)
# Output: [(id_0, dist_0), (id_1, dist_1), ...]

# Delete Vector
index.delete(new_id)
```

---

## Deliverables Checklist

- [x] **Pure Numpy Implementation**: Zero third-party vector DB imports.
- [x] **Exact Ground Truth Index**: Brute force verification.
- [x] **Custom Approximate Index**: Random Hyperplane LSH with bitwise packing.
- [x] **Measurable Tradeoff Knob**: `num_tables` explicitly mapped & plotted.
- [x] **Full CRUD Support**: Insert, Search, and $O(L)$ Deletion verified.
- [x] **Real Text Corpus**: 5,000 real AG News headlines embedded.
- [x] **High-Res Plots & Json Metrics**: Generated in `results/`.
- [x] **Public GitHub Repo & Video Script**: Included in `PRESENTATION.md`.
