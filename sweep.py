"""Optimized benchmark sweep with num_hashes=5."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import time
import json
from data.synthetic import generate_clustered_vectors, generate_query_vectors
from eval.ground_truth import compute_recall
from core.lsh_index import LSHIndex

data, labels = generate_clustered_vectors(n=50000, dim=128, seed=42)
queries = generate_query_vectors(data, n_queries=500, seed=123)
gt = np.load('results/ground_truth.npy')

results = []
table_counts = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50]
num_hashes = 5

print(f'Sweeping num_tables with num_hashes={num_hashes}')
print(f'{"tables":>6} {"recall":>8} {"qps":>8} {"cand%":>8}')
print('-' * 35)

for nt in table_counts:
    idx = LSHIndex(dim=128, num_tables=nt, num_hashes=num_hashes, seed=42)
    idx.bulk_insert(data)
    predicted = []
    total_cand = 0
    t0 = time.time()
    for q in queries:
        res = idx.search(q, k=10)
        predicted.append([r[0] for r in res])
        total_cand += idx._last_candidates_count
    elapsed = time.time() - t0
    recall = compute_recall(predicted, gt, k=10)
    qps = 500 / elapsed
    avg_cand = total_cand / 500
    cand_pct = avg_cand / 50000 * 100
    results.append({
        'num_tables': nt, 'num_hashes': num_hashes,
        'recall_at_k': float(recall), 'qps': float(qps),
        'avg_candidates': float(avg_cand),
        'candidate_ratio': float(avg_cand / 50000),
        'total_search_time': float(elapsed), 'build_time': 0
    })
    print(f'{nt:>6} {recall:>8.4f} {qps:>8.0f} {cand_pct:>7.1f}%')

with open('results/metrics.json', 'w') as f:
    json.dump(results, f, indent=4)
print('\nSaved to results/metrics.json')
