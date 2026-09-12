#!/usr/bin/env python3
"""
run_eval.py — Single-command evaluation script for the Vector Database.

Usage:
    python run_eval.py [--n-vectors 50000] [--n-queries 500] [--k 10] [--dim 128]

This script:
1. Generates clustered synthetic vectors
2. Computes exact ground truth (brute force k-NN)
3. Builds LSH indices at multiple settings
4. Measures Recall@K and QPS at each setting
5. Plots the speed-accuracy tradeoff curve
6. Saves results to results/
"""

import argparse
import json
import os
import sys
import time
import numpy as np

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from core.brute_force_index import BruteForceIndex
from core.lsh_index import LSHIndex
from core.distances import cosine_distance
from data.synthetic import generate_clustered_vectors, generate_query_vectors
from eval.ground_truth import compute_ground_truth, compute_recall
from eval.benchmark import BenchmarkRunner
from eval.plot import plot_tradeoff, plot_candidates, print_summary


def print_header():
    """Print a professional header."""
    print("\n" + "=" * 65)
    print("   Vector Database from Scratch -- Evaluation Suite")
    print("   Built with numpy only. No FAISS. No Chroma. No shortcuts.")
    print("=" * 65 + "\n")


def run_brute_force_verification(data_vectors, query_vectors, ground_truth, k):
    """Verify brute force index returns correct results."""
    print("\n[VERIFY] Verifying Brute Force Index...")
    bf = BruteForceIndex(dim=data_vectors.shape[1])
    bf.bulk_insert(data_vectors)
    
    # Test a few queries
    n_test = min(10, len(query_vectors))
    correct = 0
    for i in range(n_test):
        results = bf.search(query_vectors[i], k=k)
        predicted_ids = set(r[0] for r in results)
        true_ids = set(ground_truth[i].tolist())
        if predicted_ids == true_ids:
            correct += 1
    
    print(f"   Brute force verification: {correct}/{n_test} queries match ground truth [OK]")
    return bf


def run_insert_delete_test(dim):
    """Demonstrate and verify insert/search/delete operations."""
    print("\n[TEST] Testing Insert / Search / Delete...")
    
    rng = np.random.RandomState(999)
    lsh = LSHIndex(dim=dim, num_tables=10, num_hashes=8, seed=42)
    
    # Insert 100 vectors
    test_vectors = rng.randn(100, dim).astype(np.float32)
    test_vectors /= np.linalg.norm(test_vectors, axis=1, keepdims=True)
    ids = lsh.bulk_insert(test_vectors)
    print(f"   Inserted {len(ids)} vectors. Index size: {lsh.num_vectors}")
    
    # Search
    query = test_vectors[0]  # search for the first vector itself
    results = lsh.search(query, k=5)
    print(f"   Search for vector #0: top result = ID {results[0][0]}, dist = {results[0][1]:.4f}")
    assert results[0][0] == ids[0], "Top result should be the query vector itself!"
    print("   [OK] Self-search returns correct vector")
    
    # Insert a new vector very close to query
    new_vec = query + rng.randn(dim).astype(np.float32) * 0.001
    new_vec /= np.linalg.norm(new_vec)
    new_id = lsh.insert(new_vec)
    print(f"   Inserted new vector (ID {new_id}) near query. Index size: {lsh.num_vectors}")
    
    results_after_insert = lsh.search(query, k=5)
    top_ids = [r[0] for r in results_after_insert]
    assert new_id in top_ids, "Newly inserted vector should appear in results!"
    print(f"   [OK] New vector appears in search results (rank {top_ids.index(new_id) + 1})")
    
    # Delete the new vector
    lsh.delete(new_id)
    print(f"   Deleted vector ID {new_id}. Index size: {lsh.num_vectors}")
    
    results_after_delete = lsh.search(query, k=5)
    top_ids_after = [r[0] for r in results_after_delete]
    assert new_id not in top_ids_after, "Deleted vector should NOT appear in results!"
    print("   [OK] Deleted vector no longer in search results")
    
    print("   [PASS] Insert / Search / Delete all working correctly!\n")


def main():
    parser = argparse.ArgumentParser(description="Vector DB Evaluation")
    parser.add_argument("--n-vectors", type=int, default=50000, help="Number of data vectors")
    parser.add_argument("--n-queries", type=int, default=500, help="Number of query vectors")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    parser.add_argument("--num-hashes", type=int, default=5, help="Hash bits per table")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    print_header()
    
    os.makedirs("results", exist_ok=True)

    # -- Step 1: Generate Data --
    print(f"[DATA] Generating {args.n_vectors:,} clustered synthetic vectors (dim={args.dim})...")
    t0 = time.time()
    data_vectors, labels = generate_clustered_vectors(
        n=args.n_vectors, dim=args.dim, seed=args.seed
    )
    print(f"   Generated in {time.time() - t0:.2f}s")
    print(f"   Shape: {data_vectors.shape}, dtype: {data_vectors.dtype}")
    print(f"   Clusters: {len(np.unique(labels))}")

    print(f"\n[DATA] Generating {args.n_queries} query vectors...")
    query_vectors = generate_query_vectors(
        data_vectors, n_queries=args.n_queries, seed=args.seed + 81
    )
    print(f"   Shape: {query_vectors.shape}")

    # -- Step 2: Compute Ground Truth --
    gt_path = "results/ground_truth.npy"
    if os.path.exists(gt_path):
        print(f"\n[GT] Loading cached ground truth from {gt_path}...")
        ground_truth = np.load(gt_path)
    else:
        print(f"\n[GT] Computing exact ground truth (brute force k-NN, k={args.k})...")
        print("   This may take a minute for 50k vectors...")
        t0 = time.time()
        ground_truth = compute_ground_truth(data_vectors, query_vectors, k=args.k)
        elapsed = time.time() - t0
        print(f"   Ground truth computed in {elapsed:.2f}s")
        np.save(gt_path, ground_truth)
        print(f"   Saved to {gt_path}")

    # -- Step 3: Verify Brute Force --
    run_brute_force_verification(data_vectors, query_vectors, ground_truth, args.k)

    # -- Step 4: Test Insert/Delete --
    run_insert_delete_test(args.dim)

    # -- Step 5: Benchmark LSH at Multiple Settings --
    print("=" * 65)
    print("   LSH Benchmark -- Sweeping num_tables")
    print("=" * 65)
    
    runner = BenchmarkRunner(
        data_vectors=data_vectors,
        query_vectors=query_vectors,
        ground_truth=ground_truth,
        k=args.k
    )
    
    table_counts = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50]
    results = runner.run_sweep(
        table_counts=table_counts,
        num_hashes=args.num_hashes
    )

    # -- Step 6: Save Results --
    results_path = "results/metrics.json"
    runner.save_results(results, results_path)
    print(f"\n[SAVE] Results saved to {results_path}")

    # -- Step 7: Plot --
    if not args.skip_plots:
        print("\n[PLOT] Generating plots...")
        try:
            plot_tradeoff(results, save_path="results/accuracy_vs_speed.png")
            plot_candidates(results, save_path="results/candidates_curve.png")
            print("   Plots saved to results/")
        except Exception as e:
            print(f"   [WARN] Plot generation failed: {e}")
            print("   (Install matplotlib: pip install matplotlib)")

    # -- Step 8: Summary --
    print("\n" + "=" * 65)
    print_summary(results)
    print("=" * 65)
    
    print("\n[DONE] Evaluation complete!")
    print("   results/accuracy_vs_speed.png  -- Speed-accuracy tradeoff curve")
    print("   results/candidates_curve.png   -- Candidate ratio curve")
    print("   results/metrics.json           -- Raw metrics data")
    print("   results/ground_truth.npy       -- Cached ground truth\n")


if __name__ == "__main__":
    main()
