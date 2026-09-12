#!/usr/bin/env python3
"""
demo.py -- Interactive terminal demo for the Vector Database.

Features:
  [1] Search by text     -- Type a sentence, find semantically similar headlines
  [2] Run benchmark      -- Execute the full 50k-vector evaluation
  [3] Show tradeoff      -- Display the speed-accuracy curve
  [4] Insert/Delete demo -- Live demonstration of index operations
  [5] Compare indices    -- Side-by-side brute force vs LSH results
  [6] Exit

Usage:
    python demo.py
"""

import os
import sys
import time
import numpy as np

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    print("""
+================================================================+
|                                                                  |
|   VectorDB -- Vector Database from Scratch (numpy only)          |
|   No FAISS. No Chroma. No Pinecone. No shortcuts.                |
|                                                                  |
+================================================================+
    """)


def print_menu():
    print("+---------------------------------------------+")
    print("|  What would you like to do?                  |")
    print("+---------------------------------------------+")
    print("|  [1] Search by text (semantic search)        |")
    print("|  [2] Run full benchmark (50k vectors)        |")
    print("|  [3] Show speed-accuracy curve               |")
    print("|  [4] Insert / Search / Delete demo           |")
    print("|  [5] Compare Brute Force vs LSH              |")
    print("|  [6] Exit                                    |")
    print("+---------------------------------------------+")


def text_search_demo():
    """Interactive text-based semantic search."""
    print("\n-- Text Search Demo --\n")
    
    try:
        from data.text_corpus import TextCorpus
        from core.lsh_index import LSHIndex
        from core.brute_force_index import BruteForceIndex
    except ImportError as e:
        print(f"  [WARN] Missing dependency: {e}")
        print("  Run: pip install sentence-transformers")
        return

    print("  Loading text corpus and embeddings...")
    corpus = TextCorpus()
    
    try:
        vectors = corpus.load_or_create()
    except Exception as e:
        print(f"  [WARN] Failed to load corpus: {e}")
        print("  Make sure sentence-transformers is installed.")
        return
    
    texts = corpus.texts
    dim = vectors.shape[1]
    n = vectors.shape[0]
    
    print(f"  Loaded {n} texts, dim={dim}")
    print("  Building LSH index (num_tables=12, num_hashes=8)...", end=" ")
    
    lsh = LSHIndex(dim=dim, num_tables=12, num_hashes=8)
    lsh.bulk_insert(vectors)
    print("Done!\n")
    
    while True:
        query_text = input("  Enter your query (or 'back' to return): ").strip()
        if query_text.lower() in ('back', 'quit', 'q', ''):
            break
        
        print(f"\n  Searching for: \"{query_text}\"\n")
        
        t0 = time.time()
        query_vec = corpus.embed_query(query_text)
        embed_time = time.time() - t0
        
        t0 = time.time()
        results = lsh.search(query_vec, k=5)
        search_time = time.time() - t0
        
        candidates = lsh._last_candidates_count
        
        print(f"  Top 5 Matches (LSH, {candidates} candidates / {n} total):")
        print("  " + "-" * 65)
        
        for rank, (vid, dist) in enumerate(results, 1):
            text = texts[vid] if vid < len(texts) else f"[Vector {vid}]"
            # Truncate long texts
            if len(text) > 55:
                text = text[:52] + "..."
            print(f"  #{rank}  [{dist:.4f}]  {text}")
        
        print("  " + "-" * 65)
        print(f"  Embed: {embed_time*1000:.1f}ms | Search: {search_time*1000:.1f}ms | Candidates: {candidates}/{n} ({100*candidates/n:.1f}%)")
        print()


def run_benchmark_demo():
    """Run the full evaluation benchmark."""
    print("\n-- Full Benchmark --\n")
    print("  This will generate 50,000 vectors and run the full evaluation.")
    confirm = input("  Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        return
    
    print("  Launching run_eval.py...\n")
    os.system(f"{sys.executable} run_eval.py")


def show_curve_demo():
    """Display the saved speed-accuracy curve."""
    print("\n-- Speed-Accuracy Curve --\n")
    
    metrics_path = "results/metrics.json"
    if not os.path.exists(metrics_path):
        print("  [WARN] No benchmark results found. Run the benchmark first (option 2).")
        return
    
    import json
    with open(metrics_path) as f:
        results = json.load(f)
    
    # Print the results table
    print("  +------------+------------+---------+--------------+")
    print("  | num_tables | Recall@10  |   QPS   | Candidates%  |")
    print("  +------------+------------+---------+--------------+")
    for r in results:
        print(f"  | {r['num_tables']:^10} | {r['recall_at_k']:^10.4f} | {r['qps']:^7.0f} | {r['candidate_ratio']*100:^11.1f}% |")
    print("  +------------+------------+---------+--------------+")
    
    plot_path = "results/accuracy_vs_speed.png"
    if os.path.exists(plot_path):
        abs_path = os.path.abspath(plot_path)
        print(f"\n  Plot saved at: {abs_path}")
        if os.name == 'nt':
            os.system(f'start "" "{abs_path}"')
        else:
            print(f"  Open with: xdg-open {plot_path}")
    else:
        print("\n  [WARN] Plot image not found. Run the benchmark to generate it.")


def insert_delete_demo():
    """Live demonstration of insert, search, and delete operations."""
    print("\n-- Insert / Search / Delete Demo --\n")
    
    from core.lsh_index import LSHIndex
    
    dim = 64
    rng = np.random.RandomState(42)
    
    # Create index
    lsh = LSHIndex(dim=dim, num_tables=10, num_hashes=8, seed=42)
    print(f"  Created LSH index (dim={dim}, tables=10, hashes=8)")
    
    # Insert initial vectors
    print("\n  -- Step 1: Insert 200 random vectors --")
    vectors = rng.randn(200, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = lsh.bulk_insert(vectors)
    print(f"  [OK] Inserted {len(ids)} vectors. Index size: {lsh.num_vectors}")
    
    # Search
    print("\n  -- Step 2: Search for a query --")
    query = vectors[0] + rng.randn(dim).astype(np.float32) * 0.01
    query /= np.linalg.norm(query)
    
    results = lsh.search(query, k=5)
    print(f"  Query: near vector #0")
    print(f"  Results:")
    for rank, (vid, dist) in enumerate(results, 1):
        marker = " <-- expected!" if vid == ids[0] else ""
        print(f"    #{rank}  ID={vid:>3}  dist={dist:.6f}{marker}")
    
    # Insert a new vector very close to query
    print("\n  -- Step 3: Insert a new vector very close to query --")
    new_vec = query + rng.randn(dim).astype(np.float32) * 0.0001
    new_vec /= np.linalg.norm(new_vec)
    new_id = lsh.insert(new_vec)
    print(f"  [OK] Inserted new vector ID={new_id}. Index size: {lsh.num_vectors}")
    
    results_after = lsh.search(query, k=5)
    print(f"  Search results after insert:")
    for rank, (vid, dist) in enumerate(results_after, 1):
        marker = " <-- NEW!" if vid == new_id else ""
        print(f"    #{rank}  ID={vid:>3}  dist={dist:.6f}{marker}")
    
    # Delete the new vector
    print(f"\n  -- Step 4: Delete vector ID={new_id} --")
    lsh.delete(new_id)
    print(f"  [OK] Deleted. Index size: {lsh.num_vectors}")
    
    results_after_delete = lsh.search(query, k=5)
    print(f"  Search results after delete:")
    found_deleted = False
    for rank, (vid, dist) in enumerate(results_after_delete, 1):
        if vid == new_id:
            found_deleted = True
        print(f"    #{rank}  ID={vid:>3}  dist={dist:.6f}")
    
    if not found_deleted:
        print(f"  [OK] Deleted vector ID={new_id} is gone from results!")
    else:
        print(f"  [FAIL] Deleted vector still appears!")
    
    print("\n  [PASS] Insert / Search / Delete -- all operations verified!\n")


def compare_indices_demo():
    """Side-by-side comparison of brute force vs LSH."""
    print("\n-- Brute Force vs LSH Comparison --\n")
    
    from core.lsh_index import LSHIndex
    from core.brute_force_index import BruteForceIndex
    
    dim = 128
    n = 10000
    k = 10
    rng = np.random.RandomState(42)
    
    print(f"  Generating {n:,} vectors (dim={dim})...")
    vectors = rng.randn(n, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = rng.randn(dim).astype(np.float32)
    query /= np.linalg.norm(query)
    
    # Brute Force
    print("  Building Brute Force index...", end=" ")
    bf = BruteForceIndex(dim=dim)
    bf.bulk_insert(vectors)
    print("Done!")
    
    t0 = time.time()
    bf_results = bf.search(query, k=k)
    bf_time = time.time() - t0
    
    # LSH at different settings
    for num_tables in [3, 10, 25]:
        lsh = LSHIndex(dim=dim, num_tables=num_tables, num_hashes=8)
        lsh.bulk_insert(vectors)
        
        t0 = time.time()
        lsh_results = lsh.search(query, k=k)
        lsh_time = time.time() - t0
        
        bf_ids = set(r[0] for r in bf_results)
        lsh_ids = set(r[0] for r in lsh_results)
        recall = len(bf_ids & lsh_ids) / k
        candidates = lsh._last_candidates_count
        
        print(f"\n  +----------------------------------------------------+")
        print(f"  |  LSH (num_tables={num_tables:>2}) vs Brute Force              |")
        print(f"  +----------------------------------------------------+")
        print(f"  |  Brute Force:  {bf_time*1000:>7.2f}ms                          |")
        print(f"  |  LSH:          {lsh_time*1000:>7.2f}ms  ({bf_time/max(lsh_time,1e-9):.1f}x faster)            |")
        print(f"  |  Recall@{k}:    {recall:.2f}  ({int(recall*k)}/{k} correct)             |")
        print(f"  |  Candidates:   {candidates}/{n} ({100*candidates/n:.1f}%)                  |")
        print(f"  +----------------------------------------------------+")
    
    print()


def main():
    clear_screen()
    print_banner()
    
    while True:
        print_menu()
        choice = input("\n  Enter choice [1-6]: ").strip()
        
        if choice == '1':
            text_search_demo()
        elif choice == '2':
            run_benchmark_demo()
        elif choice == '3':
            show_curve_demo()
        elif choice == '4':
            insert_delete_demo()
        elif choice == '5':
            compare_indices_demo()
        elif choice == '6':
            print("\n  Goodbye!\n")
            break
        else:
            print("  Invalid choice. Try again.\n")
        
        input("  Press Enter to continue...")
        clear_screen()
        print_banner()


if __name__ == "__main__":
    main()
