import time
import json
import sys
import numpy as np
from core.lsh_index import LSHIndex
from eval.ground_truth import compute_recall

class BenchmarkRunner:
    """
    Runs benchmarks at multiple LSH settings and tracks performance metrics.
    """
    def __init__(self, data_vectors: np.ndarray, query_vectors: np.ndarray, ground_truth: np.ndarray, k: int = 10):
        self.data_vectors = data_vectors
        self.query_vectors = query_vectors
        self.ground_truth = ground_truth
        self.k = k
        self.n_data = data_vectors.shape[0]
        
    def run_single(self, num_tables: int, num_hashes: int = 8, seed: int = 42) -> dict:
        """
        Run a benchmark at a single LSH configuration.
        """
        d = self.data_vectors.shape[1]
        index = LSHIndex(dim=d, num_tables=num_tables, num_hashes=num_hashes, seed=seed)
        
        # Bulk insert all data vectors
        t_build = time.time()
        index.bulk_insert(self.data_vectors)
        build_time = time.time() - t_build
        
        m = self.query_vectors.shape[0]
        predicted = []
        total_candidates = 0
        
        start_time = time.time()
        for i in range(m):
            query = self.query_vectors[i]
            results = index.search(query, k=self.k)
            
            # Extract IDs from results (list of (id, distance) tuples)
            pred_ids = [r[0] for r in results]
            predicted.append(pred_ids)
            total_candidates += index._last_candidates_count
                
        end_time = time.time()
        
        search_time = end_time - start_time
        qps = m / search_time if search_time > 0 else 0.0
        
        avg_candidates = total_candidates / m if m > 0 else 0.0
        candidate_ratio = avg_candidates / self.n_data if self.n_data > 0 else 0.0
        
        recall = compute_recall(predicted, self.ground_truth, k=self.k)
        
        return {
            'num_tables': num_tables,
            'num_hashes': num_hashes,
            'recall_at_k': float(recall),
            'qps': float(qps),
            'avg_candidates': float(avg_candidates),
            'candidate_ratio': float(candidate_ratio),
            'total_search_time': float(search_time),
            'build_time': float(build_time)
        }
        
    def run_sweep(self, table_counts: list[int] = None, num_hashes: int = 8) -> list[dict]:
        """
        Run benchmarks across multiple num_tables values and print a formatted table.
        """
        if table_counts is None:
            table_counts = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30]
            
        results = []
        
        print("╔════════════╦════════════╦═════════╦══════════════╦════════════╗")
        print("║ num_tables ║ Recall@10  ║   QPS   ║ Candidates%  ║ Build (s)  ║")
        print("╠════════════╬════════════╬═════════╬══════════════╬════════════╣")
        
        for num_tables in table_counts:
            sys.stdout.write(f"║ {num_tables:^10} ║  running  ║   ...   ║     ...      ║    ...     ║\r")
            sys.stdout.flush()
            
            res = self.run_single(num_tables=num_tables, num_hashes=num_hashes)
            results.append(res)
            
            recall_str = f"{res['recall_at_k']:.4f}"
            qps_str = f"{int(res['qps'])}"
            cand_str = f"{res['candidate_ratio']*100:.1f}%"
            build_str = f"{res['build_time']:.1f}"
            
            print(f"║ {num_tables:^10} ║ {recall_str:^10} ║ {qps_str:^7} ║ {cand_str:^12} ║ {build_str:^10} ║")
            
        print("╚════════════╩════════════╩═════════╩══════════════╩════════════╝")
        
        return results

    def save_results(self, results: list[dict], filepath: str) -> None:
        """
        Save benchmark results to a JSON file.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
