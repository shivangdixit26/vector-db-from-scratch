import numpy as np
from core.distances import cosine_distance

def compute_ground_truth(data_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Compute exact nearest neighbors as ground truth using brute force search.
    
    Args:
        data_vectors: np.ndarray of shape (n, d)
        query_vectors: np.ndarray of shape (m, d)
        k: int, number of nearest neighbors to compute
        
    Returns:
        np.ndarray of shape (m, k) containing integer IDs of nearest neighbors
    """
    m = query_vectors.shape[0]
    n = data_vectors.shape[0]
    
    # Ensure k is not greater than the number of available vectors
    k = min(k, n)
    results = np.zeros((m, k), dtype=np.int32)
    
    for i in range(m):
        if i % 50 == 0:
            print(f"Computing ground truth for query {i}/{m}")
            
        dists = cosine_distance(query_vectors[i], data_vectors)
        
        # Get top-k smallest distances
        top_k = np.argpartition(dists, k)[:k]
        
        # Sort the top-k to get exact ordering
        top_k_sorted = top_k[np.argsort(dists[top_k])]
        results[i, :] = top_k_sorted
        
    return results

def compute_recall(predicted: list[list[int]], ground_truth: np.ndarray, k: int = 10) -> float:
    """
    Compute Recall@K.
    
    Args:
        predicted: list of m lists, each containing up to k predicted neighbor IDs
        ground_truth: np.ndarray of shape (m, k) containing true neighbor IDs
        k: int, max neighbors to consider
        
    Returns:
        float between 0.0 and 1.0 representing the average recall
    """
    m = len(predicted)
    if m == 0:
        return 0.0
        
    total_recall = 0.0
    for i in range(m):
        true_set = set(ground_truth[i][:k])
        pred_set = set(predicted[i][:k])
        
        if not true_set:
            continue
            
        intersection = true_set.intersection(pred_set)
        total_recall += len(intersection) / len(true_set)
        
    return total_recall / m

def save_ground_truth(gt: np.ndarray, filepath: str) -> None:
    """Save ground truth array to a .npy file."""
    np.save(filepath, gt)

def load_ground_truth(filepath: str) -> np.ndarray:
    """Load ground truth array from a .npy file."""
    return np.load(filepath)
