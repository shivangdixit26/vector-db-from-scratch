import numpy as np

def generate_clustered_vectors(n: int = 50000, dim: int = 128, num_clusters: int = 50, cluster_std: float = 0.1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate clustered synthetic vectors that mimic real embedding geometry.
    
    Real embeddings form clusters (e.g., sports sentences near sports, cooking near cooking). 
    Uniform random vectors in high dimensions are all equidistant (curse of dimensionality), 
    so clustered data is more realistic AND harder for approximate search.
    
    Args:
        n: Number of vectors to generate.
        dim: Dimensionality of the vectors.
        num_clusters: Number of cluster centers.
        cluster_std: Standard deviation of noise added to cluster centers.
        seed: Random seed for reproducibility.
        
    Returns:
        tuple containing the generated vectors and their cluster labels.
    """
    rng = np.random.RandomState(seed)
    
    # Generate random centers
    centers = rng.randn(num_clusters, dim).astype(np.float32)
    # Normalize centers to unit sphere
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    
    # Assign each vector to a random cluster
    labels = rng.randint(0, num_clusters, n)
    
    # Generate vectors
    vectors = centers[labels] + rng.randn(n, dim).astype(np.float32) * cluster_std
    
    # Normalize all vectors to unit sphere
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    
    return vectors, labels

def generate_query_vectors(data_vectors: np.ndarray, n_queries: int = 500, noise_scale: float = 0.05, seed: int = 123) -> np.ndarray:
    """
    Generate query vectors that are NEAR existing data vectors.
    
    This ensures queries have actual nearby neighbors, making recall measurement meaningful.
    
    Args:
        data_vectors: The dataset vectors.
        n_queries: Number of query vectors to generate.
        noise_scale: Amount of Gaussian noise to add to the base vectors.
        seed: Random seed for reproducibility.
        
    Returns:
        Generated query vectors as an (n_queries, dim) float32 array.
    """
    rng = np.random.RandomState(seed)
    n_data, dim = data_vectors.shape
    
    # Pick n_queries random indices from data_vectors
    indices = rng.choice(n_data, n_queries, replace=False)
    
    # Add small Gaussian noise
    queries = data_vectors[indices] + rng.randn(n_queries, dim).astype(np.float32) * noise_scale
    
    # Normalize to unit sphere
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    
    return queries

def save_vectors(vectors: np.ndarray, filepath: str) -> None:
    """Save vectors to a .npy file."""
    np.save(filepath, vectors)

def load_vectors(filepath: str) -> np.ndarray:
    """Load vectors from a .npy file."""
    return np.load(filepath)
