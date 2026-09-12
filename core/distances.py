import numpy as np

def l2_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute L2 (Euclidean) distance between a query vector and a set of vectors.
    """
    return np.sqrt(np.sum((vectors - query)**2, axis=1))

def cosine_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine distance between a query vector and a set of vectors.
    Distance is 1 - cosine_similarity.
    Handles zero-norm vectors using epsilon.
    """
    epsilon = 1e-10
    query_norm = np.linalg.norm(query)
    vectors_norm = np.linalg.norm(vectors, axis=1)
    
    query_norm = np.maximum(query_norm, epsilon)
    vectors_norm = np.maximum(vectors_norm, epsilon)
    
    dot_products = np.dot(vectors, query)
    cos_sim = dot_products / (query_norm * vectors_norm)
    # Clip to avoid floating point errors
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return 1.0 - cos_sim

def batch_cosine_distance(queries: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine distance between multiple queries and multiple vectors.
    """
    epsilon = 1e-10
    queries_norm = np.linalg.norm(queries, axis=1, keepdims=True)
    vectors_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    
    queries_norm = np.maximum(queries_norm, epsilon)
    vectors_norm = np.maximum(vectors_norm, epsilon)
    
    # Dot product of all queries with all vectors
    dot_products = np.dot(queries, vectors.T)
    cos_sim = dot_products / np.dot(queries_norm, vectors_norm.T)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return 1.0 - cos_sim
