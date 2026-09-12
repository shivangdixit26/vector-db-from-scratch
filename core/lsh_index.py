import numpy as np
from collections import defaultdict
from .distances import cosine_distance

class LSHIndex:
    """
    Locality-Sensitive Hashing (LSH) index for approximate nearest neighbors.
    
    This index uses random hyperplane projection to build hash signatures for vectors.
    Vectors that are close in cosine distance have a high probability of landing in
    the same bucket in multiple hash tables.
    """
    def __init__(self, dim: int, num_tables: int = 10, num_hashes: int = 8, seed: int = 42):
        self.dim = dim
        self.num_tables = num_tables
        self.num_hashes = num_hashes
        
        rng = np.random.RandomState(seed)
        self._planes = rng.randn(num_tables, num_hashes, dim).astype(np.float32)
        
        self._tables = [defaultdict(set) for _ in range(num_tables)]
        self._vectors = {}
        self._hash_sigs = {}
        self._next_id = 0
        self._last_candidates_count = 0

    def _hash_vector(self, vector: np.ndarray, table_idx: int) -> int:
        """Hash a single vector for a specific table."""
        projections = self._planes[table_idx] @ vector
        bits = (projections >= 0).astype(np.int64)
        powers = np.int64(1) << np.arange(self.num_hashes, dtype=np.int64)
        return int(bits @ powers)

    def _hash_all_tables(self, vector: np.ndarray) -> list[int]:
        """Hash a single vector for all tables (vectorized)."""
        # _planes: (L, K, d), vector: (d,)
        projections = np.einsum('tkd,d->tk', self._planes, vector)
        bits = (projections >= 0).astype(np.int64)
        powers = np.int64(1) << np.arange(self.num_hashes, dtype=np.int64)
        hashes = (bits @ powers).tolist()
        return hashes

    def insert(self, vector: np.ndarray, id: int = None) -> int:
        """Insert a single vector, generating an ID if none is provided."""
        if id is None:
            id = self._next_id
            self._next_id += 1
            
        vector = np.array(vector, dtype=np.float32)
        self._vectors[id] = vector
        
        hashes = self._hash_all_tables(vector)
        self._hash_sigs[id] = hashes
        
        for t, h in enumerate(hashes):
            self._tables[t][h].add(id)
            
        return id

    def bulk_insert(self, vectors: np.ndarray) -> list[int]:
        """Insert many vectors in bulk (fully vectorized hashing)."""
        vectors = np.array(vectors, dtype=np.float32)
        n = vectors.shape[0]
        
        ids = list(range(self._next_id, self._next_id + n))
        self._next_id += n
        
        # Compute hashes for all vectors across all tables
        # vectors: (n, d), planes: (L, K, d)
        projections = np.einsum('tkd,nd->tkn', self._planes, vectors)
        bits = (projections >= 0).astype(np.int64)
        powers = np.int64(1) << np.arange(self.num_hashes, dtype=np.int64)
        hash_keys = np.einsum('tkn,k->tn', bits, powers) # (L, n)
        
        # Store vectors and add to tables
        for i, idx in enumerate(ids):
            self._vectors[idx] = vectors[i]
            
            # Extract hash keys for this vector
            hashes = hash_keys[:, i].tolist()
            self._hash_sigs[idx] = hashes
            
            for t, h in enumerate(hashes):
                self._tables[t][h].add(idx)
                
        return ids

    def delete(self, id: int):
        """Remove a vector by its ID."""
        if id not in self._vectors:
            raise KeyError(f"ID {id} not found in index.")
            
        hashes = self._hash_sigs[id]
        
        # Remove from tables
        for t, h in enumerate(hashes):
            self._tables[t][h].remove(id)
            
        del self._hash_sigs[id]
        del self._vectors[id]

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Search top-k approximate nearest neighbors using LSH."""
        results, _ = self.search_with_stats(query, k)
        return results

    def search_with_stats(self, query: np.ndarray, k: int = 10) -> tuple[list[tuple[int, float]], dict]:
        """Search returning stats as well."""
        if not self._vectors:
            self._last_candidates_count = 0
            return [], {'candidates': 0, 'tables_hit': 0}
            
        query = np.array(query, dtype=np.float32)
        hashes = self._hash_all_tables(query)
        
        candidates = set()
        tables_hit = 0
        for t, h in enumerate(hashes):
            bucket = self._tables[t].get(h, set())
            if bucket:
                tables_hit += 1
                candidates.update(bucket)
                
        self._last_candidates_count = len(candidates)
        
        if not candidates:
            return [], {'candidates': 0, 'tables_hit': tables_hit}
            
        candidate_ids = list(candidates)
        candidate_vectors = np.array([self._vectors[cid] for cid in candidate_ids])
        
        distances = cosine_distance(query, candidate_vectors)
        
        actual_k = min(k, len(candidate_ids))
        if actual_k < len(candidate_ids):
            indices = np.argpartition(distances, actual_k - 1)[:actual_k]
        else:
            indices = np.arange(len(candidate_ids))
            
        sorted_indices = indices[np.argsort(distances[indices])]
        
        results = [(candidate_ids[idx], float(distances[idx])) for idx in sorted_indices]
        
        return results, {'candidates': len(candidates), 'tables_hit': tables_hit}

    @property
    def num_vectors(self) -> int:
        """Count of stored vectors."""
        return len(self._vectors)
