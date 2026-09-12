import numpy as np
from .distances import cosine_distance

class BruteForceIndex:
    """
    Exact brute-force vector index. Ground truth for everything else.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors_list = []
        self._vectors_array = None
        self._ids = []
        self._id_to_pos = {}
        self._deleted = set()
        self._next_id = 0
        
    def _materialize(self):
        """Materializes accumulated vectors into the numpy array."""
        if not self._vectors_list:
            return
        
        new_array = np.vstack(self._vectors_list).astype(np.float32)
        if self._vectors_array is None:
            self._vectors_array = new_array
        else:
            self._vectors_array = np.vstack([self._vectors_array, new_array])
            
        self._vectors_list = []

    def insert(self, vector: np.ndarray, id: int = None) -> int:
        """Insert single vector, auto-assign ID if None."""
        if id is None:
            id = self._next_id
            self._next_id += 1
            
        vector = np.array(vector, dtype=np.float32).reshape(1, self.dim)
        
        self._vectors_list.append(vector)
        pos = len(self._ids)
        self._ids.append(id)
        self._id_to_pos[id] = pos
        
        if id in self._deleted:
            self._deleted.remove(id)
            
        return id

    def bulk_insert(self, vectors: np.ndarray) -> list[int]:
        """Insert many vectors at once."""
        vectors = np.array(vectors, dtype=np.float32)
        n = vectors.shape[0]
        
        ids = list(range(self._next_id, self._next_id + n))
        self._next_id += n
        
        self._vectors_list.append(vectors)
        
        start_pos = len(self._ids)
        self._ids.extend(ids)
        for i, idx in enumerate(ids):
            self._id_to_pos[idx] = start_pos + i
            
        return ids

    def delete(self, id: int):
        """Mark ID as deleted (lazy deletion)."""
        if id not in self._id_to_pos:
            raise KeyError(f"ID {id} not found in index.")
        self._deleted.add(id)

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Search top-k nearest neighbors."""
        self._materialize()
        
        if self._vectors_array is None:
            return []
            
        query = np.array(query, dtype=np.float32)
        distances = cosine_distance(query, self._vectors_array)
        
        # Mark deleted items as infinity
        for deleted_id in self._deleted:
            if deleted_id in self._id_to_pos:
                distances[self._id_to_pos[deleted_id]] = np.inf
                
        # Number of actual elements to consider
        valid_count = len(self._ids) - len(self._deleted)
        if valid_count == 0 or k <= 0:
            return []
            
        k = min(k, len(self._ids))
            
        if k < len(self._ids):
            indices = np.argpartition(distances, k - 1)[:k]
        else:
            indices = np.arange(len(self._ids))
            
        # Sort top k
        sorted_indices = indices[np.argsort(distances[indices])]
        
        results = []
        for idx in sorted_indices:
            dist = float(distances[idx])
            if dist == np.inf:
                continue
            results.append((self._ids[idx], dist))
            
            if len(results) == k:
                break
                
        return results

    def __len__(self) -> int:
        """Return count of non-deleted vectors."""
        return len(self._ids) - len(self._deleted)

    @property
    def size(self) -> int:
        return len(self)
