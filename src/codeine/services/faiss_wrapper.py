"""
FAISS Wrapper for RAG functionality.

Provides a high-level interface for FAISS operations with support for:
- Vector addition with custom IDs (using IndexIDMap)
- Vector removal by ID
- k-NN search
- Index persistence (save/load)
- Both flat and IVF index types
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import numpy as np

# Suppress SWIG deprecation warnings from FAISS (Python 3.12+ compatibility issue)
# These warnings occur during module load, so must be set globally before import
warnings.filterwarnings("ignore", message="builtin type Swig", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="builtin type swig", category=DeprecationWarning)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a FAISS search operation."""
    vector_id: int
    distance: float
    score: float  # Normalized similarity score (0-1)


@dataclass
class ClusterInfo:
    """Information about a cluster of similar vectors."""
    cluster_id: int
    centroid: np.ndarray
    member_ids: List[int]
    member_count: int
    avg_distance_to_centroid: float


class FAISSWrapper:
    """
    Low-level wrapper for FAISS operations.

    Handles index creation, vector operations, and persistence.
    Uses IndexIDMap to support vector deletion (required for incremental updates).

    Attributes:
        dimension: Vector dimension
        index_type: Type of index ("flat" or "ivf")
        metric: Distance metric ("ip" for inner product, "l2" for Euclidean)
    """

    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "flat",
        metric: str = "ip",
        nlist: int = 100
    ):
        """
        Initialize the FAISS wrapper.

        Args:
            dimension: Vector dimension (must match embedding model)
            index_type: "flat" for exact search, "ivf" for approximate (faster for large indices)
            metric: "ip" for inner product (cosine after normalization), "l2" for Euclidean
            nlist: Number of clusters for IVF index (ignored for flat)
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required for RAG functionality. "
                "Install with: pip install faiss-cpu"
            )

        self._dimension = dimension
        self._index_type = index_type
        self._metric = metric
        self._nlist = nlist
        self._index: Optional[faiss.IndexIDMap2] = None
        self._next_id: int = 0
        self._is_trained: bool = False

    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return self._dimension

    @property
    def is_initialized(self) -> bool:
        """Check if index is initialized."""
        return self._index is not None

    @property
    def total_vectors(self) -> int:
        """Get total number of vectors in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    def create_index(self) -> None:
        """
        Create a new FAISS index.

        Creates either a flat index (exact search) or IVF index (approximate search).
        Both are wrapped in IndexIDMap to support deletion by ID.
        """
        if self._metric == "ip":
            # Inner product (use for normalized vectors / cosine similarity)
            if self._index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self._dimension)
                base_index = faiss.IndexIVFFlat(
                    quantizer, self._dimension, self._nlist, faiss.METRIC_INNER_PRODUCT
                )
                self._is_trained = False
            else:
                base_index = faiss.IndexFlatIP(self._dimension)
                self._is_trained = True
        else:
            # L2 (Euclidean) distance
            if self._index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self._dimension)
                base_index = faiss.IndexIVFFlat(
                    quantizer, self._dimension, self._nlist, faiss.METRIC_L2
                )
                self._is_trained = False
            else:
                base_index = faiss.IndexFlatL2(self._dimension)
                self._is_trained = True

        # Wrap in IndexIDMap to support custom IDs and deletion
        self._index = faiss.IndexIDMap2(base_index)
        self._next_id = 0

        logger.info(
            f"Created FAISS index: type={self._index_type}, "
            f"metric={self._metric}, dim={self._dimension}"
        )

    def _ensure_index(self) -> None:
        """Ensure index is initialized."""
        if self._index is None:
            self.create_index()

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors for cosine similarity (when using inner product).

        Args:
            vectors: Input vectors of shape (n, dimension)

        Returns:
            Normalized vectors
        """
        if self._metric != "ip":
            return vectors

        # Normalize for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        return vectors / norms

    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Add vectors to the index.

        Args:
            vectors: Numpy array of shape (n, dimension), dtype float32
            ids: Optional array of int64 IDs. If None, auto-assigns sequential IDs.

        Returns:
            Array of assigned vector IDs (int64)

        Raises:
            ValueError: If vectors have wrong dimension or shape
        """
        self._ensure_index()

        # Validate input
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {vectors.shape}")
        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._dimension}, "
                f"got {vectors.shape[1]}"
            )

        # Ensure correct dtype
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        # Normalize for cosine similarity
        vectors = self._normalize_vectors(vectors)

        # Generate IDs if not provided
        n = vectors.shape[0]
        if ids is None:
            ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
            self._next_id += n
        else:
            ids = np.asarray(ids, dtype=np.int64)
            if len(ids) != n:
                raise ValueError(f"IDs length {len(ids)} != vectors count {n}")
            # Update next_id to avoid collisions
            self._next_id = max(self._next_id, int(ids.max()) + 1)

        # Train IVF index if needed
        if self._index_type == "ivf" and not self._is_trained:
            # Need at least nlist vectors to train
            if n >= self._nlist:
                # Get the underlying index for training
                base_index = faiss.downcast_index(self._index.index)
                base_index.train(vectors)
                self._is_trained = True
                logger.info(f"Trained IVF index with {n} vectors")
            else:
                logger.warning(
                    f"Not enough vectors to train IVF index "
                    f"({n} < {self._nlist}), adding anyway"
                )

        # Add vectors with IDs
        self._index.add_with_ids(vectors, ids)

        logger.debug(f"Added {n} vectors, total now: {self._index.ntotal}")
        return ids

    def remove_vectors(self, ids: np.ndarray) -> int:
        """
        Remove vectors by ID.

        Note: FAISS IndexIDMap doesn't support true deletion.
        This uses remove_ids which marks vectors as deleted but doesn't
        reclaim space. For full cleanup, rebuild the index.

        Args:
            ids: Array of vector IDs to remove

        Returns:
            Number of vectors actually removed
        """
        if self._index is None or len(ids) == 0:
            return 0

        ids = np.asarray(ids, dtype=np.int64)
        initial_count = self._index.ntotal

        # Create ID selector for removal
        id_selector = faiss.IDSelectorArray(len(ids), faiss.swig_ptr(ids))
        removed = self._index.remove_ids(id_selector)

        logger.debug(
            f"Removed {removed} vectors, "
            f"total: {initial_count} -> {self._index.ntotal}"
        )
        return removed

    def get_vector(self, vector_id: int) -> Optional[np.ndarray]:
        """
        Retrieve a vector by its ID.

        Args:
            vector_id: ID of the vector to retrieve

        Returns:
            Vector as numpy array, or None if not found
        """
        if self._index is None or self._index.ntotal == 0:
            return None

        try:
            # IndexIDMap2 supports reconstruct_n but we need single vector
            # Use reconstruct method which takes the vector ID
            vector = np.zeros(self._dimension, dtype=np.float32)
            self._index.reconstruct(vector_id, vector)
            return vector
        except RuntimeError:
            # Vector ID not found
            return None

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.

        Args:
            query_vectors: Query vectors of shape (n, dimension)
            top_k: Number of neighbors to return per query

        Returns:
            Tuple of:
            - distances: Array of shape (n, top_k) with distances/scores
            - ids: Array of shape (n, top_k) with vector IDs (-1 for not found)
        """
        if self._index is None or self._index.ntotal == 0:
            n = query_vectors.shape[0] if query_vectors.ndim == 2 else 1
            return (
                np.full((n, top_k), -1, dtype=np.float32),
                np.full((n, top_k), -1, dtype=np.int64)
            )

        # Ensure 2D
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        # Validate dimension
        if query_vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self._dimension}, "
                f"got {query_vectors.shape[1]}"
            )

        # Ensure correct dtype and normalize
        query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
        query_vectors = self._normalize_vectors(query_vectors)

        # Limit k to available vectors
        k = min(top_k, self._index.ntotal)

        # Search
        distances, ids = self._index.search(query_vectors, k)

        # Pad if we have fewer results than requested
        if k < top_k:
            n = query_vectors.shape[0]
            padded_distances = np.full((n, top_k), -1, dtype=np.float32)
            padded_ids = np.full((n, top_k), -1, dtype=np.int64)
            padded_distances[:, :k] = distances
            padded_ids[:, :k] = ids
            distances, ids = padded_distances, padded_ids

        return distances, ids

    def search_with_scores(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search for nearest neighbors with normalized scores.

        Convenience method for single query that returns SearchResult objects.

        Args:
            query_vector: Single query vector of shape (dimension,)
            top_k: Number of neighbors to return

        Returns:
            List of SearchResult objects sorted by score (highest first)
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, ids = self.search(query_vector, top_k)

        results = []
        for i in range(len(ids[0])):
            vector_id = int(ids[0][i])
            if vector_id == -1:
                continue

            distance = float(distances[0][i])

            # Convert distance to similarity score (0-1)
            if self._metric == "ip":
                # Inner product: higher is more similar
                # Clamp to [0, 1] since vectors are normalized
                score = max(0.0, min(1.0, (distance + 1.0) / 2.0))
            else:
                # L2: lower distance is more similar
                # Convert to similarity using exponential decay
                score = float(np.exp(-distance))

            results.append(SearchResult(
                vector_id=vector_id,
                distance=distance,
                score=score
            ))

        return results

    def save(self, path: str) -> None:
        """
        Save index to file.

        Args:
            path: Path to save the index
        """
        if self._index is None:
            raise ValueError("No index to save - create or load an index first")

        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, path)
        logger.info(f"Saved FAISS index to {path} ({self._index.ntotal} vectors)")

    def load(self, path: str) -> None:
        """
        Load index from file.

        Args:
            path: Path to the saved index

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        self._index = faiss.read_index(path)

        # Infer properties from loaded index
        self._dimension = self._index.d

        # Update next_id based on existing vectors
        if self._index.ntotal > 0:
            # Try to get max ID (this may not work for all index types)
            try:
                # Search for a dummy vector to get some IDs
                dummy = np.zeros((1, self._dimension), dtype=np.float32)
                _, ids = self._index.search(dummy, min(100, self._index.ntotal))
                self._next_id = int(ids.max()) + 1
            except Exception:
                # Fallback: use ntotal as next_id
                self._next_id = self._index.ntotal

        self._is_trained = True  # Loaded index is always trained

        logger.info(
            f"Loaded FAISS index from {path} "
            f"({self._index.ntotal} vectors, dim={self._dimension})"
        )

    def clear(self) -> None:
        """Clear all vectors from the index."""
        if self._index is not None:
            self.create_index()  # Recreate empty index
            logger.info("Cleared FAISS index")

    def get_info(self) -> Dict[str, Any]:
        """
        Get index information.

        Returns:
            Dictionary with index statistics
        """
        return {
            "initialized": self.is_initialized,
            "dimension": self._dimension,
            "index_type": self._index_type,
            "metric": self._metric,
            "total_vectors": self.total_vectors,
            "is_trained": self._is_trained,
            "next_id": self._next_id,
        }
    def get_all_vectors_with_ids(self):
        """Retrieve all vectors and their IDs from the index."""
        if self._index is None or self._index.ntotal == 0:
            return np.array([], dtype=np.float32).reshape(0, self._dimension), np.array([], dtype=np.int64)

        n = self._index.ntotal
        dummy = np.zeros((1, self._dimension), dtype=np.float32)
        _, all_ids = self._index.search(dummy, n)
        all_ids = all_ids[0]

        valid_mask = all_ids != -1
        valid_ids = all_ids[valid_mask]

        vectors = np.zeros((len(valid_ids), self._dimension), dtype=np.float32)
        for i, vid in enumerate(valid_ids):
            try:
                self._index.reconstruct(int(vid), vectors[i])
            except RuntimeError as e:
                logger.warning(f"Failed to reconstruct vector {vid}: {e}")

        return vectors, valid_ids

    def cluster_vectors(self, n_clusters=50, niter=20, min_cluster_size=2, seed=42):
        """Cluster all vectors in the index using K-means."""
        if self._index is None or self._index.ntotal == 0:
            return [], np.array([], dtype=np.int64)

        vectors, ids = self.get_all_vectors_with_ids()
        n_vectors = len(vectors)

        if n_vectors < n_clusters:
            n_clusters = max(2, n_vectors // 2)

        logger.info(f"Clustering {n_vectors} vectors into {n_clusters} clusters...")

        kmeans = faiss.Kmeans(d=self._dimension, k=n_clusters, niter=niter, verbose=False, seed=seed)
        kmeans.train(vectors)

        distances, assignments = kmeans.index.search(vectors, 1)
        assignments = assignments.flatten()
        distances = distances.flatten()

        clusters = []
        for cluster_id in range(n_clusters):
            mask = assignments == cluster_id
            member_indices = np.where(mask)[0]

            if len(member_indices) < min_cluster_size:
                continue

            member_ids = [int(ids[i]) for i in member_indices]
            member_distances = distances[mask]
            avg_distance = float(np.mean(member_distances)) if len(member_distances) > 0 else 0.0

            clusters.append(ClusterInfo(
                cluster_id=cluster_id,
                centroid=kmeans.centroids[cluster_id],
                member_ids=member_ids,
                member_count=len(member_ids),
                avg_distance_to_centroid=avg_distance
            ))

        clusters.sort(key=lambda c: c.member_count, reverse=True)
        logger.info(f"Found {len(clusters)} clusters with >= {min_cluster_size} members")
        return clusters, assignments

    def find_similar_pairs(self, similarity_threshold=0.85, max_pairs=100, k=10):
        """Find pairs of vectors that are highly similar.

        Args:
            similarity_threshold: Minimum similarity score (0-1) to consider as a pair
            max_pairs: Maximum number of pairs to return
            k: Number of nearest neighbors to search per vector. Increase this when
               filtering by entity_types since the top neighbors may be dominated
               by other entity types (e.g., string_literals from log messages).
        """
        if self._index is None or self._index.ntotal < 2:
            return []

        vectors, ids = self.get_all_vectors_with_ids()
        n_vectors = len(vectors)

        k = min(k, n_vectors)
        distances, neighbor_ids = self.search(vectors, k)

        pairs = []
        seen_pairs = set()

        for i in range(n_vectors):
            vec_id = int(ids[i])
            for j in range(k):
                neighbor_id = int(neighbor_ids[i][j])
                if neighbor_id == -1 or neighbor_id == vec_id:
                    continue

                pair = (min(vec_id, neighbor_id), max(vec_id, neighbor_id))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                distance = float(distances[i][j])
                if self._metric == "ip":
                    score = max(0.0, min(1.0, (distance + 1.0) / 2.0))
                else:
                    score = float(np.exp(-distance))

                if score >= similarity_threshold:
                    pairs.append((pair[0], pair[1], score))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:max_pairs]

