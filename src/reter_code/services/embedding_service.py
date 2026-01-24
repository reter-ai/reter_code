"""
Embedding generation service for RAG functionality.

Manages embedding generation using various providers:
- Local: sentence-transformers (default, no API key needed)
- Anthropic: voyage-code-3 or voyage-3 (via Voyage AI API)
- OpenAI: text-embedding-3-small/large or ada-002

Configuration via reter.json or environment variables.
"""

import hashlib
import logging
import os
from typing import Callable, List, Dict, Optional, Any
from collections import OrderedDict

import numpy as np
from sentence_transformers import SentenceTransformer

from ..logging_config import configure_logger_for_debug_trace
logger = configure_logger_for_debug_trace(__name__)

# Provider availability
_SENTENCE_TRANSFORMERS_AVAILABLE = True  # Imported eagerly above
_VOYAGE_AVAILABLE: Optional[bool] = None
_OPENAI_AVAILABLE: Optional[bool] = None


def _check_sentence_transformers() -> bool:
    """Check for sentence-transformers availability."""
    return _SENTENCE_TRANSFORMERS_AVAILABLE


def _check_voyage() -> bool:
    """Lazy check for voyageai availability."""
    global _VOYAGE_AVAILABLE
    if _VOYAGE_AVAILABLE is None:
        try:
            import voyageai  # noqa: F401
            _VOYAGE_AVAILABLE = True
        except ImportError:
            _VOYAGE_AVAILABLE = False
    return _VOYAGE_AVAILABLE


def _check_openai() -> bool:
    """Lazy check for openai availability."""
    global _OPENAI_AVAILABLE
    if _OPENAI_AVAILABLE is None:
        try:
            import openai  # noqa: F401
            _OPENAI_AVAILABLE = True
        except ImportError:
            _OPENAI_AVAILABLE = False
    return _OPENAI_AVAILABLE


class EmbeddingService:
    """
    Manages embedding generation with caching and multiple provider support.

    Provider priority (configurable in reter.json):
    1. Local sentence-transformers (default, free, works offline)
    2. Voyage AI API (voyage-code-3, best for code)
    3. OpenAI API (text-embedding-3-small)

    Attributes:
        model_name: Name of the embedding model
        provider: Provider type ("local", "voyage", "openai")
        embedding_dim: Dimension of embeddings produced
    """

    # Model configurations
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        # Local models (sentence-transformers)
        "sentence-transformers/all-mpnet-base-v2": {"dim": 768, "provider": "local"},
        "sentence-transformers/all-MiniLM-L6-v2": {"dim": 384, "provider": "local"},
        "all-mpnet-base-v2": {"dim": 768, "provider": "local"},  # Short alias
        "all-MiniLM-L6-v2": {"dim": 384, "provider": "local"},   # Short alias

        # Voyage AI models (best for code)
        "voyage-code-3": {"dim": 1024, "provider": "voyage"},
        "voyage-3": {"dim": 1024, "provider": "voyage"},
        "voyage-3-lite": {"dim": 512, "provider": "voyage"},

        # OpenAI models
        "text-embedding-3-small": {"dim": 1536, "provider": "openai"},
        "text-embedding-3-large": {"dim": 3072, "provider": "openai"},
        "text-embedding-ada-002": {"dim": 1536, "provider": "openai"},
    }

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        cache_size: int = 1000,
        api_key: Optional[str] = None
    ):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the embedding model (see MODEL_CONFIGS)
            cache_size: Maximum number of embeddings to cache
            api_key: API key for Voyage or OpenAI (optional, uses env vars if not provided)
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self._api_key = api_key
        self._model = None
        self._client = None

        # Use OrderedDict for LRU-like cache
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # Determine provider and dimension
        config = self.MODEL_CONFIGS.get(model_name, {"dim": 768, "provider": "local"})
        self.provider = config["provider"]
        self.embedding_dim = config["dim"]

        self._initialized = False
        self._initializing = False  # Simple flag to prevent concurrent init

    def initialize(self) -> None:
        """Lazy initialization of the model/client.

        Uses simple flag-based locking to avoid blocking async event loop.
        The model is typically pre-loaded by background thread via set_preloaded_model().

        NOTE: This method is blocking. When called from async context,
        use asyncio.to_thread() at the caller level.
        """
        # Fast path - already initialized
        if self._initialized:
            return

        # Simple spin-wait if another call is initializing
        # This is acceptable because:
        # 1. Initialization happens once at startup
        # 2. Pre-loaded models skip this entirely
        if self._initializing:
            import time
            while self._initializing and not self._initialized:
                time.sleep(0.01)
            return

        self._initializing = True
        try:
            # Check again after setting flag
            if self._initialized:
                return

            logger.info(f"Initializing embedding service: {self.model_name} ({self.provider})")

            if self.provider == "local":
                self._init_local()
            elif self.provider == "voyage":
                self._init_voyage()
            elif self.provider == "openai":
                self._init_openai()
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            self._initialized = True
            logger.info(f"Embedding service ready. Dimension: {self.embedding_dim}")
        finally:
            self._initializing = False

    def set_preloaded_model(self, model) -> None:
        """Set a pre-loaded model to avoid import deadlocks in async context."""
        self._preloaded_model = model

    def _init_local(self) -> None:
        """Initialize local sentence-transformers model.

        Uses pre-loaded model if available to avoid import deadlocks in async context.
        """
        import time

        def log(msg):
            logger.info(f"[Embedding] {msg}")

        # Use pre-loaded model if available (avoids async import deadlock)
        if hasattr(self, '_preloaded_model') and self._preloaded_model is not None:
            log("_init_local: Using pre-loaded model")
            self._model = self._preloaded_model
            test_emb = self._model.encode("test", convert_to_numpy=True)
            self.embedding_dim = len(test_emb)
            log(f"_init_local: Complete! dim={self.embedding_dim}")
            return

        log("_init_local: Starting fresh load...")

        # Normalize model name for sentence-transformers
        st_model_name = self.model_name
        if st_model_name.startswith("sentence-transformers/"):
            st_model_name = st_model_name.replace("sentence-transformers/", "")

        # Set cache directory
        cache_dir = os.environ.get('TRANSFORMERS_CACHE', None)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        log(f"_init_local: Loading model '{st_model_name}'...")
        model_start = time.time()
        self._model = SentenceTransformer(st_model_name, cache_folder=cache_dir)
        log(f"_init_local: Model loaded in {time.time() - model_start:.2f}s")

        # Get actual dimension from model
        log("_init_local: Testing embedding generation...")
        test_start = time.time()
        test_emb = self._model.encode("test", convert_to_numpy=True)
        self.embedding_dim = len(test_emb)
        log(f"_init_local: Complete! dim={self.embedding_dim}, test took {time.time() - test_start:.2f}s")

    def _init_voyage(self) -> None:
        """Initialize Voyage AI client."""
        if not _check_voyage():
            raise ImportError(
                "voyageai package is required for Voyage embeddings. "
                "Install with: pip install voyageai"
            )

        # Import lazily
        import voyageai

        api_key = self._api_key or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "VOYAGE_API_KEY required for Voyage embeddings. "
                "Set via environment variable or reter.json"
            )

        self._client = voyageai.Client(api_key=api_key)

    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        if not _check_openai():
            raise ImportError(
                "openai package is required for OpenAI embeddings. "
                "Install with: pip install openai"
            )

        # Import lazily
        import openai

        api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY required for OpenAI embeddings. "
                "Set via environment variable or reter.json"
            )

        self._client = openai.OpenAI(api_key=api_key)

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _cache_get(self, key: str) -> Optional[np.ndarray]:
        """Get from cache and move to end (LRU)."""
        if key in self._embedding_cache:
            self._embedding_cache.move_to_end(key)
            return self._embedding_cache[key].copy()
        return None

    def _cache_put(self, key: str, embedding: np.ndarray) -> None:
        """Put in cache with LRU eviction."""
        if key in self._embedding_cache:
            self._embedding_cache.move_to_end(key)
        else:
            if len(self._embedding_cache) >= self.cache_size:
                # Remove oldest (first) item
                self._embedding_cache.popitem(last=False)
            self._embedding_cache[key] = embedding.copy()

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to generate embedding for

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        self.initialize()

        # Check cache
        cache_key = self._get_cache_key(text)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Generate embedding based on provider
        if self.provider == "local":
            embedding = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        elif self.provider == "voyage":
            embedding = self._embed_voyage([text])[0]
        elif self.provider == "openai":
            embedding = self._embed_openai([text])[0]
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Ensure float32
        embedding = np.asarray(embedding, dtype=np.float32)

        # Normalize for cosine similarity via inner product
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Cache
        self._cache_put(cache_key, embedding)

        return embedding

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar (local only)
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        self.initialize()

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)

        # Separate cached and uncached
        results: Dict[int, np.ndarray] = {}
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached = self._cache_get(cache_key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate for uncached
        if uncached_texts:
            cache_hits = len(texts) - len(uncached_texts)
            logger.debug(
                f"Generating embeddings for {len(uncached_texts)} texts "
                f"({cache_hits} from cache)"
            )

            if self.provider == "local":
                new_embeddings = self._embed_local_batch(
                    uncached_texts, batch_size, show_progress, progress_callback
                )
            elif self.provider == "voyage":
                new_embeddings = self._embed_voyage(uncached_texts, batch_size)
            elif self.provider == "openai":
                new_embeddings = self._embed_openai(uncached_texts, batch_size)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            # Add to cache and results (normalize for cosine similarity)
            for text, idx, embedding in zip(uncached_texts, uncached_indices, new_embeddings):
                embedding = np.asarray(embedding, dtype=np.float32)
                # Normalize for cosine similarity via inner product
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                cache_key = self._get_cache_key(text)
                self._cache_put(cache_key, embedding)
                results[idx] = embedding

        # Assemble in order
        ordered = [results[i] for i in range(len(texts))]
        return np.array(ordered, dtype=np.float32)

    def _embed_local_batch(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """Generate embeddings using local sentence-transformers with progress logging."""
        import time

        total = len(texts)

        # For small batches, just encode directly
        if total <= batch_size * 2:
            if progress_callback:
                progress_callback(total, total)
            return self._model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )

        # For large batches, process in chunks with progress logging
        all_embeddings = []
        start_time = time.time()
        num_batches = (total + batch_size - 1) // batch_size

        for i in range(0, total, batch_size):
            batch_num = i // batch_size + 1
            batch_texts = texts[i:i + batch_size]

            embeddings = self._model.encode(
                batch_texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            all_embeddings.append(embeddings)

            completed = i + len(batch_texts)

            # Call progress callback every batch
            if progress_callback:
                progress_callback(completed, total)

            # Log progress every 10 batches or at end
            if batch_num % 10 == 0 or batch_num == num_batches:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total - completed) / rate if rate > 0 else 0
                logger.info(
                    "Embedding progress: %d/%d (%.0f%%) - %.0f texts/s, ~%.0fs remaining",
                    completed, total,
                    100 * completed / total,
                    rate, remaining
                )

        return np.vstack(all_embeddings)

    def _embed_voyage(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """Generate embeddings using Voyage AI API."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Voyage AI uses embed method
            result = self._client.embed(
                texts=batch,
                model=self.model_name,
                input_type="document"  # or "query" for search queries
            )

            batch_embeddings = [np.array(e, dtype=np.float32) for e in result.embeddings]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def _embed_openai(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = self._client.embeddings.create(
                model=self.model_name,
                input=batch
            )

            # Sort by index to ensure correct order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [
                np.array(e.embedding, dtype=np.float32)
                for e in sorted_data
            ]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity score (-1 to 1, typically 0 to 1 for text)
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def compute_similarities_batch(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarities between query and multiple embeddings.

        Args:
            query_embedding: Query embedding of shape (dim,)
            embeddings: Array of embeddings of shape (n, dim)

        Returns:
            Array of similarity scores of shape (n,)
        """
        if len(embeddings) == 0:
            return np.array([], dtype=np.float32)

        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(embeddings), dtype=np.float32)

        query_normalized = query_embedding / query_norm

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1)
        norms[norms == 0] = 1.0  # Avoid division by zero
        embeddings_normalized = embeddings / norms[:, np.newaxis]

        # Compute dot products
        similarities = np.dot(embeddings_normalized, query_normalized)

        return similarities.astype(np.float32)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")

    def get_info(self) -> Dict[str, Any]:
        """
        Get service information.

        Returns:
            Dictionary with service statistics
        """
        return {
            'model': self.model_name,
            'provider': self.provider,
            'dimension': self.embedding_dim,
            'initialized': self._initialized,
            'cache_size': len(self._embedding_cache),
            'max_cache_size': self.cache_size,
        }


class LightweightEmbeddingService(EmbeddingService):
    """
    Lightweight embedding service for testing without heavy dependencies.

    Uses hash-based embeddings that maintain some semantic properties
    (same text = same embedding) but are NOT suitable for production.
    """

    def __init__(self, embedding_dim: int = 768, cache_size: int = 1000):
        """
        Initialize lightweight service.

        Args:
            embedding_dim: Dimension of embeddings to generate
            cache_size: Maximum cache size
        """
        self.model_name = "lightweight-test"
        self.provider = "test"
        self.embedding_dim = embedding_dim
        self.cache_size = cache_size
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._initialized = True

    def initialize(self) -> None:
        """No initialization needed for lightweight service."""
        pass

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate hash-based embedding (for testing only).

        Creates deterministic embeddings from text hash.
        NOT suitable for production semantic search.
        """
        # Check cache
        cache_key = self._get_cache_key(text)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Generate from hash
        text_hash = hashlib.sha256(text.encode('utf-8')).digest()

        # Convert bytes to floats
        values = [float(b) / 255.0 for b in text_hash]

        # Extend to embedding dimension
        while len(values) < self.embedding_dim:
            # Hash again for more values
            extended_hash = hashlib.sha256(
                text_hash + len(values).to_bytes(4, 'little')
            ).digest()
            values.extend([float(b) / 255.0 for b in extended_hash])

        embedding = np.array(values[:self.embedding_dim], dtype=np.float32)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Cache
        self._cache_put(cache_key, embedding)

        return embedding

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """Generate embeddings for batch (testing only)."""
        return np.array(
            [self.generate_embedding(text) for text in texts],
            dtype=np.float32
        )


# Global singleton for embedding service - avoids reloading model on every sync
_embedding_service_singleton: Optional[EmbeddingService] = None
_lightweight_singleton: Optional[EmbeddingService] = None


def get_embedding_service(
    config: Optional[Dict[str, Any]] = None,
    use_lightweight: bool = False
) -> EmbeddingService:
    """
    Factory function to get embedding service based on configuration.

    Returns a singleton instance to avoid reloading the model multiple times.

    Args:
        config: Configuration dict (from reter.json) or None for defaults
        use_lightweight: Force lightweight service for testing

    Returns:
        Configured EmbeddingService instance (singleton)
    """
    global _embedding_service_singleton, _lightweight_singleton

    if config is None:
        config = {}

    # Check for lightweight/test mode
    if use_lightweight or config.get("rag_use_lightweight", False):
        if _lightweight_singleton is None:
            logger.warning("Using lightweight embedding service (testing only)")
            dim = config.get("rag_embedding_dim", 768)
            cache_size = config.get("rag_embedding_cache_size", 1000)
            _lightweight_singleton = LightweightEmbeddingService(embedding_dim=dim, cache_size=cache_size)
        return _lightweight_singleton

    # Return existing singleton if available
    if _embedding_service_singleton is not None:
        return _embedding_service_singleton

    model_name = config.get(
        "rag_embedding_model",
        "sentence-transformers/all-mpnet-base-v2"
    )
    cache_size = config.get("rag_embedding_cache_size", 1000)
    api_key = config.get("rag_api_key")  # Optional, falls back to env vars

    _embedding_service_singleton = EmbeddingService(
        model_name=model_name,
        cache_size=cache_size,
        api_key=api_key
    )
    return _embedding_service_singleton


def reset_embedding_service_singleton() -> None:
    """Reset the embedding service singleton (for testing only)."""
    global _embedding_service_singleton, _lightweight_singleton
    _embedding_service_singleton = None
    _lightweight_singleton = None
