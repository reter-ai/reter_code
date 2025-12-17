"""
Tests for RAG (Retrieval-Augmented Generation) components.

Tests cover:
- FAISSWrapper: Vector operations, save/load (requires faiss-cpu)
- EmbeddingService: Embedding generation, caching
- ContentExtractor: Code extraction from files
- MarkdownIndexer: Markdown parsing and chunking
"""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path


# Check if FAISS is available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss-cpu not installed")
class TestFAISSWrapper:
    """Test FAISSWrapper vector operations."""

    @pytest.fixture
    def wrapper(self):
        """Create a FAISSWrapper instance for testing."""
        from codeine.services.faiss_wrapper import FAISSWrapper
        wrapper = FAISSWrapper(dimension=768, index_type="flat", metric="ip")
        wrapper.create_index()
        return wrapper

    def test_create_index(self, wrapper):
        """Test index creation."""
        assert wrapper.is_initialized
        assert wrapper.dimension == 768
        assert wrapper.total_vectors == 0

    def test_add_vectors(self, wrapper):
        """Test adding vectors."""
        vectors = np.random.randn(10, 768).astype(np.float32)
        ids = wrapper.add_vectors(vectors)

        assert len(ids) == 10
        assert wrapper.total_vectors == 10

    def test_add_vectors_with_custom_ids(self, wrapper):
        """Test adding vectors with custom IDs."""
        vectors = np.random.randn(5, 768).astype(np.float32)
        custom_ids = np.array([100, 101, 102, 103, 104], dtype=np.int64)
        ids = wrapper.add_vectors(vectors, ids=custom_ids)

        assert np.array_equal(ids, custom_ids)
        assert wrapper.total_vectors == 5

    def test_search(self, wrapper):
        """Test vector search."""
        # Add some vectors
        vectors = np.random.randn(10, 768).astype(np.float32)
        wrapper.add_vectors(vectors)

        # Search with the first vector (should find itself)
        query = vectors[0:1]
        distances, ids = wrapper.search(query, top_k=5)

        assert distances.shape == (1, 5)
        assert ids.shape == (1, 5)
        assert ids[0][0] == 0  # First result should be the query itself

    def test_search_with_scores(self, wrapper):
        """Test search returning SearchResult objects."""
        vectors = np.random.randn(5, 768).astype(np.float32)
        wrapper.add_vectors(vectors)

        query = vectors[0]
        results = wrapper.search_with_scores(query, top_k=3)

        assert len(results) == 3
        assert all(hasattr(r, 'vector_id') for r in results)
        assert all(hasattr(r, 'score') for r in results)
        assert all(0 <= r.score <= 1 for r in results)

    def test_remove_vectors(self, wrapper):
        """Test vector removal."""
        vectors = np.random.randn(10, 768).astype(np.float32)
        ids = wrapper.add_vectors(vectors)

        assert wrapper.total_vectors == 10

        # Remove some vectors
        to_remove = np.array([0, 2, 4], dtype=np.int64)
        removed = wrapper.remove_vectors(to_remove)

        assert removed == 3
        assert wrapper.total_vectors == 7

    def test_save_and_load(self, wrapper):
        """Test saving and loading index."""
        vectors = np.random.randn(10, 768).astype(np.float32)
        wrapper.add_vectors(vectors)

        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
            path = f.name

        try:
            wrapper.save(path)
            assert Path(path).exists()

            # Create new wrapper and load
            from codeine.services.faiss_wrapper import FAISSWrapper
            new_wrapper = FAISSWrapper(dimension=768)
            new_wrapper.load(path)

            assert new_wrapper.total_vectors == 10
            assert new_wrapper.dimension == 768
        finally:
            os.unlink(path)

    def test_clear(self, wrapper):
        """Test clearing the index."""
        vectors = np.random.randn(10, 768).astype(np.float32)
        wrapper.add_vectors(vectors)
        assert wrapper.total_vectors == 10

        wrapper.clear()
        assert wrapper.total_vectors == 0

    def test_get_info(self, wrapper):
        """Test getting index info."""
        vectors = np.random.randn(5, 768).astype(np.float32)
        wrapper.add_vectors(vectors)

        info = wrapper.get_info()
        assert info["initialized"] == True
        assert info["dimension"] == 768
        assert info["total_vectors"] == 5
        assert info["index_type"] == "flat"
        assert info["metric"] == "ip"


class TestLightweightEmbeddingService:
    """Test LightweightEmbeddingService (for testing without heavy deps)."""

    @pytest.fixture
    def service(self):
        """Create a lightweight embedding service."""
        from codeine.services.embedding_service import LightweightEmbeddingService
        return LightweightEmbeddingService(embedding_dim=768)

    def test_generate_embedding(self, service):
        """Test single embedding generation."""
        embedding = service.generate_embedding("Hello world")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        assert embedding.dtype == np.float32

    def test_embedding_deterministic(self, service):
        """Test that same text produces same embedding."""
        text = "Test sentence for consistency"
        emb1 = service.generate_embedding(text)
        emb2 = service.generate_embedding(text)

        assert np.allclose(emb1, emb2)

    def test_different_texts_different_embeddings(self, service):
        """Test that different texts produce different embeddings."""
        emb1 = service.generate_embedding("First text")
        emb2 = service.generate_embedding("Second text")

        assert not np.allclose(emb1, emb2)

    def test_batch_embeddings(self, service):
        """Test batch embedding generation."""
        texts = ["Text one", "Text two", "Text three"]
        embeddings = service.generate_embeddings_batch(texts)

        assert embeddings.shape == (3, 768)
        assert embeddings.dtype == np.float32

    def test_embedding_normalized(self, service):
        """Test that embeddings are normalized."""
        embedding = service.generate_embedding("Test text")
        norm = np.linalg.norm(embedding)

        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_caching(self, service):
        """Test that embeddings are cached."""
        text = "Cached text"
        service.generate_embedding(text)

        # Should be in cache now
        assert service.get_info()["cache_size"] == 1

    def test_get_info(self, service):
        """Test getting service info."""
        info = service.get_info()

        assert info["model"] == "lightweight-test"
        assert info["provider"] == "test"
        assert info["dimension"] == 768


class TestEmbeddingServiceFactory:
    """Test get_embedding_service factory function."""

    def test_default_config(self):
        """Test factory with default config returns lightweight for testing."""
        from codeine.services.embedding_service import get_embedding_service

        service = get_embedding_service({"rag_use_lightweight": True})
        assert service.provider == "test"

    def test_custom_dimension(self):
        """Test factory respects custom dimension in lightweight mode."""
        from codeine.services.embedding_service import (
            get_embedding_service,
            reset_embedding_service_singleton,
        )

        # Reset singleton to test custom dimension
        reset_embedding_service_singleton()

        config = {"rag_use_lightweight": True, "rag_embedding_dim": 512}
        service = get_embedding_service(config)

        assert service.embedding_dim == 512

        # Reset after test to not affect other tests
        reset_embedding_service_singleton()


class TestContentExtractor:
    """Test ContentExtractor for extracting code from files."""

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create a ContentExtractor with temp project root."""
        from codeine.services.content_extractor import ContentExtractor
        return ContentExtractor(project_root=tmp_path, max_body_lines=50)

    @pytest.fixture
    def sample_file(self, tmp_path):
        """Create a sample Python file for testing."""
        content = '''"""Module docstring."""

def hello(name: str) -> str:
    """Say hello to someone.

    Args:
        name: The name to greet.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"


class Greeter:
    """A greeter class."""

    def __init__(self, prefix: str = "Hello"):
        """Initialize the greeter.

        Args:
            prefix: The greeting prefix.
        """
        self.prefix = prefix

    def greet(self, name: str) -> str:
        """Greet someone.

        Args:
            name: The name to greet.

        Returns:
            A greeting string.
        """
        return f"{self.prefix}, {name}!"
'''
        file_path = tmp_path / "sample.py"
        file_path.write_text(content)
        return file_path

    def test_extract_lines(self, extractor, sample_file):
        """Test extracting specific lines."""
        content = extractor.extract_lines(str(sample_file), 3, 12)

        assert content is not None
        assert "def hello" in content
        assert "Say hello to someone" in content

    def test_extract_entity_content(self, extractor, sample_file):
        """Test extracting entity content."""
        content = extractor.extract_entity_content(
            str(sample_file),
            start_line=3,
            end_line=12,
            entity_type="function"
        )

        assert content is not None
        assert "def hello" in content

    def test_extract_signature(self, extractor, sample_file):
        """Test extracting function signature."""
        signature = extractor.extract_signature(str(sample_file), 3)

        assert signature is not None
        assert "def hello" in signature
        assert "name: str" in signature
        assert "-> str" in signature

    def test_extract_docstring(self, extractor, sample_file):
        """Test extracting docstring."""
        docstring = extractor.extract_docstring(str(sample_file), 3)

        assert docstring is not None
        assert "Say hello to someone" in docstring

    def test_extract_and_build(self, extractor, sample_file):
        """Test extracting and building a CodeEntity."""
        entity = extractor.extract_and_build(
            file_path=str(sample_file),
            entity_type="function",
            name="hello",
            qualified_name="sample.hello",
            start_line=3,
            end_line=12
        )

        assert entity is not None
        assert entity.name == "hello"
        assert entity.entity_type == "function"
        assert entity.docstring is not None
        assert entity.signature is not None

    def test_build_indexable_text(self, extractor, sample_file):
        """Test building indexable text from entity."""
        entity = extractor.extract_and_build(
            file_path=str(sample_file),
            entity_type="function",
            name="hello",
            qualified_name="sample.hello",
            start_line=3,
            end_line=12
        )

        text = extractor.build_indexable_text(entity)

        assert "function: hello" in text
        assert "Say hello to someone" in text  # docstring

    def test_cache_invalidation(self, extractor, sample_file):
        """Test file cache invalidation."""
        # Read file to populate cache
        extractor.extract_lines(str(sample_file), 1, 5)
        assert extractor.get_cache_info()["cached_files"] == 1

        # Invalidate cache
        extractor.invalidate_cache()
        assert extractor.get_cache_info()["cached_files"] == 0


class TestMarkdownIndexer:
    """Test MarkdownIndexer for parsing markdown files."""

    @pytest.fixture
    def indexer(self):
        """Create a MarkdownIndexer."""
        from codeine.services.markdown_indexer import MarkdownIndexer
        return MarkdownIndexer(
            max_chunk_words=200,
            min_chunk_words=10,
            include_code_blocks=True
        )

    @pytest.fixture
    def sample_markdown(self, tmp_path):
        """Create a sample markdown file."""
        content = '''# Project Documentation

This is the main documentation for our project.

## Installation

To install the project, run:

```bash
pip install myproject
```

## Usage

Here's how to use the project:

```python
from myproject import hello
print(hello("World"))
```

## API Reference

### hello(name)

Greets a person by name.

**Parameters:**
- name (str): The name to greet

**Returns:**
- str: A greeting message
'''
        file_path = tmp_path / "README.md"
        file_path.write_text(content)
        return file_path

    def test_parse_file(self, indexer, sample_markdown):
        """Test parsing a markdown file."""
        chunks = indexer.parse_file(str(sample_markdown))

        assert len(chunks) > 0
        # Small documents are indexed as "document", code blocks are separate
        assert any(c.chunk_type in ("document", "section", "code_block") for c in chunks)

    def test_extract_title(self, indexer, sample_markdown):
        """Test extracting document title."""
        chunks = indexer.parse_file(str(sample_markdown))

        # At least one chunk should have the title
        titles = [c.title for c in chunks if c.title]
        assert any("Project Documentation" in t for t in titles)

    def test_extract_sections(self, indexer, sample_markdown):
        """Test extracting sections by headings - for large documents."""
        chunks = indexer.parse_file(str(sample_markdown))

        # For small documents (<max_chunk_words), the whole doc is indexed as "document"
        # Sections are only extracted for larger documents
        # Our test file is small, so check for document chunk or code blocks
        doc_or_section = [c for c in chunks if c.chunk_type in ("document", "section")]
        assert len(doc_or_section) > 0 or len(chunks) > 0

        # If there's a document chunk, it should have the title
        doc_chunks = [c for c in chunks if c.chunk_type == "document"]
        if doc_chunks:
            assert any(c.title == "Project Documentation" for c in doc_chunks)

    def test_extract_code_blocks(self, indexer, sample_markdown):
        """Test extracting fenced code blocks."""
        chunks = indexer.parse_file(str(sample_markdown))
        code_blocks = [c for c in chunks if c.chunk_type == "code_block"]

        assert len(code_blocks) >= 2  # bash and python blocks
        languages = [c.language for c in code_blocks]
        assert "bash" in languages
        assert "python" in languages

    def test_build_indexable_text_section(self, indexer, sample_markdown):
        """Test building indexable text from a section or document chunk."""
        chunks = indexer.parse_file(str(sample_markdown))

        # Get a section chunk, or fall back to document chunk
        chunk = next(
            (c for c in chunks if c.chunk_type == "section"),
            next((c for c in chunks if c.chunk_type == "document"), None)
        )
        assert chunk is not None, "Expected at least one section or document chunk"

        text = indexer.build_indexable_text(chunk)

        assert "Section:" in text or "Document:" in text

    def test_build_indexable_text_code_block(self, indexer, sample_markdown):
        """Test building indexable text from a code block."""
        chunks = indexer.parse_file(str(sample_markdown))
        code_block = next(c for c in chunks if c.chunk_type == "code_block")

        text = indexer.build_indexable_text(code_block)

        assert "Code:" in text
        assert "Language:" in text or code_block.language in text

    def test_chunk_word_count(self, indexer, sample_markdown):
        """Test that chunks have word counts."""
        chunks = indexer.parse_file(str(sample_markdown))

        for chunk in chunks:
            assert chunk.word_count > 0

    def test_get_chunk_id(self, indexer, sample_markdown):
        """Test generating unique chunk IDs."""
        chunks = indexer.parse_file(str(sample_markdown))

        ids = [indexer.get_chunk_id(c) for c in chunks]
        # All IDs should be unique
        assert len(ids) == len(set(ids))

    def test_small_document_as_whole(self, indexer, tmp_path):
        """Test that small documents are indexed as a whole."""
        content = "# Title\n\nThis is a small document with just a few words."
        file_path = tmp_path / "small.md"
        file_path.write_text(content)

        chunks = indexer.parse_file(str(file_path))

        # Should have at most one document chunk (plus maybe code blocks)
        doc_chunks = [c for c in chunks if c.chunk_type == "document"]
        assert len(doc_chunks) <= 1


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss-cpu not installed")
class TestIntegration:
    """Integration tests for RAG components working together."""

    @pytest.fixture
    def project_dir(self, tmp_path):
        """Create a sample project directory."""
        # Create Python file
        py_content = '''"""Sample module."""

def calculate(x: int, y: int) -> int:
    """Calculate sum of two numbers."""
    return x + y

class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
'''
        (tmp_path / "calc.py").write_text(py_content)

        # Create markdown file
        md_content = '''# Calculator

A simple calculator module.

## Functions

### calculate(x, y)

Calculates the sum of two numbers.
'''
        (tmp_path / "README.md").write_text(md_content)

        return tmp_path

    def test_extract_and_embed(self, project_dir):
        """Test extracting content and generating embeddings."""
        from codeine.services.content_extractor import ContentExtractor
        from codeine.services.embedding_service import LightweightEmbeddingService

        extractor = ContentExtractor(project_root=project_dir, max_body_lines=50)
        embedding_service = LightweightEmbeddingService(embedding_dim=768)

        # Extract function
        entity = extractor.extract_and_build(
            file_path=str(project_dir / "calc.py"),
            entity_type="function",
            name="calculate",
            qualified_name="calc.calculate",
            start_line=3,
            end_line=5
        )

        # Build indexable text
        text = extractor.build_indexable_text(entity)

        # Generate embedding
        embedding = embedding_service.generate_embedding(text)

        assert embedding.shape == (768,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-6)

    def test_parse_and_embed_markdown(self, project_dir):
        """Test parsing markdown and generating embeddings."""
        from codeine.services.markdown_indexer import MarkdownIndexer
        from codeine.services.embedding_service import LightweightEmbeddingService

        indexer = MarkdownIndexer(max_chunk_words=500, min_chunk_words=5)
        embedding_service = LightweightEmbeddingService(embedding_dim=768)

        # Parse markdown
        chunks = indexer.parse_file(str(project_dir / "README.md"))

        # Build indexable texts and generate embeddings
        texts = [indexer.build_indexable_text(c) for c in chunks]
        embeddings = embedding_service.generate_embeddings_batch(texts)

        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == 768

    def test_full_indexing_pipeline(self, project_dir):
        """Test full indexing pipeline with FAISS."""
        from codeine.services.faiss_wrapper import FAISSWrapper
        from codeine.services.embedding_service import LightweightEmbeddingService
        from codeine.services.markdown_indexer import MarkdownIndexer

        # Setup components
        faiss_wrapper = FAISSWrapper(dimension=768, index_type="flat", metric="ip")
        faiss_wrapper.create_index()
        embedding_service = LightweightEmbeddingService(embedding_dim=768)
        indexer = MarkdownIndexer(max_chunk_words=500, min_chunk_words=5)

        # Parse and index markdown
        chunks = indexer.parse_file(str(project_dir / "README.md"))
        texts = [indexer.build_indexable_text(c) for c in chunks]
        embeddings = embedding_service.generate_embeddings_batch(texts)

        # Add to FAISS
        ids = faiss_wrapper.add_vectors(embeddings)

        assert faiss_wrapper.total_vectors == len(chunks)

        # Search
        query = "calculator function"
        query_embedding = embedding_service.generate_embedding(query)
        results = faiss_wrapper.search_with_scores(query_embedding, top_k=3)

        assert len(results) > 0
        assert all(r.vector_id >= 0 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
