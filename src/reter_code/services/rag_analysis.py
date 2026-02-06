"""
RAG Analysis - Analysis methods for RAG index operations.

Contains methods for finding similar code clusters, duplicate candidates,
and analyzing documentation relevance.

These are extracted from RAGIndexManager as a mixin to reduce file size.
"""

import time
from typing import Dict, List, Optional, Any

import numpy as np

from .initialization_progress import (
    require_rag_code_index,
    require_rag_document_index,
)


class RAGAnalysisMixin:
    """
    Mixin providing analysis methods for RAGIndexManager.

    ::: This is-in-layer Service-Layer.
    ::: This is-part-of-component RAG-Index.
    ::: This is-in-process Main-Process.
    ::: This is stateless.

    These methods require:
    - self._initialized: bool
    - self._faiss_wrapper: FAISSWrapper
    - self._metadata: Dict[str, Any]
    """

    def find_similar_clusters(
        self,
        n_clusters: int = 50,
        min_cluster_size: int = 2,
        exclude_same_file: bool = True,
        exclude_same_class: bool = True,
        entity_types: Optional[List[str]] = None,
        source_type: Optional[str] = None,  # "python", "markdown", or None for all
    ) -> Dict[str, Any]:
        """
        Find clusters of semantically similar code using K-means clustering.

        Uses FAISS K-means to group code entities by semantic similarity,
        then filters to find potential duplicates (similar code in different locations).

        Args:
            n_clusters: Number of clusters to create (auto-adjusted based on data)
            min_cluster_size: Minimum members for a cluster to be considered
            exclude_same_file: Exclude cluster members from the same file
            exclude_same_class: Exclude cluster members from the same class
            entity_types: Filter by entity type (e.g., ["method", "function"])
            source_type: Filter by source ("python" or "markdown")

        Returns:
            Dict with clusters, each containing similar code entities

        Raises:
            ComponentNotReadyError: If RAG code index is not ready
        """
        require_rag_code_index()

        start_time = time.time()

        # Defensive type coercion - parameters may come as strings from CADSL
        n_clusters = int(n_clusters) if n_clusters is not None else 50
        min_cluster_size = int(min_cluster_size) if min_cluster_size is not None else 2

        if not self._initialized or self._faiss_wrapper is None:
            return {
                "success": False,
                "error": "RAG index not initialized",
                "clusters": [],
            }

        # Get clusters from FAISS
        clusters, assignments = self._faiss_wrapper.cluster_vectors(
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
        )

        # Enrich clusters with metadata and filter
        enriched_clusters = []

        for cluster in clusters:
            members = []
            files_in_cluster = set()
            classes_in_cluster = set()

            for vector_id in cluster.member_ids:
                meta = self._metadata.get("vectors", {}).get(str(vector_id))
                if not meta:
                    continue

                # Apply filters
                if entity_types and meta.get("entity_type") not in entity_types:
                    continue
                if source_type and meta.get("source_type") != source_type:
                    continue

                file_path = meta.get("file", "")
                class_name = meta.get("class_name", "")

                members.append({
                    "vector_id": vector_id,
                    "name": meta.get("name", ""),
                    "qualified_name": meta.get("qualified_name", ""),
                    "entity_type": meta.get("entity_type", ""),
                    "file": file_path,
                    "line": meta.get("line", 0),
                    "end_line": meta.get("end_line"),
                    "class_name": class_name,
                    "source_type": meta.get("source_type", ""),
                    "docstring_preview": meta.get("docstring_preview", ""),
                })

                # Normalize path for comparison (handle / vs \ differences)
                files_in_cluster.add(file_path.replace("\\", "/"))
                if class_name:
                    classes_in_cluster.add(class_name)

            # Apply exclusion filters
            if exclude_same_file and len(files_in_cluster) <= 1:
                continue
            if exclude_same_class and len(classes_in_cluster) <= 1 and len(members) > 1:
                # All members from same class - likely expected similarity
                continue

            if len(members) >= min_cluster_size:
                # Convert L2 distance to similarity for consistency
                # For normalized vectors: L2² ranges [0, 4], convert to similarity [1, 0]
                avg_similarity = max(0.0, 1.0 - cluster.avg_distance_to_centroid / 2.0)
                enriched_clusters.append({
                    "cluster_id": cluster.cluster_id,
                    "member_count": len(members),
                    "unique_files": len(files_in_cluster),
                    "unique_classes": len(classes_in_cluster),
                    "avg_similarity": round(avg_similarity, 4),
                    "members": members,
                })

        # Sort by potential interest (more unique files = more interesting)
        enriched_clusters.sort(
            key=lambda c: (c["unique_files"], c["member_count"]),
            reverse=True
        )

        time_ms = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "total_clusters": len(enriched_clusters),
            "total_vectors_analyzed": self._faiss_wrapper.total_vectors,
            "clusters": enriched_clusters,
            "time_ms": time_ms,
            "filters": {
                "n_clusters": n_clusters,
                "min_cluster_size": min_cluster_size,
                "exclude_same_file": exclude_same_file,
                "exclude_same_class": exclude_same_class,
                "entity_types": entity_types,
                "source_type": source_type,
            }
        }

    def find_similar_clusters_dbscan(
        self,
        eps: float = 0.5,
        min_samples: int = 3,
        min_cluster_size: int = 2,
        exclude_same_file: bool = True,
        exclude_same_class: bool = True,
        entity_types: Optional[List[str]] = None,
        source_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Find clusters of semantically similar code using DBSCAN clustering.

        DBSCAN advantages over K-means:
        - No need to specify number of clusters upfront
        - Automatically discovers natural groupings
        - Identifies outliers/noise (not forced into clusters)
        - Better for finding tight clusters of truly similar code

        Args:
            eps: Maximum distance between samples to be neighbors (0.3-0.8 typical).
                 Smaller = tighter clusters, more of them.
            min_samples: Minimum points to form a dense region.
                        Higher = fewer, denser clusters.
            min_cluster_size: Minimum members for a cluster to be returned
            exclude_same_file: Exclude cluster members from the same file
            exclude_same_class: Exclude cluster members from the same class
            entity_types: Filter by entity type (e.g., ["method", "function"])
            source_type: Filter by source ("python" or "markdown")

        Returns:
            Dict with clusters, each containing similar code entities

        Raises:
            ComponentNotReadyError: If RAG code index is not ready
        """
        require_rag_code_index()

        start_time = time.time()

        # Defensive type coercion - parameters may come as strings from CADSL
        eps = float(eps) if eps is not None else 0.5
        min_samples = int(min_samples) if min_samples is not None else 3
        min_cluster_size = int(min_cluster_size) if min_cluster_size is not None else 2

        if not self._initialized or self._faiss_wrapper is None:
            return {
                "success": False,
                "error": "RAG index not initialized",
                "clusters": [],
            }

        # Get DBSCAN clusters from FAISS wrapper
        clusters, assignments = self._faiss_wrapper.cluster_vectors_dbscan(
            eps=eps,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
        )

        # Count noise points
        n_noise = int(np.sum(assignments == -1)) if len(assignments) > 0 else 0

        # Enrich clusters with metadata and filter
        enriched_clusters = []

        for cluster in clusters:
            members = []
            files_in_cluster = set()
            classes_in_cluster = set()

            for vector_id in cluster.member_ids:
                meta = self._metadata.get("vectors", {}).get(str(vector_id))
                if not meta:
                    continue

                # Apply filters
                if entity_types and meta.get("entity_type") not in entity_types:
                    continue
                if source_type and meta.get("source_type") != source_type:
                    continue

                file_path = meta.get("file", "")
                class_name = meta.get("class_name", "")

                members.append({
                    "vector_id": vector_id,
                    "name": meta.get("name", ""),
                    "qualified_name": meta.get("qualified_name", ""),
                    "entity_type": meta.get("entity_type", ""),
                    "file": file_path,
                    "line": meta.get("line", 0),
                    "end_line": meta.get("end_line"),
                    "class_name": class_name,
                    "source_type": meta.get("source_type", ""),
                    "docstring_preview": meta.get("docstring_preview", ""),
                })

                files_in_cluster.add(file_path.replace("\\", "/"))
                if class_name:
                    classes_in_cluster.add(class_name)

            # Apply exclusion filters
            if exclude_same_file and len(files_in_cluster) <= 1:
                continue
            if exclude_same_class and len(classes_in_cluster) <= 1 and len(members) > 1:
                continue

            if len(members) >= min_cluster_size:
                avg_similarity = max(0.0, 1.0 - cluster.avg_distance_to_centroid / 2.0)
                enriched_clusters.append({
                    "cluster_id": cluster.cluster_id,
                    "member_count": len(members),
                    "unique_files": len(files_in_cluster),
                    "unique_classes": len(classes_in_cluster),
                    "avg_similarity": round(avg_similarity, 4),
                    "avg_distance": round(cluster.avg_distance_to_centroid, 4),
                    "members": members,
                })

        # Sort by potential interest
        enriched_clusters.sort(
            key=lambda c: (c["unique_files"], c["member_count"]),
            reverse=True
        )

        time_ms = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "algorithm": "dbscan",
            "total_clusters": len(enriched_clusters),
            "noise_points": n_noise,
            "total_vectors_analyzed": self._faiss_wrapper.total_vectors,
            "clusters": enriched_clusters,
            "time_ms": time_ms,
            "filters": {
                "eps": eps,
                "min_samples": min_samples,
                "min_cluster_size": min_cluster_size,
                "exclude_same_file": exclude_same_file,
                "exclude_same_class": exclude_same_class,
                "entity_types": entity_types,
                "source_type": source_type,
            },
        }

    def find_duplicate_candidates(
        self,
        similarity_threshold: float = 0.85,
        max_results: int = 50,
        exclude_same_file: bool = True,
        exclude_same_class: bool = True,
        entity_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Find pairs of code entities that are highly similar (potential duplicates).

        Uses pairwise similarity search to find code that may be duplicated
        across different files or classes.

        Args:
            similarity_threshold: Minimum similarity (0-1) to consider as duplicate
            max_results: Maximum number of pairs to return
            exclude_same_file: Exclude pairs from the same file
            exclude_same_class: Exclude pairs from the same class
            entity_types: Filter by entity type (e.g., ["method", "function"])

        Returns:
            Dict with pairs of similar code entities

        Raises:
            ComponentNotReadyError: If RAG code index is not ready
        """
        require_rag_code_index()

        start_time = time.time()

        if not self._initialized or self._faiss_wrapper is None:
            return {
                "success": False,
                "error": "RAG index not initialized",
                "pairs": [],
            }

        # Find similar pairs
        # If filtering by entity_types, we need to fetch more pairs since
        # other entity types (e.g., string_literals from log messages) often
        # dominate the top similarity scores in k-NN search
        if entity_types:
            # Fetch a lot more when filtering - other types often fill top slots
            fetch_multiplier = 100
            # Also increase k for k-NN search - with mixed entity types, the
            # top-k neighbors of a method are often dominated by its own
            # string literals (log messages), not other methods
            k_neighbors = 100
        else:
            fetch_multiplier = 5
            k_neighbors = 10

        raw_pairs = self._faiss_wrapper.find_similar_pairs(
            similarity_threshold=similarity_threshold,
            max_pairs=max_results * fetch_multiplier,
            k=k_neighbors,
        )

        # Enrich and filter pairs
        enriched_pairs = []

        for id1, id2, similarity in raw_pairs:
            meta1 = self._metadata.get("vectors", {}).get(str(id1))
            meta2 = self._metadata.get("vectors", {}).get(str(id2))

            if not meta1 or not meta2:
                continue

            # Apply entity type filter
            if entity_types:
                if meta1.get("entity_type") not in entity_types:
                    continue
                if meta2.get("entity_type") not in entity_types:
                    continue

            file1 = meta1.get("file", "")
            file2 = meta2.get("file", "")
            class1 = meta1.get("class_name", "")
            class2 = meta2.get("class_name", "")

            # Normalize paths for comparison (handle / vs \ differences)
            file1_norm = file1.replace("\\", "/")
            file2_norm = file2.replace("\\", "/")

            # Apply exclusion filters
            if exclude_same_file and file1_norm == file2_norm:
                continue
            if exclude_same_class and class1 and class1 == class2:
                continue

            enriched_pairs.append({
                "similarity": round(similarity, 4),
                "entity1": {
                    "name": meta1.get("name", ""),
                    "qualified_name": meta1.get("qualified_name", ""),
                    "entity_type": meta1.get("entity_type", ""),
                    "file": file1,
                    "line": meta1.get("line", 0),
                    "class_name": class1,
                    "docstring_preview": meta1.get("docstring_preview", ""),
                },
                "entity2": {
                    "name": meta2.get("name", ""),
                    "qualified_name": meta2.get("qualified_name", ""),
                    "entity_type": meta2.get("entity_type", ""),
                    "file": file2,
                    "line": meta2.get("line", 0),
                    "class_name": class2,
                    "docstring_preview": meta2.get("docstring_preview", ""),
                },
            })

            if len(enriched_pairs) >= max_results:
                break

        time_ms = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "total_pairs": len(enriched_pairs),
            "pairs": enriched_pairs,
            "time_ms": time_ms,
            "filters": {
                "similarity_threshold": similarity_threshold,
                "exclude_same_file": exclude_same_file,
                "exclude_same_class": exclude_same_class,
                "entity_types": entity_types,
            }
        }

    def analyze_documentation_relevance(
        self,
        min_relevance: float = 0.5,
        max_results: int = 100,
        doc_entity_types: Optional[List[str]] = None,
        code_entity_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze how relevant documentation is to actual code.

        For each documentation chunk (section, code_block), find the most similar
        code entities and calculate a relevance score.

        This helps detect:
        - Documentation that doesn't match any code (orphaned docs)
        - Documentation that's closely related to specific code
        - Overall documentation coverage quality

        Args:
            min_relevance: Minimum similarity score to consider "relevant" (0-1)
            max_results: Maximum documentation chunks to analyze
            doc_entity_types: Types of doc entities to analyze (default: section, code_block)
            code_entity_types: Types of code entities to match against (default: class, method, function)

        Returns:
            Dict with:
                - relevant_docs: Docs with high relevance to code
                - orphaned_docs: Docs with low relevance (potentially outdated)
                - stats: Summary statistics

        Raises:
            ComponentNotReadyError: If RAG code or document index is not ready
        """
        # This analysis requires both code and document indexes
        require_rag_code_index()
        require_rag_document_index()

        start_time = time.time()

        if not self._initialized or self._faiss_wrapper is None:
            return {
                "success": False,
                "error": "RAG index not initialized"
            }

        # Default entity types
        if doc_entity_types is None:
            doc_entity_types = ["section", "code_block", "document"]
        if code_entity_types is None:
            code_entity_types = ["class", "method", "function"]

        # Collect all documentation vectors
        doc_vectors = []
        code_vectors = []

        for vid_str, meta in self._metadata.get("vectors", {}).items():
            vid = int(vid_str)
            entity_type = meta.get("entity_type", "")
            source_type = meta.get("source_type", "")

            if source_type == "markdown" and entity_type in doc_entity_types:
                doc_vectors.append((vid, meta))
            elif source_type == "python" and entity_type in code_entity_types:
                code_vectors.append((vid, meta))

        if not doc_vectors:
            return {
                "success": False,
                "error": "No documentation vectors found"
            }
        if not code_vectors:
            return {
                "success": False,
                "error": "No code vectors found"
            }

        # Limit doc vectors to analyze
        doc_vectors = doc_vectors[:max_results]

        # For each doc vector, find closest code vectors
        relevant_docs = []
        orphaned_docs = []

        for doc_id, doc_meta in doc_vectors:
            # Get embedding for this doc
            doc_embedding = self._faiss_wrapper.get_vector(doc_id)
            if doc_embedding is None:
                continue

            # Search for similar code (not docs)
            distances, indices = self._faiss_wrapper.search(
                np.array([doc_embedding]),
                top_k=20  # Get top 20 to filter
            )

            # Filter to only code entities
            best_code_match = None
            best_similarity = 0.0

            for i, idx in enumerate(indices[0]):
                if idx == -1:
                    continue
                match_meta = self._metadata.get("vectors", {}).get(str(idx))
                if not match_meta:
                    continue

                # Only consider code entities
                if match_meta.get("source_type") != "python":
                    continue
                if match_meta.get("entity_type") not in code_entity_types:
                    continue

                # Calculate similarity from inner product distance
                # For normalized vectors with IP metric: distance is in [-1, 1], convert to [0, 1]
                similarity = max(0.0, min(1.0, (distances[0][i] + 1.0) / 2.0))
                if similarity > best_similarity:
                    best_similarity = float(similarity)
                    best_code_match = match_meta

            # Classify as relevant or orphaned
            doc_info = {
                "doc_name": doc_meta.get("name", ""),
                "doc_file": doc_meta.get("file", ""),
                "doc_line": doc_meta.get("line", 0),
                "doc_type": doc_meta.get("entity_type", ""),
                "doc_heading": doc_meta.get("heading", ""),
                "content_preview": doc_meta.get("content_preview", "")[:80],
                "best_code_similarity": float(round(best_similarity, 4)),
            }

            if best_code_match:
                doc_info["best_code_match"] = {
                    "name": best_code_match.get("name", ""),
                    "file": best_code_match.get("file", ""),
                    "line": best_code_match.get("line", 0),
                    "type": best_code_match.get("entity_type", ""),
                    "class_name": best_code_match.get("class_name", ""),
                }

            if best_similarity >= min_relevance:
                relevant_docs.append(doc_info)
            else:
                orphaned_docs.append(doc_info)

        # Sort by similarity
        relevant_docs.sort(key=lambda x: x["best_code_similarity"], reverse=True)
        orphaned_docs.sort(key=lambda x: x["best_code_similarity"], reverse=True)

        # Calculate stats
        all_similarities = [d["best_code_similarity"] for d in relevant_docs + orphaned_docs]
        avg_similarity = sum(all_similarities) / len(all_similarities) if all_similarities else 0

        time_ms = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "relevant_docs": relevant_docs,
            "orphaned_docs": orphaned_docs,
            "stats": {
                "total_docs_analyzed": len(doc_vectors),
                "total_code_entities": len(code_vectors),
                "relevant_count": len(relevant_docs),
                "orphaned_count": len(orphaned_docs),
                "relevance_rate": float(round(len(relevant_docs) / len(doc_vectors), 4)) if doc_vectors else 0.0,
                "avg_similarity": float(round(avg_similarity, 4)),
                "min_relevance_threshold": min_relevance,
            },
            "time_ms": time_ms
        }
