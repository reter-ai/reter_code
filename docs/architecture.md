# Reter Code Architecture

Reter Code is an AI-powered MCP (Model Context Protocol) server for code reasoning, built on the RETER (RETE Rule Engine) for forward-chaining inference and FAISS for semantic vector search.

## Overview

| Metric | Count |
|--------|-------|
| Python Files | 96 |
| Classes | 194 |
| Methods | 1,108 |
| Functions | 81 |
| RAG Vectors | 7,218 |

## Layered Architecture

```
+-------------------------------------------------------------+
|                      SERVER LAYER                           |
|   server.py (Reter CodeServer) - FastMCP-based entry point    |
|   Background initialization, tool registration              |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                      SERVICES LAYER                         |
|   RAGIndexManager - Vector search with FAISS               |
|   EmbeddingService - Sentence transformers                 |
|   ReterWrapper - RETE rule engine interface                |
|   DefaultInstanceManager - Project file sync               |
|   StatePersistence - State save/load                       |
|   ConfigLoader - Configuration management                  |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                       TOOLS LAYER                           |
+----------------+----------------+---------------------------+
| python_advanced|      uml       |       refactoring        |
| - Architecture | - ClassDiagram | - RefactoringTool        |
| - ChangeImpact | - CallGraph    | - RefactoringToPatterns  |
| - CodeQuality  | - Sequence     | - TestCoverageTool       |
| - DataClump    | - Dependencies |                          |
| - Refactoring  | - Coupling     |        unified           |
| - TypeAnalysis | - Hierarchy    | - ThinkingSession        |
| - TestAnalysis |                | - UnifiedStore           |
+----------------+----------------+---------------------------+
                              |
+-------------------------------------------------------------+
|                      MODELS LAYER                           |
|   Pydantic models: LogicalThought, ThinkingRequest, WME,   |
|   QueryOutput, AddKnowledgeInput/Output, etc.              |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                     STORAGE LAYER                           |
|   SQLiteSessionStore - Session persistence                 |
|   FAISSWrapper - Vector index storage                      |
+-------------------------------------------------------------+
```

## Key Components

### Reter CodeServer (`server.py`)

The main entry point that:
- Initializes FastMCP server with Anthropic sampling handler
- Orchestrates background initialization of RETER and RAG
- Registers all MCP tools via registrars
- Handles graceful shutdown with state persistence

### Service Classes

| Service | Responsibility |
|---------|---------------|
| `ReterWrapper` | Wraps C++ RETE rule engine for forward-chaining inference |
| `RAGIndexManager` | Manages FAISS vector index for semantic code search |
| `EmbeddingService` | Generates embeddings using sentence-transformers |
| `DefaultInstanceManager` | Auto-syncs project files based on MD5 changes |
| `StatePersistenceService` | Saves/loads RETER instance state |
| `InstanceManager` | Manages multiple RETER instances |

### Tool Registrars

Tools are organized into registrar classes:

| Registrar | Tools Registered |
|-----------|-----------------|
| `UnifiedToolsRegistrar` | `thinking`, `session`, `items`, `project`, `diagram` |
| `CodeInspectionToolsRegistrar` | `code_inspection` (26 actions) |
| `RAGToolsRegistrar` | `semantic_search`, `rag_status`, `rag_reindex`, `init_status`, `find_similar_clusters`, `find_duplicate_candidates`, `analyze_documentation_relevance` |
| `RecommenderToolsRegistrar` | `recommender` (58 detectors) |
| `ToolRegistrar` | `add_knowledge`, `quick_query`, `instance_manager`, `natural_language_query`, `reter_info`, `add_external_directory` |

## Class Hierarchies

### Tool System

```
BaseTool (ABC)
+-- UMLTool
+-- RefactoringToolBase
|   +-- RefactoringTool
|   +-- RefactoringToPatternsTool
+-- TestCoverageTool
+-- DocumentationMaintenanceTool

AdvancedToolsBase
+-- AdvancedPythonTools
+-- ArchitectureAnalysisTools
+-- ChangeImpactTools
+-- CodeQualityTools
+-- DataClumpDetectionTools
+-- DependencyAnalysisTools
+-- ExceptionAnalysisTools
+-- FunctionAnalysisTools
+-- InheritanceRefactoringTools
+-- PatternDetectionTools
+-- RefactoringOpportunityDetector
+-- TestAnalysisTools
+-- TypeAnalysisTools
+-- AdvancedPythonToolsFacade
```

### UML Generators

```
UMLGeneratorBase
+-- CallGraphGenerator
+-- ClassDiagramGenerator
+-- ClassHierarchyGenerator
+-- CouplingMatrixGenerator
+-- DependencyGraphGenerator
+-- SequenceDiagramGenerator
```

### Tool Registrars

```
ToolRegistrarBase
+-- CodeInspectionToolsRegistrar
+-- RecommenderToolsRegistrar
+-- UnifiedToolsRegistrar
+-- RAGToolsRegistrar
+-- ToolsRegistrar
```

### Error Hierarchy

```
ReterError (Exception)
+-- ReterFileError
|   +-- ReterFileNotFoundError
|   +-- ReterSaveError
|   +-- ReterLoadError
+-- ReterOntologyError
+-- ReterQueryError
+-- DefaultInstanceNotInitialised
```

## Design Patterns

| Pattern | Usage |
|---------|-------|
| **Facade** | `AdvancedPythonToolsFacade` - unifies 12+ analysis tools |
| **Strategy** | Tool registrars for different tool categories |
| **Template Method** | `RefactoringToolBase` for refactoring detection |
| **Abstract Factory** | `BaseTool` with concrete implementations |
| **Singleton-like** | Default RETER instance management |

## Initialization Flow

1. **Server Start**: `Reter CodeServer.__init__()` loads config, creates services
2. **Background Init**: After 2s delay, `_async_initialize()` runs in background
3. **RETER Loading**: Python files loaded into RETER knowledge graph
4. **RAG Indexing**: Code entities indexed in FAISS for semantic search
5. **Ready State**: All components marked ready, tools become available

## Configuration

Configuration is loaded from `reter_code.json` (or `reter.json`) in project root:

```json
{
  "project_root": "/path/to/project",
  "include_patterns": ["src/**/*.py"],
  "exclude_patterns": ["**/test_*.py", "**/__pycache__/**"],
  "rag_enabled": true,
  "rag_embedding_model": "sentence-transformers/all-mpnet-base-v2"
}
```

Environment variables:
- `RETER_PROJECT_ROOT`: Project directory to analyze
- `ANTHROPIC_API_KEY`: For LLM sampling (natural language queries)
- `TRANSFORMERS_CACHE`: Cache directory for embedding models
