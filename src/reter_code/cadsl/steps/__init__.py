"""
CADSL Step Classes.

This package contains all pipeline step implementations for CADSL.
Steps are extracted from transformer.py to reduce file size.

Step categories:
- conditional: WhenStep, UnlessStep, BranchStep, CatchStep
- data_flow: CollectStep, NestStep
- graph: GraphCyclesStep, GraphClosureStep, GraphTraverseStep, ParallelStep
- render: RenderTableStep, RenderChartStep
- mermaid: RenderMermaidStep + config dataclasses
- transform: PivotStep, ComputeStep
- join: JoinStep, MergeSource, CrossJoinStep
- similarity: SetSimilarityStep, StringMatchStep
- integration: RagEnrichStep, CreateTaskStep
- io: FetchContentStep, ViewStep, WriteFileStep, PythonStep
"""

from .conditional import (
    WhenStep,
    UnlessStep,
    BranchStep,
    CatchStep,
)
from .graph import (
    GraphCyclesStep,
    GraphClosureStep,
    GraphTraverseStep,
    ParallelStep,
)
from .data_flow import (
    CollectStep,
    NestStep,
)
from .render import (
    RenderTableStep,
    RenderChartStep,
)
from .mermaid import (
    FlowchartConfig,
    SequenceConfig,
    ClassDiagramConfig,
    PieChartConfig,
    StateDiagramConfig,
    ERDiagramConfig,
    BlockBetaConfig,
    MermaidConfig,
    RenderMermaidStep,
)
from .transform import (
    PivotStep,
    ComputeStep,
)
from .join import (
    JoinStep,
    MergeSource,
    CrossJoinStep,
)
from .similarity import (
    SetSimilarityStep,
    StringMatchStep,
)
from .integration import (
    RagEnrichStep,
    CreateTaskStep,
)
from .io import (
    FetchContentStep,
    ViewStep,
    WriteFileStep,
    PythonStep,
)

__all__ = [
    # Conditional steps
    "WhenStep",
    "UnlessStep",
    "BranchStep",
    "CatchStep",
    # Graph steps
    "GraphCyclesStep",
    "GraphClosureStep",
    "GraphTraverseStep",
    "ParallelStep",
    # Data flow steps
    "CollectStep",
    "NestStep",
    # Render steps
    "RenderTableStep",
    "RenderChartStep",
    # Mermaid steps
    "FlowchartConfig",
    "SequenceConfig",
    "ClassDiagramConfig",
    "PieChartConfig",
    "StateDiagramConfig",
    "ERDiagramConfig",
    "BlockBetaConfig",
    "MermaidConfig",
    "RenderMermaidStep",
    # Transform steps
    "PivotStep",
    "ComputeStep",
    # Join steps
    "JoinStep",
    "MergeSource",
    "CrossJoinStep",
    # Similarity steps
    "SetSimilarityStep",
    "StringMatchStep",
    # Integration steps
    "RagEnrichStep",
    "CreateTaskStep",
    # I/O steps
    "FetchContentStep",
    "ViewStep",
    "WriteFileStep",
    "PythonStep",
]
