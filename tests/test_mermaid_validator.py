"""Tests for the mermaid syntax validator."""

import pytest

from reter_code.mermaid.validator import (
    MermaidValidationResult,
    MermaidValidator,
    validate_mermaid,
)


@pytest.fixture(autouse=True)
def reset_validator():
    """Reset singleton before each test to ensure clean state."""
    MermaidValidator._instance = None
    yield
    MermaidValidator._instance = None


# ============================================================
# VALID INPUTS — one per diagram type
# ============================================================


class TestValidFlowchart:
    def test_basic_flowchart(self):
        r = validate_mermaid("flowchart TB\n    A --> B\n")
        assert r.valid
        assert r.diagram_type == "flowchart"

    def test_flowchart_with_labels(self):
        r = validate_mermaid(
            'flowchart LR\n'
            '    A["Start"] --> B["Process"]\n'
            '    B --> C["End"]\n'
        )
        assert r.valid

    def test_graph_alias(self):
        r = validate_mermaid("graph TD\n    A --> B\n")
        assert r.valid
        assert r.diagram_type == "flowchart"

    def test_flowchart_with_style(self):
        r = validate_mermaid(
            "flowchart TB\n"
            "    A --> B\n"
            "    style A fill:#f9f,stroke:#333\n"
        )
        assert r.valid

    def test_flowchart_with_subgraph(self):
        r = validate_mermaid(
            "flowchart TB\n"
            "    subgraph SG1\n"
            "        A --> B\n"
            "    end\n"
        )
        assert r.valid

    def test_flowchart_shapes(self):
        r = validate_mermaid(
            'flowchart TB\n'
            '    A["rect"] --> B("rounded")\n'
            '    B --> C{{"hexagon"}}\n'
        )
        assert r.valid

    def test_flowchart_class_def(self):
        r = validate_mermaid(
            "flowchart TB\n"
            "    classDef green fill:#9f6,stroke:#333\n"
            "    A --> B\n"
        )
        assert r.valid


class TestValidSequence:
    def test_basic_sequence(self):
        r = validate_mermaid(
            "sequenceDiagram\n"
            "    participant A as Alice\n"
            "    participant B as Bob\n"
            "    A->>+B: Hello\n"
            "    B-->>-A: Hi\n"
        )
        assert r.valid
        assert r.diagram_type == "sequence"

    def test_sequence_with_loop(self):
        r = validate_mermaid(
            "sequenceDiagram\n"
            "    A->>B: Request\n"
            "    loop Every minute\n"
            "        B->>A: Heartbeat\n"
            "    end\n"
        )
        assert r.valid

    def test_sequence_with_note(self):
        r = validate_mermaid(
            "sequenceDiagram\n"
            "    participant A\n"
            "    participant B\n"
            "    Note over A,B: Both participate\n"
            "    A->>B: Message\n"
        )
        assert r.valid

    def test_sequence_with_alt(self):
        r = validate_mermaid(
            "sequenceDiagram\n"
            "    A->>B: Request\n"
            "    alt Success\n"
            "        B->>A: 200 OK\n"
            "    else Failure\n"
            "        B->>A: 500 Error\n"
            "    end\n"
        )
        assert r.valid


class TestValidClass:
    def test_basic_class(self):
        r = validate_mermaid(
            "classDiagram\n"
            "    class Animal {\n"
            "        +name\n"
            "        +speak()\n"
            "    }\n"
        )
        assert r.valid
        assert r.diagram_type == "class"

    def test_class_relationships(self):
        r = validate_mermaid(
            "classDiagram\n"
            "    Animal <|-- Dog\n"
            "    Animal <|-- Cat\n"
            "    Dog *-- Leg\n"
        )
        assert r.valid

    def test_class_with_annotation(self):
        r = validate_mermaid(
            "classDiagram\n"
            "    <<interface>> Drawable\n"
            "    class Drawable {\n"
            "        +draw()\n"
            "    }\n"
        )
        assert r.valid

    def test_class_member_line(self):
        r = validate_mermaid(
            "classDiagram\n"
            "    Animal : +name\n"
            "    Animal : +speak()\n"
        )
        assert r.valid


class TestValidState:
    def test_basic_state(self):
        r = validate_mermaid(
            "stateDiagram-v2\n"
            "    [*] --> Still\n"
            "    Still --> Moving\n"
            "    Moving --> [*]\n"
        )
        assert r.valid
        assert r.diagram_type == "state"

    def test_state_with_label(self):
        r = validate_mermaid(
            "stateDiagram-v2\n"
            "    s1 --> s2 : A transition\n"
        )
        assert r.valid

    def test_state_v1(self):
        r = validate_mermaid(
            "stateDiagram\n"
            "    [*] --> Active\n"
            "    Active --> [*]\n"
        )
        assert r.valid


class TestValidER:
    def test_basic_er(self):
        r = validate_mermaid(
            "erDiagram\n"
            "    CUSTOMER\n"
            "    ORDER\n"
        )
        assert r.valid
        assert r.diagram_type == "er"

    def test_er_with_relationship(self):
        r = validate_mermaid(
            "erDiagram\n"
            '    CUSTOMER ||--o{ ORDER : places\n'
        )
        assert r.valid

    def test_er_with_attributes(self):
        r = validate_mermaid(
            "erDiagram\n"
            "    CUSTOMER {\n"
            "        string name\n"
            "        int age\n"
            "    }\n"
        )
        assert r.valid


class TestValidGantt:
    def test_basic_gantt(self):
        r = validate_mermaid(
            "gantt\n"
            "    title My Project\n"
            "    dateFormat YYYY-MM-DD\n"
            "    section Phase 1\n"
            "    Task1 : 2024-01-01, 30d\n"
        )
        assert r.valid
        assert r.diagram_type == "gantt"

    def test_gantt_minimal(self):
        r = validate_mermaid(
            "gantt\n"
            "    dateFormat YYYY-MM-DD\n"
            "    Task1 : 2024-01-01, 2024-02-01\n"
        )
        assert r.valid


class TestValidPie:
    def test_basic_pie(self):
        r = validate_mermaid(
            "pie showData\n"
            '    "Dogs" : 45\n'
            '    "Cats" : 55\n'
        )
        assert r.valid
        assert r.diagram_type == "pie"

    def test_pie_with_title(self):
        r = validate_mermaid(
            "pie showData title Pet Distribution\n"
            '    "Dogs" : 45\n'
            '    "Cats" : 55\n'
        )
        assert r.valid

    def test_pie_no_showdata(self):
        r = validate_mermaid(
            "pie\n"
            '    "A" : 30\n'
            '    "B" : 70\n'
        )
        assert r.valid


class TestValidBlockBeta:
    def test_basic_block_beta(self):
        r = validate_mermaid(
            "block-beta\n"
            "    columns 3\n"
            '    a["Item A"]\n'
            '    b["Item B"]\n'
            '    c["Item C"]\n'
        )
        assert r.valid
        assert r.diagram_type == "block-beta"

    def test_block_beta_with_blocks(self):
        r = validate_mermaid(
            "block-beta\n"
            "    columns 1\n"
            "    block:group1\n"
            "        columns 4\n"
            '        header1["Group 1"]:4\n'
            '        a1["Item"]\n'
            "    end\n"
            "    style group1 fill:#4CAF50,color:#fff\n"
        )
        assert r.valid

    def test_block_beta_with_edges(self):
        r = validate_mermaid(
            "block-beta\n"
            "    columns 2\n"
            '    a["A"]\n'
            '    b["B"]\n'
            "    a --> b\n"
        )
        assert r.valid


class TestValidMindmap:
    def test_basic_mindmap(self):
        r = validate_mermaid(
            "mindmap\n"
            "    root(Central)\n"
            "        Branch1\n"
            "            Leaf1\n"
            "        Branch2\n"
        )
        assert r.valid
        assert r.diagram_type == "mindmap"

    def test_mindmap_with_shapes(self):
        r = validate_mermaid(
            "mindmap\n"
            "    root((Cloud))\n"
            "        [Square]\n"
            "        (Rounded)\n"
        )
        assert r.valid


class TestValidTimeline:
    def test_basic_timeline(self):
        r = validate_mermaid(
            "timeline\n"
            "    title History\n"
            "    section 2020\n"
            "    Event1 : Description\n"
        )
        assert r.valid
        assert r.diagram_type == "timeline"


class TestValidGitgraph:
    def test_basic_gitgraph(self):
        r = validate_mermaid(
            "gitgraph\n"
            "    commit\n"
            "    branch dev\n"
            "    checkout dev\n"
            "    commit\n"
            "    checkout main\n"
            "    merge dev\n"
        )
        assert r.valid
        assert r.diagram_type == "gitgraph"

    def test_gitgraph_with_opts(self):
        r = validate_mermaid(
            "gitgraph\n"
            '    commit id: "initial"\n'
            '    commit tag: "v1.0"\n'
            "    branch feature\n"
            "    commit\n"
        )
        assert r.valid

    def test_gitgraph_capital(self):
        """gitGraph (camelCase) should also work."""
        r = validate_mermaid(
            "gitGraph\n"
            "    commit\n"
        )
        assert r.valid


class TestValidJourney:
    def test_basic_journey(self):
        r = validate_mermaid(
            "journey\n"
            "    title My Journey\n"
            "    section Morning\n"
            "    Wake up: 5: Me\n"
        )
        assert r.valid
        assert r.diagram_type == "journey"


# ============================================================
# RETER-GENERATED PATTERNS — actual output from RenderMermaidStep
# ============================================================


class TestReterGenerated:
    def test_reter_flowchart(self):
        """Pattern from _render_flowchart."""
        r = validate_mermaid(
            'flowchart TB\n'
            '    Module_A["Module_A"] --> Module_B["Module_B"]\n'
            '    Module_A["Module_A"] --> Module_C["Module_C"]\n'
            '    Standalone_D["Standalone_D"]\n'
        )
        assert r.valid

    def test_reter_sequence(self):
        """Pattern from _render_sequence."""
        r = validate_mermaid(
            "sequenceDiagram\n"
            "    participant ClassA as ClassA\n"
            "    participant ClassB as ClassB\n"
            "    ClassA->>+ClassB: call_method\n"
        )
        assert r.valid

    def test_reter_class_diagram(self):
        """Pattern from _render_class."""
        r = validate_mermaid(
            "classDiagram\n"
            "    class Animal {\n"
            "        +name\n"
            "        +speak()\n"
            "    }\n"
            "    class Dog {\n"
            "        +bark()\n"
            "    }\n"
            "    Animal <|-- Dog\n"
            "    Dog *-- Leg\n"
            "    Dog --> Food\n"
        )
        assert r.valid

    def test_reter_gantt(self):
        """Pattern from _render_gantt."""
        r = validate_mermaid(
            "gantt\n"
            "    title Project Timeline\n"
            "    dateFormat YYYY-MM-DD\n"
            "    TaskA : 2024-01-01, 2024-02-01\n"
            "    TaskB : 2024-02-01, 1d\n"
        )
        assert r.valid

    def test_reter_pie(self):
        """Pattern from _render_pie."""
        r = validate_mermaid(
            "pie showData\n"
            '    "Category A" : 42\n'
            '    "Category B" : 58\n'
        )
        assert r.valid

    def test_reter_state(self):
        """Pattern from _render_state."""
        r = validate_mermaid(
            "stateDiagram-v2\n"
            "    StateA --> StateB\n"
            "    StateB --> StateC\n"
        )
        assert r.valid

    def test_reter_er(self):
        """Pattern from _render_er (minimal — just entities)."""
        r = validate_mermaid(
            "erDiagram\n"
            "    CUSTOMER\n"
            "    ORDER\n"
            "    PRODUCT\n"
        )
        assert r.valid

    def test_reter_block_beta(self):
        """Pattern from _render_block_beta."""
        r = validate_mermaid(
            "block-beta\n"
            "    columns 1\n"
            "    block:group_A\n"
            "        columns 4\n"
            '        header_A["Group A (3)"]:4\n'
            '        item_1["item_1"]\n'
            '        item_2["item_2"]\n'
            '        item_3["item_3"]\n'
            "    end\n"
            "    style group_A fill:#4CAF50,color:#fff\n"
            "    style header_A fill:#388E3C,color:#fff,font-weight:bold\n"
        )
        assert r.valid


# ============================================================
# ERROR DETECTION
# ============================================================


class TestErrorDetection:
    def test_empty_content(self):
        r = validate_mermaid("")
        assert not r.valid
        assert r.errors[0].message == "Empty content"

    def test_whitespace_only(self):
        r = validate_mermaid("   \n  \n  ")
        assert not r.valid
        assert r.errors[0].message == "Empty content"

    def test_unrecognized_type(self):
        r = validate_mermaid("foobar\n    stuff\n")
        assert not r.valid
        assert "Unrecognized diagram type" in r.errors[0].message

    def test_bad_flowchart_direction(self):
        r = validate_mermaid("flowchart XY\n    A --> B\n")
        assert not r.valid

    def test_flowchart_unclosed_bracket(self):
        r = validate_mermaid('flowchart TB\n    A["unclosed --> B\n')
        assert not r.valid

    def test_class_missing_brace(self):
        r = validate_mermaid(
            "classDiagram\n"
            "    class Animal {\n"
            "        +name\n"
        )
        assert not r.valid

    def test_sequence_bad_arrow(self):
        """Completely invalid syntax in sequence body."""
        r = validate_mermaid(
            "sequenceDiagram\n"
            "    A >>>>>> B: Hello\n"
        )
        assert not r.valid

    def test_pie_missing_value(self):
        r = validate_mermaid(
            "pie\n"
            '    "Dogs" :\n'
        )
        assert not r.valid

    def test_gantt_bad_task(self):
        """Task line without colon separator."""
        r = validate_mermaid(
            "gantt\n"
            "    dateFormat YYYY-MM-DD\n"
            "    123badtask 2024-01-01\n"
        )
        assert not r.valid

    def test_block_beta_missing_end(self):
        r = validate_mermaid(
            "block-beta\n"
            "    block:g1\n"
            '        a["item"]\n'
        )
        assert not r.valid

    def test_gitgraph_bad_command(self):
        """Invalid command in gitgraph body."""
        r = validate_mermaid(
            "gitgraph\n"
            "    push origin main\n"
        )
        assert not r.valid

    def test_state_bad_arrow(self):
        r = validate_mermaid(
            "stateDiagram-v2\n"
            "    s1 ==> s2\n"
        )
        assert not r.valid

    def test_er_bad_relationship(self):
        r = validate_mermaid(
            "erDiagram\n"
            "    CUSTOMER >>><<< ORDER : bad\n"
        )
        assert not r.valid


# ============================================================
# PASS-THROUGH (unknown but recognized types)
# ============================================================


class TestPassThrough:
    def test_sankey_passthrough(self):
        r = validate_mermaid("sankey-beta\n    data\n")
        assert r.valid
        assert r.warnings
        assert "not validated" in r.warnings[0]

    def test_xychart_passthrough(self):
        r = validate_mermaid("xychart-beta\n    data\n")
        assert r.valid
        assert r.warnings

    def test_c4context_passthrough(self):
        r = validate_mermaid("c4context\n    data\n")
        assert r.valid
        assert r.warnings

    def test_quadrantchart_passthrough(self):
        r = validate_mermaid("quadrantchart\n    data\n")
        assert r.valid
        assert r.warnings


# ============================================================
# EDGE CASES
# ============================================================


class TestEdgeCases:
    def test_comments_only(self):
        """Comments-only content after header should be valid."""
        r = validate_mermaid(
            "flowchart TB\n"
            "    %% This is a comment\n"
        )
        assert r.valid

    def test_trailing_whitespace(self):
        r = validate_mermaid(
            "flowchart TB  \n"
            "    A --> B  \n"
        )
        assert r.valid

    def test_windows_line_endings(self):
        r = validate_mermaid(
            "flowchart TB\r\n"
            "    A --> B\r\n"
        )
        assert r.valid

    def test_no_trailing_newline(self):
        r = validate_mermaid("flowchart TB\n    A --> B")
        assert r.valid

    def test_to_dict(self):
        r = validate_mermaid("flowchart TB\n    A --> B\n")
        d = r.to_dict()
        assert d["valid"] is True
        assert d["diagram_type"] == "flowchart"

    def test_error_to_dict(self):
        r = validate_mermaid("")
        d = r.to_dict()
        assert d["valid"] is False
        assert "errors" in d
        assert d["errors"][0]["message"] == "Empty content"

    def test_warning_to_dict(self):
        r = validate_mermaid("sankey-beta\n    data\n")
        d = r.to_dict()
        assert d["valid"] is True
        assert "warnings" in d

    def test_validate_mermaid_convenience(self):
        """Module-level function should work same as class method."""
        r = validate_mermaid("pie\n    \"A\" : 50\n    \"B\" : 50\n")
        assert r.valid
        assert r.diagram_type == "pie"

    def test_multiple_validations(self):
        """Singleton should handle multiple validations without issues."""
        r1 = validate_mermaid("flowchart TB\n    A --> B\n")
        r2 = validate_mermaid("pie\n    \"A\" : 50\n")
        r3 = validate_mermaid("gantt\n    title T\n    dateFormat YYYY-MM-DD\n    Task : 2024-01-01, 1d\n")
        assert r1.valid and r2.valid and r3.valid
