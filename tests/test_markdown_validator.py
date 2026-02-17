"""Tests for MarkdownValidator — markdown with embedded mermaid validation."""

from __future__ import annotations

import pytest

from reter_code.mermaid.markdown_validator import (
    CodeBlock,
    MarkdownValidationResult,
    MarkdownValidator,
    extract_code_fences,
    validate_markdown,
)


# ============================================================
# Code fence extraction
# ============================================================


class TestExtractCodeFences:
    def test_no_fences(self):
        content = "# Hello\n\nSome text\n"
        processed, blocks, errors = extract_code_fences(content)
        assert blocks == []
        assert errors == []
        assert "Hello" in processed

    def test_single_backtick_fence(self):
        content = "before\n```python\nprint('hi')\n```\nafter\n"
        processed, blocks, errors = extract_code_fences(content)
        assert len(blocks) == 1
        assert blocks[0].language == "python"
        assert blocks[0].content == "print('hi')"
        assert blocks[0].start_line == 2
        assert blocks[0].end_line == 4
        assert "___CODEBLOCK_0___" in processed
        assert errors == []

    def test_tilde_fence(self):
        content = "~~~js\nalert(1)\n~~~\n"
        processed, blocks, errors = extract_code_fences(content)
        assert len(blocks) == 1
        assert blocks[0].language == "js"
        assert blocks[0].content == "alert(1)"

    def test_no_language(self):
        content = "```\nplain code\n```\n"
        _, blocks, _ = extract_code_fences(content)
        assert len(blocks) == 1
        assert blocks[0].language is None

    def test_multiple_fences(self):
        content = "# Doc\n```python\nx=1\n```\ntext\n```mermaid\ngraph TD\n  A-->B\n```\n"
        _, blocks, errors = extract_code_fences(content)
        assert len(blocks) == 2
        assert blocks[0].language == "python"
        assert blocks[1].language == "mermaid"
        assert errors == []

    def test_unclosed_fence(self):
        content = "# Doc\n```python\nx=1\nno closing\n"
        _, blocks, errors = extract_code_fences(content)
        assert len(blocks) == 0
        assert len(errors) == 1
        assert "Unclosed" in errors[0].message
        assert errors[0].line == 2

    def test_longer_closing_fence(self):
        content = "````\ncode\n````\n"
        _, blocks, errors = extract_code_fences(content)
        assert len(blocks) == 1
        assert errors == []

    def test_language_case_insensitive(self):
        content = "```Mermaid\ngraph TD\n  A-->B\n```\n"
        _, blocks, _ = extract_code_fences(content)
        assert blocks[0].language == "mermaid"


# ============================================================
# Full markdown validation
# ============================================================


class TestMarkdownValidatorBasic:
    def test_empty_content(self):
        result = validate_markdown("")
        assert not result.valid
        assert any("Empty" in e.message for e in result.errors)

    def test_whitespace_only(self):
        result = validate_markdown("   \n  \n  ")
        assert not result.valid

    def test_plain_markdown_valid(self):
        md = "# Title\n\nSome text here.\n\n- item 1\n- item 2\n"
        result = validate_markdown(md)
        assert result.valid
        assert result.code_blocks == []
        assert result.errors == []

    def test_markdown_with_non_mermaid_code(self):
        md = "# Code\n\n```python\nprint('hello')\n```\n\nMore text.\n"
        result = validate_markdown(md)
        assert result.valid
        assert len(result.code_blocks) == 1
        assert result.code_blocks[0].language == "python"
        assert result.mermaid_results == []

    def test_markdown_no_code_blocks(self):
        md = "Just a paragraph.\n\n> Blockquote\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"
        result = validate_markdown(md)
        assert result.valid


class TestMarkdownMermaidValidation:
    def test_valid_mermaid_block(self):
        md = "# Diagram\n\n```mermaid\ngraph TD\n  A-->B\n```\n\nDone.\n"
        result = validate_markdown(md)
        assert result.valid
        assert len(result.code_blocks) == 1
        assert len(result.mermaid_results) == 1
        assert result.mermaid_results[0].valid

    def test_invalid_mermaid_block(self):
        md = "# Bad diagram\n\n```mermaid\ngraph INVALID_DIRECTION\n  A-->>\n```\n"
        result = validate_markdown(md)
        assert not result.valid
        assert len(result.errors) > 0
        # Errors should have adjusted line numbers
        for err in result.errors:
            assert err.line >= 3  # at or after the opening fence line

    def test_multiple_mermaid_one_invalid(self):
        md = (
            "# Good\n\n"
            "```mermaid\ngraph TD\n  A-->B\n```\n\n"
            "# Bad\n\n"
            "```mermaid\ngraph ZZZZZ\n  bad syntax @@@@\n```\n"
        )
        result = validate_markdown(md)
        assert not result.valid
        assert len(result.mermaid_results) == 2
        assert result.mermaid_results[0].valid
        assert not result.mermaid_results[1].valid

    def test_known_unsupported_mermaid_type(self):
        md = "# Chart\n\n```mermaid\nsankey-beta\n```\n"
        result = validate_markdown(md)
        # Known-unsupported types pass through with a warning
        assert result.valid
        assert len(result.warnings) > 0

    def test_unclosed_fence_makes_invalid(self):
        md = "# Start\n\n```mermaid\ngraph TD\n  A-->B\n"
        result = validate_markdown(md)
        assert not result.valid
        assert any("Unclosed" in e.message for e in result.errors)

    def test_mermaid_error_line_adjustment(self):
        # The mermaid block starts at line 5 in the markdown
        md = "# Title\n\nSome text\n\n```mermaid\ngraph INVALID\n  A-->>\n```\n"
        result = validate_markdown(md)
        assert not result.valid
        for err in result.errors:
            # Error lines should be >= 5 (the opening fence line)
            assert err.line >= 5

    def test_mixed_code_blocks(self):
        md = (
            "```python\nprint('hi')\n```\n\n"
            "```mermaid\ngraph TD\n  A-->B\n```\n\n"
            "```js\nconsole.log(1)\n```\n"
        )
        result = validate_markdown(md)
        assert result.valid
        assert len(result.code_blocks) == 3
        assert len(result.mermaid_results) == 1


# ============================================================
# Serialization
# ============================================================


class TestToDict:
    def test_valid_result(self):
        md = "# Hello\n\n```mermaid\ngraph TD\n  A-->B\n```\n"
        result = validate_markdown(md)
        d = result.to_dict()
        assert d["valid"] is True
        assert d["code_blocks"] == 1
        assert "mermaid_blocks" in d

    def test_invalid_result(self):
        result = validate_markdown("")
        d = result.to_dict()
        assert d["valid"] is False
        assert "errors" in d
        assert len(d["errors"]) > 0

    def test_no_mermaid_blocks_omitted(self):
        md = "# Hello\n\nJust text.\n"
        d = validate_markdown(md).to_dict()
        assert "mermaid_blocks" not in d

    def test_warnings_included(self):
        md = "```mermaid\nsankey-beta\n```\n"
        d = validate_markdown(md).to_dict()
        assert d["valid"] is True
        assert "warnings" in d


# ============================================================
# Singleton
# ============================================================


class TestSingleton:
    def test_same_instance(self):
        v1 = MarkdownValidator()
        v2 = MarkdownValidator()
        assert v1 is v2
