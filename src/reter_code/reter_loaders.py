"""
RETER Language Loader Mixins

Contains mixin classes for loading code from various languages:
- Python (load_python_file, load_python_code, load_python_directory)
- JavaScript (load_javascript_file, load_javascript_code, load_javascript_directory)
- HTML (load_html_file, load_html_code, load_html_directory)
- C# (load_csharp_file, load_csharp_code, load_csharp_directory)
- C++ (load_cpp_file, load_cpp_code, load_cpp_directory)

These are extracted from ReterWrapper to reduce file size while maintaining
backward compatibility through re-exports.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set, Callable

from .reter_utils import (
    check_initialization,
    generate_source_id,
    safe_cpp_call,
    extract_in_file_path,
    format_parse_errors,
)


class ReterPythonLoaderMixin:
    """
    Mixin providing Python file/code loading methods.

    Requires:
        - self.reasoner: Reter instance
        - self._session_stats: dict with "total_wmes" key
        - self._dirty: bool flag
        - self._load_directory_generic: directory loading helper
        - self._path_to_module_name: static method for module name calculation

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a loader.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def load_python_file(
        self,
        filepath: str,
        base_path: Optional[str] = None,
        package_roots: Optional[Set[str]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load Python source file and add semantic facts to RETER

        Args:
            filepath: Path to Python file to load
            base_path: Optional base path for calculating relative path (defaults to filepath's parent)
            package_roots: Optional set of Python package root directories (containing __init__.py).
                          If provided, enables proper Python module name calculation.

        Returns:
            Tuple[int, str, float, list]: (wme_count, source_id, time_ms, errors)

        Raises:
            Exception: If loading fails

        Extracts and adds facts about:
        - Classes, methods, functions
        - Inheritance relationships
        - Function calls
        - Imports and dependencies
        - Decorators
        - Parameters and return types

        Source ID is generated as MD5 hash of content | relative path.
        INCREMENTAL: Adds to existing knowledge, doesn't replace.
        """
        check_initialization()
        start_time = time.time()

        # Read file content
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()

        # Calculate relative path
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name

        # Generate MD5-based source ID
        source_id = generate_source_id(code, str(rel_path))

        # Use relative path with forward slashes for inFile (e.g., "path/to/file.py")
        in_file = str(rel_path).replace('\\', '/')

        # Calculate Python module name from relative path
        # With package_roots: respects __init__.py to calculate proper import paths
        # Without: falls back to simple path-to-dots conversion
        module_name = self._path_to_module_name(in_file, package_roots)

        # Load Python code - unified ReteNetwork handles hybrid mode internally
        wme_count, errors = safe_cpp_call(self.reasoner.load_python_code, code, in_file, module_name, source_id)

        self._session_stats["total_wmes"] += wme_count
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # Return errors in the response
        return wme_count, source_id, time_ms, errors

    def load_python_code(
        self,
        code: str,
        source: str = "module",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        package_roots: Optional[Set[str]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load Python code string and add semantic facts to RETER

        Args:
            code: Python source code as string
            source: Source ID for tracking (can be module name, path, or path@timestamp)
            progress_callback: Optional callback function(items_processed, total_items, message)
            package_roots: Optional set of Python package root directories (containing __init__.py).
                          If provided, enables proper Python module name calculation.

        Returns:
            Tuple[int, str, float, list]: (wme_count, source, time_ms, errors)

        Raises:
            Exception: If loading fails

        Same as load_python_file() but takes code as string.
        Useful for loading code snippets or dynamically generated code.

        INCREMENTAL: Adds to existing knowledge.
        """
        check_initialization()
        start_time = time.time()

        # Extract normalized file path from source
        in_file = extract_in_file_path(source)

        # Calculate Python module name from file path
        # With package_roots: respects __init__.py to calculate proper import paths
        # Without: falls back to simple path-to-dots conversion
        module_name = self._path_to_module_name(in_file, package_roots)

        # Load Python code - unified ReteNetwork handles hybrid mode internally
        wme_count, errors = safe_cpp_call(self.reasoner.load_python_code, code, in_file, module_name, source)

        self._session_stats["total_wmes"] += wme_count
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # Return errors in the response
        return wme_count, source, time_ms, errors

    def load_python_directory(
        self,
        directory: str,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """
        Load all Python files from a directory with optional exclusion patterns

        Args:
            directory: Path to directory containing Python files
            recursive: If True, recursively scan subdirectories
            exclude_patterns: List of patterns to exclude (e.g., ["test_*.py", "tests/**/*.py"])
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, dict, float]: (total_wmes, all_errors, time_ms)
            where all_errors is a dict mapping filepath -> list of errors

        Raises:
            Exception: If loading fails

        INCREMENTAL: Adds to existing knowledge.
        """
        return self._load_directory_generic(
            directory=directory,
            extensions=["*.py"],
            default_excludes=["__pycache__"],
            load_file_func=self.load_python_file,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterJavaScriptLoaderMixin:
    """
    Mixin providing JavaScript file/code loading methods.

    Requires:
        - self.reasoner: Reter instance
        - self._dirty: bool flag
        - self._load_directory_generic: directory loading helper

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a loader.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def load_javascript_file(
        self,
        filepath: str,
        base_path: Optional[str] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load JavaScript source file and add semantic facts to RETER

        Args:
            filepath: Path to JavaScript file to load
            base_path: Optional base path for calculating relative path (defaults to filepath's parent)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source_id, time_ms, errors)

        Raises:
            Exception: If loading fails

        Extracts and adds facts about:
        - Classes, methods, functions
        - Inheritance relationships
        - Function calls
        - Imports and exports
        - Arrow functions
        - Parameters

        Source ID is generated as MD5 hash of content | relative path.
        INCREMENTAL: Adds to existing knowledge, doesn't replace.
        """
        check_initialization()
        start_time = time.time()

        # Read file content
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()

        # Calculate relative path
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name

        # Generate MD5-based source ID
        source_id = generate_source_id(code, str(rel_path))

        # Use relative path with forward slashes for inFile (e.g., "path/to/file.js")
        in_file = str(rel_path).replace('\\', '/')

        # Load JavaScript code - use the C++ bindings
        # Returns (facts, errors, registered_methods, unresolved_calls)
        from reter import owl_rete_cpp
        facts, errors, _registered_methods, _unresolved_calls = safe_cpp_call(owl_rete_cpp.parse_javascript_code, code, in_file)

        # Add facts to the network with source tracking
        # Unified ReteNetwork handles hybrid mode internally
        wme_count = 0
        for fact in facts:
            fact_obj = owl_rete_cpp.Fact(fact)
            self.reasoner.network.add_fact_with_source(fact_obj, source_id)
            wme_count += 1

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        return wme_count, source_id, time_ms, format_parse_errors(errors)

    def load_javascript_code(
        self,
        code: str,
        source: str = "module",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load JavaScript code string and add semantic facts to RETER

        Args:
            code: JavaScript source code as string
            source: Source ID for tracking (can be module name, path, or path@timestamp)
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source, time_ms, errors)

        Raises:
            Exception: If loading fails

        Same as load_javascript_file() but takes code as string.
        Useful for loading code snippets or dynamically generated code.

        INCREMENTAL: Adds to existing knowledge.
        """
        check_initialization()
        start_time = time.time()

        # Extract normalized file path from source
        in_file = extract_in_file_path(source)

        # Load JavaScript code - use the C++ bindings (C++ derives module name from in_file)
        # Returns (facts, errors, registered_methods, unresolved_calls)
        from reter import owl_rete_cpp
        facts, errors, _registered_methods, _unresolved_calls = safe_cpp_call(owl_rete_cpp.parse_javascript_code, code, in_file)

        # Add facts to the network with source tracking
        # Unified ReteNetwork handles hybrid mode internally
        wme_count = 0
        for fact in facts:
            fact_obj = owl_rete_cpp.Fact(fact)
            self.reasoner.network.add_fact_with_source(fact_obj, source)
            wme_count += 1

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        return wme_count, source, time_ms, format_parse_errors(errors)

    def load_javascript_directory(
        self,
        directory: str,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """
        Load all JavaScript files from a directory with optional exclusion patterns

        Args:
            directory: Path to directory containing JavaScript files
            recursive: If True, recursively scan subdirectories
            exclude_patterns: List of patterns to exclude (e.g., ["test_*.js", "node_modules/**/*.js"])
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, dict, float]: (total_wmes, all_errors, time_ms)
            where all_errors is a dict mapping filepath -> list of errors

        Raises:
            Exception: If loading fails

        INCREMENTAL: Adds to existing knowledge.
        """
        return self._load_directory_generic(
            directory=directory,
            extensions=["*.js", "*.mjs", "*.jsx"],
            default_excludes=["node_modules"],
            load_file_func=self.load_javascript_file,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterHTMLLoaderMixin:
    """
    Mixin providing HTML file/code loading methods.

    Requires:
        - self.reasoner: Reter instance
        - self._dirty: bool flag
        - self._load_directory_generic: directory loading helper

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a loader.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def load_html_file(
        self,
        filepath: str,
        base_path: Optional[str] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load HTML file and add semantic facts to RETER

        Args:
            filepath: Path to HTML file to load
            base_path: Optional base path for calculating relative path (defaults to filepath's parent)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source_id, time_ms, errors)

        Raises:
            Exception: If loading fails

        Extracts and adds facts about:
        - Document structure (title, language, charset)
        - Elements (forms, inputs, links, etc.)
        - Scripts (inline and external references)
        - Event handlers (onclick, onsubmit, etc.)
        - Framework usage (Vue, Angular, HTMX, Alpine)
        - Embedded JavaScript (parsed and extracted)

        Source ID is generated as MD5 hash of content | relative path.
        INCREMENTAL: Adds to existing knowledge, doesn't replace.
        """
        check_initialization()
        start_time = time.time()

        # Read file content
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()

        # Calculate relative path
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name

        # Generate MD5-based source ID
        source_id = generate_source_id(code, str(rel_path))

        # Use relative path with forward slashes for inDocument (e.g., "path/to/file.html")
        in_file = str(rel_path).replace('\\', '/')

        # Load HTML code - use load_html_from_string directly like Python extractor
        from reter import owl_rete_cpp

        # First parse to get errors
        _, errors = safe_cpp_call(owl_rete_cpp.parse_html_code, code, in_file)

        # Load directly into network - unified ReteNetwork handles hybrid mode internally
        wme_count = safe_cpp_call(owl_rete_cpp.load_html_from_string, self.reasoner.network, code, in_file, source_id)

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        return wme_count, source_id, time_ms, format_parse_errors(errors)

    def load_html_code(
        self,
        code: str,
        source: str = "document",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load HTML code string and add semantic facts to RETER

        Args:
            code: HTML source code as string
            source: Source ID for tracking (can be document name, path, or path@timestamp)
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source, time_ms, errors)

        Raises:
            Exception: If loading fails

        Same as load_html_file() but takes code as string.
        Useful for loading HTML snippets or dynamically generated content.

        INCREMENTAL: Adds to existing knowledge.
        """
        check_initialization()
        start_time = time.time()

        # Extract normalized file path from source
        in_file = extract_in_file_path(source)

        # Load HTML code - use load_html_from_string directly (C++ derives module name from in_file)
        from reter import owl_rete_cpp

        # First parse to get errors
        _, errors = safe_cpp_call(owl_rete_cpp.parse_html_code, code, in_file)

        # Load directly into network - unified ReteNetwork handles hybrid mode internally
        wme_count = safe_cpp_call(owl_rete_cpp.load_html_from_string, self.reasoner.network, code, in_file, source)

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        return wme_count, source, time_ms, format_parse_errors(errors)

    def load_html_directory(
        self,
        directory: str,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """
        Load all HTML files from a directory with optional exclusion patterns

        Args:
            directory: Path to directory containing HTML files
            recursive: If True, recursively scan subdirectories
            exclude_patterns: List of patterns to exclude (e.g., ["test_*.html", "node_modules/**/*.html"])
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, dict, float]: (total_wmes, all_errors, time_ms)
            where all_errors is a dict mapping filepath -> list of errors

        Raises:
            Exception: If loading fails

        INCREMENTAL: Adds to existing knowledge.
        """
        return self._load_directory_generic(
            directory=directory,
            extensions=["*.html", "*.htm"],
            default_excludes=["node_modules"],
            load_file_func=self.load_html_file,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterCSharpLoaderMixin:
    """
    Mixin providing C# file/code loading methods.

    Requires:
        - self.reasoner: Reter instance
        - self._dirty: bool flag
        - self._load_directory_generic: directory loading helper

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a loader.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def load_csharp_file(
        self,
        filepath: str,
        base_path: Optional[str] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load C# source file and add semantic facts to RETER

        Args:
            filepath: Path to C# file to load
            base_path: Optional base path for calculating relative path (defaults to filepath's parent)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source_id, time_ms, errors)

        Raises:
            Exception: If loading fails

        Extracts and adds facts about:
        - Classes, interfaces, structs, enums
        - Methods, properties, fields, events
        - Inheritance relationships
        - Method calls
        - Using directives (imports)
        - Attributes (decorators)
        - Parameters and return types
        - Try/catch/finally blocks
        - Throw and return statements

        Source ID is generated as MD5 hash of content | relative path.
        INCREMENTAL: Adds to existing knowledge, doesn't replace.
        """
        check_initialization()
        start_time = time.time()

        # Read file content
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()

        # Calculate relative path
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name

        # Generate MD5-based source ID
        source_id = generate_source_id(code, str(rel_path))

        # Use relative path with forward slashes for inFile (e.g., "path/to/file.cs")
        in_file = str(rel_path).replace('\\', '/')

        # Load C# code - unified ReteNetwork handles hybrid mode internally
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(
            owl_rete_cpp.load_csharp_from_string,
            self.reasoner.network,
            code,
            in_file,
            source_id
        )

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # C# parser doesn't return errors in the same format as Python/JS
        # Return empty list for now
        error_list: List[str] = []

        return wme_count, source_id, time_ms, error_list

    def load_csharp_code(
        self,
        code: str,
        source: str = "namespace",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load C# code string and add semantic facts to RETER

        Args:
            code: C# source code as string
            source: Source ID for tracking (can be namespace name, path, or path@timestamp)
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source, time_ms, errors)

        Raises:
            Exception: If loading fails

        Same as load_csharp_file() but takes code as string.
        Useful for loading code snippets or dynamically generated code.

        INCREMENTAL: Adds to existing knowledge.
        """
        check_initialization()
        start_time = time.time()

        # Extract normalized file path from source
        in_file = extract_in_file_path(source)

        # Load C# code - unified ReteNetwork handles hybrid mode internally
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(
            owl_rete_cpp.load_csharp_from_string,
            self.reasoner.network,
            code,
            in_file,
            source
        )

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # C# parser doesn't return errors in the same format as Python/JS
        return wme_count, source, time_ms, []

    def load_csharp_directory(
        self,
        directory: str,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """
        Load all C# files from a directory with optional exclusion patterns

        Args:
            directory: Path to directory containing C# files
            recursive: If True, recursively scan subdirectories
            exclude_patterns: List of patterns to exclude (e.g., ["*.Designer.cs", "obj/**/*.cs"])
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, dict, float]: (total_wmes, all_errors, time_ms)
            where all_errors is a dict mapping filepath -> list of errors

        Raises:
            Exception: If loading fails

        INCREMENTAL: Adds to existing knowledge.
        """
        return self._load_directory_generic(
            directory=directory,
            extensions=["*.cs"],
            default_excludes=["bin", "obj"],
            load_file_func=self.load_csharp_file,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterCPPLoaderMixin:
    """
    Mixin providing C++ file/code loading methods.

    Requires:
        - self.reasoner: Reter instance
        - self._dirty: bool flag
        - self._load_directory_generic: directory loading helper

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a loader.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def load_cpp_file(
        self,
        filepath: str,
        base_path: Optional[str] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load C++ source file and add semantic facts to RETER

        Args:
            filepath: Path to C++ file to load
            base_path: Optional base path for calculating relative path (defaults to filepath's parent)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source_id, time_ms, errors)

        Raises:
            Exception: If loading fails

        Extracts and adds facts about:
        - Classes, structs, namespaces
        - Methods, functions, constructors, destructors
        - Inheritance relationships
        - Function calls
        - Using directives (imports)
        - Templates
        - Parameters and return types
        - Try/catch/throw blocks
        - Enums and enumerators
        - Literals (for magic number detection)

        Source ID is generated as MD5 hash of content | relative path.
        INCREMENTAL: Adds to existing knowledge, doesn't replace.
        """
        check_initialization()
        start_time = time.time()

        # Read file content
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()

        # Calculate relative path
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name

        # Generate MD5-based source ID
        source_id = generate_source_id(code, str(rel_path))

        # Use relative path with forward slashes for inFile (e.g., "path/to/file.cpp")
        in_file = str(rel_path).replace('\\', '/')

        # Load C++ code - unified ReteNetwork handles hybrid mode internally
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(
            owl_rete_cpp.load_cpp_from_string,
            self.reasoner.network,
            code,
            in_file,
            source_id
        )

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # C++ parser doesn't return errors in the same format as Python/JS
        # Return empty list for now
        error_list: List[str] = []

        return wme_count, source_id, time_ms, error_list

    def load_cpp_code(
        self,
        code: str,
        source: str = "namespace",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load C++ code string and add semantic facts to RETER

        Args:
            code: C++ source code as string
            source: Source ID for tracking (can be namespace name, path, or path@timestamp)
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source, time_ms, errors)

        Raises:
            Exception: If loading fails

        Same as load_cpp_file() but takes code as string.
        Useful for loading code snippets or dynamically generated code.

        INCREMENTAL: Adds to existing knowledge.
        """
        check_initialization()
        start_time = time.time()

        # Extract normalized file path from source
        in_file = extract_in_file_path(source)

        # Load C++ code - unified ReteNetwork handles hybrid mode internally
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(
            owl_rete_cpp.load_cpp_from_string,
            self.reasoner.network,
            code,
            in_file,
            source
        )

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True  # Mark instance as modified

        # C++ parser doesn't return errors in the same format as Python/JS
        return wme_count, source, time_ms, []

    def load_cpp_directory(
        self,
        directory: str,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """
        Load all C++ files from a directory with optional exclusion patterns

        Args:
            directory: Path to directory containing C++ files
            recursive: If True, recursively scan subdirectories
            exclude_patterns: List of patterns to exclude (e.g., ["test_*.cpp", "build/**/*.cpp"])
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, dict, float]: (total_wmes, all_errors, time_ms)
            where all_errors is a dict mapping filepath -> list of errors

        Raises:
            Exception: If loading fails

        INCREMENTAL: Adds to existing knowledge.
        """
        return self._load_directory_generic(
            directory=directory,
            extensions=["*.cpp", "*.cc", "*.cxx", "*.c++", "*.hpp", "*.hh", "*.hxx", "*.h++", "*.h"],
            default_excludes=["CMakeFiles", "build", "cmake-build-"],
            load_file_func=self.load_cpp_file,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterJavaLoaderMixin:
    """
    Mixin providing Java file/code loading methods.

    Requires:
        - self.reasoner: Reter instance
        - self._dirty: bool flag
        - self._load_directory_generic: directory loading helper

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a loader.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def load_java_file(
        self,
        filepath: str,
        base_path: Optional[str] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load Java source file and add semantic facts to RETER

        Args:
            filepath: Path to Java file to load
            base_path: Optional base path for calculating relative path (defaults to filepath's parent)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source_id, time_ms, errors)

        Raises:
            Exception: If loading fails

        Extracts and adds facts about:
        - Classes, interfaces, enums, records
        - Methods, constructors
        - Inheritance and interface implementation
        - Function calls
        - Imports (single, demand, static)
        - Annotations
        - Parameters and return types
        - Try/catch/throw blocks
        - Package declarations

        Source ID is generated as MD5 hash of content | relative path.
        INCREMENTAL: Adds to existing knowledge, doesn't replace.
        """
        check_initialization()
        start_time = time.time()

        # Read file content
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()

        # Calculate relative path
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name

        # Generate MD5-based source ID
        source_id = generate_source_id(code, str(rel_path))

        # Use relative path with forward slashes for inFile
        in_file = str(rel_path).replace('\\', '/')

        # Load Java code
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(
            owl_rete_cpp.load_java_from_string,
            self.reasoner.network,
            code,
            in_file,
            source_id
        )

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True

        error_list: List[str] = []
        return wme_count, source_id, time_ms, error_list

    def load_java_code(
        self,
        code: str,
        source: str = "module",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load Java code string and add semantic facts to RETER

        Args:
            code: Java source code as string
            source: Source ID for tracking (can be module name, path, or path@timestamp)
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source, time_ms, errors)

        Raises:
            Exception: If loading fails

        Same as load_java_file() but takes code as string.
        INCREMENTAL: Adds to existing knowledge.
        """
        check_initialization()
        start_time = time.time()

        # Extract normalized file path from source
        in_file = extract_in_file_path(source)

        # Load Java code
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(
            owl_rete_cpp.load_java_from_string,
            self.reasoner.network,
            code,
            in_file,
            source
        )

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True

        return wme_count, source, time_ms, []

    def load_java_directory(
        self,
        directory: str,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """
        Load all Java files from a directory with optional exclusion patterns

        Args:
            directory: Path to directory containing Java files
            recursive: If True, recursively scan subdirectories
            exclude_patterns: List of patterns to exclude
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, dict, float]: (total_wmes, all_errors, time_ms)
            where all_errors is a dict mapping filepath -> list of errors

        Raises:
            Exception: If loading fails

        INCREMENTAL: Adds to existing knowledge.
        """
        return self._load_directory_generic(
            directory=directory,
            extensions=["*.java"],
            default_excludes=["build", "target", ".gradle", "out"],
            load_file_func=self.load_java_file,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterGoLoaderMixin:
    """
    Mixin providing Go file/code loading methods.

    Requires:
        - self.reasoner: Reter instance
        - self._dirty: bool flag
        - self._load_directory_generic: directory loading helper

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a loader.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def load_go_file(
        self,
        filepath: str,
        base_path: Optional[str] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load Go source file and add semantic facts to RETER

        Args:
            filepath: Path to Go file to load
            base_path: Optional base path for calculating relative path (defaults to filepath's parent)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source_id, time_ms, errors)

        Raises:
            Exception: If loading fails

        Extracts and adds facts about:
        - Structs, interfaces, type definitions, type aliases
        - Functions, methods with receivers
        - Fields (named and embedded)
        - Function calls
        - Imports
        - Parameters and return types
        - Constants and variables
        - Goroutines and defer statements
        - Package declarations

        Source ID is generated as MD5 hash of content | relative path.
        INCREMENTAL: Adds to existing knowledge, doesn't replace.
        """
        check_initialization()
        start_time = time.time()

        # Read file content
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()

        # Calculate relative path
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name

        # Generate MD5-based source ID
        source_id = generate_source_id(code, str(rel_path))

        # Use relative path with forward slashes for inFile
        in_file = str(rel_path).replace('\\', '/')

        # Load Go code
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(
            owl_rete_cpp.load_go_from_string,
            self.reasoner.network,
            code,
            in_file,
            source_id
        )

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True

        error_list: List[str] = []
        return wme_count, source_id, time_ms, error_list

    def load_go_code(
        self,
        code: str,
        source: str = "module",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load Go code string and add semantic facts to RETER

        Args:
            code: Go source code as string
            source: Source ID for tracking (can be module name, path, or path@timestamp)
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source, time_ms, errors)

        Raises:
            Exception: If loading fails

        Same as load_go_file() but takes code as string.
        INCREMENTAL: Adds to existing knowledge.
        """
        check_initialization()
        start_time = time.time()

        # Extract normalized file path from source
        in_file = extract_in_file_path(source)

        # Load Go code
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(
            owl_rete_cpp.load_go_from_string,
            self.reasoner.network,
            code,
            in_file,
            source
        )

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True

        return wme_count, source, time_ms, []

    def load_go_directory(
        self,
        directory: str,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """
        Load all Go files from a directory with optional exclusion patterns

        Args:
            directory: Path to directory containing Go files
            recursive: If True, recursively scan subdirectories
            exclude_patterns: List of patterns to exclude
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, dict, float]: (total_wmes, all_errors, time_ms)
            where all_errors is a dict mapping filepath -> list of errors

        Raises:
            Exception: If loading fails

        INCREMENTAL: Adds to existing knowledge.
        """
        return self._load_directory_generic(
            directory=directory,
            extensions=["*.go"],
            default_excludes=["vendor", "testdata", ".git"],
            load_file_func=self.load_go_file,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterRustLoaderMixin:
    """
    Mixin providing Rust file/code loading methods.

    Requires:
        - self.reasoner: Reter instance
        - self._dirty: bool flag
        - self._load_directory_generic: directory loading helper

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a loader.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def load_rust_file(
        self,
        filepath: str,
        base_path: Optional[str] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load Rust source file and add semantic facts to RETER

        Args:
            filepath: Path to Rust file to load
            base_path: Optional base path for calculating relative path (defaults to filepath's parent)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source_id, time_ms, errors)

        Extracts and adds facts about:
        - Structs, enums, traits, unions, type aliases
        - Functions, methods (in impl blocks)
        - Fields (named and tuple)
        - Function calls and method calls
        - Use declarations and extern crates
        - Parameters and return types
        - Constants, statics, and local variables
        - Closures, match expressions, loops
        - Module declarations

        Source ID is generated as MD5 hash of content | relative path.
        INCREMENTAL: Adds to existing knowledge, doesn't replace.
        """
        check_initialization()
        start_time = time.time()

        # Read file content
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()

        # Calculate relative path
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name

        # Generate MD5-based source ID
        source_id = generate_source_id(code, str(rel_path))

        # Use relative path with forward slashes for inFile
        in_file = str(rel_path).replace('\\', '/')

        # Load Rust code
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(
            owl_rete_cpp.load_rust_from_string,
            self.reasoner.network,
            code,
            in_file,
            source_id
        )

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True

        error_list: List[str] = []
        return wme_count, source_id, time_ms, error_list

    def load_rust_code(
        self,
        code: str,
        source: str = "module",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """
        Load Rust code string and add semantic facts to RETER

        Args:
            code: Rust source code as string
            source: Source ID for tracking (can be module name, path, or path@timestamp)
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, str, float, list]: (wme_count, source, time_ms, errors)

        Same as load_rust_file() but takes code as string.
        INCREMENTAL: Adds to existing knowledge.
        """
        check_initialization()
        start_time = time.time()

        # Extract normalized file path from source
        in_file = extract_in_file_path(source)

        # Load Rust code
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(
            owl_rete_cpp.load_rust_from_string,
            self.reasoner.network,
            code,
            in_file,
            source
        )

        time_ms = (time.time() - start_time) * 1000
        self._dirty = True

        return wme_count, source, time_ms, []

    def load_rust_directory(
        self,
        directory: str,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """
        Load all Rust files from a directory with optional exclusion patterns

        Args:
            directory: Path to directory containing Rust files
            recursive: If True, recursively scan subdirectories
            exclude_patterns: List of patterns to exclude
            progress_callback: Optional callback function(items_processed, total_items, message)

        Returns:
            Tuple[int, dict, float]: (total_wmes, all_errors, time_ms)
            where all_errors is a dict mapping filepath -> list of errors

        INCREMENTAL: Adds to existing knowledge.
        """
        return self._load_directory_generic(
            directory=directory,
            extensions=["*.rs"],
            default_excludes=["target", ".git"],
            load_file_func=self.load_rust_file,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterErlangLoaderMixin:
    """Mixin providing Erlang file/code loading methods."""

    def load_erlang_file(
        self, filepath: str, base_path: Optional[str] = None
    ) -> Tuple[int, str, float, List[str]]:
        """Load Erlang source file and add semantic facts to RETER."""
        check_initialization()
        start_time = time.time()
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name
        source_id = generate_source_id(code, str(rel_path))
        in_file = str(rel_path).replace('\\', '/')
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(owl_rete_cpp.load_erlang_from_string, self.reasoner.network, code, in_file, source_id)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True
        return wme_count, source_id, time_ms, []

    def load_erlang_code(
        self, code: str, source: str = "module",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """Load Erlang code string and add semantic facts to RETER."""
        check_initialization()
        start_time = time.time()
        in_file = extract_in_file_path(source)
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(owl_rete_cpp.load_erlang_from_string, self.reasoner.network, code, in_file, source)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True
        return wme_count, source, time_ms, []

    def load_erlang_directory(
        self, directory: str, recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """Load all Erlang files from a directory."""
        return self._load_directory_generic(
            directory=directory, extensions=["*.erl", "*.hrl"],
            default_excludes=["_build", ".rebar", "deps"],
            load_file_func=self.load_erlang_file,
            recursive=recursive, exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterPHPLoaderMixin:
    """Mixin providing PHP file/code loading methods."""

    def load_php_file(
        self, filepath: str, base_path: Optional[str] = None
    ) -> Tuple[int, str, float, List[str]]:
        """Load PHP source file and add semantic facts to RETER."""
        check_initialization()
        start_time = time.time()
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name
        source_id = generate_source_id(code, str(rel_path))
        in_file = str(rel_path).replace('\\', '/')
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(owl_rete_cpp.load_php_from_string, self.reasoner.network, code, in_file, source_id)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True
        return wme_count, source_id, time_ms, []

    def load_php_code(
        self, code: str, source: str = "module",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """Load PHP code string and add semantic facts to RETER."""
        check_initialization()
        start_time = time.time()
        in_file = extract_in_file_path(source)
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(owl_rete_cpp.load_php_from_string, self.reasoner.network, code, in_file, source)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True
        return wme_count, source, time_ms, []

    def load_php_directory(
        self, directory: str, recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """Load all PHP files from a directory."""
        return self._load_directory_generic(
            directory=directory, extensions=["*.php"],
            default_excludes=["vendor", "node_modules"],
            load_file_func=self.load_php_file,
            recursive=recursive, exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterObjCLoaderMixin:
    """Mixin providing Objective-C file/code loading methods."""

    def load_objc_file(
        self, filepath: str, base_path: Optional[str] = None
    ) -> Tuple[int, str, float, List[str]]:
        """Load Objective-C source file and add semantic facts to RETER."""
        check_initialization()
        start_time = time.time()
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name
        source_id = generate_source_id(code, str(rel_path))
        in_file = str(rel_path).replace('\\', '/')
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(owl_rete_cpp.load_objc_from_string, self.reasoner.network, code, in_file, source_id)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True
        return wme_count, source_id, time_ms, []

    def load_objc_code(
        self, code: str, source: str = "module",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """Load Objective-C code string and add semantic facts to RETER."""
        check_initialization()
        start_time = time.time()
        in_file = extract_in_file_path(source)
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(owl_rete_cpp.load_objc_from_string, self.reasoner.network, code, in_file, source)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True
        return wme_count, source, time_ms, []

    def load_objc_directory(
        self, directory: str, recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """Load all Objective-C files from a directory."""
        return self._load_directory_generic(
            directory=directory, extensions=["*.m", "*.mm"],
            default_excludes=["build", "DerivedData", "Pods"],
            load_file_func=self.load_objc_file,
            recursive=recursive, exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterSwiftLoaderMixin:
    """Mixin providing Swift file/code loading methods."""

    def load_swift_file(
        self, filepath: str, base_path: Optional[str] = None
    ) -> Tuple[int, str, float, List[str]]:
        """Load Swift source file and add semantic facts to RETER."""
        check_initialization()
        start_time = time.time()
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name
        source_id = generate_source_id(code, str(rel_path))
        in_file = str(rel_path).replace('\\', '/')
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(owl_rete_cpp.load_swift_from_string, self.reasoner.network, code, in_file, source_id)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True
        return wme_count, source_id, time_ms, []

    def load_swift_code(
        self, code: str, source: str = "module",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """Load Swift code string and add semantic facts to RETER."""
        check_initialization()
        start_time = time.time()
        in_file = extract_in_file_path(source)
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(owl_rete_cpp.load_swift_from_string, self.reasoner.network, code, in_file, source)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True
        return wme_count, source, time_ms, []

    def load_swift_directory(
        self, directory: str, recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """Load all Swift files from a directory."""
        return self._load_directory_generic(
            directory=directory, extensions=["*.swift"],
            default_excludes=["build", "DerivedData", ".build"],
            load_file_func=self.load_swift_file,
            recursive=recursive, exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterVB6LoaderMixin:
    """Mixin providing Visual Basic 6 file/code loading methods."""

    def load_vb6_file(
        self, filepath: str, base_path: Optional[str] = None
    ) -> Tuple[int, str, float, List[str]]:
        """Load VB6 source file and add semantic facts to RETER."""
        check_initialization()
        start_time = time.time()
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name
        source_id = generate_source_id(code, str(rel_path))
        in_file = str(rel_path).replace('\\', '/')
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(owl_rete_cpp.load_vb6_from_string, self.reasoner.network, code, in_file, source_id)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True
        return wme_count, source_id, time_ms, []

    def load_vb6_code(
        self, code: str, source: str = "module",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """Load VB6 code string and add semantic facts to RETER."""
        check_initialization()
        start_time = time.time()
        in_file = extract_in_file_path(source)
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(owl_rete_cpp.load_vb6_from_string, self.reasoner.network, code, in_file, source)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True
        return wme_count, source, time_ms, []

    def load_vb6_directory(
        self, directory: str, recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """Load all VB6 files from a directory."""
        return self._load_directory_generic(
            directory=directory, extensions=["*.bas", "*.cls", "*.frm"],
            default_excludes=[],
            load_file_func=self.load_vb6_file,
            recursive=recursive, exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterScalaLoaderMixin:
    """Mixin providing Scala file/code loading methods."""

    def load_scala_file(
        self, filepath: str, base_path: Optional[str] = None
    ) -> Tuple[int, str, float, List[str]]:
        """Load Scala source file and add semantic facts to RETER."""
        check_initialization()
        start_time = time.time()
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name
        source_id = generate_source_id(code, str(rel_path))
        in_file = str(rel_path).replace('\\', '/')
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(owl_rete_cpp.load_scala_from_string, self.reasoner.network, code, in_file, source_id)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True
        return wme_count, source_id, time_ms, []

    def load_scala_code(
        self, code: str, source: str = "module",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """Load Scala code string and add semantic facts to RETER."""
        check_initialization()
        start_time = time.time()
        in_file = extract_in_file_path(source)
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(owl_rete_cpp.load_scala_from_string, self.reasoner.network, code, in_file, source)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True
        return wme_count, source, time_ms, []

    def load_scala_directory(
        self, directory: str, recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """Load all Scala files from a directory."""
        return self._load_directory_generic(
            directory=directory, extensions=["*.scala", "*.sc"],
            default_excludes=["target", ".bsp", ".metals"],
            load_file_func=self.load_scala_file,
            recursive=recursive, exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterHaskellLoaderMixin:
    """Mixin providing Haskell file/code loading methods."""

    def load_haskell_file(
        self, filepath: str, base_path: Optional[str] = None
    ) -> Tuple[int, str, float, List[str]]:
        """Load Haskell source file and add semantic facts to RETER."""
        check_initialization()
        start_time = time.time()
        filepath_obj = Path(filepath)
        with open(filepath_obj, 'r', encoding='utf-8') as f:
            code = f.read()
        if base_path:
            try:
                rel_path = filepath_obj.relative_to(Path(base_path))
            except ValueError:
                rel_path = filepath_obj
        else:
            rel_path = filepath_obj.name
        source_id = generate_source_id(code, str(rel_path))
        in_file = str(rel_path).replace('\\', '/')
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(owl_rete_cpp.load_haskell_from_string, self.reasoner.network, code, in_file, source_id)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True
        return wme_count, source_id, time_ms, []

    def load_haskell_code(
        self, code: str, source: str = "module",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, str, float, List[str]]:
        """Load Haskell code string and add semantic facts to RETER."""
        check_initialization()
        start_time = time.time()
        in_file = extract_in_file_path(source)
        from reter import owl_rete_cpp
        wme_count = safe_cpp_call(owl_rete_cpp.load_haskell_from_string, self.reasoner.network, code, in_file, source)
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True
        return wme_count, source, time_ms, []

    def load_haskell_directory(
        self, directory: str, recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """Load all Haskell files from a directory."""
        return self._load_directory_generic(
            directory=directory, extensions=["*.hs", "*.lhs"],
            default_excludes=[".stack-work", "dist-newstyle", "dist"],
            load_file_func=self.load_haskell_file,
            recursive=recursive, exclude_patterns=exclude_patterns,
            progress_callback=progress_callback
        )


class ReterLoaderMixin(
    ReterPythonLoaderMixin,
    ReterJavaScriptLoaderMixin,
    ReterHTMLLoaderMixin,
    ReterCSharpLoaderMixin,
    ReterCPPLoaderMixin,
    ReterJavaLoaderMixin,
    ReterGoLoaderMixin,
    ReterRustLoaderMixin,
    ReterErlangLoaderMixin,
    ReterPHPLoaderMixin,
    ReterObjCLoaderMixin,
    ReterSwiftLoaderMixin,
    ReterVB6LoaderMixin,
    ReterScalaLoaderMixin,
    ReterHaskellLoaderMixin,
):
    """
    Combined mixin providing all language loader methods.

    This single mixin can be used instead of inheriting from all individual mixins.

    ::: This is-in-layer Infrastructure-Layer.
    ::: This is a loader.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """
    pass


__all__ = [
    "ReterPythonLoaderMixin",
    "ReterJavaScriptLoaderMixin",
    "ReterHTMLLoaderMixin",
    "ReterCSharpLoaderMixin",
    "ReterCPPLoaderMixin",
    "ReterJavaLoaderMixin",
    "ReterGoLoaderMixin",
    "ReterRustLoaderMixin",
    "ReterErlangLoaderMixin",
    "ReterPHPLoaderMixin",
    "ReterObjCLoaderMixin",
    "ReterSwiftLoaderMixin",
    "ReterVB6LoaderMixin",
    "ReterScalaLoaderMixin",
    "ReterHaskellLoaderMixin",
    "ReterLoaderMixin",
]
