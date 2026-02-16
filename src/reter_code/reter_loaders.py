"""
RETER Language Loader Mixins

Data-driven loader using LANGUAGE_CONFIGS table.
All languages use owl_rete_cpp.load_LANG_from_string(network, code, in_file, source_id).
Python has one extra param (module_name).
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set, Callable

from .reter_utils import (
    check_initialization,
    generate_source_id,
    safe_cpp_call,
    extract_in_file_path,
)


# Language configuration: (cpp_func_name, extensions, default_excludes)
LANGUAGE_CONFIGS = {
    "python": ("load_python_from_string", ["*.py"], ["__pycache__"]),
    "javascript": ("load_javascript_from_string", ["*.js", "*.mjs", "*.jsx"], ["node_modules"]),
    "html": ("load_html_from_string", ["*.html", "*.htm"], ["node_modules"]),
    "csharp": ("load_csharp_from_string", ["*.cs"], ["bin", "obj"]),
    "c": ("load_cpp_from_string", ["*.c"], ["CMakeFiles", "build", "cmake-build-"]),
    "cpp": ("load_cpp_from_string", ["*.cpp", "*.cc", "*.cxx", "*.c++", "*.hpp", "*.hh", "*.hxx", "*.h++", "*.h"], ["CMakeFiles", "build", "cmake-build-"]),
    "java": ("load_java_from_string", ["*.java"], ["build", "target", ".gradle", "out"]),
    "go": ("load_go_from_string", ["*.go"], ["vendor", "testdata", ".git"]),
    "rust": ("load_rust_from_string", ["*.rs"], ["target", ".git"]),
    "erlang": ("load_erlang_from_string", ["*.erl", "*.hrl"], ["_build", ".rebar", "deps"]),
    "php": ("load_php_from_string", ["*.php"], ["vendor", "node_modules"]),
    "objc": ("load_objc_from_string", ["*.m", "*.mm"], ["build", "DerivedData", "Pods"]),
    "swift": ("load_swift_from_string", ["*.swift"], ["build", "DerivedData", ".build"]),
    "vb6": ("load_vb6_from_string", ["*.bas", "*.cls", "*.frm"], []),
    "scala": ("load_scala_from_string", ["*.scala", "*.sc"], ["target", ".bsp", ".metals"]),
    "haskell": ("load_haskell_from_string", ["*.hs", "*.lhs"], [".stack-work", "dist-newstyle", "dist"]),
    "kotlin": ("load_kotlin_from_string", ["*.kt", "*.kts"], ["build", ".gradle", ".idea"]),
    "r": ("load_r_from_string", ["*.r", "*.R"], ["renv", ".Rproj.user", "packrat"]),
    "ruby": ("load_ruby_from_string", ["*.rb", "*.rake", "*.gemspec"], ["vendor", ".bundle", "tmp"]),
    "dart": ("load_dart_from_string", ["*.dart"], ["build", ".dart_tool", ".pub-cache"]),
    "delphi": ("load_delphi_from_string", ["*.pas", "*.dpr", "*.dpk", "*.inc"], ["__history", "__recovery", "DCU"]),
    "ada": ("load_ada_from_string", ["*.adb", "*.ads", "*.ada"], ["obj", ".alire"]),
    "lua": ("load_lua_from_string", ["*.lua"], ["build", ".luarocks"]),
    "xaml": ("load_xaml_from_string", ["*.xaml"], ["bin", "obj", ".vs"]),
    "bash": ("load_bash_from_string", ["*.sh", "*.bash", "*.zsh", "*.ksh"], [".git"]),
    "eval": ("load_eval_from_string", ["*.eval"], []),
}


class ReterLoaderMixin:
    """
    Combined mixin providing all language loader methods.

    Uses a data-driven approach: LANGUAGE_CONFIGS table maps each language
    to its C++ load function, file extensions, and default directory excludes.
    All languages use the same generic _load_lang_file / _load_lang_code /
    _load_lang_directory methods.

    Requires:
        - self.reasoner: Reter instance (with .network attribute)
        - self._session_stats: dict with "total_wmes" key
        - self._dirty: bool flag
        - self._load_directory_generic: directory loading helper
        - self._path_to_module_name: static method (used by Python only)
    """

    def _load_lang_file(
        self,
        lang: str,
        filepath: str,
        base_path: Optional[str] = None,
        package_roots: Optional[Set[str]] = None,
    ) -> Tuple[int, str, float, List[str]]:
        """Generic file loader for any language."""
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
        load_func = getattr(owl_rete_cpp, LANGUAGE_CONFIGS[lang][0])

        if lang == "python":
            module_name = self._path_to_module_name(in_file, package_roots)
            wme_count = safe_cpp_call(load_func, self.reasoner.network, code, in_file, module_name, source_id)
        else:
            wme_count = safe_cpp_call(load_func, self.reasoner.network, code, in_file, source_id)

        self._session_stats["total_wmes"] += wme_count
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True

        return wme_count, source_id, time_ms, []

    def _load_lang_code(
        self,
        lang: str,
        code: str,
        source: str = "module",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        package_roots: Optional[Set[str]] = None,
    ) -> Tuple[int, str, float, List[str]]:
        """Generic code string loader for any language."""
        check_initialization()
        start_time = time.time()

        in_file = extract_in_file_path(source)

        from reter import owl_rete_cpp
        load_func = getattr(owl_rete_cpp, LANGUAGE_CONFIGS[lang][0])

        if lang == "python":
            module_name = self._path_to_module_name(in_file, package_roots)
            wme_count = safe_cpp_call(load_func, self.reasoner.network, code, in_file, module_name, source)
        else:
            wme_count = safe_cpp_call(load_func, self.reasoner.network, code, in_file, source)

        self._session_stats["total_wmes"] += wme_count
        time_ms = (time.time() - start_time) * 1000
        self._dirty = True

        return wme_count, source, time_ms, []

    def _load_lang_directory(
        self,
        lang: str,
        directory: str,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[int, Dict[str, List[str]], float]:
        """Generic directory loader for any language."""
        config = LANGUAGE_CONFIGS[lang]
        load_file_method = getattr(self, f"load_{lang}_file")
        return self._load_directory_generic(
            directory=directory,
            extensions=config[1],
            default_excludes=config[2],
            load_file_func=load_file_method,
            recursive=recursive,
            exclude_patterns=exclude_patterns,
            progress_callback=progress_callback,
        )

    # --- Python (extra package_roots param) ---

    def load_python_file(self, filepath, base_path=None, package_roots=None):
        return self._load_lang_file("python", filepath, base_path, package_roots)

    def load_python_code(self, code, source="module", progress_callback=None, package_roots=None):
        return self._load_lang_code("python", code, source, progress_callback, package_roots)

    def load_python_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("python", directory, recursive, exclude_patterns, progress_callback)

    # --- JavaScript ---

    def load_javascript_file(self, filepath, base_path=None):
        return self._load_lang_file("javascript", filepath, base_path)

    def load_javascript_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("javascript", code, source, progress_callback)

    def load_javascript_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("javascript", directory, recursive, exclude_patterns, progress_callback)

    # --- HTML ---

    def load_html_file(self, filepath, base_path=None):
        return self._load_lang_file("html", filepath, base_path)

    def load_html_code(self, code, source="document", progress_callback=None):
        return self._load_lang_code("html", code, source, progress_callback)

    def load_html_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("html", directory, recursive, exclude_patterns, progress_callback)

    # --- C# ---

    def load_csharp_file(self, filepath, base_path=None):
        return self._load_lang_file("csharp", filepath, base_path)

    def load_csharp_code(self, code, source="namespace", progress_callback=None):
        return self._load_lang_code("csharp", code, source, progress_callback)

    def load_csharp_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("csharp", directory, recursive, exclude_patterns, progress_callback)

    # --- C ---

    def load_c_file(self, filepath, base_path=None):
        return self._load_lang_file("c", filepath, base_path)

    def load_c_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("c", code, source, progress_callback)

    def load_c_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("c", directory, recursive, exclude_patterns, progress_callback)

    # --- C++ ---

    def load_cpp_file(self, filepath, base_path=None):
        return self._load_lang_file("cpp", filepath, base_path)

    def load_cpp_code(self, code, source="namespace", progress_callback=None):
        return self._load_lang_code("cpp", code, source, progress_callback)

    def load_cpp_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("cpp", directory, recursive, exclude_patterns, progress_callback)

    # --- Java ---

    def load_java_file(self, filepath, base_path=None):
        return self._load_lang_file("java", filepath, base_path)

    def load_java_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("java", code, source, progress_callback)

    def load_java_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("java", directory, recursive, exclude_patterns, progress_callback)

    # --- Go ---

    def load_go_file(self, filepath, base_path=None):
        return self._load_lang_file("go", filepath, base_path)

    def load_go_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("go", code, source, progress_callback)

    def load_go_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("go", directory, recursive, exclude_patterns, progress_callback)

    # --- Rust ---

    def load_rust_file(self, filepath, base_path=None):
        return self._load_lang_file("rust", filepath, base_path)

    def load_rust_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("rust", code, source, progress_callback)

    def load_rust_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("rust", directory, recursive, exclude_patterns, progress_callback)

    # --- Erlang ---

    def load_erlang_file(self, filepath, base_path=None):
        return self._load_lang_file("erlang", filepath, base_path)

    def load_erlang_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("erlang", code, source, progress_callback)

    def load_erlang_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("erlang", directory, recursive, exclude_patterns, progress_callback)

    # --- PHP ---

    def load_php_file(self, filepath, base_path=None):
        return self._load_lang_file("php", filepath, base_path)

    def load_php_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("php", code, source, progress_callback)

    def load_php_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("php", directory, recursive, exclude_patterns, progress_callback)

    # --- Objective-C ---

    def load_objc_file(self, filepath, base_path=None):
        return self._load_lang_file("objc", filepath, base_path)

    def load_objc_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("objc", code, source, progress_callback)

    def load_objc_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("objc", directory, recursive, exclude_patterns, progress_callback)

    # --- Swift ---

    def load_swift_file(self, filepath, base_path=None):
        return self._load_lang_file("swift", filepath, base_path)

    def load_swift_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("swift", code, source, progress_callback)

    def load_swift_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("swift", directory, recursive, exclude_patterns, progress_callback)

    # --- VB6 ---

    def load_vb6_file(self, filepath, base_path=None):
        return self._load_lang_file("vb6", filepath, base_path)

    def load_vb6_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("vb6", code, source, progress_callback)

    def load_vb6_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("vb6", directory, recursive, exclude_patterns, progress_callback)

    # --- Scala ---

    def load_scala_file(self, filepath, base_path=None):
        return self._load_lang_file("scala", filepath, base_path)

    def load_scala_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("scala", code, source, progress_callback)

    def load_scala_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("scala", directory, recursive, exclude_patterns, progress_callback)

    # --- Haskell ---

    def load_haskell_file(self, filepath, base_path=None):
        return self._load_lang_file("haskell", filepath, base_path)

    def load_haskell_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("haskell", code, source, progress_callback)

    def load_haskell_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("haskell", directory, recursive, exclude_patterns, progress_callback)

    # --- Kotlin ---

    def load_kotlin_file(self, filepath, base_path=None):
        return self._load_lang_file("kotlin", filepath, base_path)

    def load_kotlin_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("kotlin", code, source, progress_callback)

    def load_kotlin_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("kotlin", directory, recursive, exclude_patterns, progress_callback)

    # --- R ---

    def load_r_file(self, filepath, base_path=None):
        return self._load_lang_file("r", filepath, base_path)

    def load_r_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("r", code, source, progress_callback)

    def load_r_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("r", directory, recursive, exclude_patterns, progress_callback)

    # --- Ruby ---

    def load_ruby_file(self, filepath, base_path=None):
        return self._load_lang_file("ruby", filepath, base_path)

    def load_ruby_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("ruby", code, source, progress_callback)

    def load_ruby_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("ruby", directory, recursive, exclude_patterns, progress_callback)

    # --- Dart ---

    def load_dart_file(self, filepath, base_path=None):
        return self._load_lang_file("dart", filepath, base_path)

    def load_dart_code(self, code, source="module", progress_callback=None):
        return self._load_lang_code("dart", code, source, progress_callback)

    def load_dart_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("dart", directory, recursive, exclude_patterns, progress_callback)

    # --- Delphi ---

    def load_delphi_file(self, filepath, base_path=None):
        return self._load_lang_file("delphi", filepath, base_path)

    def load_delphi_code(self, code, source="unit", progress_callback=None):
        return self._load_lang_code("delphi", code, source, progress_callback)

    def load_delphi_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("delphi", directory, recursive, exclude_patterns, progress_callback)

    # --- Ada ---

    def load_ada_file(self, filepath, base_path=None):
        return self._load_lang_file("ada", filepath, base_path)

    def load_ada_code(self, code, source="unit", progress_callback=None):
        return self._load_lang_code("ada", code, source, progress_callback)

    def load_ada_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("ada", directory, recursive, exclude_patterns, progress_callback)

    # --- Lua ---

    def load_lua_file(self, filepath, base_path=None):
        return self._load_lang_file("lua", filepath, base_path)

    def load_lua_code(self, code, source="script", progress_callback=None):
        return self._load_lang_code("lua", code, source, progress_callback)

    def load_lua_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("lua", directory, recursive, exclude_patterns, progress_callback)

    # --- XAML ---

    def load_xaml_file(self, filepath, base_path=None):
        return self._load_lang_file("xaml", filepath, base_path)

    def load_xaml_code(self, code, source="document", progress_callback=None):
        return self._load_lang_code("xaml", code, source, progress_callback)

    def load_xaml_directory(self, directory, recursive=True, exclude_patterns=None, progress_callback=None):
        return self._load_lang_directory("xaml", directory, recursive, exclude_patterns, progress_callback)


__all__ = [
    "ReterLoaderMixin",
    "LANGUAGE_CONFIGS",
]
