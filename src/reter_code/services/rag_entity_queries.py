"""
RAG Entity Queries Mixin

Contains methods for querying RETER for entities to index in the RAG system.
Extracted from RAGIndexManager to reduce file size.
"""

from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..reter_wrapper import ReterWrapper

from ..logging_config import configure_logger_for_debug_trace

logger = configure_logger_for_debug_trace(__name__)


class RAGEntityQueryMixin:
    """
    Mixin providing entity querying methods for RAGIndexManager.

    ::: This is-in-layer Service-Layer.
    ::: This is-part-of-component RAG-Index.
    ::: This is-in-process Main-Process.
    ::: This is stateless.

    These methods require:
    - self._reter: ReterWrapper instance
    - self._metadata: dict mapping vector IDs to entity metadata
    """

    def _query_entities_for_source(
        self,
        reter: "ReterWrapper",
        source_id: str,
        language: str = "python"
    ) -> List[Dict[str, Any]]:
        """
        Query RETER for all indexable entities in a source.

        Args:
            reter: RETER wrapper instance
            source_id: Source ID (format: "md5|rel_path")
            language: "python", "javascript", "csharp", or "cpp"

        Returns entities with their metadata (type, name, line numbers, docstring).
        """
        entities = []

        # Determine concept prefix based on language
        _lang_to_prefix = {
            "python": "py", "javascript": "js", "csharp": "cs", "cpp": "cpp",
            "java": "java", "go": "go", "rust": "rust", "erlang": "erlang",
            "php": "php", "objc": "objc", "swift": "swift", "vb6": "vb6",
            "scala": "scala", "haskell": "haskell", "kotlin": "kotlin", "r": "r", "ruby": "ruby", "dart": "dart", "delphi": "delphi",
        }
        prefix = _lang_to_prefix.get(language, "py")

        # Convert source_id to is-in-file format
        # source_id format: "md5hash|path\\to\\file.py" or "md5hash|path\\to\\file.js" or "md5hash|path\\to\\file.cs"
        # is-in-file format: "path.to.file.ext" (all languages keep extension)
        if "|" in source_id:
            _, rel_path = source_id.split("|", 1)
        else:
            rel_path = source_id

        # Convert backslashes to dots (keep extension)
        in_file = rel_path.replace("\\", "/")

        logger.debug(f"[RAG] _query_entities: source_id={source_id}, rel_path={rel_path}, in_file={in_file}, language={language}")

        # Query classes
        class_query = f'''
        SELECT DISTINCT ?entity ?name ?line ?endLine ?docstring
        WHERE {{
            ?entity is-in-file "{in_file}" .
            ?entity type class .
            ?entity has-name ?name .
            ?entity is-at-line ?line .
            ?entity has-end-line ?endLine .
            OPTIONAL {{ ?entity has-documentation ?docstring }}
        }}
        '''
        try:
            logger.debug(f"[RAG] _query_entities: Executing class query for in_file={in_file}")
            class_table = reter.reql(class_query)
            if class_table is not None and class_table.num_rows > 0:
                class_results = class_table.to_pylist()  # PyArrow built-in: list of dicts
                logger.debug(f"[RAG] _query_entities: Class query returned {len(class_results)} results")
                for row in class_results:
                    line = int(row.get("?line", 0))
                    end_line_val = row.get("?endLine")
                    end_line = int(end_line_val) if end_line_val else None
                    entities.append({
                        "entity_type": "class",
                        "name": row.get("?name", ""),
                        "qualified_name": row.get("?entity", ""),
                        "line": line,
                        "end_line": end_line,
                        "docstring": row.get("?docstring"),
                    })
            else:
                logger.debug(f"[RAG] _query_entities: Class query returned 0 results")
        except Exception as e:
            logger.debug(f"[RAG] _query_entities: Class query FAILED for {source_id}: {e}")
            logger.debug(f"Class query failed for {source_id}: {e}")

        # Query methods
        # Note: is-defined-in is required (not OPTIONAL) because REQL OPTIONAL doesn't return bound values
        method_query = f'''
        SELECT DISTINCT ?entity ?name ?line ?endLine ?docstring ?className
        WHERE {{
            ?entity is-in-file "{in_file}" .
            ?entity type method .
            ?entity has-name ?name .
            ?entity is-at-line ?line .
            ?entity has-end-line ?endLine .
            ?entity is-defined-in ?className .
            OPTIONAL {{ ?entity has-documentation ?docstring }}
        }}
        '''
        try:
            logger.debug(f"[RAG] _query_entities: Executing method query for in_file={in_file}")
            method_table = reter.reql(method_query)
            if method_table is not None and method_table.num_rows > 0:
                method_results = method_table.to_pylist()
                logger.debug(f"[RAG] _query_entities: Method query returned {len(method_results)} results")
                for row in method_results:
                    line = int(row.get("?line", 0))
                    end_line_val = row.get("?endLine")
                    end_line = int(end_line_val) if end_line_val else None
                    class_name = row.get("?className", "")
                    entities.append({
                        "entity_type": "method",
                        "name": row.get("?name", ""),
                        "qualified_name": row.get("?entity", ""),
                        "line": line,
                        "end_line": end_line,
                        "docstring": row.get("?docstring"),
                        "class_name": class_name,
                    })
            else:
                logger.debug(f"[RAG] _query_entities: Method query returned 0 results")
        except Exception as e:
            logger.debug(f"[RAG] _query_entities: Method query FAILED for {source_id}: {e}")
            logger.debug(f"Method query failed for {source_id}: {e}")

        # Query functions (excluding methods - methods are already indexed separately)
        # Note: method is_subclass_of function, so we need to exclude methods
        func_query = f'''
        SELECT DISTINCT ?entity ?name ?line ?endLine ?docstring
        WHERE {{
            ?entity is-in-file "{in_file}" .
            ?entity type function .
            ?entity has-name ?name .
            ?entity is-at-line ?line .
            ?entity has-end-line ?endLine .
            OPTIONAL {{ ?entity has-documentation ?docstring }}
            FILTER(NOT EXISTS {{ ?entity type method }})
        }}
        '''
        try:
            logger.debug(f"[RAG] _query_entities: Executing function query for in_file={in_file}")
            func_table = reter.reql(func_query)
            if func_table is not None and func_table.num_rows > 0:
                func_results = func_table.to_pylist()
                logger.debug(f"[RAG] _query_entities: Function query returned {len(func_results)} results")
                for row in func_results:
                    line = int(row.get("?line", 0))
                    end_line_val = row.get("?endLine")
                    end_line = int(end_line_val) if end_line_val else None
                    entities.append({
                        "entity_type": "function",
                        "name": row.get("?name", ""),
                        "qualified_name": row.get("?entity", ""),
                        "line": line,
                        "end_line": end_line,
                        "docstring": row.get("?docstring"),
                    })
            else:
                logger.debug(f"[RAG] _query_entities: Function query returned 0 results")
        except Exception as e:
            logger.debug(f"[RAG] _query_entities: Function query FAILED for {source_id}: {e}")
            logger.debug(f"Function query failed for {source_id}: {e}")

        logger.debug(f"[RAG] _query_entities: Total entities found for {source_id}: {len(entities)}")
        return entities

    def _query_html_entities_for_source(
        self,
        reter: "ReterWrapper",
        source_id: str
    ) -> List[Dict[str, Any]]:
        """
        Query RETER for all indexable HTML entities in a source.

        Args:
            reter: RETER wrapper instance
            source_id: Source ID (format: "md5|rel_path")

        Returns entities with their metadata (type, name, line numbers, content).
        """
        entities = []

        # Convert source_id to is-in-file format
        if "|" in source_id:
            _, rel_path = source_id.split("|", 1)
        else:
            rel_path = source_id

        # Convert backslashes to dots
        in_file = rel_path.replace("\\", "/")

        logger.debug(f"[RAG] _query_html_entities: source_id={source_id}, in_file={in_file}")

        # Query scripts (inline JavaScript)
        script_query = f'''
        SELECT DISTINCT ?entity ?name ?line ?content
        WHERE {{
            ?entity is-in-document "{in_file}" .
            ?entity type script .
            OPTIONAL {{ ?entity has-name ?name }}
            OPTIONAL {{ ?entity is-at-line ?line }}
            OPTIONAL {{ ?entity has-content ?content }}
        }}
        '''
        try:
            script_table = reter.reql(script_query)
            if script_table is not None and script_table.num_rows > 0:
                for row in script_table.to_pylist():
                    entities.append({
                        "entity_type": "script",
                        "name": row.get("?name", "inline_script"),
                        "line": int(row.get("?line", 0)),
                        "content": row.get("?content", ""),
                        "source_type": "html",
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_html_entities: Script query failed: {e}")

        # Query event handlers
        handler_query = f'''
        SELECT DISTINCT ?entity ?event ?handler ?line
        WHERE {{
            ?entity is-in-document "{in_file}" .
            ?entity type event-handler .
            OPTIONAL {{ ?entity has-event-name ?event }}
            OPTIONAL {{ ?entity has-handler-code ?handler }}
            OPTIONAL {{ ?entity is-at-line ?line }}
        }}
        '''
        try:
            handler_table = reter.reql(handler_query)
            if handler_table is not None and handler_table.num_rows > 0:
                for row in handler_table.to_pylist():
                    event = row.get("?event", "unknown")
                    handler = row.get("?handler", "")
                    entities.append({
                        "entity_type": "event_handler",
                        "name": f"on{event}",
                        "line": int(row.get("?line", 0)),
                        "content": handler,
                        "event": event,
                        "source_type": "html",
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_html_entities: Handler query failed: {e}")

        # Query forms
        form_query = f'''
        SELECT DISTINCT ?entity ?name ?action ?method ?line
        WHERE {{
            ?entity is-in-document "{in_file}" .
            ?entity type form .
            OPTIONAL {{ ?entity has-name ?name }}
            OPTIONAL {{ ?entity has-action ?action }}
            OPTIONAL {{ ?entity has-method ?method }}
            OPTIONAL {{ ?entity is-at-line ?line }}
        }}
        '''
        try:
            form_table = reter.reql(form_query)
            if form_table is not None and form_table.num_rows > 0:
                for row in form_table.to_pylist():
                    name = row.get("?name", "")
                    action = row.get("?action", "")
                    method = row.get("?method", "GET")
                    entities.append({
                        "entity_type": "form",
                        "name": name or f"form_{action}",
                        "line": int(row.get("?line", 0)),
                        "action": action,
                        "method": method,
                        "source_type": "html",
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_html_entities: Form query failed: {e}")

        # Query framework directives (Vue, Angular, HTMX, Alpine)
        for framework, concept in [
            ("vue", "vue-directive"),
            ("angular", "angular-directive"),
            ("htmx", "htmx-attribute"),
            ("alpine", "alpine-directive"),
        ]:
            directive_query = f'''
            SELECT DISTINCT ?entity ?directive ?value ?line
            WHERE {{
                ?entity is-in-document "{in_file}" .
                ?entity type {concept} .
                OPTIONAL {{ ?entity has-directive-name ?directive }}
                OPTIONAL {{ ?entity has-directive-value ?value }}
                OPTIONAL {{ ?entity is-at-line ?line }}
            }}
            '''
            try:
                directive_table = reter.reql(directive_query)
                if directive_table is not None and directive_table.num_rows > 0:
                    for row in directive_table.to_pylist():
                        directive = row.get("?directive", "")
                        value = row.get("?value", "")
                        entities.append({
                            "entity_type": f"{framework}_directive",
                            "name": directive,
                            "line": int(row.get("?line", 0)),
                            "content": value,
                            "framework": framework,
                            "source_type": "html",
                        })
            except Exception as e:
                logger.debug(f"[RAG] _query_html_entities: {framework} query failed: {e}")

        logger.debug(f"[RAG] _query_html_entities: Found {len(entities)} total entities")
        return entities

    def _query_all_entities_bulk(self, reter: "ReterWrapper", language: str = "python") -> Dict[str, List[Dict[str, Any]]]:
        """
        Query ALL entities from RETER in bulk (3 queries instead of 3 per file).

        Args:
            reter: RETER wrapper instance
            language: Language key (e.g. "python", "javascript", "java", "go", "rust", etc.)

        Returns dict mapping inFile -> list of entities
        """
        entities_by_file: Dict[str, List[Dict[str, Any]]] = {}
        seen_entities: set = set()  # Track seen qualified_names to avoid duplicates

        # Determine concept prefix based on language
        _lang_to_prefix = {
            "python": "py", "javascript": "js", "csharp": "cs", "cpp": "cpp",
            "java": "java", "go": "go", "rust": "rust", "erlang": "erlang",
            "php": "php", "objc": "objc", "swift": "swift", "vb6": "vb6",
            "scala": "scala", "haskell": "haskell", "kotlin": "kotlin", "r": "r", "ruby": "ruby", "dart": "dart", "delphi": "delphi",
        }
        prefix = _lang_to_prefix.get(language, "py")

        # Bulk query all classes
        logger.debug(f"[RAG] _query_all_entities_bulk: Querying all {language} classes...")
        class_query = f'''
        SELECT DISTINCT ?entity ?name ?line ?endLine ?docstring ?inFile
        WHERE {{
            ?entity type class .
            ?entity has-name ?name .
            ?entity is-at-line ?line .
            ?entity is-in-file ?inFile .
            ?entity has-end-line ?endLine .
            OPTIONAL {{ ?entity has-documentation ?docstring }}
        }}
        '''
        try:
            class_table = reter.reql(class_query)
            if class_table is not None and class_table.num_rows > 0:
                class_results = class_table.to_pylist()
                logger.debug(f"[RAG] _query_all_entities_bulk: Found {len(class_results)} classes")
                for row in class_results:
                    qualified_name = row.get("?entity", "")
                    if qualified_name in seen_entities:
                        continue
                    seen_entities.add(qualified_name)
                    in_file = row.get("?inFile", "")
                    if in_file not in entities_by_file:
                        entities_by_file[in_file] = []
                    line = int(row.get("?line", 0))
                    end_line_val = row.get("?endLine")
                    end_line = int(end_line_val) if end_line_val else None
                    entities_by_file[in_file].append({
                        "entity_type": "class",
                        "name": row.get("?name", ""),
                        "qualified_name": qualified_name,
                        "line": line,
                        "end_line": end_line,
                        "docstring": row.get("?docstring"),
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_all_entities_bulk: Class query FAILED: {e}")

        # Bulk query all methods
        # Note: We query definedIn as required (not OPTIONAL) because REQL OPTIONAL
        # doesn't return bound values. All methods should have a defining class.
        logger.debug(f"[RAG] _query_all_entities_bulk: Querying all {language} methods...")
        method_query = f'''
        SELECT DISTINCT ?entity ?name ?line ?endLine ?docstring ?className ?inFile
        WHERE {{
            ?entity type method .
            ?entity has-name ?name .
            ?entity is-at-line ?line .
            ?entity is-in-file ?inFile .
            ?entity has-end-line ?endLine .
            ?entity is-defined-in ?className .
            OPTIONAL {{ ?entity has-documentation ?docstring }}
        }}
        '''
        try:
            method_table = reter.reql(method_query)
            if method_table is not None and method_table.num_rows > 0:
                method_results = method_table.to_pylist()
                logger.debug(f"[RAG] _query_all_entities_bulk: Found {len(method_results)} methods")
                for row in method_results:
                    qualified_name = row.get("?entity", "")
                    if qualified_name in seen_entities:
                        continue
                    seen_entities.add(qualified_name)
                    in_file = row.get("?inFile", "")
                    if in_file not in entities_by_file:
                        entities_by_file[in_file] = []
                    line = int(row.get("?line", 0))
                    end_line_val = row.get("?endLine")
                    end_line = int(end_line_val) if end_line_val else None
                    # Get class_name from query result
                    class_name = row.get("?className", "")
                    entities_by_file[in_file].append({
                        "entity_type": "method",
                        "name": row.get("?name", ""),
                        "qualified_name": qualified_name,
                        "line": line,
                        "end_line": end_line,
                        "docstring": row.get("?docstring"),
                        "class_name": class_name,
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_all_entities_bulk: Method query FAILED: {e}")

        # Bulk query all functions (excluding methods - they are already queried above)
        # Note: Method is_subclass_of Function, so we need to exclude methods
        logger.debug(f"[RAG] _query_all_entities_bulk: Querying all {language} functions...")
        func_query = f'''
        SELECT DISTINCT ?entity ?name ?line ?endLine ?docstring ?inFile
        WHERE {{
            ?entity type function .
            ?entity has-name ?name .
            ?entity is-at-line ?line .
            ?entity is-in-file ?inFile .
            ?entity has-end-line ?endLine .
            OPTIONAL {{ ?entity has-documentation ?docstring }}
            FILTER(NOT EXISTS {{ ?entity type method }})
        }}
        '''
        try:
            func_table = reter.reql(func_query)
            if func_table is not None and func_table.num_rows > 0:
                func_results = func_table.to_pylist()
                logger.debug(f"[RAG] _query_all_entities_bulk: Found {len(func_results)} functions")
                for row in func_results:
                    qualified_name = row.get("?entity", "")
                    if qualified_name in seen_entities:
                        continue
                    seen_entities.add(qualified_name)
                    in_file = row.get("?inFile", "")
                    if in_file not in entities_by_file:
                        entities_by_file[in_file] = []
                    line = int(row.get("?line", 0))
                    end_line_val = row.get("?endLine")
                    end_line = int(end_line_val) if end_line_val else None
                    entities_by_file[in_file].append({
                        "entity_type": "function",
                        "name": row.get("?name", ""),
                        "qualified_name": qualified_name,
                        "line": line,
                        "end_line": end_line,
                        "docstring": row.get("?docstring"),
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_all_entities_bulk: Function query FAILED: {e}")

        total_entities = sum(len(ents) for ents in entities_by_file.values())
        logger.debug(f"[RAG] _query_all_entities_bulk: Total {len(entities_by_file)} files, {total_entities} entities (deduplicated via {len(seen_entities)} unique qualified_names)")
        return entities_by_file

    def _query_all_html_entities_bulk(self, reter: "ReterWrapper") -> Dict[str, List[Dict[str, Any]]]:
        """
        Query ALL HTML entities from RETER in bulk.

        Returns dict mapping is-in-document -> list of entities
        """
        entities_by_file: Dict[str, List[Dict[str, Any]]] = {}

        # Bulk query all scripts
        logger.debug("[RAG] _query_all_html_entities_bulk: Querying all HTML scripts...")
        script_query = '''
        SELECT DISTINCT ?entity ?name ?line ?content ?inDocument
        WHERE {
            ?entity type script .
            ?entity is-in-document ?inDocument .
            OPTIONAL { ?entity has-name ?name }
            OPTIONAL { ?entity is-at-line ?line }
            OPTIONAL { ?entity has-content ?content }
        }
        '''
        try:
            script_table = reter.reql(script_query)
            if script_table is not None and script_table.num_rows > 0:
                script_results = script_table.to_pylist()
                logger.debug(f"[RAG] _query_all_html_entities_bulk: Found {len(script_results)} scripts")
                for row in script_results:
                    in_doc = row.get("?inDocument", "")
                    if in_doc not in entities_by_file:
                        entities_by_file[in_doc] = []
                    entities_by_file[in_doc].append({
                        "entity_type": "script",
                        "name": row.get("?name", "inline_script"),
                        "line": int(row.get("?line", 0)),
                        "content": row.get("?content", ""),
                        "source_type": "html",
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_all_html_entities_bulk: Script query FAILED: {e}")

        # Bulk query all event handlers
        logger.debug("[RAG] _query_all_html_entities_bulk: Querying all HTML event handlers...")
        handler_query = '''
        SELECT DISTINCT ?entity ?event ?handler ?line ?inDocument
        WHERE {
            ?entity type event-handler .
            ?entity is-in-document ?inDocument .
            OPTIONAL { ?entity has-event-name ?event }
            OPTIONAL { ?entity has-handler-code ?handler }
            OPTIONAL { ?entity is-at-line ?line }
        }
        '''
        try:
            handler_table = reter.reql(handler_query)
            if handler_table is not None and handler_table.num_rows > 0:
                handler_results = handler_table.to_pylist()
                logger.debug(f"[RAG] _query_all_html_entities_bulk: Found {len(handler_results)} event handlers")
                for row in handler_results:
                    in_doc = row.get("?inDocument", "")
                    if in_doc not in entities_by_file:
                        entities_by_file[in_doc] = []
                    event = row.get("?event", "unknown")
                    handler = row.get("?handler", "")
                    entities_by_file[in_doc].append({
                        "entity_type": "event_handler",
                        "name": f"on{event}",
                        "line": int(row.get("?line", 0)),
                        "content": handler,
                        "event": event,
                        "source_type": "html",
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_all_html_entities_bulk: Handler query FAILED: {e}")

        # Bulk query all forms
        logger.debug("[RAG] _query_all_html_entities_bulk: Querying all HTML forms...")
        form_query = '''
        SELECT DISTINCT ?entity ?name ?action ?method ?line ?inDocument
        WHERE {
            ?entity type form .
            ?entity is-in-document ?inDocument .
            OPTIONAL { ?entity has-name ?name }
            OPTIONAL { ?entity has-action ?action }
            OPTIONAL { ?entity has-method ?method }
            OPTIONAL { ?entity is-at-line ?line }
        }
        '''
        try:
            form_table = reter.reql(form_query)
            if form_table is not None and form_table.num_rows > 0:
                form_results = form_table.to_pylist()
                logger.debug(f"[RAG] _query_all_html_entities_bulk: Found {len(form_results)} forms")
                for row in form_results:
                    in_doc = row.get("?inDocument", "")
                    if in_doc not in entities_by_file:
                        entities_by_file[in_doc] = []
                    name = row.get("?name", "")
                    action = row.get("?action", "")
                    method = row.get("?method", "GET")
                    entities_by_file[in_doc].append({
                        "entity_type": "form",
                        "name": name or f"form_{action}",
                        "line": int(row.get("?line", 0)),
                        "action": action,
                        "method": method,
                        "source_type": "html",
                    })
        except Exception as e:
            logger.debug(f"[RAG] _query_all_html_entities_bulk: Form query FAILED: {e}")

        # Bulk query framework directives
        for framework, concept in [
            ("vue", "vue-directive"),
            ("angular", "angular-directive"),
            ("htmx", "htmx-attribute"),
            ("alpine", "alpine-directive"),
        ]:
            logger.debug(f"[RAG] _query_all_html_entities_bulk: Querying all {framework} directives...")
            directive_query = f'''
            SELECT DISTINCT ?entity ?directive ?value ?line ?inDocument
            WHERE {{
                ?entity type {concept} .
                ?entity is-in-document ?inDocument .
                OPTIONAL {{ ?entity has-directive-name ?directive }}
                OPTIONAL {{ ?entity has-directive-value ?value }}
                OPTIONAL {{ ?entity is-at-line ?line }}
            }}
            '''
            try:
                directive_table = reter.reql(directive_query)
                if directive_table is not None and directive_table.num_rows > 0:
                    directive_results = directive_table.to_pylist()
                    logger.debug(f"[RAG] _query_all_html_entities_bulk: Found {len(directive_results)} {framework} directives")
                    for row in directive_results:
                        in_doc = row.get("?inDocument", "")
                        if in_doc not in entities_by_file:
                            entities_by_file[in_doc] = []
                        directive = row.get("?directive", "")
                        value = row.get("?value", "")
                        entities_by_file[in_doc].append({
                            "entity_type": f"{framework}_directive",
                            "name": directive,
                            "line": int(row.get("?line", 0)),
                            "content": value,
                            "framework": framework,
                            "source_type": "html",
                        })
            except Exception as e:
                logger.debug(f"[RAG] _query_all_html_entities_bulk: {framework} directive query FAILED: {e}")

        logger.debug(f"[RAG] _query_all_html_entities_bulk: Total {len(entities_by_file)} documents with entities")
        return entities_by_file
