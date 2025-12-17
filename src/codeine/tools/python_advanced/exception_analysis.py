"""
Exception Handling Analysis Tools

Provides tools for analyzing exception handling patterns and detecting anti-patterns:
- Silent exception swallowing
- Too general exception catching
- Too general exception raising
- Error codes instead of exceptions
- Missing context managers
"""

from typing import Dict, Any
import time
from .base import AdvancedToolsBase


class ExceptionAnalysisTools(AdvancedToolsBase):
    """Exception handling analysis tools."""

    def detect_silent_exception_swallowing(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find except blocks that silently swallow exceptions (empty body or just 'pass').

        This is a common anti-pattern that hides bugs and makes debugging difficult.
        Exceptions should be either handled, logged, or re-raised.

        Args:
            instance_name: RETER instance name
            limit: Maximum results
            offset: Pagination offset

        Returns:
            dict with success, findings list, count
        """
        start_time = time.time()
        queries = []
        try:
            except_handler_concept = self._concept('ExceptHandler')
            # Use inFile (works for all languages) instead of inModule
            query = f"""
                SELECT ?handler ?function ?file ?exception_type ?line
                WHERE {{
                    ?handler type {except_handler_concept} .
                    ?handler isSilentSwallow "true" .
                    OPTIONAL {{ ?handler inFunction ?function }} .
                    OPTIONAL {{ ?handler inFile ?file }} .
                    OPTIONAL {{ ?handler exceptionType ?exception_type }} .
                    OPTIONAL {{ ?handler atLine ?line }}
                }}
                ORDER BY ?file ?function ?line
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)
            total_count = len(rows)

            findings = []
            for row in rows[offset:offset + limit]:
                findings.append({
                    "handler_id": row[0] if len(row) > 0 else None,
                    "function": row[1] if len(row) > 1 else None,
                    "file": row[2] if len(row) > 2 else None,
                    "exception_type": row[3] if len(row) > 3 else "bare",
                    "line": row[4] if len(row) > 4 else None,
                    "issue": "Silent exception swallowing - exceptions are caught but not handled",
                    "recommendation": "Log the exception, handle it appropriately, or re-raise it"
                })

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "findings": findings,
                "count": len(findings),
                "total_count": total_count,
                "has_more": offset + limit < total_count,
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "findings": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def detect_too_general_exceptions(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find except blocks that catch overly broad exceptions (Exception, BaseException, bare except).

        Catching too-general exceptions can hide bugs and make code harder to debug.
        Prefer catching specific exception types.

        Args:
            instance_name: RETER instance name
            limit: Maximum results
            offset: Pagination offset

        Returns:
            dict with success, findings list, count
        """
        start_time = time.time()
        queries = []
        try:
            except_handler_concept = self._concept('ExceptHandler')
            # Use inFile (works for all languages) instead of inModule
            query = f"""
                SELECT ?handler ?function ?file ?exception_type ?line ?is_bare
                WHERE {{
                    ?handler type {except_handler_concept} .
                    ?handler isGeneralExcept "true" .
                    OPTIONAL {{ ?handler inFunction ?function }} .
                    OPTIONAL {{ ?handler inFile ?file }} .
                    OPTIONAL {{ ?handler exceptionType ?exception_type }} .
                    OPTIONAL {{ ?handler atLine ?line }} .
                    OPTIONAL {{ ?handler isBareExcept ?is_bare }}
                }}
                ORDER BY ?file ?function ?line
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)
            total_count = len(rows)

            findings = []
            for row in rows[offset:offset + limit]:
                exc_type = row[3] if len(row) > 3 else "bare"
                is_bare = row[5] if len(row) > 5 else "false"

                if is_bare == "true" or exc_type == "bare":
                    issue = "Bare except clause catches all exceptions including KeyboardInterrupt and SystemExit"
                elif exc_type in ("Exception", "BaseException"):
                    issue = f"Catching {exc_type} is too broad - may hide unexpected errors"
                else:
                    issue = f"Catching general exception type: {exc_type}"

                findings.append({
                    "handler_id": row[0] if len(row) > 0 else None,
                    "function": row[1] if len(row) > 1 else None,
                    "file": row[2] if len(row) > 2 else None,
                    "exception_type": exc_type,
                    "line": row[4] if len(row) > 4 else None,
                    "is_bare_except": is_bare == "true",
                    "issue": issue,
                    "recommendation": "Catch specific exception types (ValueError, KeyError, etc.)"
                })

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "findings": findings,
                "count": len(findings),
                "total_count": total_count,
                "has_more": offset + limit < total_count,
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "findings": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def detect_general_exception_raising(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find raise statements that raise overly general exceptions (Exception, BaseException).

        Raising generic exceptions makes it harder for callers to handle errors appropriately.
        Define and use custom exception classes for better error handling.

        Args:
            instance_name: RETER instance name
            limit: Maximum results
            offset: Pagination offset

        Returns:
            dict with success, findings list, count
        """
        start_time = time.time()
        queries = []
        try:
            raise_concept = self._concept('RaiseStatement')
            # Use inFile (works for all languages) instead of inModule
            query = f"""
                SELECT ?raise ?function ?file ?exception_type ?line
                WHERE {{
                    ?raise type {raise_concept} .
                    ?raise isGeneralException "true" .
                    OPTIONAL {{ ?raise inFunction ?function }} .
                    OPTIONAL {{ ?raise inFile ?file }} .
                    OPTIONAL {{ ?raise exceptionType ?exception_type }} .
                    OPTIONAL {{ ?raise atLine ?line }}
                }}
                ORDER BY ?file ?function ?line
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)
            total_count = len(rows)

            findings = []
            for row in rows[offset:offset + limit]:
                exc_type = row[3] if len(row) > 3 else "Exception"
                findings.append({
                    "raise_id": row[0] if len(row) > 0 else None,
                    "function": row[1] if len(row) > 1 else None,
                    "file": row[2] if len(row) > 2 else None,
                    "exception_type": exc_type,
                    "line": row[4] if len(row) > 4 else None,
                    "issue": f"Raising generic {exc_type} - callers cannot catch specific errors",
                    "recommendation": "Define custom exception classes (e.g., class ValidationError(Exception): pass)"
                })

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "findings": findings,
                "count": len(findings),
                "total_count": total_count,
                "has_more": offset + limit < total_count,
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "findings": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def detect_error_codes_over_exceptions(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find functions that return error codes instead of raising exceptions.

        Common patterns: returning -1, None, False, or error dicts instead of raising.
        Python idiom prefers exceptions for error handling over return codes.

        Args:
            instance_name: RETER instance name
            limit: Maximum results
            offset: Pagination offset

        Returns:
            dict with success, findings list, count
        """
        start_time = time.time()
        queries = []
        try:
            return_concept = self._concept('ReturnStatement')
            # Use inFile (works for all languages) instead of inModule
            query = f"""
                SELECT ?return ?function ?file ?return_value ?line
                WHERE {{
                    ?return type {return_concept} .
                    ?return looksLikeErrorCode "true" .
                    OPTIONAL {{ ?return inFunction ?function }} .
                    OPTIONAL {{ ?return inFile ?file }} .
                    OPTIONAL {{ ?return returnValue ?return_value }} .
                    OPTIONAL {{ ?return atLine ?line }}
                }}
                ORDER BY ?file ?function ?line
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)
            total_count = len(rows)

            findings = []
            for row in rows[offset:offset + limit]:
                return_value = row[3] if len(row) > 3 else "None"
                findings.append({
                    "return_id": row[0] if len(row) > 0 else None,
                    "function": row[1] if len(row) > 1 else None,
                    "file": row[2] if len(row) > 2 else None,
                    "return_value": return_value,
                    "line": row[4] if len(row) > 4 else None,
                    "issue": f"Returning error code '{return_value}' instead of raising exception",
                    "recommendation": "Use exceptions for error handling (raise ValueError, raise CustomError)"
                })

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "findings": findings,
                "count": len(findings),
                "total_count": total_count,
                "has_more": offset + limit < total_count,
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "findings": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def detect_finally_without_context_manager(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Find try/finally blocks that could use context managers (with statement).

        RAII cleanup in finally blocks (close, release, unlock) is often better
        expressed using the 'with' statement for automatic resource management.

        Args:
            instance_name: RETER instance name
            limit: Maximum results
            offset: Pagination offset

        Returns:
            dict with success, findings list, count
        """
        start_time = time.time()
        queries = []
        try:
            finally_concept = self._concept('FinallyBlock')
            # Use inFile (works for all languages) instead of inModule
            query = f"""
                SELECT ?finally ?function ?file ?line ?has_close ?has_release ?has_unlock
                WHERE {{
                    ?finally type {finally_concept} .
                    ?finally isRAIICleanup "true" .
                    OPTIONAL {{ ?finally inFunction ?function }} .
                    OPTIONAL {{ ?finally inFile ?file }} .
                    OPTIONAL {{ ?finally atLine ?line }} .
                    OPTIONAL {{ ?finally hasCloseCall ?has_close }} .
                    OPTIONAL {{ ?finally hasReleaseCall ?has_release }} .
                    OPTIONAL {{ ?finally hasUnlockCall ?has_unlock }}
                }}
                ORDER BY ?file ?function ?line
            """
            queries.append(query.strip())

            result = self.reter.reql(query)
            rows = self._query_to_list(result)
            total_count = len(rows)

            findings = []
            for row in rows[offset:offset + limit]:
                cleanup_types = []
                if len(row) > 4 and row[4] == "true":
                    cleanup_types.append("close()")
                if len(row) > 5 and row[5] == "true":
                    cleanup_types.append("release()")
                if len(row) > 6 and row[6] == "true":
                    cleanup_types.append("unlock()")

                findings.append({
                    "finally_id": row[0] if len(row) > 0 else None,
                    "function": row[1] if len(row) > 1 else None,
                    "file": row[2] if len(row) > 2 else None,
                    "line": row[3] if len(row) > 3 else None,
                    "cleanup_operations": cleanup_types,
                    "issue": f"Manual cleanup in finally block: {', '.join(cleanup_types)}",
                    "recommendation": "Use 'with' statement for automatic resource management (context managers)"
                })

            time_ms = (time.time() - start_time) * 1000
            return {
                "success": True,
                "findings": findings,
                "count": len(findings),
                "total_count": total_count,
                "has_more": offset + limit < total_count,
                "queries": queries,
                "time_ms": time_ms
            }
        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "findings": [],
                "count": 0,
                "queries": queries,
                "time_ms": time_ms
            }

    def analyze_exception_handling(
        self,
        instance_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Comprehensive exception handling analysis - runs all exception detectors.

        Provides a complete overview of exception handling issues in the codebase:
        - Silent exception swallowing
        - Too general exception catching
        - Too general exception raising
        - Error codes instead of exceptions
        - Missing context managers

        Args:
            instance_name: RETER instance name
            limit: Max results per category
            offset: Pagination offset

        Returns:
            dict with all findings grouped by category
        """
        start_time = time.time()

        results = {
            "silent_swallowing": self.detect_silent_exception_swallowing(instance_name, limit, offset),
            "too_general_catching": self.detect_too_general_exceptions(instance_name, limit, offset),
            "too_general_raising": self.detect_general_exception_raising(instance_name, limit, offset),
            "error_codes": self.detect_error_codes_over_exceptions(instance_name, limit, offset),
            "missing_context_managers": self.detect_finally_without_context_manager(instance_name, limit, offset)
        }

        # Calculate totals
        total_issues = sum(r.get("count", 0) for r in results.values())

        # Count by severity (all exception issues are high/medium priority)
        critical_count = results["silent_swallowing"].get("count", 0)
        high_count = results["too_general_catching"].get("count", 0) + results["too_general_raising"].get("count", 0)
        medium_count = results["error_codes"].get("count", 0) + results["missing_context_managers"].get("count", 0)

        time_ms = (time.time() - start_time) * 1000
        return {
            "success": True,
            "total_issues": total_issues,
            "by_severity": {
                "critical": critical_count,
                "high": high_count,
                "medium": medium_count
            },
            "categories": results,
            "time_ms": time_ms
        }
