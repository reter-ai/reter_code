"""
Thinking Session - Central Orchestrator

Manages the complete lifecycle of thinking sessions:
- Session management (start, context, end, clear)
- Thought creation with operations
- Context generation for restoration
- Project health/analytics
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from .store import UnifiedStore
from .operations import OperationsHandler
from ..dataclasses import ThoughtInput

logger = logging.getLogger(__name__)


class ThinkingSession:
    """
    Central orchestrator for unified thinking sessions.

    Provides:
    - Session lifecycle management
    - Thinking with operations
    - Full context restoration
    - Project analytics
    """

    def __init__(self, store: UnifiedStore, reter_engine: Optional[Any] = None):
        """
        Initialize thinking session.

        Args:
            store: UnifiedStore instance
            reter_engine: Optional RETER engine for knowledge operations
        """
        self.store = store
        self.reter_engine = reter_engine
        self.operations_handler = OperationsHandler(store, reter_engine)

    # =========================================================================
    # Session Lifecycle
    # =========================================================================

    def start_session(
        self,
        instance_name: str,
        goal: Optional[str] = None,
        project_start: Optional[str] = None,
        project_end: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start or get existing session.

        Args:
            instance_name: RETER instance name
            goal: Session goal/objective
            project_start: Project start date (ISO format)
            project_end: Project end date (ISO format)

        Returns:
            Session info with session_id and status
        """
        session_id = self.store.get_or_create_session(
            instance_name=instance_name,
            goal=goal,
            project_start=project_start,
            project_end=project_end
        )

        session = self.store.get_session(session_id)

        return {
            "success": True,
            "session_id": session_id,
            "instance_name": instance_name,
            "goal": session.get("goal"),
            "status": session.get("status"),
            "created_at": session.get("created_at"),
            "project_start": session.get("project_start"),
            "project_end": session.get("project_end"),
            "_tip": "Call session(action='context') to restore full context",
            "_resources": {
                "guide://reter/session-context": "Session Context Guide",
                "guide://logical-thinking/usage": "Complete Usage Guide"
            }
        }

    def end_session(self, instance_name: str) -> Dict[str, Any]:
        """
        End and archive a session.

        Args:
            instance_name: RETER instance name

        Returns:
            End status and summary
        """
        session_id = self.store.get_or_create_session(instance_name)
        session = self.store.get_session(session_id)

        if not session:
            return {"success": False, "error": "Session not found"}

        self.store.end_session(session_id)

        # Get summary stats
        summary = self.store.get_session_summary(session_id)

        return {
            "success": True,
            "session_id": session_id,
            "status": "completed",
            "ended_at": datetime.now().isoformat(),
            "summary": summary
        }

    def clear_session(self, instance_name: str) -> Dict[str, Any]:
        """
        Clear all session data.

        Args:
            instance_name: RETER instance name

        Returns:
            Clear status and items deleted
        """
        session_id = self.store.get_or_create_session(instance_name)
        result = self.store.clear_session(session_id)

        return {
            "success": True,
            "session_id": session_id,
            "items_deleted": result.get("items_deleted", 0),
            "relations_deleted": result.get("relations_deleted", 0)
        }

    def get_context(self, instance_name: str) -> Dict[str, Any]:
        """
        Get full context for session restoration.

        CRITICAL: This must be called at session start and after compactification.

        Args:
            instance_name: RETER instance name

        Returns:
            Complete context including thoughts, requirements, recommendations,
            project status, artifacts, activities, and suggestions
        """
        session_id = self.store.get_or_create_session(instance_name)
        session = self.store.get_session(session_id)

        if not session:
            return {"success": False, "error": "Session not found"}

        # Get session summary
        summary = self.store.get_session_summary(session_id)

        # Build context response
        context = {
            "success": True,

            # Session state
            "session": {
                "session_id": session_id,
                "instance_name": instance_name,
                "goal": session.get("goal"),
                "status": session.get("status"),
                "created_at": session.get("created_at"),
                "project_start": session.get("project_start"),
                "project_end": session.get("project_end")
            },

            # Thought chain summary
            "thoughts": self._get_thoughts_context(session_id, summary),

            # Requirements status
            "requirements": self._get_requirements_context(session_id, summary),

            # Recommendations progress
            "recommendations": self._get_recommendations_context(session_id, summary),

            # Project status (tasks/milestones)
            "project": self._get_project_context(session_id, summary),

            # Artifacts with freshness
            "artifacts": self._get_artifacts_context(session_id),

            # Recent activities
            "recent_activities": self._get_activities_context(session_id),

            # RETER state (if available)
            "reter": self._get_reter_context(instance_name),

            # Actionable suggestions
            "suggestions": self._generate_suggestions(session_id, summary),

            # MCP usage guide
            "mcp_guide": self._get_mcp_guide()
        }

        return context

    # =========================================================================
    # Thinking
    # =========================================================================

    def think(
        self,
        instance_name: str,
        thought: str,
        thought_number: int,
        total_thoughts: int,
        thought_type: str = "reasoning",
        next_thought_needed: bool = True,
        branch_id: Optional[str] = None,
        branch_from: Optional[int] = None,
        is_revision: bool = False,
        revises_thought: Optional[int] = None,
        needs_more_thoughts: bool = False,
        operations: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a thought with optional operations.

        Args:
            instance_name: RETER instance name
            thought: Thought content
            thought_number: Current thought number (1-indexed)
            total_thoughts: Estimated total thoughts
            thought_type: Type of thought (reasoning, analysis, decision, planning, verification)
            next_thought_needed: Whether more thoughts are needed
            branch_id: ID for branching
            branch_from: Thought number to branch from
            is_revision: Whether this revises a previous thought
            revises_thought: Which thought number is being revised
            needs_more_thoughts: Signal that more analysis is needed
            operations: Operations dict (requirements, tasks, relations, etc.)

        Returns:
            Thought result with created items and relations
        """
        # Create thought input object to reduce parameter passing
        thought_input = ThoughtInput(
            thought=thought,
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            thought_type=thought_type,
            next_thought_needed=next_thought_needed,
            needs_more_thoughts=needs_more_thoughts,
            branch_id=branch_id,
            branch_from=branch_from,
            is_revision=is_revision,
            revises_thought=revises_thought,
            operations=operations
        )
        return self._create_thought(instance_name, thought_input)

    def _create_thought(
        self,
        instance_name: str,
        thought_input: ThoughtInput
    ) -> Dict[str, Any]:
        """Create a thought using ThoughtInput object.

        Args:
            instance_name: RETER instance name
            thought_input: ThoughtInput with all thought parameters

        Returns:
            Thought result with created items and relations
        """
        session_id = self.store.get_or_create_session(instance_name)

        # Create the thought item
        thought_id = self.store.add_item(
            session_id=session_id,
            item_type="thought",
            content=thought_input.thought,
            thought_number=thought_input.thought_number,
            total_thoughts=thought_input.total_thoughts,
            next_thought_needed=1 if thought_input.next_thought_needed else 0,
            thought_type=thought_input.thought_type,
            is_revision=1 if thought_input.is_revision else 0,
            revises_thought=thought_input.revises_thought,
            branch_from_thought=thought_input.branch_from,
            branch_id=thought_input.branch_id,
            needs_more_thoughts=1 if thought_input.needs_more_thoughts else 0,
            status="completed"
        )

        result = {
            "success": True,
            "thought_id": thought_id,
            "thought_number": thought_input.thought_number,
            "total_thoughts": thought_input.total_thoughts,
            "next_thought_needed": thought_input.next_thought_needed,
            "thought_type": thought_input.thought_type,
            "items_created": [],
            "items_updated": [],
            "relations_created": [],
            "reter_operations": {}
        }

        # Execute operations if provided
        if thought_input.operations:
            ops_result = self.operations_handler.execute(
                session_id=session_id,
                thought_id=thought_id,
                operations=thought_input.operations
            )
            result["items_created"] = ops_result.get("items_created", [])
            result["items_updated"] = ops_result.get("items_updated", [])
            result["relations_created"] = ops_result.get("relations_created", [])
            result["reter_operations"] = ops_result.get("reter", {})

            if ops_result.get("errors"):
                result["errors"] = ops_result["errors"]

        # Add session status snapshot
        summary = self.store.get_session_summary(session_id)
        result["session_status"] = {
            "total_items": summary.get("total_items", 0),
            "by_type": summary.get("by_type", {}),
            "by_status": summary.get("by_status", {}),
            "thought_chain": summary.get("thought_chain", {})
        }

        return result

    # =========================================================================
    # Project Analytics
    # =========================================================================

    def get_project_health(self, instance_name: str) -> Dict[str, Any]:
        """
        Get overall project health status.

        Args:
            instance_name: RETER instance name

        Returns:
            Health metrics including completion, status, milestones
        """
        session_id = self.store.get_or_create_session(instance_name)
        session = self.store.get_session(session_id)

        # Get tasks and milestones
        tasks = self.store.get_items(session_id, item_type="task")
        milestones = self.store.get_items(session_id, item_type="milestone")

        # Calculate task metrics
        total_tasks = len(tasks)
        completed = len([t for t in tasks if t.get("status") == "completed"])
        in_progress = len([t for t in tasks if t.get("status") == "in_progress"])
        blocked = len([t for t in tasks if t.get("status") == "blocked"])
        pending = len([t for t in tasks if t.get("status") == "pending"])

        percent_complete = (completed / total_tasks * 100) if total_tasks > 0 else 0

        # Calculate project dates
        project_start = session.get("project_start")
        project_end = session.get("project_end")
        days_remaining = None
        on_track = True

        if project_end:
            try:
                end_date = datetime.strptime(project_end, "%Y-%m-%d")
                days_remaining = (end_date - datetime.now()).days
                on_track = days_remaining >= 0
            except ValueError as e:
                logger.warning(f"Invalid project_end date format '{project_end}': {e}")

        # Get overdue tasks
        overdue = []
        today = datetime.now().strftime("%Y-%m-%d")
        for task in tasks:
            if task.get("end_date") and task.get("end_date") < today and task.get("status") != "completed":
                overdue.append({
                    "id": task["item_id"],
                    "name": task["content"],
                    "end_date": task["end_date"],
                    "days_overdue": (datetime.now() - datetime.strptime(task["end_date"], "%Y-%m-%d")).days
                })

        # Milestone status
        milestones_status = []
        for ms in milestones:
            status = {
                "id": ms["item_id"],
                "name": ms["content"],
                "target_date": ms.get("end_date"),
                "status": ms.get("status", "pending")
            }
            if ms.get("end_date"):
                try:
                    target = datetime.strptime(ms["end_date"], "%Y-%m-%d")
                    status["days_until"] = (target - datetime.now()).days
                except ValueError as e:
                    logger.warning(f"Invalid milestone end_date format '{ms['end_date']}' for {ms['item_id']}: {e}")
            milestones_status.append(status)

        # Get recommendations progress
        recommendations = self.store.get_items(session_id, item_type="recommendation")
        rec_pending = len([r for r in recommendations if r.get("status") == "pending"])
        rec_completed = len([r for r in recommendations if r.get("status") == "completed"])
        rec_total = len(recommendations)

        return {
            "success": True,
            "tasks": {
                "total": total_tasks,
                "completed": completed,
                "in_progress": in_progress,
                "blocked": blocked,
                "pending": pending,
                "percent_complete": round(percent_complete, 1)
            },
            "timeline": {
                "project_start": project_start,
                "project_end": project_end,
                "days_remaining": days_remaining,
                "on_track": on_track
            },
            "overdue": overdue,
            "milestones": milestones_status,
            "recommendations": {
                "total": rec_total,
                "pending": rec_pending,
                "completed": rec_completed,
                "progress_percent": round((rec_completed / rec_total * 100) if rec_total > 0 else 0, 1)
            }
        }

    def get_critical_path(self, instance_name: str) -> Dict[str, Any]:
        """
        Calculate critical path for tasks.

        Algorithm:
        1. Build dependency graph
        2. Topological sort
        3. Forward pass (earliest start/finish)
        4. Backward pass (latest start/finish)
        5. Identify zero-float tasks

        Args:
            instance_name: RETER instance name

        Returns:
            Critical path tasks and total duration
        """
        session_id = self.store.get_or_create_session(instance_name)

        # Get all tasks
        tasks = self.store.get_items(session_id, item_type="task")
        if not tasks:
            return {"success": True, "critical_tasks": [], "total_duration": 0}

        # Build task lookup and dependency graph
        task_map = {t["item_id"]: t for t in tasks}
        dependencies = {}  # task_id -> list of dependency task_ids
        dependents = {}    # task_id -> list of tasks that depend on this

        for task in tasks:
            task_id = task["item_id"]
            dependencies[task_id] = []
            dependents[task_id] = []

        # Get dependencies from relations
        for task in tasks:
            task_id = task["item_id"]
            relations = self.store.get_relations(task_id, direction="outgoing")
            for rel in relations:
                if rel["relation_type"] == "depends_on" and rel["target_type"] == "item":
                    dep_id = rel["target_id"]
                    if dep_id in task_map:
                        dependencies[task_id].append(dep_id)
                        dependents[dep_id].append(task_id)

        # Topological sort (Kahn's algorithm)
        in_degree = {t: len(dependencies[t]) for t in task_map}
        queue = [t for t in task_map if in_degree[t] == 0]
        sorted_tasks = []

        while queue:
            task_id = queue.pop(0)
            sorted_tasks.append(task_id)
            for dependent in dependents[task_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for cycles
        if len(sorted_tasks) != len(task_map):
            return {"success": False, "error": "Circular dependency detected"}

        # Forward pass - calculate early start/finish
        early_start = {}
        early_finish = {}

        for task_id in sorted_tasks:
            task = task_map[task_id]
            duration = task.get("duration_days") or 1

            # Early start is max of all dependency early finishes
            if dependencies[task_id]:
                es = max(early_finish.get(dep, 0) for dep in dependencies[task_id])
            else:
                es = 0

            early_start[task_id] = es
            early_finish[task_id] = es + duration

        # Project duration
        total_duration = max(early_finish.values()) if early_finish else 0

        # Backward pass - calculate late start/finish
        late_finish = {}
        late_start = {}

        for task_id in reversed(sorted_tasks):
            task = task_map[task_id]
            duration = task.get("duration_days") or 1

            # Late finish is min of all dependent late starts
            if dependents[task_id]:
                lf = min(late_start.get(dep, total_duration) for dep in dependents[task_id])
            else:
                lf = total_duration

            late_finish[task_id] = lf
            late_start[task_id] = lf - duration

        # Calculate float and identify critical path
        critical_tasks = []
        for task_id in sorted_tasks:
            float_time = late_start[task_id] - early_start[task_id]
            if float_time == 0:
                task = task_map[task_id]
                critical_tasks.append({
                    "id": task_id,
                    "name": task["content"],
                    "duration": task.get("duration_days") or 1,
                    "early_start": early_start[task_id],
                    "early_finish": early_finish[task_id],
                    "late_start": late_start[task_id],
                    "late_finish": late_finish[task_id],
                    "status": task.get("status", "pending")
                })

        return {
            "success": True,
            "critical_tasks": critical_tasks,
            "total_duration": total_duration,
            "all_tasks_count": len(tasks),
            "critical_count": len(critical_tasks)
        }

    def get_overdue_tasks(self, instance_name: str) -> Dict[str, Any]:
        """
        Get all overdue tasks.

        Args:
            instance_name: RETER instance name

        Returns:
            List of overdue tasks with days overdue
        """
        session_id = self.store.get_or_create_session(instance_name)
        tasks = self.store.get_items(session_id, item_type="task")

        overdue = []
        today = datetime.now().strftime("%Y-%m-%d")

        for task in tasks:
            if task.get("end_date") and task.get("end_date") < today and task.get("status") != "completed":
                days_overdue = (datetime.now() - datetime.strptime(task["end_date"], "%Y-%m-%d")).days
                overdue.append({
                    "id": task["item_id"],
                    "name": task["content"],
                    "end_date": task["end_date"],
                    "days_overdue": days_overdue,
                    "status": task.get("status"),
                    "assigned_to": task.get("assigned_to")
                })

        # Sort by days overdue descending
        overdue.sort(key=lambda x: x["days_overdue"], reverse=True)

        return {
            "success": True,
            "overdue_tasks": overdue,
            "total_overdue": len(overdue),
            "max_days_overdue": overdue[0]["days_overdue"] if overdue else 0
        }

    def analyze_impact(
        self,
        instance_name: str,
        task_id: str,
        delay_days: int
    ) -> Dict[str, Any]:
        """
        Analyze impact of delaying a task.

        Args:
            instance_name: RETER instance name
            task_id: Task to analyze
            delay_days: Number of days delay

        Returns:
            Affected tasks, new end dates, milestone impacts
        """
        session_id = self.store.get_or_create_session(instance_name)

        # Get the task
        task = self.store.get_item(task_id)
        if not task:
            return {"success": False, "error": f"Task {task_id} not found"}

        # Get all dependent tasks (recursively)
        affected = []
        visited = set()

        def get_dependents(tid):
            if tid in visited:
                return
            visited.add(tid)

            # Find tasks that depend on this one
            all_tasks = self.store.get_items(session_id, item_type="task")
            for t in all_tasks:
                relations = self.store.get_relations(t["item_id"], direction="outgoing")
                for rel in relations:
                    if rel["relation_type"] == "depends_on" and rel["target_id"] == tid:
                        affected.append(t)
                        get_dependents(t["item_id"])

        get_dependents(task_id)

        # Calculate new end dates
        new_end_dates = {}
        original_end = task.get("end_date")
        if original_end:
            try:
                new_end = datetime.strptime(original_end, "%Y-%m-%d") + timedelta(days=delay_days)
                new_end_dates[task_id] = new_end.strftime("%Y-%m-%d")
            except ValueError as e:
                logger.warning(f"Invalid task end_date format '{original_end}' for {task_id}: {e}")

        for affected_task in affected:
            if affected_task.get("end_date"):
                try:
                    new_end = datetime.strptime(affected_task["end_date"], "%Y-%m-%d") + timedelta(days=delay_days)
                    new_end_dates[affected_task["item_id"]] = new_end.strftime("%Y-%m-%d")
                except ValueError as e:
                    logger.warning(f"Invalid task end_date format '{affected_task['end_date']}' for {affected_task['item_id']}: {e}")

        # Check milestone impacts
        milestones = self.store.get_items(session_id, item_type="milestone")
        affected_milestones = []

        for ms in milestones:
            ms_deps = self.store.get_relations(ms["item_id"], direction="outgoing")
            for rel in ms_deps:
                if rel["relation_type"] == "depends_on":
                    if rel["target_id"] == task_id or rel["target_id"] in [a["item_id"] for a in affected]:
                        affected_milestones.append({
                            "id": ms["item_id"],
                            "name": ms["content"],
                            "original_target": ms.get("end_date"),
                            "may_slip": True
                        })
                        break

        return {
            "success": True,
            "delayed_task": {
                "id": task_id,
                "name": task["content"],
                "original_end": original_end,
                "new_end": new_end_dates.get(task_id),
                "delay_days": delay_days
            },
            "affected_tasks": [
                {
                    "id": t["item_id"],
                    "name": t["content"],
                    "original_end": t.get("end_date"),
                    "new_end": new_end_dates.get(t["item_id"])
                }
                for t in affected
            ],
            "affected_milestones": affected_milestones,
            "total_affected": len(affected)
        }

    # =========================================================================
    # Context Generation Helpers
    # =========================================================================

    def _get_thoughts_context(
        self,
        session_id: str,
        summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get thought chain context."""
        chain = self.store.get_thought_chain(session_id)
        thought_count = summary.get("by_type", {}).get("thought", 0)

        # Get last 3 thoughts as summary
        latest = []
        for t in chain[-3:]:
            latest.append({
                "num": t.get("thought_number"),
                "type": t.get("thought_type", "reasoning"),
                "summary": t.get("content", "")[:100] + "..." if len(t.get("content", "")) > 100 else t.get("content", "")
            })

        # Get decisions
        decisions = self.store.get_items(session_id, item_type="decision")
        key_decisions = [
            {
                "id": d["item_id"],
                "text": d["content"][:100],
                "created_at": d.get("created_at")
            }
            for d in decisions[:5]
        ]

        return {
            "total": thought_count,
            "max_number": summary.get("thought_chain", {}).get("max_number", 0),
            "latest_chain": latest,
            "key_decisions": key_decisions
        }

    def _get_requirements_context(
        self,
        session_id: str,
        summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get requirements context."""
        requirements = self.store.get_items(session_id, item_type="requirement")

        verified = len([r for r in requirements if r.get("status") == "verified"])
        pending = len([r for r in requirements if r.get("status") in ("pending", "active")])

        items = [
            {
                "id": r["item_id"],
                "text": r["content"][:100],
                "status": r.get("status", "pending"),
                "risk": r.get("risk", "medium")
            }
            for r in requirements[:10]
        ]

        return {
            "total": len(requirements),
            "verified": verified,
            "pending_verification": pending,
            "items": items
        }

    def _get_recommendations_context(
        self,
        session_id: str,
        summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get recommendations context."""
        recommendations = self.store.get_items(session_id, item_type="recommendation")

        total = len(recommendations)
        completed = len([r for r in recommendations if r.get("status") == "completed"])
        in_progress = len([r for r in recommendations if r.get("status") == "in_progress"])
        pending = len([r for r in recommendations if r.get("status") == "pending"])

        # Count by priority
        by_priority = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for r in recommendations:
            p = r.get("priority", "medium")
            if p in by_priority:
                by_priority[p] += 1

        # Get highest priority pending
        highest = [
            {
                "id": r["item_id"],
                "text": r["content"][:100],
                "priority": r.get("priority", "medium")
            }
            for r in recommendations
            if r.get("status") == "pending" and r.get("priority") in ("critical", "high")
        ][:5]

        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "progress_percent": round((completed / total * 100) if total > 0 else 0, 1),
            "by_priority": by_priority,
            "highest_priority": highest
        }

    def _get_project_context(
        self,
        session_id: str,
        summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get project/task context."""
        tasks = self.store.get_items(session_id, item_type="task")
        milestones = self.store.get_items(session_id, item_type="milestone")

        total = len(tasks)
        completed = len([t for t in tasks if t.get("status") == "completed"])
        in_progress = len([t for t in tasks if t.get("status") == "in_progress"])
        blocked = len([t for t in tasks if t.get("status") == "blocked"])

        # Get overdue
        overdue = []
        today = datetime.now().strftime("%Y-%m-%d")
        for task in tasks:
            if task.get("end_date") and task.get("end_date") < today and task.get("status") != "completed":
                overdue.append(task["item_id"])

        # Get upcoming milestones
        upcoming = []
        for ms in milestones:
            if ms.get("end_date") and ms.get("status") != "completed":
                try:
                    target = datetime.strptime(ms["end_date"], "%Y-%m-%d")
                    days = (target - datetime.now()).days
                    if days >= 0:
                        upcoming.append({
                            "id": ms["item_id"],
                            "name": ms["content"],
                            "target": ms["end_date"],
                            "days_until": days
                        })
                except ValueError as e:
                    logger.warning(f"Invalid milestone end_date format '{ms['end_date']}' for {ms['item_id']}: {e}")
        upcoming.sort(key=lambda x: x["days_until"])

        return {
            "total_tasks": total,
            "completed": completed,
            "in_progress": in_progress,
            "blocked": blocked,
            "percent_complete": round((completed / total * 100) if total > 0 else 0, 1),
            "overdue": overdue,
            "upcoming_milestones": upcoming[:3]
        }

    def _get_artifacts_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Get artifacts with freshness."""
        artifacts = self.store.get_artifacts(session_id)

        result = []
        for a in artifacts:
            result.append({
                "file_path": a.get("file_path"),
                "type": a.get("artifact_type"),
                "created_at": a.get("created_at"),
                "fresh": True  # Would need file checking for real freshness
            })

        return result

    def _get_activities_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Get recent activities."""
        activities = self.store.get_items(session_id, item_type="activity")

        # Sort by created_at descending and take last 5
        activities.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        result = []
        for a in activities[:5]:
            result.append({
                "tool": a.get("source_tool"),
                "content": a.get("content"),
                "time": a.get("created_at")
            })

        return result

    def _get_reter_context(self, instance_name: str) -> Dict[str, Any]:
        """Get RETER instance context."""
        if not self.reter_engine:
            return {"instance": instance_name, "available": False}

        # Would query RETER for loaded sources and stats
        return {
            "instance": instance_name,
            "available": True
        }

    def _generate_suggestions(
        self,
        session_id: str,
        summary: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable suggestions."""
        suggestions = []

        # Thought chain suggestion
        max_thought = summary.get("thought_chain", {}).get("max_number", 0)
        if max_thought > 0:
            suggestions.append(f"Continue thought chain from #{max_thought}")

        # Pending recommendations
        rec_pending = summary.get("by_status", {}).get("pending", 0)
        if rec_pending > 0:
            suggestions.append(f"{rec_pending} pending items to address")

        # Blocked tasks
        tasks = self.store.get_items(session_id, item_type="task")
        blocked = [t for t in tasks if t.get("status") == "blocked"]
        if blocked:
            suggestions.append(f"{len(blocked)} task(s) blocked - resolve dependencies")

        # Overdue tasks
        today = datetime.now().strftime("%Y-%m-%d")
        overdue = [t for t in tasks if t.get("end_date") and t.get("end_date") < today and t.get("status") != "completed"]
        if overdue:
            suggestions.append(f"{len(overdue)} task(s) overdue - update or complete")

        # Requirements needing verification
        requirements = self.store.get_items(session_id, item_type="requirement")
        unverified = [r for r in requirements if r.get("status") not in ("verified", "rejected")]
        if unverified:
            suggestions.append(f"{len(unverified)} requirement(s) need verification")

        if not suggestions:
            suggestions.append("Session is up to date - continue reasoning")

        return suggestions

    def _get_mcp_guide(self) -> Dict[str, Any]:
        """Get MCP usage guide with resource references."""
        return {
            "tools": {
                "thinking": "PRIMARY - Create thoughts with operations (requirements, tasks, traces)",
                "session": "Lifecycle: start, context, end, clear",
                "items": "Query/manage: list, get, delete, update",
                "project": "Analytics: health, critical_path, overdue, impact",
                "diagram": "Visualize: gantt, class_hierarchy, sequence, traceability",
                "code_inspection": "Python analysis (26 actions)",
                "recommender": "Code quality: refactoring, test_coverage",
                "natural_language_query": "RECOMMENDED - Plain English queries",
                "instance_manager": "Manage instances/sources"
            },
            "resources": {
                "guide://logical-thinking/usage": "Complete AI Agent Usage Guide",
                "guide://reter/session-context": "Session Context (CRITICAL)",
                "python://reter/tools": "Python Analysis Tools Reference",
                "recipe://refactoring/index": "Refactoring Recipes Index",
                "reference://reter/syntax-quick": "Syntax Quick Reference"
            },
            "recommended_workflow": [
                "1. session(action='context') - restore context (YOU ARE HERE)",
                "2. thinking(...) - continue reasoning chain",
                "3. code_inspection/natural_language_query - analyze code",
                "4. recommender('refactoring'/'test_coverage') - find issues",
                "5. thinking(..., operations={...}) - record findings",
                "6. session(action='end') - archive when done"
            ]
        }
