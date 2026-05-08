"""Loop middleware: extract compliance checks from the core agent loop.

Inspired by Claude Code's clean separation where QueryEngine manages
session state and query.ts only does "call model → execute tools → return".

This module provides a middleware pattern for the agent loop:
  - pre_tool_call hooks run BEFORE each tool execution
  - post_tool_call hooks run AFTER each tool execution
  - Hooks can modify args, block execution, or inject side effects

This moves compliance checks, skill eval gates, and guardrails OUT of
the core loop, making it easier to add/remove checks without modifying
run_agent.py.

Usage:
    from agent.loop_middleware import LoopMiddleware

    middleware = LoopMiddleware()
    middleware.register_pre("skill_eval_gate", skill_eval_hook, priority=10)
    middleware.register_post("compliance_check", compliance_hook, priority=5)

    # In the agent loop:
    for tool_call in tool_calls:
        pre = middleware.run_pre(tool_call.name, tool_call.args, context)
        if pre.blocked:
            return pre.error_message
        result = execute_tool(tool_call)
        middleware.run_post(tool_call.name, result, context)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HookResult:
    """Result of a middleware hook."""
    blocked: bool = False
    error_message: str = ""
    modified_args: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookEntry:
    """A registered middleware hook."""
    name: str
    fn: Callable[..., HookResult]
    priority: int = 0  # lower = runs first
    enabled: bool = True


class LoopMiddleware:
    """Middleware processor for the agent loop.

    Pre-hooks run before tool execution (can block or modify args).
    Post-hooks run after tool execution (for logging, tracking, etc.).
    """

    def __init__(self) -> None:
        self._pre_hooks: List[HookEntry] = []
        self._post_hooks: List[HookEntry] = []

    def register_pre(
        self,
        name: str,
        fn: Callable[..., HookResult],
        priority: int = 0,
        enabled: bool = True,
    ) -> None:
        """Register a pre-execution hook."""
        self._pre_hooks.append(HookEntry(name=name, fn=fn, priority=priority, enabled=enabled))
        self._pre_hooks.sort(key=lambda h: h.priority)

    def register_post(
        self,
        name: str,
        fn: Callable[..., HookResult],
        priority: int = 0,
        enabled: bool = True,
    ) -> None:
        """Register a post-execution hook."""
        self._post_hooks.append(HookEntry(name=name, fn=fn, priority=priority, enabled=enabled))
        self._post_hooks.sort(key=lambda h: h.priority)

    def unregister(self, name: str) -> bool:
        """Unregister a hook by name. Returns True if found."""
        old_pre = len(self._pre_hooks)
        old_post = len(self._post_hooks)
        self._pre_hooks = [h for h in self._pre_hooks if h.name != name]
        self._post_hooks = [h for h in self._post_hooks if h.name != name]
        return len(self._pre_hooks) < old_pre or len(self._post_hooks) < old_post

    def enable(self, name: str) -> None:
        """Enable a hook by name."""
        for h in self._pre_hooks + self._post_hooks:
            if h.name == name:
                h.enabled = True

    def disable(self, name: str) -> None:
        """Disable a hook by name."""
        for h in self._pre_hooks + self._post_hooks:
            if h.name == name:
                h.enabled = False

    def run_pre(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> HookResult:
        """Run all pre-execution hooks in priority order.

        If any hook returns blocked=True, execution stops and the error
        is returned. Otherwise, args may be modified by hooks.

        Returns:
            HookResult with blocked/modified_args/metadata.
        """
        context = context or {}
        final_args = dict(args)

        for entry in self._pre_hooks:
            if not entry.enabled:
                continue
            try:
                result = entry.fn(tool_name, final_args, context)
                if result.blocked:
                    logger.info("pre-hook '%s' blocked '%s': %s",
                                entry.name, tool_name, result.error_message)
                    return result
                if result.modified_args:
                    final_args = result.modified_args
            except Exception as e:
                logger.warning("pre-hook '%s' failed (non-fatal): %s", entry.name, e)

        return HookResult(modified_args=final_args)

    def run_post(
        self,
        tool_name: str,
        result: Any,
        context: Dict[str, Any] = None,
    ) -> None:
        """Run all post-execution hooks in priority order."""
        context = context or {}

        for entry in self._post_hooks:
            if not entry.enabled:
                continue
            try:
                entry.fn(tool_name, result, context)
            except Exception as e:
                logger.warning("post-hook '%s' failed (non-fatal): %s", entry.name, e)

    @property
    def pre_hook_names(self) -> List[str]:
        """List of registered pre-hook names."""
        return [h.name for h in self._pre_hooks]

    @property
    def post_hook_names(self) -> List[str]:
        """List of registered post-hook names."""
        return [h.name for h in self._post_hooks]

    def stats(self) -> Dict[str, Any]:
        """Return middleware statistics."""
        return {
            "pre_hooks": len(self._pre_hooks),
            "post_hooks": len(self._post_hooks),
            "pre_hook_names": self.pre_hook_names,
            "post_hook_names": self.post_hook_names,
        }


# ── Built-in hooks ───────────────────────────────────────────────────

def tool_permission_hook(
    tool_name: str,
    args: Dict[str, Any],
    context: Dict[str, Any],
) -> HookResult:
    """Pre-hook: check tool permissions via tool_permissions module."""
    try:
        from agent.tool_permissions import check_tool_permission

        yolo = context.get("yolo_mode", False)
        interactive = context.get("interactive", True)

        decision = check_tool_permission(
            tool_name, args=args, yolo_mode=yolo, interactive=interactive,
        )

        if decision.blocked:
            return HookResult(
                blocked=True,
                error_message=decision.reason,
            )
        if decision.needs_approval:
            return HookResult(
                metadata={"needs_approval": True, "reason": decision.reason},
            )
        return HookResult()
    except Exception as e:
        logger.debug("tool_permission_hook failed: %s", e)
        # Fail-safe: on exception, require approval rather than auto-approve
        return HookResult(
            blocked=True,
            error_message=f"Permission check failed (fail-safe): {e}",
        )


def tool_result_tracking_hook(
    tool_name: str,
    result: Any,
    context: Dict[str, Any],
) -> HookResult:
    """Post-hook: track tool results for analytics/debugging."""
    # This is a placeholder for future analytics integration.
    # Currently just logs the tool call.
    return HookResult()
