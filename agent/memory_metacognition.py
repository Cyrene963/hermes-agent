"""Memory Metacognition Framework — interfaces + policy loader.

Provides three pluggable extension points for enhancing the agent's
awareness of its own persistent memory:

1. MemoryIndexProvider  — generates a compact index of what's in the memory store
2. RecallQueryExpander  — expands user messages into better hindsight queries
3. MemoryPreflightPolicy — gates dangerous operations based on memory state

All three are loaded from a YAML policy file. The framework ships with
no-op defaults; users/deployers customize via ~/.hermes/memory_policy.yaml.

Design principles:
- Core code contains ZERO hardcoded deployment-specific data (no user IDs,
  no platform-specific rules, no language-specific mappings).
- Default behavior is conservative: warn-only, never block.
- Policy is loaded once per process and cached.
- All failures are non-blocking (agent continues without enhancement).
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Data types ──────────────────────────────────────────────────────

@dataclass
class PreflightCheck:
    """Result of a single preflight check."""
    check_type: str           # e.g. "identity", "method", "safety"
    query: str                # hindsight query used
    required: bool            # whether this check is mandatory
    found: bool               # whether relevant memory was found
    status: str               # "PASS" | "WARN" | "FAIL"
    message: str = ""         # human-readable explanation
    top_memory: str = ""      # relevant memory snippet if found


@dataclass
class PreflightResult:
    """Aggregated preflight result for a high-risk task."""
    decision: str             # "allow" | "warn" | "block"
    reason: str
    task_type: str
    checks: List[PreflightCheck] = field(default_factory=list)


# ─── Abstract interfaces ─────────────────────────────────────────────

class MemoryIndexProvider(ABC):
    """Generates a compact index of the agent's memory store.

    The index is injected into the system prompt once per session so the
    model knows what categories and recent memories exist, without needing
    to search proactively.
    """

    @abstractmethod
    def build_index(self) -> str:
        """Return compact memory index text (ideally 200-300 tokens).

        Returns empty string on failure. Must never raise.
        """
        ...


class RecallQueryExpander(ABC):
    """Expands a user message into better hindsight search queries.

    The default implementation returns [original_message] only.
    Policies can add keyword→expansion mappings for their language/domain.
    """

    @abstractmethod
    def expand(self, user_message: str, max_queries: int = 8) -> List[str]:
        """Return [original, expanded1, expanded2, ...] capped at max_queries."""
        ...


class MemoryPreflightPolicy(ABC):
    """Gates dangerous operations based on memory state.

    The default implementation allows all operations (no-op).
    Policies define which tool+arg combinations are high-risk and what
    memory checks are required.
    """

    @abstractmethod
    def get_task_type(self, tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
        """Return task type string if this tool call is high-risk, else None."""
        ...

    @abstractmethod
    def run_checks(self, task_type: str, context: Dict[str, Any]) -> PreflightResult:
        """Run preflight checks for a high-risk task.

        Returns PreflightResult with decision=allow/warn/block.
        """
        ...


# ─── Default no-op implementations ───────────────────────────────────

class NoOpIndexProvider(MemoryIndexProvider):
    """Default: no memory index injected."""
    def build_index(self) -> str:
        return ""


class PassthroughExpander(RecallQueryExpander):
    """Default: returns only the original message (no expansion)."""
    def expand(self, user_message: str, max_queries: int = 8) -> List[str]:
        if not user_message:
            return []
        return [user_message]


class NoOpPreflightPolicy(MemoryPreflightPolicy):
    """Default: all operations allowed, no checks."""
    def get_task_type(self, tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
        return None

    def run_checks(self, task_type: str, context: Dict[str, Any]) -> PreflightResult:
        return PreflightResult(
            decision="allow",
            reason="No preflight policy configured",
            task_type=task_type or "",
        )


# ─── Script-backed implementations ──────────────────────────────────

class ScriptIndexProvider(MemoryIndexProvider):
    """Generates memory index by calling an external script.

    Looks for the script at ~/.hermes/scripts/memory_index_generator.py.
    Falls back to no-op if script is missing or fails.
    """

    def __init__(self, script_dir: Optional[str] = None, timeout: int = 5):
        self._script_dir = script_dir or os.path.join(
            os.path.expanduser("~"), ".hermes", "scripts"
        )
        self._timeout = timeout

    def build_index(self) -> str:
        try:
            import subprocess
            import sys as _sys
            script = os.path.join(self._script_dir, "memory_index_generator.py")
            if not os.path.exists(script):
                return ""
            result = subprocess.run(
                [_sys.executable, script, "--limit", "10"],
                capture_output=True, text=True, timeout=self._timeout,
            )
            if result.returncode == 0 and result.stdout.strip():
                output = result.stdout.strip()
                if len(output) > 800:
                    output = output[:797] + "..."
                return f"# Memory Index (auto-generated, session start)\n{output}"
        except Exception:
            pass
        return ""


class PolicyQueryExpander(RecallQueryExpander):
    """Expands queries using keyword→terms mappings from policy."""

    def __init__(self, expansions: Optional[Dict[str, List[str]]] = None,
                 max_queries: int = 8):
        self._expansions = expansions or {}
        self._max_queries = max_queries

    def expand(self, user_message: str, max_queries: int = 8) -> List[str]:
        if not user_message:
            return []
        cap = min(max_queries, self._max_queries)
        queries = [user_message]
        msg_lower = user_message.lower()
        for keyword, terms in self._expansions.items():
            if keyword.lower() in msg_lower:
                for term in terms:
                    if term.lower() not in [q.lower() for q in queries]:
                        queries.append(term)
                        if len(queries) >= cap:
                            return queries
        return queries


class PolicyPreflightPolicy(MemoryPreflightPolicy):
    """Preflight policy loaded from YAML config.

    Supports two categories of checks:

    1. Memory recall checks (backward compatible):
       - type: memory / memory_recall / identity / method / safety
       - Searches hindsight for the query string
       - PASS if found, FAIL if required and not found

    2. Structured argument checks (new):
       - type: field_required      — field must exist in context
       - type: field_equals        — field must equal value
       - type: field_not_equals    — field must NOT equal value
       - type: field_contains      — field must contain value
       - type: field_not_contains  — field must NOT contain value

    Config format:
      preflight_rules:
        - tool: send_message
          task_type: outgoing_message
          checks:
            - type: field_required
              field: chat_id
              required: true
              error: "chat_id is required"
            - type: field_equals
              field: method
              value: sendDocument
              required: true
              error: "must use sendDocument"
            - type: memory_recall
              query: "sendDocument usage"
              required: false
              error: "no related memory"
          block_on_failure: true
    """

    def __init__(self, rules: Optional[List[Dict]] = None,
                 hindsight_api: str = "http://localhost:9177",
                 bank_id: str = "hindsight"):
        self._rules = rules or []
        self._hindsight_api = hindsight_api
        self._bank_id = bank_id
        # Build lookup: tool_name → [rule, ...]
        self._tool_map: Dict[str, List[Dict]] = {}
        for rule in self._rules:
            tool = rule.get("tool", "")
            if tool:
                self._tool_map.setdefault(tool, []).append(rule)

    def get_task_type(self, tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
        rules = self._tool_map.get(tool_name, [])
        for rule in rules:
            patterns = rule.get("trigger_patterns", [])
            if patterns:
                # Build searchable string from ALL tool_args keys AND values
                # so patterns can match field names or field values.
                parts = []
                for k, v in tool_args.items():
                    parts.append(str(k))
                    if isinstance(v, str):
                        parts.append(v)
                arg_text = " ".join(parts).lower()
                if not any(p.lower() in arg_text for p in patterns):
                    continue
            return rule.get("task_type", tool_name)
        return None

    def run_checks(self, task_type: str, context: Dict[str, Any]) -> PreflightResult:
        # Find the rule for this task type
        rule = None
        for r in self._rules:
            if r.get("task_type") == task_type:
                rule = r
                break
        if not rule:
            return PreflightResult(
                decision="allow",
                reason=f"No rule for task type: {task_type}",
                task_type=task_type,
            )

        # Build enriched context: top-level fields + tool_args sub-dict
        # This allows structured checks to access both context["method"]
        # and context["tool_args"]["method"] patterns.
        enriched = dict(context)
        if "tool_args" not in enriched:
            enriched["tool_args"] = dict(context)

        checks = []
        all_passed = True
        should_block = rule.get("block_on_failure", False)

        for check_def in rule.get("checks", []):
            check = self._run_single_check(check_def, enriched)
            checks.append(check)
            if check.status == "FAIL":
                all_passed = False

        if not all_passed and should_block:
            decision = "block"
            reason = "Critical preflight checks failed"
        elif not all_passed:
            decision = "warn"
            reason = "Some recommended checks failed"
        else:
            decision = "allow"
            reason = "All checks passed"

        return PreflightResult(
            decision=decision,
            reason=reason,
            task_type=task_type,
            checks=checks,
        )

    def _run_single_check(self, check_def: Dict, context: Dict) -> PreflightCheck:
        """Dispatch a single preflight check to the appropriate handler."""
        check_type = check_def.get("type", "unknown")
        required = check_def.get("required", False)
        error_msg = check_def.get("error", "")

        # ── Structured argument checks ──
        if check_type == "field_required":
            return self._check_field_required(check_def, context, required, error_msg)
        if check_type == "field_equals":
            return self._check_field_equals(check_def, context, required, error_msg)
        if check_type == "field_not_equals":
            return self._check_field_not_equals(check_def, context, required, error_msg)
        if check_type == "field_contains":
            return self._check_field_contains(check_def, context, required, error_msg)
        if check_type == "field_not_contains":
            return self._check_field_not_contains(check_def, context, required, error_msg)

        # ── Memory recall check (backward compatible) ──
        # Covers: memory, memory_recall, identity, method, safety, unknown
        return self._check_memory_recall(check_def, context, required, error_msg, check_type)

    def _resolve_field(self, context: Dict, field: str) -> Any:
        """Resolve a field from context. Checks top-level first, then tool_args."""
        if field in context:
            return context[field]
        tool_args = context.get("tool_args", {})
        if field in tool_args:
            return tool_args[field]
        return None

    def _check_field_required(self, check_def: Dict, context: Dict,
                               required: bool, error_msg: str) -> PreflightCheck:
        field = check_def.get("field", "")
        value = self._resolve_field(context, field)
        found = value is not None and value != ""
        status = "PASS" if found else ("FAIL" if required else "WARN")
        return PreflightCheck(
            check_type="field_required", query=field,
            required=required, found=found, status=status,
            message=error_msg if status != "PASS" else "",
        )

    def _check_field_equals(self, check_def: Dict, context: Dict,
                             required: bool, error_msg: str) -> PreflightCheck:
        field = check_def.get("field", "")
        expected = check_def.get("value", "")
        actual = self._resolve_field(context, field)
        found = actual is not None and str(actual) == str(expected)
        status = "PASS" if found else ("FAIL" if required else "WARN")
        return PreflightCheck(
            check_type="field_equals", query=f"{field}=={expected}",
            required=required, found=found, status=status,
            message=error_msg if status != "PASS" else "",
        )

    def _check_field_not_equals(self, check_def: Dict, context: Dict,
                                 required: bool, error_msg: str) -> PreflightCheck:
        field = check_def.get("field", "")
        forbidden = check_def.get("value", "")
        actual = self._resolve_field(context, field)
        # PASS if field doesn't exist OR doesn't equal forbidden value
        found = actual is not None and str(actual) == str(forbidden)
        status = "FAIL" if (found and required) else ("WARN" if found else "PASS")
        return PreflightCheck(
            check_type="field_not_equals", query=f"{field}!={forbidden}",
            required=required, found=not found, status=status,
            message=error_msg if status != "PASS" else "",
        )

    def _check_field_contains(self, check_def: Dict, context: Dict,
                               required: bool, error_msg: str) -> PreflightCheck:
        field = check_def.get("field", "")
        needle = check_def.get("value", "")
        actual = self._resolve_field(context, field)
        found = actual is not None and needle in str(actual)
        status = "PASS" if found else ("FAIL" if required else "WARN")
        return PreflightCheck(
            check_type="field_contains", query=f"{field}~={needle}",
            required=required, found=found, status=status,
            message=error_msg if status != "PASS" else "",
        )

    def _check_field_not_contains(self, check_def: Dict, context: Dict,
                                   required: bool, error_msg: str) -> PreflightCheck:
        field = check_def.get("field", "")
        needle = check_def.get("value", "")
        actual = self._resolve_field(context, field)
        contains = actual is not None and needle in str(actual)
        status = "FAIL" if (contains and required) else ("WARN" if contains else "PASS")
        return PreflightCheck(
            check_type="field_not_contains", query=f"{field}!~={needle}",
            required=required, found=not contains, status=status,
            message=error_msg if status != "PASS" else "",
        )

    def _check_memory_recall(self, check_def: Dict, context: Dict,
                              required: bool, error_msg: str,
                              check_type: str) -> PreflightCheck:
        """Legacy memory recall check. Searches hindsight for query."""
        query_template = check_def.get("query", "")
        query = query_template
        for k, v in context.items():
            if isinstance(v, str):
                query = query.replace(f"{{{k}}}", v)

        found = False
        top_memory = ""
        try:
            import json
            import urllib.request
            url = f"{self._hindsight_api}/v1/default/banks/{self._bank_id}/memories/recall"
            data = json.dumps({"query": query, "limit": 2}).encode()
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read())
                memories = result.get("memories", result.get("results", []))
                found = len(memories) > 0
                if found:
                    top_memory = memories[0].get("text", "")[:100]
        except Exception:
            pass

        if required and not found:
            status = "FAIL"
        elif not found:
            status = "WARN"
        else:
            status = "PASS"

        return PreflightCheck(
            check_type=check_type, query=query,
            required=required, found=found, status=status,
            message=error_msg if status != "PASS" else "",
            top_memory=top_memory,
        )


# ─── Policy loader ───────────────────────────────────────────────────

_POLICY_CACHE: Dict[Optional[str], Dict] = {}  # keyed by user_id (None = global)
_POLICY_FILE_PATHS = [
    # Local private policy (user-specific, not committed)
    os.path.join(os.path.expanduser("~"), ".hermes", "memory_policy.yaml"),
]


def _find_default_policy() -> Optional[str]:
    """Find the default policy YAML bundled with the agent."""
    # Look relative to this file
    here = Path(__file__).parent
    candidate = here / "memory_policy.default.yaml"
    if candidate.exists():
        return str(candidate)
    return None


def _get_user_policy_path(user_id: Optional[str]) -> Optional[str]:
    """Get user-specific policy file path."""
    if not user_id:
        return None
    return os.path.join(
        os.path.expanduser("~"), ".hermes", "memories",
        f"user_{user_id}", "memory_policy.yaml"
    )


def _resolve_bank_id(user_context: Optional[Dict] = None) -> str:
    """Resolve hindsight bank_id with per-user isolation.

    Priority:
    1. user_context["bank_id"] if explicitly provided
    2. "hindsight-{user_id}" if user_id is available
    3. "hindsight" (default, no isolation)
    """
    if user_context:
        if user_context.get("bank_id"):
            return user_context["bank_id"]
        if user_context.get("user_id"):
            return f"hindsight-{user_context['user_id']}"
    return "hindsight"


def load_policy(force_reload: bool = False,
                user_context: Optional[Dict] = None) -> Dict[str, Any]:
    """Load and merge memory metacognition policy.

    With user_context, loads per-user policy overlay on top of global policy.
    Cache is per-user (keyed by user_id).

    Priority: user private > local private > default bundled.
    """
    cache_key = user_context.get("user_id") if user_context else None

    if cache_key in _POLICY_CACHE and not force_reload:
        return _POLICY_CACHE[cache_key]

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML unavailable; memory metacognition policy disabled")
        _POLICY_CACHE[cache_key] = {}
        return _POLICY_CACHE[cache_key]

    merged: Dict[str, Any] = {}

    # 1. Load default (bundled) policy
    default_path = _find_default_policy()
    if default_path:
        try:
            with open(default_path) as f:
                merged = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Failed to load default memory policy: %s", e)

    # 2. Overlay local private policy (skipped if HERMES_MEMORY_POLICY_DISABLE_LOCAL=1)
    if os.environ.get("HERMES_MEMORY_POLICY_DISABLE_LOCAL") == "1":
        logger.debug("Local memory policy disabled via env var")
    else:
        for path in _POLICY_FILE_PATHS:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        local = yaml.safe_load(f) or {}
                    _deep_merge(merged, local)
                    logger.info("Loaded memory policy from %s", path)
                except Exception as e:
                    logger.warning("Failed to load memory policy from %s: %s", path, e)

        # 3. Overlay user-specific policy (if user_context provided)
        if user_context and user_context.get("user_id"):
            user_path = _get_user_policy_path(user_context["user_id"])
            if user_path and os.path.exists(user_path):
                try:
                    with open(user_path) as f:
                        user_policy = yaml.safe_load(f) or {}
                    _deep_merge(merged, user_policy)
                    logger.info("Loaded user policy from %s", user_path)
                except Exception as e:
                    logger.warning("Failed to load user policy from %s: %s", user_path, e)

    _POLICY_CACHE[cache_key] = merged
    return merged


def _deep_merge(base: Dict, overlay: Dict) -> Dict:
    """Recursively merge overlay into base (overlay wins)."""
    for k, v in overlay.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def build_index_provider(user_context: Optional[Dict] = None) -> MemoryIndexProvider:
    """Build MemoryIndexProvider from policy config."""
    policy = load_policy(user_context=user_context)
    index_cfg = policy.get("memory_index", {})
    if not index_cfg.get("enabled", False):
        return NoOpIndexProvider()
    script_dir = index_cfg.get("script_dir")
    timeout = index_cfg.get("timeout", 5)
    return ScriptIndexProvider(script_dir=script_dir, timeout=timeout)


def build_query_expander(user_context: Optional[Dict] = None) -> RecallQueryExpander:
    """Build RecallQueryExpander from policy config.

    With user_context, merges global expansions with user-specific expansions.
    """
    policy = load_policy(user_context=user_context)
    expansion_cfg = policy.get("query_expansion", {})
    if not expansion_cfg.get("enabled", False):
        return PassthroughExpander()
    expansions = expansion_cfg.get("expansions", {})
    max_queries = expansion_cfg.get("max_queries", 8)
    return PolicyQueryExpander(expansions=expansions, max_queries=max_queries)


def build_preflight_policy(user_context: Optional[Dict] = None) -> MemoryPreflightPolicy:
    """Build MemoryPreflightPolicy from policy config.

    With user_context, uses per-user bank_id for memory recall checks.
    """
    policy = load_policy(user_context=user_context)
    preflight_cfg = policy.get("preflight", {})
    if not preflight_cfg.get("enabled", False):
        return NoOpPreflightPolicy()
    rules = preflight_cfg.get("rules", [])
    api = preflight_cfg.get("hindsight_api", "http://localhost:9177")
    bank = _resolve_bank_id(user_context) or preflight_cfg.get("bank_id", "hindsight")
    return PolicyPreflightPolicy(rules=rules, hindsight_api=api, bank_id=bank)


# ─── Task Routing / Strategy Recall ──────────────────────────────────

@dataclass
class StrategyHint:
    """Output from task routing preflight. Injected into agent planning."""
    task_type: str                    # e.g. "cloudflare_site_access"
    recommended_strategy: str         # e.g. "use_camoufox"
    avoid_methods: List[str]          # e.g. ["browser_navigate", "curl"]
    preferred_method: str             # e.g. "camoufox"
    reason: str                       # human-readable explanation
    confidence: str                   # "high", "medium", "low"
    recall_hits: int                  # how many memories matched


class TaskRoutingPreflight:
    """Strategy recall gate — runs BEFORE tool selection.

    Matches user_message against routing rules. If triggered, searches
    hindsight for strategy memories and returns a StrategyHint that
    should be injected into the agent's planning context.

    Default: disabled (no-op). Controlled by routing_preflight in policy.
    """

    def __init__(self, rules: Optional[List[Dict]] = None,
                 hindsight_api: str = "http://localhost:9177",
                 bank_id: str = "hindsight"):
        self._rules = rules or []
        self._hindsight_api = hindsight_api
        self._bank_id = bank_id

    def check(self, user_message: str) -> Optional[StrategyHint]:
        """Check user_message against routing rules. Returns StrategyHint or None."""
        if not user_message or not self._rules:
            return None

        msg_lower = user_message.lower()

        for rule in self._rules:
            patterns = rule.get("trigger_patterns", [])
            if not patterns:
                continue

            # Match trigger patterns against user message
            if not any(p.lower() in msg_lower for p in patterns):
                continue

            # Triggered — run recall for strategy memories
            recall_queries = rule.get("recall_queries", [])
            if not recall_queries:
                recall_queries = [user_message]

            total_hits = 0
            top_reasons = []
            for query in recall_queries[:3]:  # Cap at 3 queries
                try:
                    hits = self._recall(query)
                    total_hits += len(hits)
                    for h in hits[:2]:
                        text = h.get("text", "")[:150]
                        if text:
                            top_reasons.append(text)
                except Exception:
                    pass

            # Build confidence from hit count
            if total_hits >= 3:
                confidence = "high"
            elif total_hits >= 1:
                confidence = "medium"
            else:
                confidence = "low"

            return StrategyHint(
                task_type=rule.get("task_type", "unknown"),
                recommended_strategy=rule.get("preferred_method", ""),
                avoid_methods=rule.get("avoid_methods", []),
                preferred_method=rule.get("preferred_method", ""),
                reason=rule.get("strategy_hint", "") or "; ".join(top_reasons[:2]),
                confidence=confidence,
                recall_hits=total_hits,
            )

        return None

    def _recall(self, query: str) -> List[Dict]:
        """Search hindsight for strategy memories."""
        try:
            import json
            import urllib.request
            url = f"{self._hindsight_api}/v1/default/banks/{self._bank_id}/memories/recall"
            data = json.dumps({"query": query, "budget": "low", "max_tokens": 512}).encode()
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=8) as resp:
                result = json.loads(resp.read())
                return result.get("results", result.get("memories", []))
        except Exception:
            return []


def build_strategy_preflight(user_context: Optional[Dict] = None) -> TaskRoutingPreflight:
    """Build TaskRoutingPreflight from policy config."""
    policy = load_policy(user_context=user_context)
    routing_cfg = policy.get("routing_preflight", {})
    if not routing_cfg.get("enabled", False):
        return TaskRoutingPreflight()  # no-op (empty rules)
    rules = routing_cfg.get("rules", [])
    api = routing_cfg.get("hindsight_api",
                           policy.get("preflight", {}).get("hindsight_api", "http://localhost:9177"))
    bank = _resolve_bank_id(user_context) or routing_cfg.get("bank_id",
                           policy.get("preflight", {}).get("bank_id", "hindsight"))
    return TaskRoutingPreflight(rules=rules, hindsight_api=api, bank_id=bank)
