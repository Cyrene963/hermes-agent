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

    Config format:
      preflight_rules:
        - tool: send_message
          task_type: outgoing_message
          checks:
            - type: identity
              query: recipient
              required: true
              error: "recipient not verified"
          block_on_failure: true
    """

    def __init__(self, rules: Optional[List[Dict]] = None,
                 hindsight_api: str = "http://localhost:9177",
                 bank_id: str = "hindsight"):
        self._rules = rules or []
        self._hindsight_api = hindsight_api
        self._bank_id = bank_id
        # Build lookup: tool_name → rule
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
                cmd = tool_args.get("command", "").lower()
                if not any(p.lower() in cmd for p in patterns):
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

        checks = []
        all_passed = True
        should_block = rule.get("block_on_failure", False)

        for check_def in rule.get("checks", []):
            check = self._run_single_check(check_def, context)
            checks.append(check)
            if check.status == "FAIL":
                all_passed = False

        if not all_passed and should_block:
            decision = "block"
            reason = "Critical memory checks failed"
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
        """Run a single preflight check against hindsight."""
        check_type = check_def.get("type", "unknown")
        query_template = check_def.get("query", "")
        required = check_def.get("required", False)
        error_msg = check_def.get("error", "")

        # Interpolate context into query
        query = query_template
        for k, v in context.items():
            query = query.replace(f"{{{k}}}", str(v))

        # Search hindsight
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
                memories = result.get("memories", [])
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
            check_type=check_type,
            query=query,
            required=required,
            found=found,
            status=status,
            message=error_msg if status != "PASS" else "",
            top_memory=top_memory,
        )


# ─── Policy loader ───────────────────────────────────────────────────

_POLICY_CACHE: Optional[Dict] = None
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


def load_policy(force_reload: bool = False) -> Dict[str, Any]:
    """Load and merge memory metacognition policy.

    Priority: local private > default bundled.
    Returns dict with keys: index_provider, expander, preflight_policy.
    Cached after first load unless force_reload=True.
    """
    global _POLICY_CACHE
    if _POLICY_CACHE is not None and not force_reload:
        return _POLICY_CACHE

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML unavailable; memory metacognition policy disabled")
        _POLICY_CACHE = {}
        return _POLICY_CACHE

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

    _POLICY_CACHE = merged
    return merged


def _deep_merge(base: Dict, overlay: Dict) -> Dict:
    """Recursively merge overlay into base (overlay wins)."""
    for k, v in overlay.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def build_index_provider() -> MemoryIndexProvider:
    """Build MemoryIndexProvider from policy config."""
    policy = load_policy()
    index_cfg = policy.get("memory_index", {})
    if not index_cfg.get("enabled", False):
        return NoOpIndexProvider()
    script_dir = index_cfg.get("script_dir")
    timeout = index_cfg.get("timeout", 5)
    return ScriptIndexProvider(script_dir=script_dir, timeout=timeout)


def build_query_expander() -> RecallQueryExpander:
    """Build RecallQueryExpander from policy config."""
    policy = load_policy()
    expansion_cfg = policy.get("query_expansion", {})
    if not expansion_cfg.get("enabled", False):
        return PassthroughExpander()
    expansions = expansion_cfg.get("expansions", {})
    max_queries = expansion_cfg.get("max_queries", 8)
    return PolicyQueryExpander(expansions=expansions, max_queries=max_queries)


def build_preflight_policy() -> MemoryPreflightPolicy:
    """Build MemoryPreflightPolicy from policy config."""
    policy = load_policy()
    preflight_cfg = policy.get("preflight", {})
    if not preflight_cfg.get("enabled", False):
        return NoOpPreflightPolicy()
    rules = preflight_cfg.get("rules", [])
    api = preflight_cfg.get("hindsight_api", "http://localhost:9177")
    bank = preflight_cfg.get("bank_id", "hindsight")
    return PolicyPreflightPolicy(rules=rules, hindsight_api=api, bank_id=bank)
