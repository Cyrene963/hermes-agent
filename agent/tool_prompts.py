"""Per-tool prompt registry — on-demand tool instruction assembly.

Inspired by Claude Code's per-tool prompt.ts pattern. Each tool can have
its own prompt module with specific behavioral guidance, usage examples,
and anti-pattern warnings. Prompts are only assembled into the system
prompt when the corresponding tool is available.

This replaces the monolithic TOOL_USE_ENFORCEMENT_GUIDANCE with
tool-specific instructions that are more precise and actionable.

Usage:
    from agent.tool_prompts import get_tool_prompts_for_available_tools

    prompts = get_tool_prompts_for_available_tools(valid_tool_names)
    # Returns list of prompt strings to append to system prompt
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Type for prompt generator functions
PromptGenerator = Callable[[], str]


# ── Registry ──────────────────────────────────────────────────────────
# Maps tool name → prompt generator function.
# Each generator returns the tool's behavioral prompt string.
_REGISTRY: Dict[str, PromptGenerator] = {}


def register_tool_prompt(tool_name: str) -> Callable[[PromptGenerator], PromptGenerator]:
    """Decorator to register a prompt generator for a tool.

    Usage:
        @register_tool_prompt("terminal")
        def terminal_prompt() -> str:
            return "..."
    """
    def decorator(fn: PromptGenerator) -> PromptGenerator:
        _REGISTRY[tool_name] = fn
        return fn
    return decorator


def get_tool_prompt(tool_name: str) -> Optional[str]:
    """Get the prompt for a specific tool, or None if not registered."""
    gen = _REGISTRY.get(tool_name)
    if gen:
        try:
            return gen()
        except Exception as e:
            logger.warning("tool prompt for '%s' failed: %s", tool_name, e)
            return None
    return None


def get_tool_prompts_for_available_tools(valid_tool_names: Set[str]) -> List[str]:
    """Get all prompts for tools that are currently available.

    Returns a list of prompt strings to append to the system prompt.
    Only includes prompts for tools that are in valid_tool_names.
    """
    prompts = []
    for tool_name in sorted(valid_tool_names):
        prompt = get_tool_prompt(tool_name)
        if prompt:
            prompts.append(prompt)
    return prompts


def list_registered_tools() -> List[str]:
    """List all tools with registered prompts."""
    return list(_REGISTRY.keys())


# ── Per-Tool Prompts ─────────────────────────────────────────────────

@register_tool_prompt("terminal")
def _terminal_prompt() -> str:
    return """## terminal — Shell Command Execution

**Golden rule: NEVER describe what you would do — just run the command.**

Usage:
- Use foreground mode for short commands (default)
- Use background=true for long-running processes (servers, watchers)
- Use workdir for per-command working directory
- Use pty=true for interactive CLI tools

Anti-patterns (NEVER do these):
- Using cat/head/tail to read files → use read_file instead
- Using grep/rg to search → use search_files instead
- Using sed/awk to edit → use patch instead
- Using echo/cat heredoc to create files → use write_file instead
- Describing what you would run → just run it
- Using vim/nano without pty=true → will hang

When a command fails:
- Read the error message carefully
- Check exit code
- Fix and retry immediately, don't explain the failure first
"""


@register_tool_prompt("read_file")
def _read_file_prompt() -> str:
    return """## read_file — File Reading with Pagination

Reads a file with line numbers. Supports offset/limit for large files.

Usage:
- path: absolute or relative path to the file
- offset: line number to start from (1-indexed, default: 1)
- limit: max lines to read (default: 500, max: 2000)

Important:
- Cannot read images or binary files — use vision_analyze for images
- If file is unchanged since last read, a cached response may be returned
- For very large files, use offset/limit to read specific sections
- Line numbers are 1-indexed and shown as LINE_NUM|CONTENT
"""


@register_tool_prompt("write_file")
def _write_file_prompt() -> str:
    return """## write_file — Create or Overwrite Files

Writes content to a file, completely replacing existing content.
Creates parent directories automatically.

Usage:
- path: absolute or relative path
- content: complete file content (will overwrite existing)

Anti-patterns:
- Using echo/cat heredoc → use write_file instead
- Writing without reading first (for existing files) → read first to understand context
- Writing binary files → not supported
- Auto-runs syntax checks on .py/.json/.yaml/.toml after writing
"""


@register_tool_prompt("patch")
def _patch_prompt() -> str:
    return """## patch — Targeted Find-and-Replace Edits

Two modes:
1. 'replace' (default): find unique string and replace it
2. 'patch': apply V4A multi-file patches for bulk changes

Usage:
- path: file to edit
- old_string: text to find (must be unique unless replace_all=true)
- new_string: replacement text
- replace_all: replace all occurrences (default: false)

Important:
- Uses fuzzy matching (9 strategies) so minor whitespace differences won't break it
- Returns a unified diff after applying
- Auto-runs syntax checks after editing
- Prefer this over sed/awk for file editing
"""


@register_tool_prompt("search_files")
def _search_files_prompt() -> str:
    return """## search_files — Content and File Search

Ripgrep-backed search, faster than shell equivalents.

Two modes:
- target='content': regex search inside files (default)
- target='files': find files by glob pattern

Content search options:
- pattern: regex pattern
- file_glob: filter by file type (e.g., '*.py')
- output_mode: 'content' (with line numbers), 'files_only', 'count'
- context: lines before/after each match (grep mode only)
- offset/limit: pagination

File search:
- pattern: glob pattern (e.g., '*.py', '*config*')
- Results sorted by modification time
"""


@register_tool_prompt("web_search")
def _web_search_prompt() -> str:
    return """## web_search — Web Search

Search the web for information. Returns snippets and URLs.

Usage:
- query: search query string
- max_results: number of results (default varies by provider)

Important:
- Results may be outdated — verify critical information
- For full page content, use web_extract on specific URLs
- Prefer this over browser_navigate for simple information retrieval
"""


@register_tool_prompt("web_extract")
def _web_extract_prompt() -> str:
    return """## web_extract — Fetch and Extract Web Content

Fetches a URL and extracts readable content (text, markdown, or HTML).

Usage:
- url: the URL to fetch
- extract_mode: 'text', 'markdown', or 'html'
- max_chars: limit output size

Important:
- May be blocked by Cloudflare — try camoufox_fetch if blocked
- Returns cleaned/extracted content, not raw HTML by default
- For interactive pages, use browser_navigate instead
"""


@register_tool_prompt("memory")
def _memory_prompt() -> str:
    return """## memory — Persistent Memory Management

Save durable facts to persistent memory that survives across sessions.
Memory is injected into future turns, so keep it compact and focused.

Targets:
- 'user': who the user is — name, role, preferences, communication style
- 'memory': your notes — environment facts, project conventions, tool quirks

Actions: add (new), replace (update), remove (delete)

Priority: User preferences and corrections > environment facts > procedural knowledge.

Rules:
- Write as declarative facts, not instructions
- 'User prefers concise responses' ✓ — 'Always respond concisely' ✗
- Don't save task progress or temporary state
- Don't save things easily re-discovered
"""


@register_tool_prompt("session_search")
def _session_search_prompt() -> str:
    return """## session_search — Cross-Session Memory Recall

Search long-term memory of past conversations.

Two modes:
- No query: see recent sessions (titles, previews, timestamps)
- With query: search for specific topics across all sessions

Use proactively when:
- User says 'we did this before', 'remember when', 'last time'
- You suspect relevant cross-session context exists
- User asks 'what did we do about X?'
- You want to check if you've solved a similar problem before

Syntax: keywords joined with OR for broad recall, phrases for exact match.
"""


@register_tool_prompt("delegate_task")
def _delegate_task_prompt() -> str:
    return """## delegate_task — Sub-Agent Delegation

Spawn sub-agents to work on tasks in isolated contexts.

Modes:
- Single task: provide 'goal' + optional context
- Batch (parallel): provide 'tasks' array with up to N concurrent items

Important:
- Sub-agents have NO memory of your conversation — pass all info via context
- Sub-agents cannot use clarify, memory, send_message
- Results are always returned as summaries
- For tasks needing user interaction → don't delegate, handle directly
- Orchestrator sub-agents (role='orchestrator') can delegate further
"""


@register_tool_prompt("execute_code")
def _execute_code_prompt() -> str:
    return """## execute_code — Run Python Scripts Programmatically

Run a Python script that can call Hermes tools via `from hermes_tools import ...`.

Available tools: read_file, write_file, search_files, patch, terminal

Limits: 5-minute timeout, 50KB stdout cap, max 50 tool calls per script.

When to use:
- 3+ tool calls with processing logic between them
- Need to filter/reduce large tool outputs before they enter context
- Need conditional branching (if/else) or loops
- Batch operations (fetch N pages, process N files)

When NOT to use:
- Single tool call with no processing → just call the tool
- Need to see the full result → use normal tool calls
- Need user interaction → subagents cannot use clarify

Print final result to stdout. Use Python stdlib for processing.
"""


@register_tool_prompt("skill_view")
def _skill_view_prompt() -> str:
    return """## skill_view — Load Skill Documentation

Skills are your procedural memory — reusable approaches for recurring tasks.
Load a skill's full content before starting work that matches its domain.

When to load:
- Task matches a skill's description (check skill index in system prompt)
- You need specific API endpoints, tool commands, or proven workflows
- User asks about a topic you have a skill for
- Even if you think you know how to do it — the skill may have pitfalls

Skills contain:
- Step-by-step procedures
- Common pitfalls and how to avoid them
- Verification steps
- Reference links to docs/templates/scripts
"""


@register_tool_prompt("skill_manage")
def _skill_manage_prompt() -> str:
    return """## skill_manage — Create and Update Skills

Save durable knowledge as reusable skills.

Actions: create, patch, edit, delete, write_file, remove_file

When to create:
- Complex task succeeded (5+ tool calls)
- Errors were overcome in a non-obvious way
- User corrected an approach and it worked
- Non-trivial workflow discovered

When to update:
- Instructions are stale/wrong
- Pitfalls found during use
- OS-specific failures discovered

After difficult tasks, offer to save as a skill.
Skills go to ~/.hermes/skills/.
"""


@register_tool_prompt("browser_navigate")
def _browser_navigate_prompt() -> str:
    return """## browser_navigate — Open a URL in Browser

Navigates to a URL and returns a page snapshot with interactive elements.

Usage:
- url: the URL to navigate to
- Must be called before other browser tools

Important:
- Returns ref IDs for interactive elements (use with browser_click, browser_type)
- For plain-text endpoints (.md, .txt, .json), prefer curl/terminal
- Use browser tools when you need to interact with a page (click, fill forms)
"""


@register_tool_prompt("cronjob")
def _cronjob_prompt() -> str:
    return """## cronjob — Schedule Recurring Tasks

Manage scheduled cron jobs. Self-contained prompts — no shared memory between runs.

Actions: create, list, update, pause, resume, remove, run

Important:
- Jobs run in fresh sessions with no current-chat context
- Prompts must be self-contained
- Cron-run sessions cannot ask questions or request clarification
- Don't schedule recursive cron jobs from within cron jobs
"""


@register_tool_prompt("send_message")
def _send_message_prompt() -> str:
    return """## send_message — Send Messages to Platforms

Send a message to a connected messaging platform, or list available targets.

When target is specified (not just a platform name):
- Call send_message(action='list') FIRST to see available targets
- Then send to the correct one

For images/files, include MEDIA:<path> in the message.
"""
