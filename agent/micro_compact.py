"""Micro-compact: lightweight per-turn tool result pruning.

Inspired by Claude Code's microCompact strategy — clears the content of
old tool results to free context space BEFORE expensive LLM-based
compression kicks in.

Unlike full context compression (which uses an LLM to summarize),
microcompact is a cheap mechanical pass:
  - Keeps the N most recent tool results intact
  - Replaces older tool result content with a placeholder
  - Runs before each API call (zero LLM cost)

This dramatically reduces context bloat from repeated file reads,
terminal outputs, and web search results that accumulate in long sessions.

Configuration (config.yaml):
  agent:
    compression:
      microcompact:
        enabled: true        # default true
        keep_recent: 5       # keep this many recent tool results (default 5)
        min_result_chars: 200  # only compact results larger than this (default 200)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Placeholder text for cleared tool results
CLEARED_PLACEHOLDER = "[Old tool output cleared to save context space]"

# Tools whose results are worth compacting (high-output tools)
COMPACTABLE_TOOLS = {
    "read_file", "terminal", "search_files", "web_search", "web_extract",
    "browser_snapshot", "browser_vision", "vision_analyze",
    "execute_code", "delegate_task", "skill_view",
    "process", "send_message",
}


def microcompact_messages(
    messages: list,
    keep_recent: int = 5,
    min_result_chars: int = 200,
    compactable_tools: set = None,
) -> Tuple[list, Dict]:
    """Clear old tool results to free context space.

    Walks through messages and replaces the content of tool results
    that are older than the `keep_recent` most recent ones.

    Args:
        messages: The conversation messages list (mutated in-place).
        keep_recent: Number of most recent tool results to keep intact.
        min_result_chars: Only compact results with content longer than this.
        compactable_tools: Set of tool names to compact. Defaults to COMPACTABLE_TOOLS.

    Returns:
        (messages, stats) tuple. Messages is the same list (mutated).
        Stats dict has: compacted_count, chars_saved, total_tool_results.
    """
    if compactable_tools is None:
        compactable_tools = COMPACTABLE_TOOLS

    stats = {
        "compacted_count": 0,
        "chars_saved": 0,
        "total_tool_results": 0,
        "kept_recent": 0,
    }

    # First pass: collect all tool result positions (by message index)
    tool_result_positions: List[Tuple[int, int, str]] = []  # (msg_idx, block_idx, tool_name)

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")

        # Tool results are in user messages with role="tool" or in
        # user messages with tool_result content blocks
        if role == "tool":
            # OpenAI format: standalone tool message
            tool_name = msg.get("name", "")
            tool_call_id = msg.get("tool_call_id", "")
            tool_result_positions.append((msg_idx, -1, tool_name))
            stats["total_tool_results"] += 1
        elif role == "user" and isinstance(content, list):
            # Anthropic format: tool_result blocks inside user message
            for block_idx, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_result_positions.append((msg_idx, block_idx, ""))
                    stats["total_tool_results"] += 1

    # If we have fewer results than keep_recent, nothing to compact
    if len(tool_result_positions) <= keep_recent:
        return messages, stats

    # Determine which results to compact (all except the most recent N)
    # Note: list[:-0] returns empty in Python, handle specially
    if keep_recent == 0:
        to_compact = tool_result_positions
    else:
        to_compact = tool_result_positions[:-keep_recent]
    stats["kept_recent"] = min(keep_recent, len(tool_result_positions))

    # Second pass: compact old results
    for msg_idx, block_idx, tool_name in to_compact:
        msg = messages[msg_idx]
        if not isinstance(msg, dict):
            continue

        if block_idx == -1:
            # OpenAI format: standalone tool message
            content = msg.get("content", "")
            content_len = _content_length(content)
            if content_len >= min_result_chars:
                msg["content"] = CLEARED_PLACEHOLDER
                stats["compacted_count"] += 1
                stats["chars_saved"] += content_len - len(CLEARED_PLACEHOLDER)
        else:
            # Anthropic format: tool_result block inside user message
            content = msg.get("content", [])
            if isinstance(content, list) and block_idx < len(content):
                block = content[block_idx]
                if isinstance(block, dict):
                    block_content = block.get("content", "")
                    content_len = _content_length(block_content)
                    if content_len >= min_result_chars:
                        block["content"] = CLEARED_PLACEHOLDER
                        stats["compacted_count"] += 1
                        stats["chars_saved"] += content_len - len(CLEARED_PLACEHOLDER)

    if stats["compacted_count"] > 0:
        logger.info(
            "microcompact: compacted %d tool results, saved ~%d chars "
            "(kept %d recent, %d total)",
            stats["compacted_count"],
            stats["chars_saved"],
            stats["kept_recent"],
            stats["total_tool_results"],
        )

    return messages, stats


def _content_length(content) -> int:
    """Get the effective character length of content."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict):
                total += len(str(item.get("text", "")))
            elif isinstance(item, str):
                total += len(item)
        return total
    return len(str(content or ""))


def estimate_context_tokens(messages: list) -> int:
    """Rough token estimate for a message list.

    Uses ~4 chars per token as a heuristic. Better than nothing for
    threshold checks; the API returns exact counts after the call.
    """
    total_chars = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        total_chars += _content_length(content)
        # Add overhead for role, tool calls, etc.
        total_chars += 50
    return total_chars // 4
