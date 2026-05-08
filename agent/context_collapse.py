"""Context collapse: fold long tool outputs into compact summaries.

Inspired by Claude Code's CONTEXT_COLLAPSE strategy — replaces very long
tool results with a compact representation (head + tail + metadata) to
free context space without losing critical information.

Unlike microcompact (which clears old results entirely), contextCollapse
preserves the beginning and end of each result, which typically contain
the most important information (file headers, final output, error messages).

This is a zero-LLM-cost operation — no summarization model needed.

Usage:
    from agent.context_collapse import collapse_messages

    messages, stats = collapse_messages(messages, max_chars=2000, head_chars=500, tail_chars=500)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Template for collapsed content
_COLLAPSE_TEMPLATE = (
    "{head}\n\n"
    "... [{omitted} chars, {omitted_lines} lines] ...\n\n"
    "{tail}"
)


def collapse_tool_content(
    content: str,
    max_chars: int = 2000,
    head_chars: int = 500,
    tail_chars: int = 500,
) -> Tuple[str, bool]:
    """Collapse a single tool result content if it exceeds max_chars.

    Args:
        content: The tool result content string.
        max_chars: Only collapse if content exceeds this length.
        head_chars: Number of characters to keep from the beginning.
        tail_chars: Number of characters to keep from the end.

    Returns:
        (collapsed_content, was_collapsed) tuple.
    """
    if not content or len(content) <= max_chars:
        return content, False

    # Calculate what we're omitting
    head = content[:head_chars]
    tail = content[-tail_chars:]
    omitted = len(content) - head_chars - tail_chars
    omitted_lines = content[head_chars:-tail_chars].count('\n') + 1

    collapsed = _COLLAPSE_TEMPLATE.format(
        head=head,
        tail=tail,
        omitted=max(0, omitted),
        omitted_lines=max(1, omitted_lines),
    )

    return collapsed, True


def collapse_messages(
    messages: list,
    max_chars: int = 2000,
    head_chars: int = 500,
    tail_chars: int = 500,
    compactable_tools: set = None,
) -> Tuple[list, Dict]:
    """Collapse long tool outputs in-place across all messages.

    Walks through messages and collapses tool results that exceed max_chars.
    Only affects old tool results (keeps the most recent N intact via
    integration with microcompact's keep_recent).

    Args:
        messages: The conversation messages list (mutated in-place).
        max_chars: Only collapse results longer than this.
        head_chars: Characters to keep from start.
        tail_chars: Characters to keep from end.
        compactable_tools: Tool names to collapse. None = all tools.

    Returns:
        (messages, stats) tuple. Messages is the same list (mutated).
        Stats dict has: collapsed_count, chars_saved, total_checked.
    """
    stats = {
        "collapsed_count": 0,
        "chars_saved": 0,
        "total_checked": 0,
    }

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = msg.get("content")

        if role == "tool":
            stats["total_checked"] += 1
            if isinstance(content, str) and len(content) > max_chars:
                collapsed, was_collapsed = collapse_tool_content(
                    content, max_chars, head_chars, tail_chars,
                )
                if was_collapsed:
                    msg["content"] = collapsed
                    stats["collapsed_count"] += 1
                    stats["chars_saved"] += len(content) - len(collapsed)

        elif role == "user" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    stats["total_checked"] += 1
                    block_content = block.get("content", "")
                    if isinstance(block_content, str) and len(block_content) > max_chars:
                        collapsed, was_collapsed = collapse_tool_content(
                            block_content, max_chars, head_chars, tail_chars,
                        )
                        if was_collapsed:
                            block["content"] = collapsed
                            stats["collapsed_count"] += 1
                            stats["chars_saved"] += len(block_content) - len(collapsed)

    if stats["collapsed_count"] > 0:
        logger.info(
            "context_collapse: collapsed %d tool results, saved ~%d chars "
            "(checked %d total)",
            stats["collapsed_count"],
            stats["chars_saved"],
            stats["total_checked"],
        )

    return messages, stats
