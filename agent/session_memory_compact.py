"""Session memory compaction — preserve memory authority during compression.

Inspired by Claude Code's sessionMemoryCompact. Ensures that when context
is compressed, the user's persistent memory (MEMORY.md, USER.md) and
critical rules are preserved with full authority — not demoted to
"background reference" by the compression summary.

The problem:
  When context_compressor.py compresses messages, the SUMMARY_PREFIX tells
  the model to treat compressed content as "background reference". This
  inadvertently demotes memory authority — the model starts ignoring
  memory rules after compression.

The solution:
  1. Before compression: capture current memory state
  2. After compression: re-inject memory with STRONGER authority markers
  3. Add a post-compaction memory block that overrides the summary's
     "background reference" framing

Usage:
    from agent.session_memory_compact import (
        extract_memory_before_compact,
        reinject_memory_after_compact,
    )

    # Before compression:
    memory_snapshot = extract_memory_before_compact(memory_store, memory_manager)

    # After compression:
    messages = reinject_memory_after_compact(messages, memory_snapshot)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Strong authority marker injected after compression
MEMORY_AUTHORITY_MARKER = (
    "## ⚠️ ACTIVE MEMORY — AUTHORITATIVE (POST-COMPACTION)\n\n"
    "The following memory rules are MANDATORY and OVERRIDE any "
    "'background reference' framing in the compaction summary above. "
    "These rules are ALWAYS ACTIVE regardless of context compression:\n\n"
)


def extract_memory_before_compact(
    memory_store: Any = None,
    memory_manager: Any = None,
) -> Dict[str, str]:
    """Capture current memory state before compression.

    Returns a dict with keys: 'memory', 'user', 'external'.
    Each value is the formatted memory content, or empty string if unavailable.
    """
    snapshot = {
        "memory": "",
        "user": "",
        "external": "",
    }

    if memory_store:
        try:
            mem_block = memory_store.format_for_system_prompt("memory")
            if mem_block:
                snapshot["memory"] = mem_block
        except Exception as e:
            logger.debug("memory extraction failed: %s", e)

        try:
            user_block = memory_store.format_for_system_prompt("user")
            if user_block:
                snapshot["user"] = user_block
        except Exception as e:
            logger.debug("user profile extraction failed: %s", e)

    if memory_manager:
        try:
            ext_block = memory_manager.build_system_prompt()
            if ext_block:
                snapshot["external"] = ext_block
        except Exception as e:
            logger.debug("external memory extraction failed: %s", e)

    return snapshot


def reinject_memory_after_compact(
    messages: list,
    memory_snapshot: Dict[str, str],
) -> list:
    """Re-inject memory into messages after compression with authority markers.

    Finds the compaction summary message and prepends a strong memory
    authority block that overrides the summary's "background reference" framing.

    Args:
        messages: The compressed message list.
        memory_snapshot: Output from extract_memory_before_compact().

    Returns:
        The messages list with memory re-injected (mutated in-place).
    """
    # Build the memory re-injection block
    memory_parts = []
    if memory_snapshot.get("memory"):
        memory_parts.append(memory_snapshot["memory"])
    if memory_snapshot.get("user"):
        memory_parts.append(memory_snapshot["user"])
    if memory_snapshot.get("external"):
        memory_parts.append(memory_snapshot["external"])

    if not memory_parts:
        return messages

    # Find the compaction summary message (usually the first user message
    # after the system prompt, or a message containing the SUMMARY_PREFIX)
    summary_idx = _find_compaction_summary(messages)
    if summary_idx is None:
        return messages

    # Prepend the memory authority block to the summary message
    memory_block = MEMORY_AUTHORITY_MARKER + "\n\n".join(memory_parts)
    _prepend_to_message_content(messages[summary_idx], memory_block)

    logger.info(
        "session_memory_compact: re-injected %d memory blocks into "
        "compaction summary at message index %d",
        len(memory_parts),
        summary_idx,
    )

    return messages


def _find_compaction_summary(messages: list) -> Optional[int]:
    """Find the index of the compaction summary message."""
    from agent.context_compressor import SUMMARY_PREFIX, LEGACY_SUMMARY_PREFIX

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            if content.startswith(SUMMARY_PREFIX) or content.startswith(LEGACY_SUMMARY_PREFIX):
                return i
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if isinstance(text, str) and (
                        text.startswith(SUMMARY_PREFIX) or text.startswith(LEGACY_SUMMARY_PREFIX)
                    ):
                        return i

    return None


def _prepend_to_message_content(message: dict, text: str) -> None:
    """Prepend text to a message's content."""
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = text + "\n\n" + content
    elif isinstance(content, list):
        content.insert(0, {"type": "text", "text": text})
