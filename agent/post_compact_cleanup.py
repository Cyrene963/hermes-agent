"""Post-compact cleanup and time-based micro-compact config.

Post-compact cleanup: clears caches and resets state after context
compression to prevent stale data from polluting the fresh context.

Time-based micro-compact: when the gap since the last assistant message
exceeds a threshold, the server cache has expired and old tool results
should be cleared proactively (before the next API call).

Inspired by Claude Code's postCompactCleanup.ts and timeBasedMCConfig.ts.

Usage:
    from agent.post_compact_cleanup import run_post_compact_cleanup
    from agent.time_based_mc import should_time_based_compact, TimeBasedMCConfig

    # After compression:
    run_post_compact_cleanup(file_state_cache, read_tracker)

    # Before API call:
    if should_time_based_compact(messages, config):
        messages = apply_time_based_compact(messages)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Post-Compact Cleanup ─────────────────────────────────────────────

def run_post_compact_cleanup(
    file_state_cache: Any = None,
    read_tracker: Any = None,
    microcompact_state: Any = None,
) -> Dict[str, Any]:
    """Clean up state after context compression.

    Resets caches and tracking state that referenced the pre-compaction
    message indices. Without this, stale dedup entries and file cache
    references would persist and cause incorrect behavior.

    Args:
        file_state_cache: The FileStateCache singleton (if available).
        read_tracker: The _read_tracker dict from file_tools.py.
        microcompact_state: Any microcompact tracking state.

    Returns:
        Stats dict with cleanup counts.
    """
    stats = {
        "file_cache_cleared": False,
        "read_tracker_cleared": False,
        "microcompact_reset": False,
    }

    # Clear file state cache — compressed context means old file reads
    # are no longer in the conversation, so cached content should be
    # re-fetched on next read.
    if file_state_cache is not None:
        try:
            old_size = file_state_cache.size
            file_state_cache.clear()
            stats["file_cache_cleared"] = True
            logger.info("post_compact: cleared file_state_cache (%d entries)", old_size)
        except Exception as e:
            logger.debug("file_state_cache clear failed: %s", e)

    # Clear read tracker — dedup entries reference pre-compaction
    # message positions and are no longer valid.
    if read_tracker is not None:
        try:
            if isinstance(read_tracker, dict):
                old_count = len(read_tracker)
                read_tracker.clear()
                stats["read_tracker_cleared"] = True
                logger.info("post_compact: cleared read_tracker (%d entries)", old_count)
        except Exception as e:
            logger.debug("read_tracker clear failed: %s", e)

    # Reset microcompact state — tool tracking references pre-compaction
    # tool results that no longer exist.
    if microcompact_state is not None:
        try:
            if hasattr(microcompact_state, 'clear'):
                microcompact_state.clear()
            stats["microcompact_reset"] = True
            logger.info("post_compact: reset microcompact state")
        except Exception as e:
            logger.debug("microcompact reset failed: %s", e)

    return stats


# ── Time-Based Micro-Compact ─────────────────────────────────────────

@dataclass
class TimeBasedMCConfig:
    """Configuration for time-based micro-compact triggers."""
    # Gap in seconds since last assistant message that triggers compaction
    # Default: 5 minutes (300 seconds)
    gap_threshold_seconds: int = 300
    # Whether time-based compaction is enabled
    enabled: bool = True


def should_time_based_compact(
    messages: list,
    config: TimeBasedMCConfig = None,
) -> bool:
    """Check if time-based micro-compact should trigger.

    When the gap since the last assistant message exceeds the threshold,
    the server cache has expired and old tool results should be cleared.

    Args:
        messages: The conversation messages.
        config: Time-based MC configuration.

    Returns:
        True if time-based compaction should trigger.
    """
    if config is None:
        config = TimeBasedMCConfig()

    if not config.enabled:
        return False

    # Find the last assistant message timestamp
    last_assistant_ts = _find_last_assistant_timestamp(messages)
    if last_assistant_ts is None:
        return False

    gap = time.time() - last_assistant_ts
    return gap > config.gap_threshold_seconds


def _find_last_assistant_timestamp(messages: list) -> Optional[float]:
    """Find the timestamp of the last assistant message."""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            # Check for timestamp in various formats
            ts = msg.get("timestamp")
            if ts is not None:
                try:
                    return float(ts)
                except (ValueError, TypeError):
                    pass
            # Check for created_at
            ts = msg.get("created_at")
            if ts is not None:
                try:
                    return float(ts)
                except (ValueError, TypeError):
                    pass
    return None


def apply_time_based_compact(
    messages: list,
    config: TimeBasedMCConfig = None,
) -> Tuple[list, Dict[str, Any]]:
    """Apply time-based micro-compact: clear old tool results.

    This is a lightweight pass that clears tool results that are older
    than the time gap threshold. It's cheaper than full compression and
    prevents stale tool outputs from consuming context space.

    Args:
        messages: The conversation messages (mutated in-place).
        config: Time-based MC configuration.

    Returns:
        (messages, stats) tuple.
    """
    if config is None:
        config = TimeBasedMCConfig()

    stats = {
        "triggered": False,
        "cleared_count": 0,
        "chars_saved": 0,
    }

    if not config.enabled:
        return messages, stats

    # Find the last assistant message index
    last_asst_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], dict) and messages[i].get("role") == "assistant":
            last_asst_idx = i
            break

    if last_asst_idx is None:
        return messages, stats

    # Clear tool results before the last assistant message
    placeholder = "[Old tool output cleared — cache expired due to time gap]"
    for i in range(last_asst_idx):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 200:
                msg["content"] = placeholder
                stats["cleared_count"] += 1
                stats["chars_saved"] += len(content) - len(placeholder)

    if stats["cleared_count"] > 0:
        stats["triggered"] = True
        logger.info(
            "time_based_mc: cleared %d old tool results, saved ~%d chars",
            stats["cleared_count"],
            stats["chars_saved"],
        )

    return messages, stats
