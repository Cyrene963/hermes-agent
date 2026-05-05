#!/usr/bin/env python3
"""Hybrid skill selector — semantic-only, no regex/keyword matching.

Layer 1: Fast greeting detection (skip skills for "hi"/"hello")
Layer 2: FTS5 semantic search (SkillDB full-text search)
Layer 3: Broadcast fallback (if SkillDB empty)

NO regex patterns. NO keyword matching. The model decides what's relevant
by seeing skill names and descriptions in the system prompt. This module
only handles the INDEX (which skills to show), not the SELECTION (which
to load — that's the model's job via skill_eval_gate).
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Layer 1: Greeting detection — only used to SKIP skills entirely
# This is NOT skill selection. It's "don't waste tokens on 'hi'".
_GREETING_PATTERN = re.compile(
    r'^(hi|hello|hey|你好|嗨|谢谢|感谢|bye|再见|'
    r'good morning|good afternoon|good evening|早上好|下午好|晚上好|'
    r'ok|好的|嗯|啊|哦|知道了|收到)\s*[!！。.？?]*$',
    re.IGNORECASE
)


def is_greeting(text: str) -> bool:
    """Check if text is a simple greeting or acknowledgment."""
    return bool(_GREETING_PATTERN.match(text.strip()))


def should_skip_skills(user_message: str) -> bool:
    """Quick check if skills should be skipped entirely."""
    return is_greeting(user_message)


def ai_inference_select(user_message: str, context: str = "", max_skills: int = 5) -> List[str]:
    """Semantic skill selection via FTS5 full-text search.

    No regex. No keyword matching. Pure database search.
    """
    try:
        from agent.skill_db import SkillDB

        db = SkillDB()
        if db.get_skill_count() == 0:
            logger.debug("SkillDB empty, cannot select skills")
            return []

        query = user_message.strip()
        if not query:
            return []

        results = db.search(query=query, limit=max_skills, boost_recent=True)
        skill_names = [skill["name"] for skill in results]

        logger.debug(
            "FTS5 selected %d skills for query: %r",
            len(skill_names), query[:50]
        )
        return skill_names

    except ImportError:
        logger.debug("SkillDB not available")
        return []
    except Exception as e:
        logger.warning("FTS5 search failed: %s", e)
        return []


def hybrid_skill_select(user_message: str, context: str = "") -> Dict[str, Any]:
    """Skill selection with two layers: greeting skip + FTS5 semantic search.

    Returns:
        {
            "selected_skills": List[str],
            "method": "fast_rule" | "ai_inference" | "fts5_fallback",
            "confidence": float
        }
    """
    result = {
        "selected_skills": [],
        "method": "fast_rule",
        "confidence": 1.0
    }

    # Layer 1: Greeting detection — skip skills for simple messages
    if is_greeting(user_message):
        result["method"] = "fast_rule"
        result["confidence"] = 0.95
        return result

    # Layer 2: FTS5 semantic search
    selected_skills = ai_inference_select(user_message, context)
    if selected_skills:
        result["selected_skills"] = selected_skills
        result["method"] = "ai_inference"
        result["confidence"] = 0.75
    else:
        result["method"] = "fts5_fallback"
        result["confidence"] = 0.6

    return result


def get_skills_for_message(user_message: str, context: str = "", use_ai: bool = True) -> List[str]:
    """Main entry point. Returns skill names to inject into system prompt."""
    selection = hybrid_skill_select(user_message, context)

    if selection["method"] == "fast_rule":
        return []

    return selection["selected_skills"]
