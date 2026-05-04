"""
Skill auto-injection module for Hermes Agent.
Analyzes the user's first message and injects matching skill content
directly into the system prompt — no agent self-discipline required.
"""
from __future__ import annotations
import re
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

_MAX_INJECT_CHARS = 8000
_MAX_INJECT_SKILLS = 2


def _parse_skill_triggers(skill_path: Path) -> Tuple[str, str, List[str]]:
    """Parse SKILL.md YAML front matter for name, description, triggers."""
    try:
        content = skill_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ("", "", [])

    fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not fm_match:
        return ("", "", [])

    fm_text = fm_match.group(1)
    name = description = ""
    triggers = []

    m = re.search(r"^name:\s*(.+)", fm_text, re.MULTILINE)
    if m:
        name = m.group(1).strip().strip("\"'")

    m = re.search(r"^description:\s*[>|]?\s*\n((?:\s+.+\n?)+)", fm_text, re.MULTILINE)
    if m:
        description = " ".join(l.strip() for l in m.group(1).strip().split("\n")).strip("\"'")
    else:
        m = re.search(r'^description:\s*"(.+)"', fm_text, re.MULTILINE)
        if m:
            description = m.group(1).strip()

    in_triggers = False
    for line in fm_text.split("\n"):
        if re.match(r"^triggers:", line):
            in_triggers = True
            continue
        if in_triggers:
            if re.match(r"^\s+-\s+", line):
                t = re.sub(r"^\s+-\s+", "", line).strip().strip("\"'")
                if t:
                    triggers.append(t)
            elif not line.startswith(" ") and line.strip():
                break

    return (name, description, triggers)


def _extract_phrases(text: str) -> set:
    """Extract meaningful phrases from text (Chinese 2+ char sequences, English words)."""
    # Chinese phrases: 2+ consecutive Chinese characters
    chinese = set(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    # English words: 3+ char words
    english = set(w.lower() for w in re.findall(r"[A-Za-z]{3,}", text))
    return chinese | english


def _match_score(message: str, triggers: List[str], description: str) -> float:
    """Score how well a message matches a skill's triggers (0.0-1.0)."""
    if not triggers:
        return 0.0

    msg_phrases = _extract_phrases(message)
    if not msg_phrases:
        return 0.0

    best_score = 0.0

    for trigger in triggers:
        trigger_phrases = _extract_phrases(trigger)
        if not trigger_phrases:
            continue

        # Exact substring match (highest confidence)
        if trigger.lower() in message.lower():
            best_score = max(best_score, 1.0)
            continue

        # Phrase overlap
        overlap = msg_phrases & trigger_phrases
        if overlap:
            score = len(overlap) / max(len(trigger_phrases), 1)
            best_score = max(best_score, score * 0.8)

    # Bonus for description match
    desc_phrases = _extract_phrases(description)
    desc_overlap = msg_phrases & desc_phrases
    if len(desc_overlap) >= 2:
        best_score = min(best_score + 0.15, 1.0)

    return best_score


def auto_inject_skills(
    user_message: str,
    skills_dir: Path,
    max_skills: int = _MAX_INJECT_SKILLS,
    min_score: float = 0.3,
    max_chars: int = _MAX_INJECT_CHARS,
) -> str:
    """Analyze user message and return matching skill content for system prompt injection.

    Called during _build_system_prompt. The returned content is injected
    directly — the agent doesn't need to call skill_view.
    """
    if not user_message or not skills_dir.exists():
        return ""

    candidates: List[Tuple[float, str, str]] = []

    for skill_file in sorted(skills_dir.rglob("SKILL.md")):
        try:
            name, desc, triggers = _parse_skill_triggers(skill_file)
            if not name or not triggers:
                continue
            score = _match_score(user_message, triggers, desc)
            if score >= min_score:
                content = skill_file.read_text(encoding="utf-8", errors="replace")
                candidates.append((score, name, content))
        except Exception as e:
            logger.debug("skill-auto-inject: skip %s: %s", skill_file, e)

    if not candidates:
        return ""

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = candidates[:max_skills]

    parts = ["<auto_loaded_skills>"]
    parts.append(
        "The following skills were automatically loaded based on your task. "
        "You MUST follow their instructions. Do NOT skip quality gates."
    )

    total = 0
    for score, name, content in selected:
        if total + len(content) > max_chars:
            remaining = max_chars - total
            if remaining < 200:
                break
            content = content[:remaining] + "\n... (truncated)"
        parts.append(f"\n## Skill: {name} (relevance: {score:.0%})\n")
        parts.append(content)
        total += len(content)

    parts.append("</auto_loaded_skills>")

    logger.info(
        "skill-auto-inject: injected %d skills (%d chars) for: %.80s",
        len(selected), total, user_message,
    )
    return "\n".join(parts)
