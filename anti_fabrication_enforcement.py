"""
Anti-Fabrication Enforcement v2 — Universal Approach

v1的问题：regex匹配特定模式（PR状态、价格等），是面向结果编程。
v2的思路：不验证就不准说。不是匹配"什么话需要验证"，
而是检查"模型有没有调用验证工具"。如果没有任何验证工具调用，
就限制模型只能输出：
- 基于对话上下文的信息（用户刚说的话）
- 明确标注为"未验证"的推测
- 建议用户自己去验证

核心逻辑：
- 如果本轮对话中模型调用了验证工具 → 放行
- 如果本轮对话中模型没有调用任何验证工具 → 只允许非事实性输出
- "非事实性输出"定义：提问、建议、分析用户提供的信息、标注为推测的内容

这不是regex匹配，是基于工具调用历史的通用拦截。
"""

import json
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Tools that count as "the model has verified something"
VERIFICATION_TOOLS = {
    # Web
    'web_search', 'web_extract',
    # Browser
    'browser_navigate', 'browser_click', 'browser_snapshot', 'browser_vision',
    # Vision
    'vision_analyze',
    # Terminal (for curl, API calls, git commands with verification intent)
    'terminal',
    # File reading (for checking config, code, docs)
    'read_file',
    # Session search (for checking past conversations)
    'session_search',
    # Memory recall
    'hindsight_recall', 'hindsight_reflect',
}

# Tools that are NOT verification (even if they involve data)
NON_VERIFICATION_TOOLS = {
    'write_file', 'patch', 'send_message', 'todo', 'cronjob',
    'skill_manage', 'skill_view', 'memory', 'delegate_task',
    'execute_code', 'browser_type', 'browser_press', 'browser_scroll',
    'text_to_speech', 'clarify', 'process', 'skills_list',
}


def _count_verification_calls(messages: List[dict], lookback: int = 20) -> int:
    """Count how many verification tool calls were made in recent messages."""
    count = 0
    recent = messages[-lookback:] if len(messages) > lookback else messages
    for msg in recent:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            for tc in msg['tool_calls']:
                tool_name = tc.get('function', {}).get('name', '')
                if tool_name in VERIFICATION_TOOLS:
                    count += 1
    return count


def _has_factual_content(text: str) -> bool:
    """
    Check if the response contains factual claims (not just opinions/questions).
    This is intentionally broad — any statement that could be a fact.
    """
    # Questions are OK (not factual claims)
    # Count question marks vs periods
    questions = text.count('?') + text.count('？')
    statements = text.count('。') + text.count('.') - questions
    
    # If mostly questions, it's not factual
    if questions > 0 and statements <= 0:
        return False
    
    # Very short responses are usually OK (acks, greetings)
    if len(text.strip()) < 30:
        return False
    
    # Check for hedging language that indicates uncertainty
    uncertainty_markers = [
        '我认为', '我觉得', '我猜', '可能是', '也许', '不确定',
        'I think', 'I believe', 'maybe', 'perhaps', 'not sure',
        '据我所知', '未经验证', '需要确认', '待确认',
    ]
    text_lower = text.lower()
    has_hedging = any(m.lower() in text_lower for m in uncertainty_markers)
    
    # If it has hedging language, it's presenting as opinion — OK
    if has_hedging:
        return False
    
    return True


def enforce(
    response_text: str,
    messages: List[dict],
    enabled: bool = True,
    min_verification_calls: int = 1,
) -> Tuple[bool, Optional[str]]:
    """
    Universal anti-fabrication enforcement.
    
    Logic:
    - If model used verification tools → allow (it checked something)
    - If model didn't use any tools AND response has factual content → block
    
    This is NOT pattern matching. It's a simple rule:
    "If you didn't look anything up, don't state anything as fact."
    
    Returns:
        (should_block, reason)
    """
    if not enabled:
        return False, None
    
    # Count verification tool calls
    verification_count = _count_verification_calls(messages)
    
    if verification_count >= min_verification_calls:
        # Model has verified something — allow
        return False, None
    
    # Model hasn't verified anything
    # Check if response contains factual content
    if not _has_factual_content(response_text):
        # Just questions or hedging — allow
        return False, None
    
    # Block: model is stating facts without having verified anything
    return True, (
        "你正在输出事实性声明，但本轮对话中没有调用过任何验证工具。"
        "请先验证再回复。可选方式：\n"
        "1. web_search 搜索相关信息\n"
        "2. terminal 执行curl或API调用\n"
        "3. browser_navigate 访问官方文档\n"
        "4. read_file 检查本地文件\n"
        "5. session_search/hindsight_recall 检查历史记录\n"
        "如果确实是基于对话上下文的分析或个人观点，请明确标注。"
    )


# ═══════════════════════════════════════════════════════════════
# Integration: Same hook point as v1, but simpler logic
# ═══════════════════════════════════════════════════════════════
"""
In run_agent.py (same location as v1 patch):

    from anti_fabrication_enforcement import enforce as _fab_enforce
    _fab_block, _fab_reason = _fab_enforce(
        response_text=final_response,
        messages=messages,
        enabled=True,
        min_verification_calls=1,
    )
    if _fab_block:
        messages.append({
            "role": "system",
            "content": f"[VERIFICATION REQUIRED] {_fab_reason}"
        })
        continue  # re-run, don't break
"""
