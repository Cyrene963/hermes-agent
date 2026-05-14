#!/usr/bin/env python3
"""Hybrid skill selector - combines rules, keyword matching, and AI inference."""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Layer 1: Fast rules (0 token cost, <10ms)
FAST_RULES = {
    # Greetings and simple exchanges
    r'^(hi|hello|hey|你好|嗨|谢谢|感谢|bye|再见)$': [],
    r'^(good morning|good afternoon|good evening|早上好|下午好|晚上好)$': [],
    r'^(ok|好的|嗯|啊|哦|知道了|收到)$': [],
    r'^(what|how|why|who|where|when|什么|怎么|为什么|谁|哪里|何时)\\??$': [],  # Simple questions
}

# Layer 2: High confidence task patterns (0 token cost, <50ms)
TASK_PATTERNS = {
    # Debugging tasks
    r'(debug|调试|错误|exception|error|traceback|crash|内存泄漏|memory leak|segmentation fault)': 
        ['python-debugpy', 'debugging-hermes-tui-commands', 'test-driven-development'],
    
    # GitHub operations  
    r'(github|pull request|pr|merge|commit|git|仓库|repository|branch)':
        ['github-pr-workflow', 'github-code-review', 'requesting-code-review'],
    
    # System administration
    r'(systemd|gateway|restart|service|process|kill|port|config|配置|系统)':
        ['hermes-agent', 'cron-management', 'gateway-troubleshooting'],
    
    # Research tasks
    r'(research|研究|web search|scrape|crawl|cloudflare|bypass|数据|data|信息|information)':
        ['research-workflows', 'deep-research-v4'],
    
    # Skill management
    r'(skill|技能|add skill|create skill|update skill|skill factory)':
        ['hermes-skill-factory', 'skill-authoring'],
    
    # Testing
    r'(test|测试|pytest|unit test|integration test|coverage|assert)':
        ['test-driven-development', 'pytest-parallel'],
    
    # AI model/cost
    r'(model|模型|provider|api|key|token|credit|pricing|cost|成本|费用)':
        ['token-cost-analysis', 'model-provider-setup'],
}

# Layer 3: Complex task indicators (may need AI inference)
COMPLEX_INDICATORS = [
    '设计', '架构', '优化', '重构', '实现', '分析', '规划', '部署',
    'design', 'architecture', 'optimize', 'refactor', 'implement', 'analyze', 'plan', 'deploy'
]

def is_greeting(text: str) -> bool:
    """Check if text is a simple greeting or acknowledgment."""
    text_lower = text.lower().strip()
    for pattern in FAST_RULES.keys():
        if re.match(pattern, text_lower):
            return True
    return False

def get_task_skills(text: str) -> List[str]:
    """Get skills based on high-confidence task patterns."""
    text_lower = text.lower()
    for pattern, skills in TASK_PATTERNS.items():
        if re.search(pattern, text_lower):
            return skills[:3]  # Return top 3 for this category
    return []

def is_complex_task(text: str) -> bool:
    """Check if task is complex and may need AI reasoning."""
    text_lower = text.lower()
    # Check for complex indicators
    for indicator in COMPLEX_INDICATORS:
        if indicator in text_lower:
            return True
    # Check for long, detailed requests (>50 chars)
    if len(text) > 50:
        return True
    return False

def ai_inference_select(user_message: str, context: str = "", max_skills: int = 5) -> List[str]:
    """
    AI inference layer: use SkillDB FTS5 search for intelligent skill selection.
    This is a "smart" search that extracts key concepts from the user message.
    
    Args:
        user_message: User's message
        context: Recent conversation context
        max_skills: Maximum number of skills to return
        
    Returns:
        List of skill names
    """
    try:
        from agent.skill_db import SkillDB
        
        # Get SkillDB instance
        db = SkillDB()
        
        # Check if DB has any skills
        if db.get_skill_count() == 0:
            logger.debug("SkillDB empty, AI inference cannot work")
            return []
        
        # Extract key concepts from user message
        # Simple approach: use the message itself as query
        # For better results, we could extract keywords or use NLP
        query = user_message.strip()
        
        if not query:
            return []
        
        # Search using FTS5 with boosted recent usage
        results = db.search(
            query=query,
            limit=max_skills,
            boost_recent=True
        )
        
        # Extract skill names
        skill_names = [skill["name"] for skill in results]
        
        logger.debug(
            "AI inference selected %d skills for query: %r",
            len(skill_names),
            query[:50]
        )
        
        return skill_names
        
    except ImportError:
        logger.debug("SkillDB not available, AI inference failed")
        return []
    except Exception as e:
        logger.warning("AI inference failed: %s", e)
        return []

def hybrid_skill_select(user_message: str, context: str = "") -> Dict[str, Any]:
    """
    Hybrid skill selection with three layers.
    
    Returns:
        {
            "selected_skills": List[str],
            "method": "fast_rule" | "task_pattern" | "ai_inference" | "fts5_fallback",
            "confidence": float  # 0.0 to 1.0
        }
    """
    result = {
        "selected_skills": [],
        "method": "fast_rule",
        "confidence": 1.0
    }
    
    # Layer 1: Fast rules (greetings, simple questions)
    if is_greeting(user_message):
        result["method"] = "fast_rule"
        result["confidence"] = 0.95
        return result
    
    # Layer 2: High confidence task patterns
    task_skills = get_task_skills(user_message)
    if task_skills:
        result["selected_skills"] = task_skills
        result["method"] = "task_pattern"
        result["confidence"] = 0.85
        return result
    
    # Layer 3: Complex task detection -> AI inference
    if is_complex_task(user_message):
        selected_skills = ai_inference_select(user_message, context)
        if selected_skills:
            result["selected_skills"] = selected_skills
            result["method"] = "ai_inference"
            result["confidence"] = 0.75
        else:
            # AI inference didn't find skills, fall back to FTS5
            result["method"] = "fts5_fallback"
            result["confidence"] = 0.6
        return result
    
    # Default: needs FTS5 search
    result["method"] = "fts5_fallback"
    result["confidence"] = 0.7
    return result

def should_skip_skills(user_message: str) -> bool:
    """Quick check if skills should be skipped entirely."""
    return is_greeting(user_message)

def get_skills_for_message(user_message: str, context: str = "", use_ai: bool = True) -> List[str]:
    """
    Main entry point for hybrid skill selection.
    
    Args:
        user_message: Current user message
        context: Recent conversation context
        use_ai: Whether to use AI inference for complex tasks (default True)
    
    Returns:
        List of skill names to load
    """
    selection = hybrid_skill_select(user_message, context)
    
    if selection["method"] == "fast_rule":
        return []  # No skills needed
    
    if selection["method"] == "task_pattern":
        return selection["selected_skills"]
    
    if selection["method"] == "ai_inference":
        return selection["selected_skills"]
    
    # Default: empty list, let FTS5 handle it
    return []