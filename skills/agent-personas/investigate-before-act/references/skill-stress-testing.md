# Skill Stress-Testing Methodology

## Problem
Writing skills and assuming they work is unreliable. Skills need to be tested against
REAL failure scenarios from hindsight/session history, not theoretical ones.

## Method

### Step 1: Pull real failure scenarios from hindsight
```python
hindsight_recall(query="agent failure error mistake")
hindsight_recall(query="investigate skipped acted without checking")
```

### Step 2: For each scenario, ask three questions
1. Would the new skill have PREVENTED this failure?
2. If yes, HOW? (trace the exact path through the skill)
3. If no, WHY NOT? (what's the gap?)

### Step 3: Classify results
- **Can prevent**: Technical improvements (checkpoint, retry, file-based recovery)
- **Partially can**: Rules are clear but depend on agent loading the skill
- **Cannot prevent**: Agent "knows" but doesn't "do" (need technical enforcement)

### Step 4: For "cannot prevent" — escalate to technical enforcement
Don't just add more text. Create a plugin with pre_tool_call hook.
See: references/skill-enforcement-architecture.md

## Example: 6 real scenarios tested (2026-05-02)

| Scenario | Result | Why |
|----------|--------|-----|
| Fabricated "5块钱服务器" | Partial | Rules exist but agent didn't load skill |
| Waited for user on error | Partial | Decision tree clear but depends on loading |
| Confused Dashboard vs API Server | Cannot | Agent "knew" but didn't check (Dunning-Kruger) |
| Used browser despite having token | Cannot | Step 0 exists but wasn't executed |
| Long task lost progress | Can | Checkpoint is file-based, survives context |
| Merged closed/duplicate PR | Can | PR status is objective data, easy to check |

## Key Insight
The gap between "can prevent" and "cannot prevent" is the gap between
TECHNICAL enforcement and BEHAVIORAL rules. Always prefer technical.
