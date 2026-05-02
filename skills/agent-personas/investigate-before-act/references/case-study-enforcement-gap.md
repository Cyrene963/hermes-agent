# Case Study: Enforcement Gap Investigation (2026-05-02)

## Scenario
User asked why ChatGPT Codex can work 24/7 but I can't. I hallucinated about OpenCode Go and server pricing. User pointed out I already have anti-hallucination rules in my skills but didn't follow them.

## Investigation Path

### Step 1: Check existing skills
- Loaded `precision-and-verification` — found comprehensive rules about not hallucinating
- Loaded `product-identity-rules.md` — found previous incident documentation
- Loaded `investigate-before-act` — found Step 0 self-check requirements

### Step 2: Find root cause
User said: "你的memory永久记忆文件，以及hindsight recall不是都有提醒过你要先查再答吗？你自己查一下先吧。"

Investigation revealed:
1. Rules exist in skills ✅
2. Rules are injected into system prompt ✅
3. But enforcement is model-specific ❌

### Step 3: Code investigation
Found `run_agent.py` line 4893:
```python
TOOL_USE_ENFORCEMENT_MODELS = ("gpt", "codex", "gemini", "gemma", "grok")
```

**"mimo" is NOT in the list.** MiMo models get NO enforcement guidance.

### Step 4: Check if limitation was already fixed
User had merged PR `e2c614d93` — "fix: preserve memory authority across context compaction"

**Key insight:** I used "context compaction degrades rule authority" as an excuse, but this was already fixed. Don't use technical limitations as excuses without checking if they've been resolved.

## Lessons Learned

1. **Rules in skills are useless if not loaded** — Must call `skill_view()` before answering factual questions
2. **Model-specific enforcement exists** — Some models get enforcement guidance, others don't
3. **Check if limitations are already fixed** — Don't use outdated excuses
4. **Code-level gaps exist** — Even with good skills, the underlying system may not enforce them

## Action Items

1. Add "mimo" to `TOOL_USE_ENFORCEMENT_MODELS` (quick fix)
2. Create `MIMO_MODEL_EXECUTION_GUIDANCE` (better fix)
3. Implement `mandatory_skills` config option (best fix)
