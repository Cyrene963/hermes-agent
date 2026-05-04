# skill-router Plugin

## Problem

Hermes Agent has a rich skill system with 60+ skills covering coding, writing, research, task management, and more. But there's a fundamental gap: **the agent often skips loading relevant skills** because:

1. The agent sees 60+ skills listed in the system prompt but doesn't systematically evaluate which are relevant
2. "Self-discipline" loading (trusting the agent to call `skill_view`) fails under time pressure or confidence
3. Even with compliance checkpoints (skill-enforcer), the agent may not have loaded the right skills in the first place

## Solution

**skill-router** forces a skill-loading checkpoint on the **first action tool call** in a session. It uses the same `pre_tool_call` hook pattern as skill-enforcer, but fires once at session start instead of periodically.

```
User message arrives
  |
  ├─ Agent starts processing
  ├─ Agent tries first action (e.g., terminal, write_file)
  ├─ skill-router BLOCKS with checkpoint message
  ├─ Agent must load task-orchestrator (or confirm trivial task)
  ├─ task-orchestrator tells agent which skills to load
  ├─ Agent loads skills
  ├─ Gate satisfied, agent proceeds
  └─ skill-enforcer takes over for periodic compliance
```

## Relationship to Other Plugins

| Plugin | When it fires | What it does |
|--------|--------------|--------------|
| **skill-router** | Once at session start | Ensures skills are loaded |
| **skill-enforcer** | Every N action calls | Ensures skills are followed |

Together they form a complete enforcement chain:
1. **Load skills** (skill-router) → ensures the right skills are in context
2. **Follow skills** (skill-enforcer) → ensures the agent actually follows the rules
3. **Periodic verification** (skill-enforcer) → catches drift mid-session

## How It Works

1. The plugin tracks per-session state (whether the gate has been satisfied)
2. On the first action tool call (terminal, write_file, delegate_task, etc.), it blocks with a checkpoint message
3. The checkpoint asks the agent to analyze task complexity and load `task-orchestrator` if needed
4. Any `skill_view` call satisfies the gate (the agent doesn't have to load task-orchestrator specifically)
5. After the gate is satisfied, all subsequent action tool calls pass through normally

## Configuration

Add to `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - skill-enforcer
    - skill-router  # Add this line
```

Or use the CLI:
```bash
hermes plugins enable skill-router
```

## Limitations

- The plugin can **block** the agent, but it can't **force** the agent to load the right skills. The agent must still choose to call `skill_view`. However, the checkpoint makes this choice explicit and unavoidable.
- For simple tasks (< 2 tool calls), the checkpoint adds one extra round-trip. This is acceptable for the safety it provides.
- The plugin depends on `task-orchestrator` being available as a skill. If it's not installed, the checkpoint will still fire but the agent may not know which skills to load.

## Related Skills

- `task-orchestrator` — the entry point skill that tells the agent which skills to load
- `skill-enforcer` — periodic compliance checkpoints (complementary)
