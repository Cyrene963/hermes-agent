---
name: long-task-execution
description: >
  Long task execution engine — mandatory checklist for any task expected to take 10+ minutes
  or involve 15+ tool calls. Prevents the "write script then forget" failure mode.
  Trigger: when the user says "do X in the background", "take your time", "run this long task",
  or when you recognize a task will require many steps (scanning large datasets, batch processing,
  multi-phase research, complex builds, PR evaluation at scale).
triggers:
  - 长任务
  - 后台跑
  - 大活
  - 批量处理
  - 深度研究
  - PR扫描
  - 全量扫描
  - background task
  - long-running
  - batch process
version: "1.0"
pinned: false
---

# Long Task Execution Engine

## WHEN TO LOAD THIS SKILL

Any task that matches ANY of:
- Estimated 10+ minutes runtime
- 15+ tool calls expected
- Batch processing (N items to evaluate/process)
- Multi-phase research or analysis
- User explicitly says "run in background", "take your time", "no rush"

## FAILURE MODES TO PREVENT

| Failure Mode | Root Cause | Prevention |
|---|---|---|
| Script runs then dies silently | No notification set | Rule 1: notify_on_complete |
| Script can't adapt to errors | Script is deterministic | Rule 2: Script collects, Agent decides |
| Waste full day, user discovers failure | No intermediate checkpoints | Rule 3: Write state to disk every step |
| Context overflow loses progress | Too much in one context | Rule 4: Delegate + checkpoint |
| Forget to load this skill | No enforcement | Rule 5: Pre-flight checklist |
| User comes back to nothing done | No progress visibility | Rule 6: Progress file |

## MANDATORY RULES (non-negotiable)

### Rule 1: NEVER background without notify

```
# WRONG
terminal("long_script.sh", background=True)

# RIGHT
terminal("long_script.sh", background=True, notify_on_complete=True)
```

For critical tasks, also set watch_patterns for error keywords:
```python
terminal("important_script.sh", background=True, 
         notify_on_complete=True,
         watch_patterns=["ERROR", "Traceback", "FATAL", "FAILED"])
```

### Rule 2: Script collects data, Agent makes decisions

Scripts are data collectors, not decision makers. The agent must:
- Read script output
- Evaluate results
- Decide next action
- Log decision rationale

Pattern:
```
1. Script scans/collects → writes results to file
2. Agent reads file → evaluates → decides
3. Agent writes decision to file → triggers next step
4. Repeat
```

NEVER: Script does everything end-to-end without agent checkpoints.
ALWAYS: Agent has decision points between major phases.

### Rule 3: Checkpoint after every major step

After completing any meaningful step, write state to disk:

```
/root/.hermes/tasks/{task_name}/
  plan.md          — what we're doing and why
  progress.json    — current step, completed steps, next step
  results/         — intermediate outputs
  decisions.md     — what we decided and why
```

progress.json schema:
```json
{
  "task": "scan_github_prs",
  "started": "2026-05-02T20:07:00+08:00",
  "total_steps": 5,
  "current_step": 3,
  "steps": [
    {"name": "scan_all_prs", "status": "completed", "output": "results/scan.json"},
    {"name": "filter_candidates", "status": "completed", "output": "results/candidates.json"},
    {"name": "deep_evaluate", "status": "in_progress", "output": null},
    {"name": "select_and_merge", "status": "pending", "output": null},
    {"name": "test_and_verify", "status": "pending", "output": null}
  ],
  "last_checkpoint": "2026-05-02T20:35:00+08:00"
}
```

### Rule 4: Use delegation for parallel evaluation

When evaluating N items (PRs, files, pages, etc.):
- Batch items into groups of 3-5
- Use delegate_task with parallel tasks (up to max_concurrent_children)
- Each subtask evaluates its batch independently
- Agent synthesizes results

NEVER: Evaluate 23 items sequentially in one context.
ALWAYS: Split into 3-5 parallel batches, merge results.

### Rule 5: Pre-flight checklist (BEFORE starting)

Before ANY long task, verify:

```
□ Task broken into discrete steps?
□ Each step has expected output format?
□ Failure handling for each step defined?
□ notify_on_complete set for all background processes?
□ Checkpoint directory created?
□ Recovery plan if session interrupted?
□ Timeout reasonable (not too short)?
```

If any checkbox is empty, fill it before proceeding.

### Rule 6: Progress visibility

Create and maintain a progress file that the user can check:
- Write to a known location (e.g., /root/.hermes/tasks/{task_name}/progress.md)
- Update it after each checkpoint
- Include: what's done, what's next, any blockers, estimated time remaining

### Rule 7: Health checks during execution

For background processes, poll periodically:
```
# After starting a background process
process(action="poll", session_id=...)  # Check it's still alive
process(action="log", session_id=..., limit=10)  # Check recent output
```

Schedule checks:
- First check: 30 seconds after start (confirm it launched correctly)
- Subsequent: every 2-5 minutes during execution
- On any error signal: immediate investigation

### Rule 8: Graceful degradation

When something fails mid-task:
1. DON'T panic or restart from scratch
2. Read the checkpoint — what was the last successful step?
3. Identify what failed and why
4. Try to fix and continue from the failure point
5. If unfixable, save all progress and report to user with:
   - What completed successfully
   - What failed and why
   - What remains to do
   - Suggested next steps

## RECOVERY FROM INTERRUPTED SESSIONS

If a new session starts and the user mentions a previous long task:
1. Check /root/.hermes/tasks/ for existing task directories
2. Read progress.json to see where we left off
3. Resume from the last completed checkpoint
4. Don't re-do completed work

## 自动重试与自愈机制（Auto-Retry & Self-Healing）

**相关skills:** autonomous-decision-boundary（决定能不能自主做）, skill-enforcer plugin（强制加载skills）

### 重试策略

遇到可重试错误时，不要停下来等用户：

| 错误类型 | 重试策略 | 最大次数 |
|---------|---------|---------|
| 网络超时 (timeout) | 指数退避: 2s → 4s → 8s | 3 |
| Rate limit (429) | 等待 Retry-After 头指定时间，或 30s → 60s → 120s | 3 |
| 服务不可用 (503/502) | 30s → 60s → 120s | 3 |
| 连接拒绝 | 检查服务是否启动，尝试启动后重试 | 2 |
| 权限错误 (403/401) | 检查memory里有没有凭证，有就用 | 1 |
| 文件不存在 | 检查路径是否正确，尝试创建目录 | 1 |

**不可重试的错误（立即报告）：**
- 逻辑错误（代码bug）→ 尝试debug修复，不是重试
- 用户输入错误 → 报告并说明正确用法
- 资源耗尽（磁盘满、内存不足）→ 尝试清理后报告

### 自愈流程

```
步骤失败
  │
  ├─ 1. 记录错误详情到 progress.json
  │
  ├─ 2. 判断错误类型（参考上面的表格）
  │
  ├─ 3. 可重试？
  │   └─ YES → 执行重试策略 → 成功？→ 继续下一个步骤
  │                                    └─ 仍然失败 → 进入4
  │   └─ NO → 进入4
  │
  ├─ 4. 有替代方案？
  │   └─ 换工具/方法/API → 尝试 → 成功？→ 记录替代方案，继续
  │                                      └─ 失败 → 进入5
  │
  ├─ 5. 能降级？
  │   └─ 减少功能/精度/范围 → 尝试 → 记录降级原因，继续
  │
  └─ 6. 真的无法解决
      └─ 写完整报告到 checkpoint 文件
      └─ 报告格式：
         - 已完成的步骤（附结果）
         - 失败的步骤（附错误详情）
         - 已尝试的修复方案（附结果）
         - 建议的下一步（如果有）
```

### 进度恢复协议

当session中断后重新开始，用户提到之前的任务：

1. **搜索任务目录** — `ls /root/.hermes/tasks/`
2. **读取progress.json** — 确认最后成功的步骤
3. **跳过已完成步骤** — 不要重复工作
4. **从checkpoint继续** — 保持上下文连续性
5. **自主决策** — 恢复过程中的问题按 autonomous-decision-boundary skill 处理

## POST-TASK

After completing a long task:
1. Write a summary to the task directory
2. If the methodology was novel or had pitfalls, offer to save as a skill
3. Update progress.json with final status: "completed" or "partial"

## Plugin Enforcement

skill-enforcer plugin (pre_tool_call hook) periodically blocks action tools to force
compliance checkpoints during long tasks. When blocked:
1. Call skill_view/hindsight_recall/session_search to acknowledge compliance
2. Then retry the original tool call
3. This is NOT an error — it's a mandatory quality gate

See: hermes-agent skill `references/plugin-enforcement-pattern.md` for implementation details.
