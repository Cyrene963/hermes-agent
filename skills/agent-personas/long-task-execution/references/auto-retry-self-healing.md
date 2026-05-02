# Auto-Retry & Self-Healing Decision Matrix

## Retry Strategy Table

| Error Type | Strategy | Max Retries | Backoff |
|-----------|----------|-------------|---------|
| Network timeout | Retry same | 3 | 2s → 4s → 8s |
| Rate limit (429) | Wait + retry | 3 | 30s → 60s → 120s |
| Service unavailable (503/502) | Wait + retry | 3 | 30s → 60s → 120s |
| Connection refused | Check service + retry | 2 | immediate |
| Auth error (403/401) | Check memory for creds | 1 | immediate |
| File not found | Check path + retry | 1 | immediate |
| Logic error (bug) | Debug + fix | N/A | N/A |

## Non-Retryable Errors (escalate immediately)
- Logic errors → try debug, not retry
- User input errors → report and explain
- Resource exhaustion (disk full, OOM) → try cleanup, then report

## Self-Healing Flow

```
Step failed
  │
  ├─ 1. Record error in progress.json
  │
  ├─ 2. Classify error (see table above)
  │
  ├─ 3. Retryable? → execute strategy → success? → continue
  │                                        └─ still fail → go to 4
  │
  ├─ 4. Has alternative? → try different tool/method → success? → record, continue
  │                                                          └─ fail → go to 5
  │
  ├─ 5. Can degrade? → reduce scope/precision → record reason, continue
  │
  └─ 6. Unrecoverable → write report to checkpoint, notify user
```

## Session Recovery Protocol

When resuming after interruption:
1. Check `/root/.hermes/tasks/` for existing task directories
2. Read `progress.json` for last successful checkpoint
3. Skip completed steps
4. Resume from failure point
5. Apply autonomous-decision-boundary rules for recovery decisions

## Integration with Other Skills

- **autonomous-decision-boundary**: Determines WHAT you can fix autonomously
- **This skill**: Determines HOW to fix it (retry/alternative/degrade)
- **skill-enforcer plugin**: Reminds you to CHECK rules periodically during execution
- **fact verification gate**: Catches fabrication BEFORE delivery
