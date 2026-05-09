# Memory Metacognition Framework — Deep Acceptance Test Report

**Date:** 2026-05-09
**PR:** [#22516](https://github.com/NousResearch/hermes-agent/pull/22516)
**Branch:** `memory-metacognition-framework`
**HEAD:** `7b7315de4`
**Tester:** Hermes Agent (automated)

---

## Executive Summary

PR #22516 adds a **five-layer memory metacognition framework** to Hermes Agent. All layers are **disabled by default (no-op)**. Local policy opt-in enables each layer independently. The PR has been verified across 150 automated test cases with zero failures.

**Verdict: PASS. Ready for review.**

---

## Test Results Overview

| Test Suite | Tests | Passed | Failed | Notes |
|-----------|-------|--------|--------|-------|
| Upstream pytest | 61 | 61 | 0 | Full suite, isolated temp dirs |
| Phase 1: Default no-op | 16 | 16 | 0 | DISABLE_LOCAL=1, pure default policy |
| Phase 1: Local policy overlay | 44 | 44 | 0 | Real ~/.hermes/memory_policy.yaml |
| Phase 3: Multi-user isolation | 29 | 29 | 0 | Bank ID, cache, path isolation |
| **Total** | **150** | **150** | **0** | |

---

## Phase 1: Default No-Op Baseline

**Test:** `HERMES_MEMORY_POLICY_DISABLE_LOCAL=1` → pure default policy (all disabled)

| Check | Result |
|-------|--------|
| memory_index.enabled = False | ✅ |
| query_expansion.enabled = False | ✅ |
| preflight.enabled = False | ✅ |
| routing_preflight absent/disabled | ✅ |
| build_index_provider → NoOpIndexProvider | ✅ |
| build_index() → "" | ✅ |
| build_query_expander → PassthroughExpander | ✅ |
| expand("发文件") → ["发文件"] (no expansion) | ✅ |
| build_preflight_policy → NoOpPreflightPolicy | ✅ |
| get_task_type → None for all tools | ✅ |
| run_checks → decision="allow" | ✅ |
| build_strategy_preflight → empty rules | ✅ |
| check("linux.do") → None (no rules) | ✅ |

**Conclusion:** With default policy, the framework has ZERO side effects. No extra tokens in prompt, no blocking, no expansions, no strategy hints.

---

## Phase 1: Local Policy Overlay (Enabled Layers)

**Test:** Real `~/.hermes/memory_policy.yaml` with all layers enabled.

### Layer 1: Memory Index
| Check | Result |
|-------|--------|
| Returns ScriptIndexProvider when enabled | ✅ |
| build_index returns string | ✅ |

### Layer 2: Query Expansion
| Check | Result |
|-------|--------|
| Returns PolicyQueryExpander when enabled | ✅ |
| "发文件到 Telegram" → expands to include "sendDocument" | ✅ |
| "今天天气怎么样" → returns original only (no trigger) | ✅ |
| Empty message → [] | ✅ |

### Layer 3: Structured Preflight
| Check | Result |
|-------|--------|
| Returns PolicyPreflightPolicy when enabled | ✅ |
| send_message + file_path → triggers preflight | ✅ |
| terminal + "rm -rf" → triggers preflight | ✅ |
| terminal + "ls -la" → does NOT trigger | ✅ |
| send_message + text only → does NOT trigger | ✅ |
| read_file → does NOT trigger | ✅ |
| run_checks returns PreflightResult with checks list | ✅ |

### Layer 4: Task Routing
| Check | Result |
|-------|--------|
| Has routing rules from policy | ✅ |
| Rule has task_type, trigger_patterns, preferred_method, strategy_hint | ✅ |
| "linux.do" message matches trigger pattern | ✅ |
| "what is 2+2" → no match | ✅ |

### Layer 5: Lesson Promotion
| Check | Result |
|-------|--------|
| suggest_policy_patch returns LessonSuggestion | ✅ |
| Classifies lesson_type correctly | ✅ |
| Has suggestion dict, not auto-applied | ✅ |
| Task routing lesson classified as "task_routing" | ✅ |

---

## Phase 3: Multi-User Isolation

### 3.1 Bank ID Isolation
| Input | Expected | Got | Result |
|-------|----------|-----|--------|
| None | "hindsight" | "hindsight" | ✅ |
| user_id="alice" | "hindsight-alice" | "hindsight-alice" | ✅ |
| user_id="bob" | "hindsight-bob" | "hindsight-bob" | ✅ |
| user_id="alice", bank_id="custom" | "custom" | "custom" | ✅ |
| alice ≠ bob | different | different | ✅ |

### 3.2 Policy Cache Isolation
| Check | Result |
|-------|--------|
| Global, alice, bob have separate cache entries | ✅ |
| Separate objects (not same reference) | ✅ |
| force_reload alice doesn't invalidate global | ✅ |
| force_reload alice doesn't invalidate bob | ✅ |

### 3.3 build_* user_context Propagation
| Function | None bank_id | alice bank_id | Result |
|----------|-------------|---------------|--------|
| build_preflight_policy | "hindsight" | "hindsight-alice" | ✅ |
| build_strategy_preflight | "hindsight" | "hindsight-alice" | ✅ |

### 3.4 Lesson Promotion Cross-User Isolation
| Check | Result |
|-------|--------|
| alice dry_run → file_path contains "alice" | ✅ |
| bob dry_run → file_path contains "bob" | ✅ |
| alice file_path ≠ bob file_path | ✅ |
| private lesson without user_context → refused | ✅ |
| public lesson without shared=True → refused | ✅ |
| public + shared + dry_run → allowed | ✅ |

### 3.5 Per-User Policy Path
| Check | Result |
|-------|--------|
| _get_user_policy_path("alice") contains "alice" | ✅ |
| _get_user_policy_path("bob") contains "bob" | ✅ |
| Paths differ | ✅ |

---

## Security Boundary Verification

| Rule | Implementation | Verified |
|------|---------------|----------|
| All layers disabled by default | Default policy YAML: all `enabled: false` | ✅ |
| `HERMES_MEMORY_POLICY_DISABLE_LOCAL=1` forces off | Checked: all build_* return NoOp | ✅ |
| Private scope without user_context → refused | `apply_suggestion` returns status="refused" | ✅ |
| Public scope without shared=True → refused | `apply_suggestion` returns status="refused" | ✅ |
| dry_run=True by default | `apply_suggestion` default parameter | ✅ |
| suggest_policy_patch doesn't write files | Returns LessonSuggestion only | ✅ |
| Lesson text sanitized (chat_id/token masked) | `_sanitize_lesson_text` redacts patterns | ✅ |
| user_id from runtime metadata only | `user_context` dict, not lesson text | ✅ |
| No private data in upstream code | Grep: no chat IDs, user names, personal paths | ✅ |
| Per-user bank_id isolation | `_resolve_bank_id` → `hindsight-{user_id}` | ✅ |
| Per-user policy cache isolation | Cache keyed by user_id, separate objects | ✅ |

---

## Before/After Comparison

### What this PR adds (was not possible before)

| Capability | Before PR | After PR |
|-----------|-----------|----------|
| **Memory awareness** | Agent doesn't know what it remembers | Compact index injected into system prompt (~200 tokens) |
| **Query quality** | Original message only for hindsight search | Keyword→term expansion for better recall |
| **Preflight safety** | No argument validation before tool execution | Structured field checks (required/equals/contains) |
| **Strategy recall** | Agent discovers strategies by failing first | Strategy hints injected before tool selection |
| **Lesson capture** | Lessons lost after session | Explicit lessons → reviewable policy suggestions |
| **Multi-user** | Single global policy | Per-user bank_id, per-user policy overlay, per-user cache |

### Concrete code changes

**agent/prompt_builder.py** (+28 lines):
- `expand_recall_queries()` — delegates to PolicyQueryExpander
- `build_memory_index_block()` — delegates to ScriptIndexProvider

**run_agent.py** (+98 lines):
- Memory Index injection (cached per session, ~200 tokens)
- Query Expansion before memory prefetch (up to 5 expanded queries)
- Preflight Gate before tool execution (both sequential and concurrent paths)
- Task Routing strategy hint injection before tool loop

**agent/memory_metacognition.py** (39,279 bytes, new file):
- All five layers: interfaces, no-op defaults, policy-backed implementations
- Policy loader with default → local → user overlay chain
- Lesson Promotion with classify → suggest → apply (dry_run) pipeline
- Multi-user isolation: bank_id, cache, paths

**agent/memory_policy.default.yaml** (4,622 bytes, new file):
- Conservative defaults (all disabled)
- Ships with agent, not user-specific

**tests/test_memory_metacognition_upstream.py** (23,061 bytes, new file):
- 61 tests covering all layers + security boundaries

### Integration safety

| Concern | How it's handled |
|---------|-----------------|
| Prompt caching breakage | Memory Index cached on `self._memory_index_cached`, same content per session |
| Exception safety | Every integration point wrapped in try/except → pass |
| Token overhead when disabled | Zero (NoOp returns empty string / None) |
| Token overhead when enabled | Memory Index: ~200 tokens. Query Expansion: 0 extra (used for prefetch only). Task Routing: ~100 tokens if triggered. Preflight: 0 extra tokens. |

---

## Known Limitations (Documented)

1. **Task routing provides hints, not mandatory tool locks.** Model retains autonomy.
2. **Lesson Promotion handles explicit lessons only.** Implicit lesson mining is follow-up.
3. **Policy quality depends on user-maintained local policy.** Framework enforces structure, not content.
4. **User identity from trusted runtime metadata.** Not validated against external identity providers.
5. **Shared policy writes require explicit `shared=True`.** No automatic sharing.

---

## Follow-up Items (NOT in this PR)

- [ ] Implicit Lesson Detection (pattern-based extraction from conversation history)
- [ ] Session Lesson Mining (post-session failure analysis → policy suggestions)
- [ ] Policy Quality Scoring
- [ ] Cross-Session Lesson Deduplication

---

## Final Verdict

```
PASS — 150/150 tests, 0 failures

- Default: no-op (zero side effects)
- Enabled: all five layers functional
- Multi-user: isolated bank_id, cache, paths
- Security: private/refuse/shared gates verified
- No private data in upstream code
- Integration: exception-safe, prompt-cache-safe
```

**Recommendation: Merge.**
