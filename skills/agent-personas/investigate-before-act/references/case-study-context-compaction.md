# Case Study: Context Compaction Memory Demotion

## What Happened (2026-05-02)

User asked "为什么你的 memory 不遵守" — memory is forcefully injected every conversation but the LLM doesn't follow it.

## Investigation Process

1. User pointed out that memory IS injected but not followed
2. Searched GitHub issues for NousResearch/hermes-agent
3. Found Issue #17251 (P1 Bug): "Context Compaction Demotes Memory to Background Reference"
4. Root cause: `SUMMARY_PREFIX` in `context_compressor.py` labels everything as "background reference, NOT active instructions"
5. LLM follows literally → ignores memory after compaction

## Solution

Applied upstream PRs #17380 (memory authority) and #17349 (recompaction detection) locally.

## Key Lesson

The problem was NOT "memory not injected" — it was "memory demoted by compaction prompt". 
Investigation required searching upstream issues, not just checking local config.

When a user reports that injected rules/instructions aren't being followed:
1. Check if context compaction has triggered (long sessions)
2. Check `SUMMARY_PREFIX` in `context_compressor.py` for "background reference"
3. Check `memory_manager.py` for "informational background data"
4. Search upstream issues: `#17251`, `#17344`
