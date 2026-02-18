# Code Review — 2026-02-18

Scope: Agent SDK (Phase 2), Patch T+ (selfie framing, pre-flush, decay), GLM-TTS, memory search, Clawra persona, Telegram, Heartbeat

## Severe Issues (Fixed Directly)

### 1. `core/ceo_agent.py:234` — `_last_emotion` not initialized
- **Problem**: `_last_emotion` is set in `_process_message()` but never declared in `__init__`. If accessed before the first message (e.g., by main.py voice passthrough), raises `AttributeError`.
- **Fix**: Added `self._last_emotion: str = "normal"` to `__init__`.

### 2. `core/ceo_agent.py:1029` — Dead code: `elif` inside wrong `if` block
- **Problem**: `elif not task_result.success` was indented inside `if task_result.success`, making the condition always False. Failed reactive tool calls (`[FETCH:]`/`[SEARCH:]`) never saved to pending tasks.
- **Fix**: Separated into independent `if` blocks:
  ```python
  if task_result.success and isinstance(task_result.result, dict):
      ...
      return None
  if not task_result.success and self.pending:
      self.pending.add(...)
  return None
  ```

### 3. `core/ceo_agent.py:1349` — Empty base prompt when soul not loaded
- **Problem**: Python ternary `"AB" f"\n{extra}" if extra else ""` evaluates to empty string `""` when `extra` is falsy. When soul is not loaded AND no context provided, the system prompt becomes empty.
- **Fix**: Split into explicit `base = "..."` + `if extra: base += f"\n{extra}"`.

---

## Medium Issues (For Ted's Review)

### 4. `core/ceo_agent.py:589` — Fire-and-forget asyncio.create_task
```python
asyncio.create_task(self._compressor.flush_pending())
```
If `flush_pending()` raises, the exception is silently swallowed (Python default for unhandled task exceptions). Suggest wrapping in a helper that logs errors:
```python
async def _safe_flush():
    try:
        await self._compressor.flush_pending()
    except Exception as e:
        logger.warning(f"Pre-flush failed: {e}")
asyncio.create_task(_safe_flush())
```

### 5. `core/agent_executor.py` — `_prepare_env()` modifies global env vars
Every `AgentExecutor` instantiation calls `_prepare_env()` which modifies `os.environ` globally (sets `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, etc.). If multiple components instantiate AgentExecutor, or if other code reads these env vars, there could be side effects. Consider making env var mapping lazy (only when `run()` is called) and restoring after.

### 6. `core/heartbeat.py:144` — Memory cleanup scheduling edge case
```python
minute=bm + 15 if bm + 15 < 60 else 0,
```
When backup is at `03:45`, memory_cleanup would be at `03:00` (same hour, minute=0), which is BEFORE backup. The hour is not incremented. Fix:
```python
cleanup_minute = bm + 15
cleanup_hour = bh + (cleanup_minute // 60)
cleanup_minute = cleanup_minute % 60
```

### 7. `skills/selfie/main.py:220` — Deprecated asyncio pattern
```python
asyncio.get_event_loop().run_in_executor(None, _call)
```
Python 3.10+ deprecates `get_event_loop()` in favor of `asyncio.get_running_loop()`. Since the project is on Python 3.14, use `asyncio.to_thread(_call)` instead.

### 8. `clients/telegram_client.py:424` — Batch task error handling
```python
batch["task"] = asyncio.create_task(self._process_batch(chat_id))
```
Same fire-and-forget pattern as #4. `_process_batch()` has its own try/except, but if the outer logic fails before reaching it, the exception is swallowed. The method does have internal error handling, so this is lower risk than #4.

### 9. `workers/voice_worker.py` — Uncommitted `highpass=f=200` filter
The uncommitted change adds `highpass=f=200` to ffmpeg. This is aggressive — male voice fundamental frequency can be 85-155Hz, so some speech energy could be lost. Since the GLM-TTS rewrite (Task #16) will replace this approach entirely, recommend **reverting** this uncommitted change to keep the codebase clean.

### 10. `core/heartbeat.py:310-316` — Hidden dependency on `self.ceo`
```python
ceo = getattr(self, "ceo", None)
```
`self.ceo` is never set in `__init__`. It relies on external code setting `heartbeat.ceo = ceo_agent`. This is a hidden dependency. Consider adding `self.ceo = None` to `__init__` and a setter method.

---

## Low Issues (Informational)

### 11. `core/embedding_search.py` — Large batch embedding
`build_index()` embeds all texts in a single batch. For very large memory directories, this could hit Gemini API rate limits. Currently not an issue at JARVIS's memory size, but worth noting for future growth.

### 12. `workers/selfie_worker.py` — Broad medium framing regex
`detect_framing()` medium regex includes common words like `咖啡`, `自拍`. This could interfere with non-selfie messages containing these words, but since `detect_framing()` is only called after selfie intent is confirmed, the blast radius is contained.

### 13. `skills/selfie/main.py:201` — Hardcoded model name
`"gemini-2.0-flash-exp"` may not exist in the future. Consider using an env var or config value.

### 14. `clients/telegram_client.py:356-359` — Long minimum typing delay
`_simulate_typing()` has a 15-60 second delay for ALL Clawra replies. Even a 1-character response waits 15 seconds. Consider lowering the minimum to ~5 seconds for short replies.

### 15. `config/SOUL_CLAWRA.md:86` — Confusing example
`「東西」→「東西」` is listed as a simplified→traditional example, but both forms are identical in this case. The example doesn't illustrate the rule.

---

## Architecture Notes

### Agent SDK Integration (Phase 2)
- Clean separation: `AgentExecutor` is lazily initialized, fails gracefully if SDK not installed
- Token tracking with daily limits is well-designed
- Bash security whitelisting (`BASH_ALLOWED_PREFIXES`, `BASH_BLOCKED`) is good defense
- Complexity classification uses regex (no LLM cost), appropriate for routing

### Selfie Framing (Patch T+)
- 4-framing system (mirror/full_body/medium/closeup) is well-structured
- `detect_framing()` → `build_framing_prompt()` chain is clean
- Backward-compat aliases (`detect_mode`, `build_prompt`) properly maintained
- `AppearanceBuilder.select_scene(framing)` correctly uses per-framing scene pools

### Pre-flush Memory (Patch T+)
- `set_pre_flush_callback()` pattern is clean and testable
- Extraction prompt is well-scoped (FACT: lines only)
- Full try/except protection — failure only logs warning

### Memory Search (Patch T)
- Temporal decay + MMR post-processing is cleanly isolated
- Decay lambda 0.0154 (45-day half-life) is reasonable for personal memory
- MMR SequenceMatcher threshold 0.7 prevents near-duplicate results

---

## Summary

| Severity | Count | Status |
|----------|-------|--------|
| Severe   | 3     | Fixed  |
| Medium   | 7     | Listed for Ted |
| Low      | 5     | Informational |

Total test count before review: 1077 (1072 passing, 5 live deselected)
