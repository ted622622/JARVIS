# JARVIS 全面體檢 + OpenClaw 對標分析

> 日期：2026-02-19
> 執行者：Claude Sonnet 4.6 (3 parallel agents)
> 模式：Read-only diagnostic，無 API 測試，無重啟

---

## Part A — JARVIS 功能體檢

---

### A1. 架構層 (Architecture)

#### A1a. CEO Agent 完整訊息路徑

```
TG message
  → TelegramClient._handle_text_message()     [batch accumulator]
  → TelegramClient._process_batch()            [6s debounce, combine with \n]
  → on_telegram_message() closure in main.py    [line 436]
  → CEOAgent.handle_message()                   [line 254]
     1. Silent mode check                        [line 279]
     2. _process_message()                       [line 292]
        2a. Compressor tracking                  [line 351]
        2b. Emotion classification               [line 354]
        2c. Skill match (regex + LLM judge)      [line 360]  — early return if matched
        2d. Turn count / memory flush            [line 365]
        2e. Session transcript idle check        [line 372]
        2f. Memory search (BM25 or Hybrid)       [line 381]
        2g. Complexity classification            [line 396]
            → Agent SDK dispatch if COMPLEX      [line 397]  — early return if success
        2h. Proactive web search                 [line 446]
            → Booking short-circuit              [line 462]  — early return if booking_url
        2i. Long-content detection (Patch P)     [line 487]
            → _handle_long_content() chunking    [line 524]  — early return if triggered
        2j. Build system prompt                  [line 538]
        2k. Build messages (system + 6 history)  [line 543]
        2l. CEO LLM call                         [line 550]
        2m. Tool-use loop (3 rounds max)         [line 562]
        2n. Empty reply guard                    [line 600]
        2o. Store to MemOS + daily log           [line 608]
        2p. Emotion passthrough for TTS          [line 611]
        2q. Pre-compaction flush (async task)    [line 617]
        2r. Soul growth learning                 [line 626]
        2s. Shared memory (Clawra only)          [line 637]
        2t. Booking dict wrapping                [line 646]
     3. Clawra s2t conversion (OpenCC)           [line 304]
  → return to TelegramClient._send_reply()
```

#### A1b. Error Handling 清單

| Step | 機制 | 狀態 |
|------|------|------|
| Silent mode outer wrap | `try/except RouterError` → 15-min silent mode | OK |
| Emotion classify | **無 try/except** — 靠外層 RouterError catch | LOW risk |
| Skill match | `try/except Exception` + `asyncio.wait_for(timeout=45)` | OK |
| Memory search | `try/except Exception` → DEBUG log | OK |
| Agent SDK | `try/except Exception` → fallthrough | OK |
| Proactive web | 多層 nested try/except | OK |
| **CEO LLM call** | **無 try/except，無 asyncio.wait_for** | **HIGH** |
| Tool-use followup | **同樣無 timeout** | **HIGH** |
| Store conversation | `try/except Exception` → DEBUG log | OK |
| Soul growth | `try/except Exception` → WARNING log | OK |
| Shared memory | `try/except Exception` → WARNING log | OK |

#### A1c. Tool Call 系統

| Tag | Regex | Handler | 內容上限 |
|-----|-------|---------|----------|
| `[FETCH:url]` | `\[(?:FETCH\|SEARCH\|MAPS):([^\]]+)\]` | ReactExecutor `web_browse` chain | 50,000 chars |
| `[SEARCH:query]` | 同上 | ReactExecutor `web_search` chain | 3,000 chars |
| `[MAPS:query]` | 同上 | `browser.search_google_maps()` | N/A |

Tool-use loop: 最多 3 rounds，每 round 掃全部 tag → 執行 → 結果回餵 LLM (max_tokens=4096)。
**3 rounds 用完後**：最後一輪 reply 原封不動回傳用戶，可能殘留未解析的 `[SEARCH:...]` tag。

#### A1d. ReactExecutor Fallback Chains

```
web_browse:     browser → knowledge
web_search:     browser → search → knowledge
maps_search:    browser → knowledge
file_operation: interpreter → code → knowledge
code_task:      code → knowledge
calendar:       gog → knowledge        ← gog execute() 是 stub，永遠 fallback
email:          gog → knowledge        ← 同上
booking:        browser → assist
ticket:         browser → assist
general:        knowledge
```

Fuse 三層保護：per-task 3 rounds / 60s，sliding window 5 tasks/5min，daily budget 10K tokens。

#### A1e. Model Router / Balancer

- **CEO chain**: zhipu_ceo (GLM-4.6v/4.7 via balancer) → groq (llama-3.3-70b) → openrouter (deepseek-chat)
- **Lite tasks**: GLM-4.5-air (emotion classify, skill judge, pref extract, caring msg)
- **Balancer**: 選 remaining tokens 較高的 model，但 `ZHIPU_CEO_MODEL` 非 `"auto"` 時直接跳過 balancer

#### A1f. Worker 註冊表

| Worker | ReactExecutor chains | 實際被呼叫 |
|--------|---------------------|-----------|
| code | code_task, file_operation | YES (via ReactExecutor) |
| interpreter | file_operation | **NO** (TaskRouter 從不分類為 file_operation) |
| browser | web_browse, web_search, maps, booking, ticket | YES |
| vision | (無) | YES (Telegram photo handler 直接呼叫) |
| selfie | (無) | YES (SkillRegistry) |
| voice | (無) | YES (Telegram 直接呼叫) |
| knowledge | 所有 chain 的最後一站 | YES |
| gog | calendar, email | **stub** (execute() 永遠回 `{"status":"ready"}`) |
| assist | booking, ticket | YES |
| search | web_search | YES |
| transcribe | (無) | YES (Telegram document handler) |

**ParallelDispatcher**: 完整實作 79 行，但從未被 instantiate 或呼叫。dead code。

---

### A1 Issues Summary

| ID | 嚴重度 | 問題 | 位置 |
|----|--------|------|------|
| A1-1 | **HIGH** | CEO 主 LLM call 無 timeout — API hang 會永久阻塞 bot | `ceo_agent.py:550` |
| A1-2 | **MEDIUM** | 未解析 tool tag 洩漏給用戶 (3 rounds 用完後) | `ceo_agent.py:562-593` |
| A1-3 | **MEDIUM** | ParallelDispatcher + TaskRouter.build_ceo_context() 是 dead code | `parallel_dispatcher.py`, `task_router.py` |
| A1-4 | **HIGH** | GogWorker.execute() 是 stub — calendar/email chain 永遠 fallback 到 knowledge | `gog_worker.py:199-207` |
| A1-5 | **MEDIUM** | GoogleCalendarClient 376 行完整實作但從未 wire 進 main.py (dead code) | `clients/google_calendar.py` |
| A1-6 | **LOW** | Emotion classify 無獨立 try/except | `ceo_agent.py:354-357` |

---

### A2. Google 整合層

| 整合項 | 狀態 |
|--------|------|
| Calendar read (today/upcoming) | **OK** — via gog CLI (Heartbeat 直接呼叫 specific methods) |
| Calendar create | **OK** — via gog CLI (PostActionChain) |
| Calendar 雙帳號 + 衝突偵測 | **Dead code** — `clients/google_calendar.py` 未 wire |
| Gmail search/send | **OK** — via gog CLI (有 worker method，但 execute() stub 導致 ReactExecutor 無法使用) |
| Google Drive search | **OK** — via gog CLI (同上問題) |
| Google Contacts | gog CLI 支援但 **無 worker method** |
| Gemini Embedding | **OK** — HybridSearch |

#### Morning Brief 數據源

```
heartbeat.morning_brief()
  → gog_worker.get_today_events()     ← 直接呼叫，不經 execute()，正常
  → fallback: MemOS cache
  → fallback: "行事曆未連線"
```

Hourly patrol 用 `get_upcoming_events(30)` (30 分鐘預警)，evening summary 用 `get_events_for_date(tomorrow)`。
這些都直接呼叫 specific methods，**不受 execute() stub 影響**。

### A2 Issues Summary

| ID | 嚴重度 | 問題 | 位置 |
|----|--------|------|------|
| A2-1 | **HIGH** | GogWorker.execute() stub 使 ReactExecutor calendar/email chain 永遠失敗 | `gog_worker.py:199` |
| A2-2 | **MEDIUM** | GoogleCalendarClient dead code (376 行，含雙帳號/衝突偵測) | `clients/google_calendar.py` |
| A2-3 | **LOW** | Google Contacts 未暴露 worker method | `gog_worker.py` |

---

### A3. 研究深度層

#### 研究流程 ("幫我研究 XXX")

```
1. _classify_complexity() → COMPLEX (需含"幫我"前綴)
2. AgentExecutor.run(tier="complex") → 40 turns / 420s
   ↓ 失敗
3. _proactive_web_search() → 只觸發天氣/股價/新聞 keyword
4. CEO LLM call → tool-use loop (3 rounds max)
   ↓ 每 round
   LLM outputs [SEARCH:query] → ReactExecutor → results
   → 回餵 LLM "根據以上資訊回答用戶的問題..."
   → LLM 可再產生新 [SEARCH:] tag → 下一 round
```

#### 研究品質維度

| 維度 | 狀態 |
|------|------|
| 結果摘要 | PARTIAL — 純靠 LLM 自由生成，無 template |
| 交叉驗證 | **NONE** — 多個搜尋結果直接 concat，無比對邏輯 |
| 結構化輸出 | **NONE** — 無強制段落 (背景/發現/來源/信心度) |
| 來源標註 | **NONE** — raw text 注入，不追蹤來源 |
| 查詢分解 | **NONE** — 不分解子問題，完全靠 LLM 自主行為 |
| 深度控制 | **NONE** — 無 "夠了嗎?" 中間檢查 |

### A3 Issues Summary

| ID | 嚴重度 | 問題 | 位置 |
|----|--------|------|------|
| A3-1 | **HIGH** | "幫我" 前綴必要 — "研究一下 X"、"調查 X" 分類為 SIMPLE | `ceo_agent.py:52-60` |
| A3-2 | **HIGH** | 無研究框架 — 無 query decomposition, 無 structured output, 無 source tracking | `ceo_agent.py:562-593` |
| A3-3 | **MEDIUM** | 僅 3 rounds tool-use，無深度判斷 | `ceo_agent.py:562` |
| A3-4 | **MEDIUM** | _proactive_web_search() 不觸發通用研究 keyword | `ceo_agent.py:446-459` |
| A3-5 | **LOW** | ReactExecutor daily budget 10K tokens 偏緊 | `react_executor.py:92` |

---

### A4. 排程 / Heartbeat 層

#### 完整 Cron Job 清單 (14 jobs)

| # | Job ID | 排程 | 來源 |
|---|--------|------|------|
| 1 | `hourly_patrol` | 每 60 min | heartbeat.py |
| 2 | `morning_brief` | 每日 07:30 | heartbeat.py |
| 3 | `health_check` | 每 6 小時 | heartbeat.py |
| 4 | `nightly_backup` | 每日 03:00 | heartbeat.py |
| 5 | `night_owl` | 00:00-04:59, 每 30 min | heartbeat.py |
| 6 | `evening_summary` | 每日 23:00 | heartbeat.py |
| 7 | `memory_cleanup` | 每日 03:15 | heartbeat.py |
| 8 | `pending_tasks` | 每 15 min | heartbeat.py |
| 9 | `pending_selfies_check` | 每 5 min | heartbeat.py |
| 10 | `clawra_morning` | 每日 08:30 | heartbeat.py |
| 11 | `clawra_daily_share` | 每日 13-17 隨機 (啟動時固定) | heartbeat.py |
| 12 | `clawra_evening` | 每日 22:00 | heartbeat.py |
| 13 | `clawra_missing_check` | 08:00-22:00, 每 2h at :15 | heartbeat.py |
| 14 | `skill_learner_propose` | 每日 03:30 | **main.py** (非 heartbeat) |

#### Morning Brief 內容

1. Header: "早安，Ted！"
2. Weather: Open-Meteo / fallback
3. Calendar: gog.get_today_events() → MemOS cache → "行事曆未連線"
4. Reminders: ReminderManager.get_today()
5. Trading hint: 交易日提示
6. Token saving: survival.tracker.daily_report()
7. Agent SDK usage: token_usage.json
8. Token pool balance: model_balancer.get_status() + alert

### A4 Issues Summary

| ID | 嚴重度 | 問題 | 位置 |
|----|--------|------|------|
| A4-1 | **MEDIUM** | memory_cleanup 無 try/except — 異常會無聲失敗 | `heartbeat.py:650-659` |
| A4-2 | **LOW** | clawra_daily_share 時間啟動時固定，非每日隨機 | `heartbeat.py:206-207` |
| A4-3 | **LOW** | skill_learner_propose 在 main.py 註冊，脫離 Heartbeat 抽象 | `main.py:425-432` |
| A4-4 | **LOW** | _clawra_daily_count 純記憶體，重啟歸零 | `heartbeat.py:87-89` |
| A4-5 | **LOW** | Backup 加密靜默降級 — 用戶不知道備份未加密 | `memos_manager.py:164-167` |

---

### A5. 記憶層

#### MemOS Database Schema

| 表格 | 主鍵 | 用途 |
|------|------|------|
| `working_memory` | `key TEXT` | 跨 agent 共享狀態 (RAM cached) |
| `long_term` | `(category, key)` | 持久偏好/決策 |
| `conversation_log` | `id AUTOINCREMENT` | 對話歷史 |

#### HybridSearch 參數

| 參數 | 值 | 說明 |
|------|-----|------|
| BM25 weight | 0.3 | keyword 匹配 |
| Embedding weight | 0.7 | 語意搜尋 (Gemini) |
| Both-engine boost | +0.1 | 雙引擎命中加分 |
| DECAY_LAMBDA | 0.0154 | 半衰期 ~45 天 |
| MMR_THRESHOLD | 0.7 | SequenceMatcher 去重 |

#### MEMORY.md 狀態

```markdown
# 長期記憶
## 用戶偏好
- （待記錄）
- 不要用語音回覆，要用打字的
- 回覆時要用Typing的方式
## 重要決策
- （待記錄）
## 常用設定
- （待記錄）
```

**僅 2 筆真實記錄**，且是同一個偏好 (text-over-voice) 的兩種說法。

#### ConversationCompressor

- `recent_turns_keep = 10` (保留最近 10 組 user+assistant)
- `max_summary_lines = 30` (舊對話壓縮為單行摘要)
- Pre-flush callback 已 wire 到 CEO `_pre_flush_extract()`
- **但 `get_context_for_ceo()` 從未被呼叫** — 壓縮後的 context 是 dead code
- CEO 自己用 `SELECT ... FROM conversation_log ... LIMIT 6` 取歷史，完全繞過 compressor

### A5 Issues Summary

| ID | 嚴重度 | 問題 | 位置 |
|----|--------|------|------|
| A5-1 | **HIGH** | `get_context_for_ceo()` dead code — 壓縮後的歷史從未注入 LLM prompt | `conversation_compressor.py:55-79` |
| A5-2 | **HIGH** | MEMORY.md 幾乎空白 — 長期記憶提取機制未有效運作 | `memory/MEMORY.md` |
| A5-3 | **MEDIUM** | Daily log 有 `{當前時間}` 未替換的 template placeholder (4+ 處) | `memory/daily/2026-02-18.md` |
| A5-4 | **LOW** | conversation_log 表無 timestamp index | `memos_manager.py:42-48` |

---

## Part B — OpenClaw 對標分析

---

### B1. System Prompt 組裝

| 面向 | OpenClaw | JARVIS |
|------|----------|--------|
| 結構 | 25 個 section，Markdown headers | 無 section headers，串接式 |
| 工具列表 | 動態生成 tool list 含 priority | Tag-based，不列在 prompt |
| Prompt 模式 | full / minimal / none (sub-agent 用 minimal 省 token) | 單一模式 |
| 安全聲明 | 硬編碼 safety section | SecurityGate runtime 檢查 |
| 靜默回覆 | `SILENT_REPLY_TOKEN` 讓 LLM 明確不回覆 | 無，有時生成不必要回覆 |
| Runtime metadata | 一行注入 agent ID / model / OS / channel | 無 |

**可以抄**:
- Prompt 加 Markdown section headers (改動量: S，效果: 中)
- Sub-agent dispatch 用 minimal prompt 省 token (改動量: M，效果: 高)
- 加 SILENT_REPLY_TOKEN 讓 LLM 可以選擇不回覆 (改動量: S，效果: 中)

---

### B2. Tool Calling

| 面向 | OpenClaw | JARVIS |
|------|----------|--------|
| 機制 | Claude native `tool_use` API (JSON Schema) | Text regex tag `[FETCH:]` / `[SEARCH:]` / `[MAPS:]` |
| 並行 | 單次 response 可含多個 tool calls | 順序執行所有 tag |
| Schema 驗證 | TypeBox JSON Schema per tool | 無 |
| Policy | Per-session allowlist/blocklist pipeline | SecurityGate 三級判定 |

**可以抄**:
- Tag 內多個匹配可用 `asyncio.gather()` 並行執行 (改動量: S，效果: 中)
- Tool-level policy layer (e.g. Clawra 不能用 code worker) (改動量: M，效果: 中)
- 遷移到 native tool_use 是最高影響改進，但需換 API provider 支援 (改動量: XL，效果: 極高)

---

### B3. Skill 系統

| 面向 | OpenClaw | JARVIS |
|------|----------|--------|
| 格式 | SKILL.md (YAML frontmatter + Markdown) | skill.yaml + main.py |
| 載入 | 6 層 precedence (bundled→user→project→workspace) | 單一 `skills/` 目錄 |
| LLM 互動 | LLM 自主讀 SKILL.md → 自己決定執行 | CEO regex 判斷 → 呼叫 registry |
| 依賴宣告 | `requires.bins` + install 指令 | 無 |
| Prompt budget | 30K chars，binary search 裁切 | 無限制 |
| Auto-learn | 無 | SkillLearner (MIN_REPEAT=3，WINDOW_DAYS=14) |

**可以抄**:
- Skill summary 注入 system prompt 讓 LLM 自主判斷 (改動量: M，效果: 高)
- 支援 user-level skills `~/.jarvis/skills/` (改動量: M，效果: 中)
- Prompt char budget 防 context overflow (改動量: S，效果: 低，目前 skill 少)

---

### B4. 記憶系統

| 面向 | OpenClaw | JARVIS |
|------|----------|--------|
| 向量儲存 | SQLite + sqlite-vec (本地) | Gemini Embedding API (雲端) |
| Keyword | SQLite FTS5 | rank-bm25 Python lib |
| Hybrid 權重 | 可設定 vectorWeight / textWeight | 硬編碼 BM25=0.3, Emb=0.7 |
| Temporal decay | 可設定 halfLifeDays (預設 30), 預設 disabled | 固定 λ=0.0154 (~45d), always on |
| MMR | Jaccard token similarity, λ=0.7 | SequenceMatcher 500 chars |
| File watcher | chokidar live sync, debounce re-embed | 無 (啟動時 build 一次) |
| 記憶交付 | LLM 用 `memory_search` tool (pull model) | Top-k 注入 prompt (push model) |
| Evergreen | MEMORY.md 明確標記不 decay | 無日期檔案不 decay (隱式) |
| 離線能力 | sqlite-vec 全離線 | 依賴 Gemini API |

**可以抄**:
- File watcher (`watchdog`) 自動 re-embed (改動量: M，效果: 高)
- Memory 改 tool-based pull model 省 token (改動量: L，效果: 高)
- HybridSearch 權重可設定 (改動量: S，效果: 低)
- MMR 升級 Jaccard token similarity (改動量: S，效果: 中)

---

### B5. Multi-Agent / Sub-Agent

| 面向 | OpenClaw | JARVIS |
|------|----------|--------|
| 觸發 | LLM 自主呼叫 `sessions_spawn` tool | CEO regex `_classify_complexity()` |
| 持久化 | SubagentRegistry disk JSON | 無 (fire-and-forget) |
| 完成通知 | `runSubagentAnnounceFlow()` push 回 requester | 無 (同步等待或丟失) |
| 操控 | steer (重新指令) / kill | 無 |
| Depth limit | 可設定遞迴深度上限 | 無 |
| Sub-agent prompt | minimal mode (省 60%+ token) | 完整 prompt |

**可以抄**:
- Sub-agent registry `data/agent_runs.json` + heartbeat 完成通知 (改動量: M，效果: 高)
- Sub-agent minimal prompt mode (改動量: M，效果: 高 — 直接省 token)
- `/cancel` 指令停止進行中任務 (改動量: S，效果: 中)

---

### B6. Bootstrap / Config

| 面向 | OpenClaw | JARVIS |
|------|----------|--------|
| 檔案數 | 8 types (AGENTS/SOUL/TOOLS/IDENTITY/USER/HEARTBEAT/BOOTSTRAP/MEMORY) | 6 types (SOUL×2/IDENTITY/USER/config/MEMORY) |
| Profile 支援 | `OPENCLAW_PROFILE=<name>` 切換 workspace | 無 |
| TOOLS.md | 獨立工具指引檔 | 無 (工具知識嵌在 code) |
| HEARTBEAT.md | 可自訂 heartbeat 提示文字 | 硬編碼 |
| Template 產生 | 缺檔自動從 template scaffold | 無 |

**可以抄**:
- `config/TOOLS.md` 列出各 worker/tool 使用方式 (改動量: S，效果: 中)
- `config/HEARTBEAT.md` 可自訂 heartbeat 人格 (改動量: S，效果: 低)

---

### B7. 安全機制

| 面向 | OpenClaw | JARVIS |
|------|----------|--------|
| Startup audit | `runSecurityAudit()` 掃描 690 行 (config/permission/token/exposure) | 無 |
| 嚴重度分級 | info / warn / critical + structured findings | AUTO_ALLOW / CONFIRM / AUTO_BLOCK |
| Tool policy | Per-session/agent/group allowlist + blocklist pipeline | SecurityGate 三級 + Bash prefix list |
| Skill 掃描 | Skill code safety scanner (deep mode 讀原始碼) | 無 (SkillLearner 產生但不驗證) |
| 文件權限 | Windows ACL / Unix permissions check | 無 |
| Rate limiting | Auth attempt rate limiting | 無 |
| SoulGuard | 無 (SOUL.md 不設為 immutable) | SoulGuard (CORE file 不可寫, growth 驗證) |

**可以抄**:
- 啟動時 `security_audit()` 檢查 config/data 權限 + API key exposure (改動量: M，效果: 中)
- SkillLearner 產生 code 前做 safety scan (改動量: S，效果: 中)
- Auth rate limiting (改動量: S，效果: 低)

---

### B8. Google 整合

| 面向 | OpenClaw | JARVIS |
|------|----------|--------|
| 整合方式 | SKILL.md + gog CLI (LLM 自主讀指令 → exec) | GogWorker Python wrapper |
| Calendar | 完整 (list/create/update/colors) | list + create (update 未實作) |
| Gmail | 完整 (search/send/draft/reply) | search + send (draft/reply 未實作) |
| Drive | search | search |
| Contacts | list | 無 |
| Sheets | get/update/append/clear/metadata | 無 |
| Docs | export/cat/copy | 無 |
| 執行問題 | LLM 透過 `exec` tool 直接呼叫 gog CLI | **execute() stub 導致 ReactExecutor 無法使用 gog** |

**可以抄**:
- 修復 `GogWorker.execute()` — parse task string dispatch 到 specific methods (改動量: S，效果: 極高)
- 擴展 gog worker: Calendar update, Gmail draft/reply, Contacts list (改動量: M，效果: 高)
- 新增 Google Sheets/Docs 整合 (改動量: L，效果: 中)

---

## Part C — 特定缺口分析

---

### C1. Calendar → Brief 數據鏈路

**現狀**：Morning brief 透過 `gog_worker.get_today_events()` 直接呼叫 gog CLI → 解析 JSON → 格式化。
這條路徑 **正常運作** (不經 ReactExecutor)。

**缺口**：
1. `GogWorker.execute()` stub 導致 ReactExecutor 的 `calendar` chain 永遠失敗 → fallback 到 knowledge (LLM 胡說)
2. `GoogleCalendarClient` (雙帳號 + 衝突偵測) 是 dead code
3. Calendar update 未實作 (gog CLI 支援但 worker 未暴露)

**修復優先度**: HIGH — `execute()` stub 修復只需 ~20 行 parse logic

---

### C2. 研究增強

**現狀**：研究品質完全依賴 LLM 自主行為 + 3-round tool loop。無 query decomposition、無 structured output、無 source tracking。

**缺口**：
1. Complexity 分類要求 "幫我" 前綴 (A3-1)
2. 無研究專用 prompt template (A3-2)
3. Proactive web search 不觸發通用研究 keyword (A3-4)
4. Agent SDK 是唯一的深度研究管道，但觸發條件太窄

**建議改進**:
- 擴展 `_COMPLEX_PATTERNS`: 加入 `研究|分析|比較|調查|評估` (去掉 "幫我" 前綴需求)
- 研究任務加 structured output prompt: "請用以下格式回答：## 背景 / ## 發現 / ## 來源 / ## 信心度"
- 擴展 `_WEB_NEED_PATTERNS` 含通用研究 keyword

---

### C3. Tool Call 穩定性

**現狀**：Text regex tag parsing 有固有脆弱性 — tag 可能被 hallucinate、malform、或 3 rounds 後殘留。

**缺口**：
1. 未解析 tag 洩漏給用戶 (A1-2)
2. CEO LLM call 無 timeout (A1-1)
3. 同一 round 內多 tag 順序執行，不並行

**建議改進**:
- 3 rounds 後 strip 殘留 `[FETCH:]`/`[SEARCH:]`/`[MAPS:]` tag (regex replace)
- CEO LLM call 加 `asyncio.wait_for(timeout=120)`
- 同一 round 內多 tag 用 `asyncio.gather()` 並行

---

## Priority Fix List — 按影響排序

| # | 問題 | 嚴重度 | 改動量 | 能否從 OpenClaw 抄 | 預期效果 |
|---|------|--------|--------|-------------------|----------|
| 1 | CEO LLM call 無 timeout (A1-1) | HIGH | **S** (2 行 asyncio.wait_for) | 不需要 | 防止 bot 永久阻塞 |
| 2 | GogWorker.execute() stub (A1-4/A2-1) | HIGH | **S** (~20 行 parse logic) | 參考 OpenClaw SKILL.md 的 task routing | calendar/email chain 真正可用 |
| 3 | 未解析 tool tag 洩漏 (A1-2) | MEDIUM | **S** (5 行 regex strip) | 不需要 | 用戶不再看到 raw tag |
| 4 | Complexity 分類擴展 (A3-1) | HIGH | **S** (regex 調整) | 不需要 | 更多研究任務觸發 Agent SDK |
| 5 | MEMORY.md 長期記憶修復 (A5-2) | HIGH | **M** (pre-flush → MEMORY.md 寫入路徑) | 參考 OpenClaw file watcher 概念 | 長期記憶累積 |
| 6 | Compressor context 接入 CEO (A5-1) | HIGH | **M** (改 _build_messages() 用 compressor) | 參考 OpenClaw memory-as-tool | CEO 看到更多對話歷史 |
| 7 | memory_cleanup 加 try/except (A4-1) | MEDIUM | **S** (5 行) | 不需要 | 防靜默失敗 |
| 8 | Daily log template placeholder (A5-3) | MEDIUM | **S** (find+fix 產生源) | 不需要 | 日誌品質 |
| 9 | Sub-agent minimal prompt mode (B5) | — | **M** | 抄 OpenClaw 的 prompt mode 概念 | Agent SDK 省 60%+ token |
| 10 | File watcher auto re-embed (B4) | — | **M** | 抄 OpenClaw chokidar → Python watchdog | Embedding index 即時更新 |
| 11 | ParallelDispatcher 接入或移除 (A1-3) | MEDIUM | **M** (接入) / **S** (移除) | 參考 OpenClaw parallel tool calls | 多任務加速或減少 dead code |
| 12 | GoogleCalendarClient 決定去留 (A1-5) | MEDIUM | **S** (移除) / **L** (接入) | — | 減少 dead code 或啟用雙帳號 |
| 13 | 研究 structured output template (A3-2) | HIGH | **S** (prompt engineering) | — | 研究品質大幅提升 |
| 14 | Tag 並行執行 (C3) | — | **S** (asyncio.gather) | 參考 OpenClaw parallel tool calls | 多 tool call 加速 |
| 15 | SkillLearner safety scan (B7) | — | **S** | 參考 OpenClaw skill-scanner.ts | 防止自動生成危險 code |

---

## 整體評估

### JARVIS 優勢 (vs OpenClaw)

1. **SoulGuard + SOUL_CORE/GROWTH 分離** — OpenClaw 的 SOUL.md 無 immutability 保護
2. **SkillLearner 自動學習** — OpenClaw 無自動 skill proposal 機制
3. **AppearanceBuilder + 分鏡系統** — 高度客製化的 selfie 生成 pipeline
4. **SharedMemory (Clawra moments)** — 關係記憶追蹤
5. **Message Batching + Split Reply** — 更自然的對話節奏
6. **PostActionChain** — booking → calendar + reminders 自動串聯

### JARVIS 劣勢 (vs OpenClaw)

1. **Text tag parsing vs native tool_use** — 最大架構差距
2. **Memory push vs pull model** — 每次都注入 top-k 浪費 token
3. **無 sub-agent persistence** — 長任務結果可能遺失
4. **研究框架空白** — 無 query decomposition, 無 structured output
5. **GogWorker execute() stub** — Google 整合斷裂
6. **CEO LLM call 無 timeout** — 唯一的 hanging risk

### 結論

JARVIS 的 persona/emotion/relationship 系統遠超 OpenClaw，但在 **核心 agent 基礎設施** (tool calling, memory retrieval, sub-agent management, research framework) 方面有明顯差距。上方 Priority Fix List 前 6 項（全部 S-M 改動量）能解決最關鍵的功能斷裂。
