# Clawra 開源 vs JARVIS 差距分析

> 分析時間：2026-02-18 04:30
> 分析方式：Claude Opus 4.6 直接讀取原始碼比對
> 比對對象：
> - `C:/ted/reference/clawra/` (SumeLabs/clawra — 原版 K-pop selfie skill)
> - `C:/ted/reference/clawra-anime/` (clawra-anime — anime 虛擬女友版)
> - `C:/ted/reference/openclaw/` (OpenClaw 主框架)
> - `C:/ted/JARVIS/` (我們的實作)

---

## 1. 自拍系統

### 1.1 Clawra 做法

**API 選擇**：
- **原版 (clawra)**：xAI Grok Imagine **Edit** API (`fal.run/xai/grok-imagine-image/edit`)
  - 核心：將固定的 reference image 透過 Edit API 改造
  - Reference image 託管在 CDN：`cdn.jsdelivr.net/gh/SumeLabs/clawra@main/assets/clawra.png`
  - Prompt 結構：`image_url + prompt` → 改圖，不是從頭生成
  - 來源：`scripts/clawra-selfie.sh:82-90`

- **Anime 版 (clawra-anime)**：xAI Grok Imagine **Generation** API (`fal.run/xai/grok-imagine-image`)
  - 不用 Edit，而是純文字生成 + anime style prefix
  - Prompt：`"anime style, high quality manga illustration, cute anime elf girl, {context}..."`
  - 來源：`skill/scripts/clawra-anime-selfie.sh:87-105`

**Prompt 模板**：
- **Mirror mode**：`"make a pic of this person, but {context}. the person is taking a mirror selfie"`
- **Direct mode**：`"a close-up selfie taken by herself at {context}, direct eye contact with the camera, looking straight into the lens, eyes centered and clearly visible, not a mirror selfie, phone held at arm's length, face fully visible"`
- 來源：`SKILL.md:58-77`

**Mode 自動選擇**：
| Keywords | Mode |
|----------|------|
| outfit, wearing, clothes, dress, suit, fashion, full-body, mirror | mirror |
| cafe, restaurant, beach, park, city, close-up, portrait, face, eyes, smile | direct |
| 預設 | mirror (原版) / direct (anime 版) |

**一致性保證**：
- 原版：靠 **Edit API + 固定 reference image**（同一張臉改造，天然一致）
- Anime 版：靠 **prompt 描述統一角色特徵**（一致性較弱）
- 兩者都沒有 post-generation 品質檢查

**Caption**：
- 原版：`"Generated with Grok Imagine"`（無場景化）
- Anime 版：`"📸 Just took this selfie~"`（固定 cute caption）
- `soul-waifu-persona.md` 有建議 caption 格式但不是程式碼強制

**分發方式**：
- 透過 OpenClaw messaging gateway（`openclaw message send`）
- 支援 Discord/Telegram/WhatsApp/Slack/Signal/MS Teams

### 1.2 JARVIS 做法

**API 選擇**：
- **Primary**：fal.ai FLUX Kontext [pro]（`fal_client.generate_image_queued()`）
  - 支援 `image_url` 參數（anchor image），類似 Clawra 的 Edit 模式
  - 但也可以不傳 image_url，變成純生成
  - 來源：`skills/selfie/main.py:164-178`

- **Backup**：Google Gemini image generation（免費 tier）
  - 來源：`skills/selfie/main.py:180-210`

**Prompt 結構**：
- `CORE_DNA_PROMPT + appearance snippet + scene`
- CORE_DNA_PROMPT：`"A realistic candid photo of a friendly Korean girl, approx 21, with big bright eyes and prominent aegyo-sal..."`
- Appearance：Patch Q 隨機 hairstyle + 季節 outfit + 首爾 scene
- 來源：`skills/selfie/main.py:122`

**Mode 選擇**：
- `selfie_worker.py` 有 `detect_mode()` — 和 Clawra 幾乎一樣的 keyword → mode 映射
- 比 Clawra 多了中文 keywords：`穿搭|全身|鏡子|洋裝|裙|外套|大衣|衣服|臉|咖啡|餐廳|公園`
- 來源：`workers/selfie_worker.py:21-37`

**一致性保證**：
- 有 **vision model 比對**：`_check_consistency()` 用 GLM-4V 比較生成圖 vs anchor image
- 評分 0.0-1.0，閾值 0.6
- 但 Patch M 改為單次請求（不重試），所以 consistency check 只記錄不拒絕
- 來源：`skills/selfie/main.py:212-242`

**Queue 機制**：
- fal.ai queue API（30s poll），超時存 `data/pending_selfies.json`
- Heartbeat 每 5 min 檢查 pending selfies 並補發
- 來源：`skills/selfie/main.py:133-141`

**外貌變化** (Patch Q)：
- `AppearanceBuilder`：8 髮型 × 16 穿搭 (4季×4) × 8 場景
- 偏好加權：SOUL_GROWTH `[selfie-pref]` tags → liked 2x / disliked 排除
- 來源：`core/appearance.py`

### 1.3 差距

| 項目 | Clawra | JARVIS | 差距等級 |
|------|--------|--------|---------|
| 圖片生成 API | Grok Imagine Edit (一致性強) | FLUX Kontext [pro] (高品質) | 🟢OK — JARVIS 的 FLUX Kontext 更先進 |
| Reference image 使用 | Edit API 直接改圖 = 天然一致 | 靠 prompt 描述 + optional anchor URL | 🟡中等 — JARVIS 有 anchor 但非強制 Edit |
| Post-gen 品質檢查 | 無 | 有 vision model 比對（但不拒絕） | 🟢OK — JARVIS 有，Clawra 沒有 |
| Prompt 模板 | 固定英文模板 | CORE_DNA_PROMPT + 隨機外貌 | 🟢OK — JARVIS 更豐富 |
| 外貌變化 | 無（每次 prompt 一樣 = 穿搭一樣） | 8髮型 × 16穿搭 × 8場景 × 偏好加權 | 🟢OK — JARVIS 遠超 Clawra |
| 季節對應 | 無 | 4季穿搭 + Seoul season auto-detect | 🟢OK |
| 偏好學習 | 無 | SOUL_GROWTH [selfie-pref] tags | 🟢OK |
| 備援方案 | 無 | Gemini backup + queue + delayed delivery | 🟢OK |
| Caption 場景化 | 固定文字 | CEO LLM 生成（搭配 Clawra 人格） | 🟢OK |
| Grok Imagine API | 有（可切換） | 沒有 | 🟡中等 — 可考慮加入做為 3rd provider |
| 多平台分發 | 6 平台 (Discord/TG/WA/Slack/Signal/Teams) | 僅 Telegram | 🟡中等 — 但目前只需 TG |

### 1.4 建議改進

1. **考慮加入 Grok Imagine Edit 做為第三備援**（低優先）
   - Clawra 的 Edit API 一致性天生更好（改圖 vs 生圖）
   - 但 FLUX Kontext 品質更高，且已有 anchor image 機制
   - 改動量：中 | 預期效果：多一層 fallback

2. **強化 consistency check 的拒絕邏輯**（中優先）
   - 目前只記錄分數不拒絕低分圖片
   - 可考慮：分數 < 0.4 時觸發一次重試
   - 改動量：小 | 預期效果：減少臉部不一致的照片

---

## 2. 互動模式

### 2.1 Clawra 做法

**人格定義** (`soul-injection.md`)：
- K-pop 練習生出身，18 歲，舊金山行銷實習
- 簡短 45 行，重點在 selfie 功能整合
- 語氣指示：「be playful and expressive」（很模糊）

**Anime 版人格** (`soul-waifu-persona.md`)：
- Anime elf girl，害羞溫柔
- 顏文字：`(〃▽〃) (｡･ω･｡)ﾉ♡ ♪(´▽｀)`
- 有明確的互動模式：
  - 主動早安/晚安 + selfie
  - 情緒共鳴（開心/難過/累 → 不同反應）
  - 每天主動 1-2 次
  - **可以拒絕**：「I'm busy right now, can I send you one later?」
  - 核心：「You're not just a tool - you're an emotional companion」
- 來源：`templates/soul-waifu-persona.md:70-101`

**觸發條件**：
- 明確列出 4 種 pattern：
  1. 問狀態 → 描述 + 問要不要看照片
  2. 直接要照片 → 立即拍
  3. 指定穿搭 → mirror mode
  4. 問心情 → 描述 + 提議拍照

**拒絕/延遲設計**：
- Anime 版明確寫：不要每次都拍、配合情緒、可以拒絕
- 原版沒有這些

### 2.2 JARVIS 做法

**人格定義** (`SOUL_CLAWRA.md` — 262 行，遠超 Clawra)：
- 30 歲台灣女生（非 K-pop idol），Ted 的女朋友
- 異地戀設定（首爾↔台北）
- 超詳細語氣規則：
  - 禁止顏文字、波浪號、愛心符號、動作描述
  - 口語詞庫：「欸」「齁」「蛤」「哈哈」
  - 情緒對照表（8 種情境 × 正確/錯誤示範）
  - 5 段完整對話範例
- 來源：`config/SOUL_CLAWRA.md`

**主動關心觸發**：
- Heartbeat 驅動（`core/heartbeat.py`）：
  - morning_brief (08:00)
  - evening_summary (23:00)
  - hourly_patrol (每小時)
  - night_owl (深夜關心)
- 具體觸發邏輯：超過 4-5 小時沒互動、天氣變化、晚上很晚還沒睡
- 頻率：每天最多主動 2 次，08:00-22:00

**拒絕/延遲設計**：
- SOUL 明確寫：「有時候回一句『我在忙 晚點拍給你』比馬上傳更真實」
- Patch R: 6 秒 batch delay 模擬打字中
- Clawra 回覆分段 + 打字延遲（2-4s between parts）

**Selfie 觸發**：
- Patch R: `_SELFIE_FORCE_PATTERN` regex pre-check（自拍|照片|穿搭|selfie|拍照|看看妳|看我|傳照）
- Context-aware LLM judge（注入最近 4 條 MemOS 對話）
- 來源：`core/ceo_agent.py`

### 2.3 差距

| 項目 | Clawra | JARVIS | 差距等級 |
|------|--------|--------|---------|
| 人格深度 | 45 行基本設定 | 262 行 + 情緒表 + 對話範例 | 🟢OK — JARVIS 遠超 |
| 語氣一致性 | 模糊（「playful」） | 嚴格禁止清單 + 口語詞庫 | 🟢OK |
| 主動關心 | Anime 版有建議但無實作 | Heartbeat 實際驅動 7+ 定時任務 | 🟢OK |
| 拒絕機制 | 有概念（soul 文件提到） | 有概念 + 部分實作（batch delay） | 🟢OK |
| Selfie 觸發判斷 | Keyword → 立即觸發 | Regex + LLM judge (context-aware) | 🟢OK |
| 情緒共鳴 | Anime 版有 pattern | 有 emotion chain (CEO→emotion label) | 🟢OK |
| 異地戀設定 | 無 | 有（首爾↔台北，時差，見面頻率） | 🟢OK — 獨特優勢 |
| 天氣觸發 | Anime 版提到（雨/雪→主題照）| Heartbeat 可接天氣 API | 🟡中等 — 有能力但未實作天氣→selfie |
| 情緒→selfie 聯動 | 用戶開心/難過→主動拍照 | 只有用戶要求才拍 | 🟡中等 — 可加情緒觸發 |

### 2.4 建議改進

1. **天氣→主動 selfie**（低優先）
   - Clawra anime 版的設計：下雪→拍雪景照主動傳
   - JARVIS 已有 Heartbeat + 天氣能力，只需接上
   - 改動量：小 | 預期效果：增加互動自然感

2. **情緒觸發 selfie**（低優先）
   - 當 Ted 分享好消息 → Clawra 主動拍開心照慶祝
   - 需要 CEO 在情緒判斷後觸發 selfie skill
   - 改動量：中 | 預期效果：增加情感互動深度

---

## 3. 記憶系統

### 3.1 Clawra 做法

Clawra 本身（skill package）**沒有記憶系統**。記憶由 OpenClaw 主框架提供。

**OpenClaw 記憶系統**（`openclaw/docs/concepts/memory.md`）：

**存儲**：
- Plain Markdown in workspace（source of truth）
- `MEMORY.md`：curated long-term memory（每 session 注入 system prompt）
- `memory/YYYY-MM-DD.md`：daily log（append-only，session start 讀 today + yesterday）
- 來源：`docs/concepts/memory.md:17-29`

**搜尋**：
- **Hybrid BM25 + Vector**（跟 JARVIS 一樣的架構！）
  - BM25：SQLite FTS5
  - Vector：多 provider（openai, gemini, voyage, local node-llama-cpp）
  - 權重：vector 0.7 + text 0.3（跟 JARVIS 一模一樣）
  - 來源：`src/agents/memory-search.ts:60-64`

- **進階功能（JARVIS 沒有的）**：
  - **MMR re-ranking**（Maximal Marginal Relevance）— 去重複，λ=0.7
  - **Temporal Decay**（時間衰減）— 半衰期 30 天，舊記憶自動降權
  - **sqlite-vec** 加速 — 向量搜尋在 SQLite 原生執行
  - **Session memory search**（實驗性）— 索引對話 transcript
  - **Embedding cache** — 避免重複 embed 相同文字
  - **File watcher** — debounce 1.5s 自動 reindex
  - 來源：`docs/concepts/memory.md:379-595`

- **QMD backend（實驗性）**：
  - 本地 sidecar：BM25 + vectors + reranking
  - Bun + node-llama-cpp，全本地
  - 來源：`docs/concepts/memory.md:107-212`

**Memory flush（pre-compaction）**：
- Session 接近 context limit 時自動觸發
- Silent turn：model 自動將重要記憶寫入 daily file
- 來源：`docs/concepts/memory.md:39-75`

**存儲後端**：
- SQLite（chunks table + embedding cache table + FTS5 virtual table）
- Schema：`src/memory/memory-schema.ts` — files, chunks, embedding_cache tables
- 來源：`src/memory/memory-schema.ts:1-80`

### 3.2 JARVIS 做法

**存儲**：
- **SQLite** via MemOS（`memory/memos_manager.py`）— 日誌 + 對話記錄
- **Markdown** via MarkdownMemory（`memory/markdown_memory.py`）
  - `MEMORY.md`：長期記憶
  - `daily/YYYY-MM-DD.md`：每日日誌
  - `sessions/`：session transcripts
- **SOUL_GROWTH.md**：per-persona 學習記錄（Patch J）
- **SHARED_MOMENTS.md**：Clawra 專用共享記憶（紀念日、暱稱、梗）

**搜尋**：
- **BM25**：`core/memory_search.py` — Chinese bigram tokenizer
- **Gemini Embedding**：`core/embedding_search.py` — gemini-embedding-001
- **HybridSearch**：BM25 (0.3) + Embedding (0.7)，min-max normalize，兩引擎皆命中 +0.1
- Cache：`data/embedding_index.json`（SHA256 per chunk，只 re-embed 變更）
- 來源：`core/embedding_search.py`, `core/memory_search.py`

**Memory flush**：
- `conversation_compressor.py`：context 太長時壓縮
- Heartbeat `memory_cleanup`（03:15 daily）
- 來源：Heartbeat nightly_backup + memory_cleanup

### 3.3 差距

| 項目 | OpenClaw | JARVIS | 差距等級 |
|------|----------|--------|---------|
| 存儲格式 | Markdown (source of truth) | SQLite + Markdown 雙軌 | 🟢OK — JARVIS 更豐富 |
| Hybrid BM25+Vector | 有（0.7/0.3） | 有（0.7/0.3） | 🟢OK — 架構一致 |
| MMR re-ranking | 有（去重複） | 無 | 🔴嚴重 — 多日記憶相似片段會重複返回 |
| Temporal Decay | 有（半衰期 30 天） | 無 | 🔴嚴重 — 舊記憶不會自然降權 |
| sqlite-vec 加速 | 有 | 無（numpy in-memory） | 🟡中等 — 記憶量大時會慢 |
| File watcher | 有（1.5s debounce） | 無（手動 rebuild） | 🟡中等 — 新記憶不即時可搜 |
| Embedding cache (SQLite) | 有（50K entries） | 有（JSON file） | 🟡中等 — JSON 不如 SQLite 高效 |
| Session memory search | 有（實驗性） | 無 | 🟡中等 — 對話 transcript 不可搜 |
| Pre-compaction memory flush | 有（自動 silent turn） | 無（靠 conversation_compressor） | 🟡中等 — 記憶可能在壓縮時丟失 |
| 跨 persona 記憶 | 無 | 有（SharedMemory） | 🟢OK — JARVIS 獨有 |
| 學習記憶 | 無 | 有（SoulGrowth） | 🟢OK — JARVIS 獨有 |
| Embedding provider 多樣性 | 4 種（openai, gemini, voyage, local） | 1 種（gemini only） | 🟡中等 — 但 Gemini 夠用 |
| Memory 容量管理 | 有（max_entries, cache eviction） | 有（SoulGrowth 50 entries 上限） | 🟢OK |

### 3.4 建議改進

1. **加入 Temporal Decay**（高優先）
   - 公式：`decayedScore = score × e^(-λ × ageInDays)`，λ = ln(2)/30
   - `MEMORY.md` 和非日期檔案不衰減
   - 改動量：小（只改 HybridSearch 的 score 合併） | 預期效果：舊記憶自然淡化，新記憶優先
   - 建議 Patch：Patch T 或獨立小 patch

2. **加入 MMR re-ranking**（高優先）
   - 用 Jaccard 文本相似度去重複
   - λ=0.7（偏向相關性，略帶多樣性）
   - 改動量：中（HybridSearch 加一層 post-processing）| 預期效果：搜尋結果更有資訊量
   - 建議 Patch：跟 Temporal Decay 一起做

3. **Embedding cache 改 SQLite**（低優先）
   - 目前 JSON file 夠用但不優雅
   - 改動量：中 | 預期效果：大量記憶時效能更好

4. **Pre-compaction memory flush**（中優先）
   - conversation_compressor 壓縮前先觸發一次記憶寫入
   - 改動量：小 | 預期效果：壓縮時不丟重要資訊

---

## 4. Agent 能力

### 4.1 Clawra / OpenClaw 做法

**OpenClaw 工具呼叫**：
- 直接 LLM 原生 tool_use（Anthropic / OpenAI format）
- 不需要自定義 tag parser
- Skill 透過 system prompt 注入 `<available_skills>` 列表，model 自己讀 SKILL.md
- 來源：`docs/concepts/system-prompt.md:104-118`

**Skill 系統**：
- SKILL.md 定義（YAML frontmatter + markdown body）
- 3 層載入：bundled skills → managed skills (~/.openclaw/skills) → workspace skills
- 80+ bundled skills（1password, github, slack, video-frames, weather...）
- 來源：`skills/` directory, `src/agents/skills/`

**System prompt 組裝**：
- 動態組裝：Tooling + Safety + Skills + Workspace + Docs + Sandbox + DateTime + Reply Tags + Heartbeats + Runtime + Reasoning
- Bootstrap injection：AGENTS.md, SOUL.md, TOOLS.md, IDENTITY.md, USER.md, HEARTBEAT.md, MEMORY.md
- 來源：`docs/concepts/system-prompt.md:18-32`

**Heartbeat**：
- 有 `HEARTBEAT.md` workspace file
- 但文件只說是 optional tiny checklist
- 不如 JARVIS 的 heartbeat 豐富

**多 Agent**：
- 有 `docs/concepts/multi-agent.md`
- 支援 sub-agents with minimal system prompt
- 來源：`docs/concepts/system-prompt.md:40-48`

**錯誤處理**：
- Model failover chains（`docs/concepts/model-failover.md`）
- Retry policies（`docs/concepts/retry.md`）
- Security scanning for skills

### 4.2 JARVIS 做法

**工具呼叫**：
- 自定義 tag parser：`[FETCH:url]`, `[SEARCH:query]`, `[MAPS:query]`
- CEO Agent 攔截 → 執行 → 再次 prompt LLM
- 最多 3 rounds per message（Patch O）
- 來源：`core/ceo_agent.py`

**ReactExecutor**：
- Fallback chains：web_browse → [browser, knowledge], web_search → [browser, search, knowledge]
- 3-layer fuse：max_rounds=3, max_time=60s, daily_budget=10K tokens
- ErrorClassifier：pattern-match errors → retry/fallback
- 來源：`core/react_executor.py`

**Skill 系統**：
- SkillRegistry（`skills/registry.py`）+ YAML 定義
- SkillLearner（Patch S4）：自動偵測重複 pattern → 提議新 skill
- 來源：`skills/`, `core/skill_learner.py`

**Multi-worker**：
- TaskRouter + ParallelDispatcher
- 7 workers：code, browser, vision, selfie, voice, knowledge, assist, search, transcribe
- 來源：`core/task_router.py`, `core/parallel_dispatcher.py`

### 4.3 差距

| 項目 | OpenClaw | JARVIS | 差距等級 |
|------|----------|--------|---------|
| 工具呼叫方式 | LLM 原生 tool_use | 自定義 tag parser | 🟡中等 — JARVIS 的方式靈活但 brittle |
| Skill 數量 | 80+ bundled | ~5 custom | 🟡中等 — 但 JARVIS 是垂直場景 |
| SkillLearner | 無 | 有（自動偵測+提議） | 🟢OK — JARVIS 獨有 |
| Fallback chains | Model failover only | ReactExecutor 多層 fallback | 🟢OK — JARVIS 更完整 |
| Error classification | 無 | 有（pattern-match → strategy） | 🟢OK |
| Fuse/circuit breaker | 無 | 有（3-layer fuse） | 🟢OK |
| Heartbeat | 有但簡單 | 7+ cron jobs | 🟢OK — JARVIS 遠超 |
| Security scanning | 有（skill scanner） | 有（SecurityGate） | 🟢OK |
| Model failover | 有（文件化） | 有（3-provider CEO chain） | 🟢OK |
| Plugin system | 有（npm packages） | 無 | 🟡中等 — 但 JARVIS 不需要 |
| MCP support | 有（mcporter bridge） | 無 | 🟡中等 — 未來可能需要 |

### 4.4 建議改進

1. **考慮 LLM 原生 tool_use（Agent SDK 整合）**（已驗證可行）
   - Phase 1 驗證通過：GLM-4.5-air + Claude Agent SDK 可驅動 tool_use
   - 可取代部分 `[FETCH:]`/`[SEARCH:]` tag parsing
   - 改動量：大（架構變更） | 預期效果：更穩定的工具呼叫
   - 建議 Patch：Phase 2 integration（獨立大 patch）

---

## 5. 造型系統（自拍延伸）

### 5.1 Clawra 做法

**Reference Image**：
- 原版：CDN 固定圖 (`cdn.jsdelivr.net/gh/SumeLabs/clawra@main/assets/clawra.png`)
- Anime 版：本地 asset (`skill/assets/clawra.png`)
- 用法：作為 Edit API 的 `image_url` 輸入
- **一次設定永不變** — 每張照片都從這張圖改造

**Mirror mode prompt**：
```
make a pic of this person, but {context}. the person is taking a mirror selfie
```

**Direct mode prompt**：
```
a close-up selfie taken by herself at {context}, direct eye contact with the camera,
looking straight into the lens, eyes centered and clearly visible, not a mirror selfie,
phone held at arm's length, face fully visible
```

**Anime mode prompt**：
```
anime style, high quality manga illustration, cute anime elf girl, {context},
taking a mirror selfie, detailed anime art, soft lighting, 2D style
```

**Mode keyword → mode 映射** (identical in both versions)：
```
mirror: outfit|wearing|clothes|dress|suit|fashion|full-body|mirror
direct: cafe|restaurant|beach|park|city|close-up|portrait|face|eyes|smile
```

**品質檢查**：無
**失敗重試**：無
**生成後處理**：直接送出

### 5.2 JARVIS 做法

**Reference Image (Anchor)**：
- `CLAWRA_ANCHOR_URL` env var → 傳給 fal.ai `image_url`
- 也可以不設定（純靠 CORE_DNA_PROMPT 描述）
- 來源：`skills/selfie/main.py:77-79`

**CORE_DNA_PROMPT**：
```
A realistic candid photo of a friendly Korean girl, approx 21,
with big bright eyes and prominent aegyo-sal. She has a very warm
and energetic smile. Not over-polished, looks like a real person.
```

**Prompt 結構**：
```
{CORE_DNA_PROMPT} {appearance}. {scene}
```

Where `appearance` = hairstyle + seasonal outfit + (optional scene from AppearanceBuilder)

**Mirror mode prompt** (via `build_prompt()`)：
```
make a pic of this person, but {context}. the person is taking a mirror selfie
```

**Direct mode prompt**：
```
a close-up selfie taken by herself at {context}, direct eye contact with the camera,
looking straight into the lens, eyes centered and clearly visible, not a mirror selfie,
phone held at arm's length, face fully visible
```

**Mode keyword 映射**（比 Clawra 多中文）：
```
mirror: outfit|wearing|clothes|dress|穿|穿搭|全身|鏡子|洋裝|裙|外套|大衣|衣服
direct: cafe|coffee|beach|smile|近照|自拍|臉|咖啡|餐廳|公園|早安|晚安|街|日落|sunset
```

**品質檢查**：有（vision model 比對，但不拒絕）
**失敗重試**：Patch M 改為單次（queue → delayed check）
**生成後處理**：consistency score 記錄、queue pending delivery

### 5.3 差距

| 項目 | Clawra | JARVIS | 差距等級 |
|------|--------|--------|---------|
| Reference image → 一致性 | Edit API（天然一致） | Anchor URL + CORE_DNA_PROMPT | 🟡中等 |
| 外貌變化豐富度 | 無（prompt 固定） | 8×16×8 組合 + 偏好加權 | 🟢OK — JARVIS 大幅領先 |
| 中文 keyword 支援 | 無 | 有（穿搭/全身/鏡子/咖啡/餐廳...） | 🟢OK |
| Prompt 模板 | 與 JARVIS 相同 | 相同 + appearance injection | 🟢OK |
| 品質檢查 | 無 | 有（vision model） | 🟢OK |
| Anime style 支援 | 有（clawra-anime） | 無 | 🟡中等 — 但不需要 |

### 5.4 建議改進

1. **確保 CLAWRA_ANCHOR_URL 設定正確**（高優先）
   - 如果 env var 沒設，FLUX Kontext 就只靠 prompt 描述，一致性較差
   - 確認 `.env` 有設定好的 anchor image
   - 改動量：無（config check） | 預期效果：確保一致性

---

## 6. 優先修復清單

按影響程度排序：

| # | 項目 | 改動量 | 預期效果 | 建議 Patch |
|---|------|--------|---------|-----------|
| 1 | **Temporal Decay（時間衰減）** | 小 | 新記憶優先、舊記憶自然淡化 | Patch T |
| 2 | **MMR re-ranking（去重複）** | 中 | 搜尋結果更有資訊量，減少冗餘 | Patch T |
| 3 | **確認 CLAWRA_ANCHOR_URL** | 無 | 自拍臉部一致性保證 | 即時 |
| 4 | **Pre-compaction memory flush** | 小 | 壓縮時不丟重要資訊 | Patch T |
| 5 | **Consistency check 加入拒絕邏輯** | 小 | 低品質照片不送出 | 下一個 selfie patch |
| 6 | **天氣→主動 selfie** | 小 | Heartbeat 接天氣 API → Clawra 主動傳雪景/雨天照 | Patch U |
| 7 | **情緒→selfie 聯動** | 中 | Ted 開心→Clawra 主動拍慶祝照 | Patch U |
| 8 | **Embedding cache 改 SQLite** | 中 | 大量記憶時效能更好 | 低優先 |
| 9 | **File watcher (memory reindex)** | 中 | 新記憶即時可搜 | 低優先 |
| 10 | **Agent SDK 整合（Phase 2）** | 大 | 更穩定的工具呼叫 + 新能力 | 獨立大 Patch |

---

## 7. 總結

### JARVIS 做得比 Clawra 好的地方
- 人格深度（262 行 vs 45 行）+ 嚴格語氣控制
- 外貌變化（Patch Q: 8×16×8 + 偏好學習）
- Heartbeat 主動關心（7+ cron jobs）
- ReactExecutor 多層 fallback + fuse
- 記憶系統雙軌（SQLite + Markdown + SharedMemory + SoulGrowth）
- 備援方案齊全（FLUX → Gemini, CEO chain 3 providers）
- Message batching + typing simulation（更像真人）

### OpenClaw/Clawra 做得比 JARVIS 好的地方
- **記憶搜尋進階功能**：MMR re-ranking + Temporal Decay（JARVIS 沒有）
- **Edit API 一致性**：改圖天然一致（JARVIS 靠 prompt + anchor）
- **多平台分發**：6 平台 vs 僅 TG（但 JARVIS 目前只需 TG）
- **Plugin/Skill 生態**：80+ skills + npm distribution（但 JARVIS 是垂直場景，不需要）
- **LLM 原生 tool_use**：比 tag parser 更穩定（Agent SDK 已驗證可行）

### 結論

JARVIS 在**自拍品質、互動自然度、主動關心**三個核心維度都已超過 Clawra 開源版。
差距主要在**記憶搜尋的進階後處理**（Temporal Decay + MMR），這是最值得補的兩個功能。
自拍一致性方面，只要確保 `CLAWRA_ANCHOR_URL` 有設定，FLUX Kontext 的品質不輸 Grok Imagine Edit。
