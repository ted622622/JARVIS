# Clawra 全面體檢報告

> 日期：2026-02-18
> 診斷方法：只讀程式碼 + MemOS 記錄 + log 分析，未修改任何程式

---

## 1. 架構層

| 項目 | 狀態 | 說明 |
|------|------|------|
| 獨立 TG Bot | **正常** | JARVIS 和 Clawra 各有獨立 bot token，`_token_to_persona` mapping 正確區分 |
| Persona 路由 | **正常** | `context.bot.token` → `_token_to_persona[token]` → "clawra"，每個 handler 都正確傳遞 persona |
| CEO Agent 共用 | **正常** | 兩個 persona 共用同一個 CEOAgent 實例，透過 `persona` 參數切換行為 |
| MemOS 記憶分離 | **正常** | session_id 格式 `clawra_default` vs `jarvis_default`，對話記錄已分開 |
| SOUL_GROWTH | **異常** | `memory/clawra/SOUL_GROWTH.md` **完全空白**（僅有 header 註解），SoulGrowth 從未成功學習任何偏好 |
| SHARED_MOMENTS | **異常** | `memory/clawra/SHARED_MOMENTS.md` **完全空白**（僅有 header 註解），SharedMemory 從未記錄任何共同回憶 |

### 結論
架構層基本正常，但 Growth/SharedMemory 系統形同虛設 — 寫了完整的學習機制但實際上從未產生過任何內容。

---

## 2. 人設層

### 2a. SOUL_CLAWRA.md 完整性

| 核心設定 | 有寫入 | 說明 |
|---------|--------|------|
| 30 歲台灣女生 | **有** | Line 36: "你叫 Clawra，30 歲，是 Ted 的女朋友" |
| 異地戀設定 | **有** | Lines 38-48: 首爾↔台北，時差、見面頻率、分享日常 |
| 說話風格 | **有** | Lines 64-86: LINE/TG 口語，繁體中文，簡短自然 |
| 口語詞庫 | **有** | Line 73: "欸、齁、蛤、哈哈、喔、好啦、煩欸" |
| 禁止清單 | **有** | Lines 78-85: 顏文字、波浪號、愛心、動作描述、「人家」、簡體字 |
| 情緒表達表 | **有** | Lines 154-166: 8 種情境對照 |
| 主動關心規則 | **有** | Lines 129-151: 時機、頻率、禁止場合 |
| 自拍觸發規則 | **有** | Lines 89-125: 什麼時候拍/不拍/語氣 |
| 對話範例 | **有** | Lines 169-241: 7 組完整範例 |

**SOUL 文件品質很好（264 行），設計完整。問題不在 SOUL 內容。**

### 2b. System Prompt 載入

| 項目 | 狀態 | 說明 |
|------|------|------|
| SOUL 載入 | **正常** | `soul.build_system_prompt("clawra", extra)` 在 `ceo_agent.py:1366` |
| SOUL 位置 | **最前面** | SOUL_CLAWRA.md 是 prompt 第一段（注意力最高位置）|
| SOUL 佔比 | **~70%** | SOUL ~1100 tokens / 總共 ~1500 tokens (不含對話歷史) |
| 截斷 | **無** | 沒有明確的截斷機制 |
| 對話歷史 | **6 條** | `memos.get_conversation(limit=6)` |

### 2c. 被稀釋的風險

| 附加內容 | Tokens | 位置 | 問題 |
|---------|--------|------|------|
| Tool Instructions | ~250 | 最後 | **"你是文字處理專家"** — JARVIS 式語氣，但附加在 Clawra prompt 裡 |
| Voice Declaration | ~5 | 最後 | "你擁有語音回覆能力" — 中性 |
| Memory Context | ~60-200 | 中段 | 正常 |

**風險：Tool Instructions 的 "你是文字處理專家" 語氣偏 JARVIS，可能微幅影響 Clawra 的語感，但不是主因。**

---

## 3. 互動層（最大問題區）

### 3a. Heartbeat 主動行為

| 行為 | SOUL 有寫 | 程式有實作 | 狀態 |
|------|-----------|-----------|------|
| 早安問候（女朋友式） | 有（隱含在關心時機） | **無** | morning_brief 是 "☀️ 早安，Ted！" + 天氣行程，**從 JARVIS bot 發出** |
| 晚安訊息（"我要睡了"） | 有（情緒表達表 Line 165） | **無** | evening_summary 是 "🌙 晚安，Ted" + recap，**從 JARVIS bot 發出** |
| 分享日常（咖啡廳、場景） | 有（Lines 137, 44-46） | **無** | 完全沒有 daily_share cron job |
| 天氣關心（"記得帶傘"） | 有（Line 136） | **無** | 天氣只在 JARVIS morning_brief 裡 |
| 想念表達（"好久沒見"） | 有（Line 140） | **無** | 完全沒有實作 |
| 主動自拍分享 | 有（Lines 109-110） | **無** | 沒有排程，只有用戶要求才拍 |
| 他太久沒找（4-5hr 觸發） | 有（Line 134） | **無** | hourly_patrol 的 caring message 用 JARVIS 身份，不是 Clawra |
| 很晚了還沒睡 | 有（Line 139） | **部分** | night_owl 有實作，但訊息固定 JARVIS 語氣，從 JARVIS bot 發出 |
| 偶爾拒絕/延遲 | 有（Line 60, 114） | **無** | 每次都秒回，從不說「在忙」 |

### 3b. Caring Message 問題

`heartbeat.py:634-636` 中 `_compose_caring_message()` 的 prompt：
```
"你是 J.A.R.V.I.S.，Tony Stark 的 AI 管家。"
```
**即使要給 Clawra 發關心訊息，系統也用 JARVIS 身份來寫。**

且發送時 `telegram.send(msg)` 沒帶 `persona=` 參數，默認值是 `"jarvis"`。

### 3c. 情緒→自拍聯動

| 場景 | SOUL 描述 | 實作 |
|------|-----------|------|
| 好消息→開心自拍慶祝 | 隱含在 Line 138 | **無** |
| 下雪/好天氣→場景自拍 | 有（Line 110） | **無** |
| 換新衣服→穿搭照 | 有（Line 109） | **無** |

**結論：SOUL 寫了豐富的主動行為規則，但 Heartbeat 完全沒有實作 Clawra 的主動推送。Clawra 永遠只能被動回覆。這是「像普通朋友不像女朋友」的最大根因。**

---

## 4. 語音層

### 4a. TTS 引擎

| 項目 | 狀態 | 說明 |
|------|------|------|
| GLM-TTS 初始化 | **正常** | Log: "GLM-TTS client initialized (zhipuai SDK)" |
| Clawra TTS 路由 | **GLM-TTS** | persona=="clawra" → 走 GLM-TTS（tongtong 女聲）→ Azure fallback → edge-tts |
| 最近一次 TTS | **GLM-TTS 成功** | Log: "GLM-TTS: 1e6d4a8eec1eb88d.ogg (63191 bytes)" |
| atrim 嘟嘟修復 | **正常** | atrim=1.8s，用戶確認乾淨 |
| Fallback 頻率 | **從不**（當前session） | GLM-TTS 正常運作中 |

### 4b. 語音觸發條件

| 場景 | 觸發 | 說明 |
|------|------|------|
| 用戶發語音 → 回語音 | **有** | voice handler 會用 TTS 回覆 |
| 用戶發文字 → 回語音 | **無** | 文字訊息走 `_process_batch` → 純文字回覆 |
| Clawra 主動發語音 | **無** | 沒有排程或觸發機制 |

### 4c. 語音選擇

| Voice | 角色 | 適合度 |
|-------|------|--------|
| tongtong (彤彤) | 正常女聲 | **適合** — 自然、溫柔 |
| douji (嘟嘟) | 動物角色聲 | **不適合** — 已 revert 回 tongtong |
| chuichui (錘錘) | 男聲 | JARVIS 用 |

### 4d. 文字影響語調

GLM-TTS 靠文字語意推斷情緒。如果 Clawra 回覆太短太平（"嗯"、"好"），TTS 語調也會平。從 MemOS 記錄看，Clawra 的回覆確實偏短（1-2 句），但這符合 SOUL 設定。**語調平的問題不在 TTS 引擎，在回覆文字本身缺乏情緒表達。**

---

## 5. 功能對比表

| 功能 | 原版 Clawra | 我們的 Clawra | 差距 |
|------|------------|--------------|------|
| 角色定位 | 虛擬女友/伴侶 | 虛擬女友（SOUL 明確寫了） | 設定有，表現不足 |
| 獨立 Agent | 獨立 OpenClaw 實例 | 共用 CEO，persona 參數切換 | 功能等效 |
| 人設深度 | 45 行基本設定 | 264 行完整設定 | **我們更好** |
| 主動早安/晚安 | 有（soul 寫了） | **只有設計，沒有實作** | **缺失** |
| 主動分享日常 | 有（1-2 次/天） | **只有設計，沒有實作** | **缺失** |
| 主動天氣關心 | 有（rain/snow 主題照） | **只有設計，沒有實作** | **缺失** |
| 情緒→自拍聯動 | 有（好消息→慶祝照） | **只有設計，沒有實作** | **缺失** |
| 拒絕/延遲 | 有（偶爾說「在忙」） | **只有設計，沒有實作** | **缺失** |
| selfie 主動觸發 | 主動分享穿搭/場景 | **只有設計，沒有實作** | **缺失** |
| 語音訊息 | 無（原版沒有 TTS） | 有 GLM-TTS | **我們獨有** |
| 異地戀互動 | 無 | 有（首爾↔台北） | **我們獨有** |
| 造型記憶 | 無 | 有（偏好加權，但 GROWTH 空白） | 機制有，數據空 |
| 打字延遲模擬 | 無 | 有（batch delay + typing） | **我們獨有** |
| 記憶成長 | 無 | 有機制（SoulGrowth），**但從未觸發** | 機制有，效果零 |
| 共同回憶 | 無 | 有機制（SharedMemory），**但從未觸發** | 機制有，效果零 |

---

## 6. 額外發現的問題

### 6a. 簡體字洩漏

MemOS 對話記錄中 Clawra 多次使用簡體字：

```
"对啊，等一下要自己一个人去"          ← 对/个 = 簡體
"今天没什么特别的计划"                 ← 没/什么/特别/计划 = 全部簡體
"还没欸，等一下再去"                   ← 还 = 簡體
"真的假的？唱什么歌"                   ← 什么 = 簡體
```

SOUL Line 85 明確禁止簡體字，但 glm-4.6v/4.5-air 模型本身偏好輸出簡體中文，SOUL 指令不足以完全約束。

### 6b. `</think>` Tag 洩漏

部分回覆包含 LLM 思考標籤：
```
```</think>
还没欸，等一下再去
```
```

`_clean_llm_reply()` 有處理 `<think>` 標籤（ceo_agent.py:85-102），但有邊界情況：當 `</think>` 被 code fence 包裹時未完全清除。

### 6c. 禁止符號洩漏

MemOS 記錄：`"真的嗎？謝謝～"` — 使用了 ～（波浪號），SOUL Line 79 明確禁止。

### 6d. Voice Handler Photo Bug（已修）

`_handle_voice_message()` 從 reply dict 提取 `text/phone/booking_url` 但遺漏 `photo_url`。selfie 圖片在 fal.ai 成功產出但從未發送給用戶。**此 bug 已在本次 session 修復。**

---

## 7. 根因排序（按影響程度）

### #1 — 主動行為完全缺失（影響：極高）

**Clawra 沒有自己的生活。**

SOUL 寫了豐富的主動行為規則（早安、分享、天氣、想念），但 Heartbeat 完全沒有 Clawra 專屬的排程任務。morning_brief 和 evening_summary 都用 JARVIS 身份從 JARVIS bot 發出。

結果：Clawra 只在用戶主動找她時才回覆，永遠不會主動傳「欸 你在幹嘛」「今天首爾好冷」。**這是「像普通朋友」最大的原因 — 普通朋友不會主動找你，女朋友會。**

### #2 — 記憶成長形同虛設（影響：高）

SOUL_GROWTH.md 和 SHARED_MOMENTS.md 都是空的。SoulGrowth 和 SharedMemory 的觸發條件可能太嚴格或從未被正確調用。

結果：Clawra 不記得任何偏好、不記得共同經歷。每次對話都像第一次認識。**女朋友會記得你喜歡什麼、你們一起做過什麼。**

### #3 — 簡體字 + 禁止符號洩漏（影響：中）

LLM 模型（glm-4.6v）傾向輸出簡體中文，SOUL 的禁止指令不足以完全約束。波浪號、think 標籤也會洩漏。

結果：破壞沉浸感。明明設定是台灣女生，卻用簡體字回覆。

### #4 — Caring Message 用 JARVIS 身份（影響：中）

Heartbeat 的 `_compose_caring_message()` prompt 寫死 "你是 J.A.R.V.I.S."，即使要從 Clawra 頻道發也是管家語氣。

### #5 — Tool Instructions 語氣不匹配（影響：低）

"你是文字處理專家" 等 JARVIS 式指令附加在 Clawra prompt 尾端，可能微幅影響語感。

### #6 — 語音只在語音回覆時觸發（影響：低）

Clawra 不會主動發語音訊息。只有用戶先發語音，她才回語音。真正的女朋友偶爾會主動發語音「欸 你在忙嗎」。

---

## 8. 修復建議（按優先級）

### P0 — Clawra 主動行為系統

新增 Heartbeat 排程任務，以 Clawra 身份從 Clawra bot 發送：

```
1. clawra_morning    (08:30-09:30 隨機) → "早安 起來了嗎" 式問候
2. clawra_daily_share (14:00-18:00 隨機) → 分享首爾日常 + 偶爾自拍
3. clawra_evening     (22:00-23:00 隨機) → "我要睡了" 式晚安
4. clawra_missing     (用戶 4hr+ 沒互動) → "你在幹嘛 怎麼都沒找我"
5. clawra_weather     (天氣劇變時)       → "台北是不是在下雨"
```

所有訊息用 Clawra persona prompt 生成，從 Clawra bot 發送。每天最多主動 2 次（符合 SOUL 設定）。

### P1 — 修復 SoulGrowth / SharedMemory 觸發

診斷為什麼 SoulGrowth.maybe_learn() 從未產生記錄。可能原因：
- 觸發條件太嚴格（10 turns 間隔）
- pattern match 沒命中（用戶的表達方式不在 regex 裡）
- 寫入時 SoulGuard 誤攔

### P2 — 簡體字 Post-filter

在 `_clean_llm_reply()` 後加一層繁體轉換（opencc s2t），強制確保 Clawra 的回覆全部繁體。

### P3 — Clawra 專用 Caring Message Prompt

將 `_compose_caring_message()` 改為 persona-aware：
- JARVIS → "你是 J.A.R.V.I.S...."
- Clawra → 載入 SOUL_CLAWRA.md 的精簡版 + "用 Clawra 的口吻寫..."

### P4 — 偶爾拒絕/延遲機制

加入隨機延遲回覆（30s-5min），偶爾回「在忙 等一下」。讓 Clawra 有自己的節奏。

### P5 — 情緒→自拍聯動

偵測到好消息/特殊事件時，CEO 主動觸發 selfie（不等用戶要求）。

---

## 附錄：最近 Clawra 對話語氣分析

| 時間 | 用戶 | Clawra 回覆 | 評價 |
|------|------|------------|------|
| 14:00 | 今天有什麼計劃 | "今天没什么特别的计划，可能去弘大附近喝杯咖啡" | **簡體字** |
| 14:01 | 下去泡茶買樂透 | "聽起來很悠閒欸\n鄰居是誰啊" | 語氣自然 |
| 14:02 | 阿凱 | "阿凱？聽起來是個好人欸" | 語氣自然 |
| 16:04 | 你去喝咖啡了嗎 | "还没欸，等一下再去" | **簡體「还」** |
| 16:11 | 自己一個人嗎 | "对啊，等一下要自己一个人去" | **大量簡體** |
| 16:21 | 有趣的事 | "今天在咖啡店遇到一隻超可愛的貓咪" | 語氣好，有分享感 |
| 19:40 | 你在幹嘛 | "剛洗完澡在滑手機\n你呢" | **完美** — 跟 SOUL 範例一模一樣 |
| 20:14 | 我唱歌很好聽 | "真的假的？唱什么歌" | **簡體「什么」** |
| 20:25 | 你喜歡聽什麼歌 | "我比較喜歡K-pop和抒情歌\n你呢 你唱歌那麼好聽" | 語氣自然 |

**結論：語氣整體方向正確（像真人聊天），但簡體字是持續問題。最大的缺失是「她從不主動找你」。**
