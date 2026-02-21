# 長期記憶

> 此處記錄用戶偏好、重要決策、常用設定等長期有用的資訊。

## 系統狀態

- OpenClaw 版本: 2026.2.19-2
- 上線日期: 2026-02-20
- 架構: 4-Agent（JARVIS + Clawra + Marcus + Quant）

## Agent 狀態

| Agent | 模型 | TG Bot | 狀態 |
|-------|------|--------|------|
| JARVIS (main) | GLM-4.7 (primary) | JARVIS bot | ✅ 運行中 |
| Clawra | GLM-4.7 | Clawra bot | ✅ 運行中 |
| Marcus | GLM-4.7 | 無 | ✅ 運行中 |
| Quant | GLM-4.7 (primary) | 無 | ✅ 運行中 |

**升級規則**：JARVIS / Quant 遇複雜任務時自動升級 GLM-5

## Sub-Agent 派工解法（2026-02-20）

### 錯誤
```
gateway closed (1008): pairing required
```

### 解法
Sub-agent 首次使用需要 approve pairing request：

1. 檢查 pending requests：`openclaw pairing list`
2. 批准 request：`openclaw pairing approve <REQUEST_ID>`
3. 重試派工

### 例子
```
openclaw pairing list
# 看到 pending request: 09e03276-a5f2-42e2-8a3e-33fc4617aa3d

openclaw pairing approve 09e03276-a5f2-42e2-8a3e-33fc4617aa3d
# 批准後即可正常派工
```

## 用戶偏好

- 不吃香菜
- 不用語音回覆，要用打字的
- 回覆時要用 Typing 的方式
- 結論先行，不廢話

## 工作時間限制

- **下午 14:00-18:00（台北時間）**：高峰時段，GLM-5 token 1.5 倍
- 此時段不工作，包括所有 Agent
- 可工作時間：18:00 後

## 今日已完成（2026-02-21）

1. ✅ Git 配置 + 清除 secrets
2. ✅ memoryFlush + local embedding
3. ✅ GLM-TTS 腳本（xiaochen + tongtong）
4. ✅ 自拍 SKILL.md 重寫
5. ✅ Clawra 語音測試成功
6. ✅ 自拍測試成功
7. ✅ MCP 工具安裝（4 個）
8. ✅ coding-agent skill（JARVIS → OpenCode）
9. ✅ YouTube 策略驗證依賴安裝
10. ✅ Obsidian 文章整理設定

### 高優先級

1. **OpenClaw 維運排程實作**（已完成）
   - ✅ 每日備份（Marcus）15:00
   - ✅ 版本更新前備份
   - ✅ 記憶維護（每兩週，先備份再清理）
   - ✅ Gateway 重啟（>500MB 觸發）
   - ✅ Heartbeat 配置（Clawra）

### 已知問題

#### 2026-02-21 Clawra 問題

1. **20:03 消息** - sendChatAction 网络抖动，但消息有发送
2. **20:04 语音没回** - `MediaFetchError` Telegram 语音下载失败（网络问题）
3. **整天没主动找** - HEARTBEAT.md 是空的 → 已修复

---

## ⚠️ 重要教訓：Git 絕對不能泄露 API Key

**已發生 2 次泄露**：
1. Telegram Bot Token
2. Gemini API Key

**預防措施**：
- 所有 `.env` 檔案在 `.gitignore`
- `openclaw.json` 在 `.gitignore`
- Commit 前檢查：`git diff --staged`
- 若已泄露：立即重新產生 key + git filter-branch 或 reset --hard

## 已知問題

### Web Search

- **Brave API 需付費**，OpenClaw 內建 `web_search` 不能用
- **解決方案**：調度 OpenCode 用 `web-search-prime` MCP
- JARVIS/Marcus/Quant 需要搜索時 → 調度 OpenCode

## 待優化：Agent 靈魂設計參考（來源：GLM 官方文檔）

### 專業化智能體設計範例

**前端專家智能體**：
```
你是一個專注於 React/TypeScript 開發的前端專家智能體。

你的專業領域：
- React 組件設計和優化
- TypeScript 類型系統
- CSS-in-JS 和 Tailwind CSS
- 前端性能優化
- 用戶體驗設計

工作原則：
1. 始終考慮組件的可復用性
2. 遵循 React 最佳實踐
3. 確保類型安全
4. 優先考慮用戶體驗
5. 代碼要易於測試
```

**後端專家智能體**：
```
你是一個專注於 Node.js/Python 後端開發的專家智能體。

你的職責範圍：
- RESTful API 設計
- 數據庫設計和優化
- 安全性和認證
- 性能監控和優化
- 微服務架構

核心原則：
1. API 設計要符合 RESTful 規範
2. 數據安全是第一優先級
3. 代碼要有充分的錯誤處理
4. 性能優化從設計開始
5. 可擴展性要考慮在內
```

### 層次化架構

```
         ┌──────────┐
         │   Ted    │
         └────┬─────┘
              │
   ┌──────────▼──────────┐
   │      JARVIS         │
   │    CEO Agent        │
   └──┬──────┬───────┬───┘
      │      │       │
   ┌──▼──┐ ┌─▼──┐ ┌──▼──┐
   │ C   │ │ M  │ │  Q  │
   └─────┘ └────┘ └─────┘
```

### OpenClaw + OpenCode 協同

JARVIS 可以調度 OpenCode 執行需要 MCP 的任務：
- 視覺理解（GLM-4.6V）
- 網絡搜索
- 網頁讀取
- GitHub 倉庫分析

## Clawra 自拍規則（重要）

| 規則 | 說明 |
|------|------|
| 每日上限 | 3 張，**只扣成功的** |
| fal timeout | 不 retry，回「不方便拍」或「網路卡」|
| timeout 計數 | **不扣上限** |
| 補發 | Heartbeat 檢查 pending，完成則補發 + 女朋友的話 |
| 失敗處理 | 放棄，不 retry，**不扣上限** |

## TTS 規則（重要）

| Agent | 聲音 | 失敗處理 |
|-------|------|---------|
| JARVIS | 雲哲（唯一）| 打字回覆 |
| Clawra | tongtong（唯一）| 「喉嚨不舒服」+ 打字回覆 |

**回覆規則**：文字→文字，語音→語音，除非 Ted 指定

## 重要決策

### 2026-02-20 系統上線
- 完成 4-agent 架構部署
- TG 雙 bot 整合完成
- Google 整合待設定（需要 OAuth）

### 2026-02-20 Sub-Agent 問題調試
- 已完成配置和檢查
- 已讀取所有相關文檔
- 已生成詳細調試報告
- 問題尚未解決，需進一步調查
- **詳情**：見 `memory/subagent-issue-complete.md`

## 常用設定

- （待記錄）

## OpenClaw 記憶系統配置（2026-02-21）

### memoryFlush（pre-compaction 自動存記憶）

```json
"compaction": {
  "mode": "safeguard",
  "memoryFlush": {
    "enabled": true,
    "softThresholdTokens": 6000
  }
}
```

- 快壓縮前自動把重要記憶寫入 `memory/YYYY-MM-DD.md`
- 解決之前 JARVIS SOUL_GROWTH 一直是空的問題

### Local Embedding（不花 token）

```json
"memorySearch": {
  "provider": "local",
  "fallback": "none",
  "query": {
    "hybrid": {
      "enabled": true,
      "mmr": { "enabled": true },
      "temporalDecay": { "enabled": true, "halfLifeDays": 30 }
    }
  }
}
```

- 使用本地 GGUF 模型（auto-download）
- BM25 + 向量混合搜尋
- 30 天時間衰減
- MMR 去重

### memory-growth skill

- 位置：`~/.openclaw/skills/memory-growth/`（所有 agent 共享）
- 所有 agent（JARVIS, Clawra, Marcus, Quant）都會自動載入

## JARVIS 成長記錄

- 以後推薦餐廳不要有香菜的（2026-02-19）

## 共識會議記錄

見 consensus_log.md
