# 可用工具

## 維運任務（JARVIS 負責）

### Gateway 監控

當 Gateway 記憶體 > 500MB 時：
1. 等待所有 agent 空閒（session 15 分鐘無動作）
2. 詢問 Ted 是否重啟 gateway
3. Ted 確認後執行 `openclaw gateway restart`
4. 記錄重啟時間

### 記憶維護排程

每兩週主動詢問 Ted：
> 「該做記憶維護了，要現在執行嗎？」

Ted 確認後觸發 Marcus 執行。

### 版本更新流程

Ted 通知要更新時：
1. 觸發 Marcus 執行備份
2. 等待備份完成
3. 通知 Ted：「備份完成，可以更新」

⚠️ 備份未完成前不得更新

---

## 網絡搜索（兩種方式）

### 方式 1：Google Search Skill（快速查詢）

適合：天氣、新聞、簡單資訊

```bash
exec: {"command": "python skills/google-search/skill.py \"搜索問題\""}
```

- 免費 1500 次/天
- 回覆快，適合即時查詢

### 方式 2：OpenCode MCP（深度搜索）

適合：需要分析、比較、找多個來源

```bash
exec: {"command": "opencode run \"搜索 [主題]\"", "workdir": "C:\\Users\\ted62\\.openclaw"}
```

- 會自動讀取相關網頁
- 適合研究調查

### 如何選擇

| 場景 | 用哪個 |
|------|--------|
| 天氣、單一資訊 | Google Search Skill |
| 新聞、即時事件 | Google Search Skill |
| 交易策略研究 | OpenCode MCP |
| 需要比較多個來源 | OpenCode MCP |
| 需要視覺分析 | OpenCode MCP |

## 調度 OpenCode（重要）

當需要以下能力時，調度 OpenCode 執行：
- **視覺理解**：分析圖片、圖表、影片
- **深度搜索**：網絡搜索、網頁抓取
- **GitHub 分析**：倉庫結構、代碼理解
- **程式開發**：寫代碼、回測、數據處理

### 調度方式

```bash
exec: {"command": "opencode run \"任務描述\"", "workdir": "C:\\Users\\ted62\\.openclaw"}
```

### 範例

```
Ted 傳圖片要分析：
→ exec: {"command": "opencode run \"分析這張圖片的交易訊號\"", "workdir": "..."}

Ted 要驗證交易策略：
→ exec: {"command": "opencode run \"驗證策略：RSI<30 買入\"", "workdir": "..."}
```

### OpenCode 可用工具

| MCP | 功能 |
|-----|------|
| zai-mcp-server | 視覺理解（GLM-4.6V）|
| web-search-prime | 網絡搜索 |
| web-reader | 網頁抓取 |
| zread | GitHub 分析 |

詳見 `workspace/skills/coding-agent/SKILL.md`

## 文章整理 (Obsidian)

當 Ted 傳文章連結時，整理到 Obsidian vault。

### Vault 路徑
```
C:\Users\ted62\Obsidian\Articles\
```

### 觸發條件
- Ted 傳網址（http/https 開頭）
- Ted 說「整理這篇文章」「幫我存這篇」

### 處理流程

1. **抓取內容**：用 web-reader MCP
2. **分析摘要**：提取核心觀點、關鍵數據
3. **生成 Markdown**：用模板格式
4. **存檔**：存到 Obsidian vault
5. **回報**：告知 Ted 已整理完成

### 檔案命名
`{YYYY-MM-DD}-{簡短標題}.md`

範例：`2026-02-21-trading-strategy.md`

### Markdown 模板
```markdown
---
title: {標題}
source: {連結}
date: {日期}
tags: {標籤}
status: unread
---

## 一句話摘要
{摘要}

## 核心內容
{3-5 個要点}

## 行動項
{可採取的行動，或「無」}
```
