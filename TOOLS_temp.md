# 可用工具

## 網頁搜尋

⚠️ **內建 web_search 需要 Brave API（付費），不能用。**

所有搜索都要調度 OpenCode 用 MCP：
```
exec: {"command": "opencode run \"用 web-search-prime MCP 搜索 [主題]\"", "workdir": "..."}
```

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
