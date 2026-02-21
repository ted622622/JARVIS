---
name: coding-agent
description: 調度 OpenCode 執行複雜任務。當需要視覺理解、深度搜索、GitHub 分析、程式碼開發時，讓 OpenCode 處理。
---

# Coding Agent — OpenCode 調度 Skill

讓 JARVIS 能夠調度 OpenCode 執行需要 MCP 或複雜處理的任務。

## 何時使用

| 情況 | 調度 OpenCode |
|------|--------------|
| Ted 傳圖片/影片需要分析 | ✅ 視覺理解 |
| Ted 需要深度網絡搜索 | ✅ 網絡搜索 MCP |
| Ted 要分析 GitHub 倉庫 | ✅ zread MCP |
| Ted 要驗證交易策略 | ✅ 回測 + MT5 |
| 需要寫/改程式碼 | ✅ |
| 需要執行多步驟任務 | ✅ |

## 調度方式

使用 `exec` 啟動 OpenCode：

```bash
exec: {
  "command": "opencode run \"任務描述\"",
  "workdir": "C:\\Users\\ted62\\.openclaw"
}
```

## 範例

### 網絡搜索（重要）
```
Ted：「幫我搜尋 [某主題]」

JARVIS 調度：
exec: {
  "command": "opencode run \"用 web-search-prime MCP 搜索 [主題]\"",
  "workdir": "C:\\Users\\ted62\\.openclaw"
}
```

**注意**：OpenClaw 內建 `web_search` 需要 Brave API（付費），不能用。
所有搜索都要調度 OpenCode 用 MCP。

### 視覺理解
```
Ted 傳圖片：「幫我看這張圖表的交易訊號」

JARVIS 調度：
exec: {
  "command": "opencode run \"分析圖片 /path/to/image.png，找出交易訊號\"",
  "workdir": "C:\\Users\\ted62\\.openclaw"
}
```

### GitHub 倉庫分析
```
Ted：「幫我看看這個 repo 的架構」

JARVIS 調度：
exec: {
  "command": "opencode run \"用 zread MCP 分析 https://github.com/user/repo 的架構\"",
  "workdir": "C:\\Users\\ted62\\.openclaw"
}
```

### YouTube 策略驗證
```
Ted 傳 YouTube 連結

JARVIS 調度：
exec: {
  "command": "opencode run \"驗證 YouTube 策略：https://youtube.com/xxx\"",
  "workdir": "C:\\Users\\ted62\\.openclaw"
}
```

## OpenCode 可用的 MCP 工具

| MCP | 功能 |
|-----|------|
| **zai-mcp-server** | 視覺理解（圖片/影片分析）|
| **web-search-prime** | 網絡搜索 |
| **web-reader** | 網頁內容抓取 |
| **zread** | GitHub 倉庫分析 |

## 回報機制

OpenCode 執行完成後，會返回結果。JARVIS 需要：

1. 等待 OpenCode 完成
2. 獲取結果
3. 整理後回報 Ted

## 注意事項

- OpenCode 執行時間可能較長（30秒 ~ 數分鐘）
- 先告知 Ted 正在處理
- 如果超時，告知 Ted 並提供進度
