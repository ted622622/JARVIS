---
name: article-organizer
description: 整理文章到 Obsidian vault。当 Ted 丢文章链接时，抓取内容、摘要、整理成 Markdown 笔记。
---

# Article Organizer — 文章整理 Skill

将 Ted 传来的文章链接整理成 Obsidian 笔记。

## Obsidian Vault 路径

```
C:\Users\ted62\Obsidian\Articles\
```

## 文章处理流程

### Step 1: 接收链接

Ted 传来文章链接（YouTube、博客、新闻等）

### Step 2: 抓取内容

根据链接类型选择工具：

| 来源 | 工具 |
|------|------|
| 一般网页 | web-reader MCP |
| YouTube | yt-dlp 抓字幕 |
| GitHub | zread MCP |

### Step 3: 分析并摘要

使用 AI 分析内容，提取：
- 核心观点
- 关键数据
- 行动建议（如有）

### Step 4: 生成 Markdown

使用以下模板：

```markdown
---
title: {文章标题}
source: {原文链接}
date: {今天日期 YYYY-MM-DD}
tags: {自动判断，2-5个}
status: unread
---

## 一句话摘要

{用一句话说明这篇文章在讲什么}

---

## 核心内容

{3-5 个要点，用 bullet points}

## 重要引用

{如果有值得记录的原文引用}

## 我的行动项

{如果有可以采取的行动，列出；没有则写「无」}

## 原始内容

{完整内容或摘要，依长度决定}
```

### Step 5: 存档

1. 生成文件名：`{YYYY-MM-DD}-{简短标题}.md`
2. 存到 `C:\Users\ted62\Obsidian\Articles\`
3. 回报 Ted：「已整理到 Obsidian：{标题}」

## 文件名规则

- 使用日期前缀：`2026-02-21-`
- 标题转成英文或拼音（避免中文文件名问题）
- 限制长度：30 字符以内

范例：
```
2026-02-21-rsi-macd-strategy.md
2026-02-21-youtube-trading-tips.md
2026-02-21-ai-agent-trends.md
```

## 标签建议

自动判断文章类型并添加标签：

| 类型 | 标签 |
|------|------|
| 交易/投资 | `#交易` `#投资` `#策略` |
| 程式/技术 | `#技术` `#程式` `#AI` |
| 生活/健康 | `#生活` `#健康` |
| 商业/财经 | `#商业` `#财经` |
| 学习/教育 | `#学习` |

## 回覆 Ted

整理完成后，简短回报：

```
已整理到 Obsidian ✅
标题：{标题}
标签：{标签}
档案：{文件名}
```

## 特殊处理

### YouTube 影片

如果 Ted 传 YouTube 链接：
1. 用 yt-dlp 抓字幕
2. 如果没有字幕，用 zai-mcp-server 分析截图
3. 额外记录：频道名称、影片时长

### 长文章

如果文章超过 3000 字：
- 摘要为主，不存完整内容
- 在「原始内容」注明：「内容过长，仅存摘要」

### 付费墙/无法抓取

如果无法抓取内容：
1. 告知 Ted：「无法抓取此链接内容」
2. 请 Ted 贴上原文或截图
