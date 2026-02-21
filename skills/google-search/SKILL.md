# Google Search Skill

用 Gemini 2.5 Flash + Google Search Grounding 执行搜索。

## 用途

查询即时信息、最新新闻、当前价格、近期事件。
静态知识或历史信息不需要呼叫此 Skill。

## 触发情境

- 「今天」「最新」「现在」「最近」等时间敏感词
- 股价、汇率、天气等即时数据
- 新闻事件查询
- 验证某项信息是否仍然有效

## 呼叫方式

```bash
exec: {"command": "python skills/google-search/skill.py \"搜索问题\""}
```

## 回传

已整理的搜索摘要，可直接引用。

## 注意

- **免费额度**：每天 1500 次，每分钟 15 次
- 需要 GEMINI_API_KEY 环境变量
