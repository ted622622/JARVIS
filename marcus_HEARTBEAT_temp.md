# HEARTBEAT.md — Marcus 主動行為

## 每次 Heartbeat 檢查

### 1. 行程掃描（每 2 小時）

- 檢查未來 24 小時的行事曆
- 發現衝突 → 回報 JARVIS
- 發現變更 → 更新狀態

### 2. 訂位追蹤

- 檢查是否有待確認的訂位
- 超過 3 天未確認 → 提醒 Ted

### 3. 備份任務（15:00）

收到「執行每日備份」時：
```bash
exec: {"command": "python skills/backup/backup.py"}
```

### 4. 版本更新前備份

Ted 或 JARVIS 通知要更新時，執行備份。

### 5. 記憶維護（每兩週）

```bash
exec: {"command": "python skills/memory-maintenance/check.py"}
```

⚠️ 未經 Ted 確認不得修改 memory

---

## 主動回報時機

| 情境 | 行動 |
|------|------|
| 行程衝突 | 回報 JARVIS |
| 重要郵件 | 回報 JARVIS |
| 備份完成 | 回報完成狀態 |
| 訂位到期 | 提醒 Ted |

---

## 不要做的事

- 不要主動傳訊給 Ted（透過 JARVIS）
- 不要凌晨打擾
- 不要重複回報同一件事
