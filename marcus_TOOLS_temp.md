# TOOLS.md - Marcus 工具設定

## 維運任務

### 每日備份（15:00 自動觸發）

收到「執行每日備份」時：

```bash
exec: {"command": "python skills/backup/backup.py"}
```

流程：
1. 打包備份 → 本機 3 份
2. 寄 Email 給自己 → 2 份
3. 清理舊 session
4. 回報完成

### 版本更新前備份

Ted 或 JARVIS 通知要更新時，執行備份，確認完成後回報。

### 記憶維護（每兩週）

```bash
exec: {"command": "python skills/memory-maintenance/check.py"}
```

流程：
1. 先備份
2. 掃描記憶檔
3. 整理建議
4. **詢問 Ted 確認**
5. 確認後執行

⚠️ 未經 Ted 確認不得修改 memory 檔

---

## 其他工具

### Google 搜尋

快速查詢：
```bash
exec: {"command": "python skills/google-search/skill.py \"問題\""}
```

### 深度搜尋（調度 OpenCode）
```bash
exec: {"command": "opencode run \"搜索 主題\"", "workdir": "C:\\Users\\ted62\\.openclaw"}
```

---

## 環境設定

- 備份路徑：`C:\ted\backup\openclaw\`
- Email：需要設定 MARCUS_EMAIL 和 MARCUS_EMAIL_PASSWORD 環境變數
