---
name: pdm-onboard-scenario
description: Safely preview, validate, and add a new PDM equipment/measurement scenario to configs/model_registry.yaml. Use when a user asks to onboard, register, copy, or add a new device and measurement configuration, validate its effective DB/YAML/defaults resolution, or diagnose why a candidate scenario cannot be added. This MVP only creates new scenarios, uses offline DB/time-series fixtures, rejects duplicates and unsupported model contracts, and never updates existing scenarios or writes databases.
---

# PDM 场景接入

安全新增一个设备/测点场景。默认只预览候选配置和 diff；确认后才原子写入 YAML。

## 执行流程

1. 读取 [config-contract.md](references/config-contract.md)，按其中 schema 收集候选字段。
2. 确认任务是“新增”。发现 YAML 或 DB fixture 已有同名设备+测点时立即停止；不要转为更新。
3. 将候选写入临时 YAML/JSON 文件，不要把真实连接配置、凭证或原始生产数据写入文件或回复。
4. 使用项目 Python 运行预览：

   ```powershell
   uv run python .agents/skills/pdm-onboard-scenario/scripts/onboard_scenario.py `
     --repo-root . `
     --candidate <candidate.yaml> `
     --db-models <db_models.json> `
     --timeseries-fixture <timeseries.csv> `
     --format json
   ```

5. 对 CSV 场景省略 `--timeseries-fixture`；脚本从候选的 `data_path` 读取数据。相对路径以仓库根目录解析。
6. 向用户展示候选 diff、最终生效配置、模型支持交集、清洗统计和窗口切分。不要隐藏警告。
7. 记录预览返回的 `diff_hash`。仅在 `can_apply=true`、所有静态/数据验证通过且用户明确确认该 diff 后执行：

   ```powershell
   uv run python .agents/skills/pdm-onboard-scenario/scripts/onboard_scenario.py `
     --repo-root . `
     --candidate <candidate.yaml> `
     --db-models <db_models.json> `
     --timeseries-fixture <timeseries.csv> `
     --apply --confirmed --expected-diff-hash <confirmed-diff-hash> --format json
   ```

8. 写入后复核结果。脚本使用同目录临时副本、原子替换和失败回滚；不要绕过该写入路径手工拼接 YAML。

## 安全边界

- 默认完全断开真实数据库和网络。`db_models.json` 与本地时序 fixture 是唯一 DB 模拟输入。
- DB fixture 未提供时只允许预览，并明确标记 `db_status=unknown`；禁止自动写入。
- 仓库外绝对数据路径默认拒绝。只有用户明确授权只读后才传 `--allow-external-data`。
- 不写 SQL Server/PostgreSQL，不运行训练，不上传附件，不修改现有场景。
- 不输出本地配置文件内容、连接串、凭证或原始时序行。

## 结果处理

- 脚本成功时读取 `status`、`can_apply`、`effective_config`、`data_quality`、`window_counts` 和 `diff`。
- 脚本失败时保留稳定错误码并向用户解释。错误码见 [config-contract.md](references/config-contract.md)。
- 需要命令范例或 fixture 格式时读取 [usage-examples.md](references/usage-examples.md)。
- 真实数据库只读验证属于后续集成测试；在本 MVP 中停止并请求额外授权，不自行实现网络访问。
