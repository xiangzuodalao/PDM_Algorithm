# 使用示例

## CSV 场景预览

```powershell
uv run python .agents/skills/pdm-onboard-scenario/scripts/onboard_scenario.py `
  --repo-root . `
  --candidate tests/fixtures/onboard_scenario/candidate_csv.yaml `
  --db-models tests/fixtures/onboard_scenario/db_models.json `
  --format json
```

CSV 数据由候选配置的 `data_path` 指定。

## DB 场景预览

```powershell
uv run python .agents/skills/pdm-onboard-scenario/scripts/onboard_scenario.py `
  --repo-root . `
  --candidate tests/fixtures/onboard_scenario/candidate_db.yaml `
  --db-models tests/fixtures/onboard_scenario/db_models.json `
  --timeseries-fixture tests/fixtures/onboard_scenario/db_timeseries.csv `
  --format json
```

## 确认后应用

先运行预览并向用户展示 diff，记录返回的 `diff_hash`。该 hash 同时绑定 diff、最终生效配置和
registry 快照；用户明确确认后，在同一组参数后追加：

```text
--apply --confirmed --expected-diff-hash <confirmed-diff-hash>
```

## 仅生成候选但 DB 状态未知

省略 `--db-models` 可生成 diff 和完成其余检查，但结果为 `can_apply=false`，不得继续写入。

## 仓库外数据

默认拒绝仓库外绝对路径。只有用户明确授权对指定文件只读后，才追加：

```text
--allow-external-data
```

## 独立的真实库只读集成测试

默认 pytest 通过 `-m 'not integration_db'` 排除真实库测试，并在普通测试中封锁网络。只有用户明确授权后，才可设置以下环境变量并单独运行：

```powershell
$env:VALEO_PDM_ALLOW_READONLY_DB_TESTS = "1"
$env:VALEO_PDM_READONLY_SQLSERVER_CONFIG = "<authorized-config.json>"
$env:VALEO_PDM_READONLY_EQUIPMENT_CODE = "<equipment>"
$env:VALEO_PDM_READONLY_MEAS_CODE = "<measurement>"
uv run pytest tests/integration/test_readonly_db.py -m integration_db
```

该测试只执行 `SELECT COUNT(1)` 并断言待新增键不存在；不要在默认测试或 skill 预览中运行。
