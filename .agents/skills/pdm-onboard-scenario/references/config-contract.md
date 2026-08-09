# 场景配置契约

## 候选文件

候选使用 YAML 或 JSON：

```yaml
equipment_code: EQ-NEW
meas_code: SENSOR-A
config:
  model_type: informer
  freq: 15min
  days_back: 30
  source: csv
  data_path: tests/fixtures/onboard_scenario/csv_timeseries.csv
  train_params:
    seq_len: 4
    label_len: 2
    pred_len: 2
    stride: 1
```

必须显式提供 `equipment_code`、`meas_code` 和
`config.train_params.seq_len/label_len/pred_len`。最终 `source=csv` 时，候选块还必须显式提供
`data_path`。`model_type/freq/days_back/source` 允许由 registry 顶层 `defaults` 补齐。
只有这四个字段允许从顶层 defaults 继承；`train_params` 与其他字段不会继承。

## 生效顺序

1. 同一设备+测点优先选择 DB 整块；DB 不继承同名 YAML 的任何字段。
2. DB 不存在时选择 YAML 整块。
3. 对选中的整块应用顶层 `defaults` 补缺；嵌套字典不做字段级深度合并。

“新增”要求目标键在 YAML 和 DB fixture 中都不存在。任一来源重复即硬失败；DB 状态未知时只能预览。

## 模型支持

模型支持集是以下三者交集：

- `training/registry.py` 中 Trainer 目标源码存在，并能在隔离子进程中实际导入为 callable。
- `api/router.py` 的 `MODEL_PREDICT_FUNCS` 存在预测映射。
- `api/router.py` 的 `API_TRAIN_MODEL_TYPES` 声明 API 训练支持。

当前 MVP 权威 allowlist 仅为 `informer`、`autoformer`，最终支持集还必须位于上述三方交集中；
`autoformer` 仅允许 CSV。`testmodel`、`xlstm`、`timellm` 不可用。

## 窗口规则

- `seq_len/label_len/pred_len` 必须是非布尔正整数。
- `0 < label_len <= seq_len`。
- `stride` 缺省为 1；存在时必须是非布尔正整数。
- 清洗重采样后的窗口数：`floor((rows - seq_len - pred_len) / stride) + 1`，下限为 0。
- 使用真实 80/10/10 顺序切分，train/val/test 均至少有一个窗口。

## 数据规则

- 时间列不区分大小写，接受 `collect_time/collecttime/timestamp`。
- 数值列不区分大小写，接受 `value/values`。
- 无法解析时间或非数值行直接丢弃并报告数量，不插值。
- 重复时间与同一重采样桶按均值聚合。
- DB fixture 时序行使用 `equipment_code/meas_code/value/unit/collect_time`，并先过滤目标场景。

## DB fixture

`db_models.json`：

```json
{
  "models": {
    "EXISTING-EQ": {
      "EXISTING-MEAS": {
        "model_type": "informer",
        "source": "db",
        "train_params": {"seq_len": 4, "label_len": 2, "pred_len": 2}
      }
    }
  }
}
```

顶层必须且只能包含 `models`；其下必须严格为设备 → 测点 → 配置块三层映射。连接串或其他本地配置不能作为 fixture。

DB 时序 fixture 可为 CSV，或 JSON 行数组/`{"rows": [...]}`。它只模拟查询结果，不包含连接信息。

## 稳定错误码

- `E_CANDIDATE_SCHEMA`：候选结构或字段类型错误。
- `E_DUPLICATE_YAML`：YAML 已存在同名场景。
- `E_DUPLICATE_DB`：DB fixture 已存在同名场景。
- `E_DB_STATUS_UNKNOWN`：未提供 DB fixture，禁止 apply。
- `E_MODEL_UNSUPPORTED`：模型不在三方支持交集。
- `E_SOURCE_UNSUPPORTED`：数据源不支持，或 Autoformer 使用非 CSV。
- `E_WINDOW_INVALID`：窗口参数类型/关系错误。
- `E_WINDOW_INSUFFICIENT`：清洗后 train/val/test 任一无窗口。
- `E_DATA_PATH_REQUIRED`：CSV 未显式提供 `data_path`。
- `E_DATA_FILE_MISSING`：数据或 fixture 文件不存在。
- `E_EXTERNAL_DATA_PATH`：未经授权读取仓库外绝对路径。
- `E_DATA_SCHEMA`：时序列缺失或频率无效。
- `E_CONFIRM_REQUIRED`：apply 未附带确认标记。
- `E_DIFF_MISMATCH`：apply 的 diff 与用户确认的预览 hash 不一致。
- `E_REGISTRY_CHANGED`：registry 在预览后变化或已有另一个写入进行中。
- `E_TEMP_VALIDATION`：临时副本无法解析或最终配置不一致。
- `E_POST_WRITE_VALIDATION`：写入后验证失败且已回滚。
