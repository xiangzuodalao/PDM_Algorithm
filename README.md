# valeo-pdm

面向预测性维护的时序预测项目，包含训练、推理与 FastAPI 服务。

## 目录结构

- `src/valeo_pdm/`：可安装的 Python 包（核心代码）
- `configs/`：运行时配置（`model_registry.yaml` 等）
- `artifacts/`：训练产物（checkpoint 等，默认不入库）
- `data/`：数据文件（默认不入库）

## 安装（本地开发）

### 推荐：使用 `uv`

Python 3.12、运行时依赖及 CPU 版 PyTorch 均由 `uv.lock` 固定。按锁文件同步环境：

```bash
uv sync --frozen
```

`pyproject.toml` 的显式 `pytorch-cpu` 索引会安装锁定的 CPU wheel；不要注释
`torch`、另行安装 Torch/torchvision，或在此环境中混入 CUDA 包。

同步完成后运行：

```bash
uv run valeo-pdm list-models
uv run valeo-pdm train -e V-SZ-ISD-102 -m CCD-Score1
# uv run --index-url https://pypi.tuna.tsinghua.edu.cn/simple valeo-pdm train -e V-SZ-ISD-102 -m CCD-Score3
uv run valeo-pdm serve --host 0.0.0.0 --port 8000
```

如果不想每次都写 `uv run`，也可以激活环境后直接用 `valeo-pdm ...`。

## 训练

列出已配置的模型：

```bash
valeo-pdm list-models
```

训练单个配置：

```bash
valeo-pdm train -e V-SZ-ISD-102 -m CCD-Score1
```

训练全部配置：

```bash
valeo-pdm train --all
```

## 启动 API

```bash
valeo-pdm serve --host 0.0.0.0 --port 8000
```

或直接用 uvicorn：

```bash
uvicorn valeo_pdm.api.app:app --host 0.0.0.0 --port 8000
```

## Docker 部署（生产）

见 `docs/docker-deploy.md`。

默认 Docker 镜像使用仅 CPU 的 PyTorch，不会请求 NVIDIA 设备；训练与预测 API 均保持可用，但生产规模训练可能较慢。如需 GPU 镜像，必须使用独立的 Dockerfile/profile，并且只配置一个 CUDA/PyTorch 索引；不要向默认 lock 文件添加 CUDA 包。

默认 Compose 服务仅监听 `127.0.0.1:10021`。为兼容既有部署，
`docker-compose.yml` 保留 `./src:/app/src` 源码挂载；运行时代码来自宿主机当前
checkout，部署时应确保它与构建版本一致。Dockerfile 的外部镜像摘要已在
Linux/amd64 本地构建中验证，其他架构需要单独验证。

## 配置

- `configs/model_registry.yaml`：设备/参数 -> 模型配置与训练超参数
- `configs/postgres_config.json`：数据库连接（不入库），参考 `configs/postgres_config.example.json`
- `configs/sqlserver_config.json`：训练状态写回 SQL Server 的连接与列信息（不入库），参考 `configs/sqlserver_config.example.json`

环境变量：

- `VALEO_PDM_MODEL_REGISTRY`：指定 `model_registry.yaml` 路径
- `VALEO_PDM_POSTGRES_CONFIG`：指定 PostgreSQL 配置 JSON 路径
- `VALEO_PDM_SQLSERVER_CONFIG`：指定 SQL Server 状态更新配置 JSON 路径
- `VALEO_PDM_ROOT`：显式指定项目根目录（通常不需要）
- `VALEO_PDM_REQUIRE_TRAIN_PLAN_HASH`：设为 `1` 时，训练必须携带预览返回的计划哈希；默认 `0` 以兼容既有平台调用

## 快速使用

### CLI

- 列表：`valeo-pdm list-models`
- 训练（默认从 `configs/model_registry.yaml` 读取）：`valeo-pdm train -e <设备> -m <参数>`
- 指定模型类型/状态回写：`valeo-pdm train -e V-SZ-ISD-102 -m CCD-Score1 --model-type testmodel --model-info-id Run_001 --sqlserver-config configs/sqlserver_config.json`
- 全量训练：`valeo-pdm train --all`

### FastAPI

- 启动：`valeo-pdm serve --host 0.0.0.0 --port 8000`
- 训练预览：`POST /measPredict/train/preview`，返回最终生效参数和 `plan_hash`，不读取训练数据、不启动训练；数据检查由 `/checkData` 或 MCP 的 `pdm_prepare_training` 编排完成。
- 训练接口：`POST /measPredict/train`，必填 `EquipmentCode/MeasCode/ModelInfoID/ParamArr/DataSource`；API 训练仅支持 `informer`/`autoformer`。
- 训练状态：`GET /measPredict/train/status`
- 预测接口：`POST /measPredict/predict`，可选 `ModelInfoID`（为空使用固定默认 checkpoint）、`ModelType`。
- 模型列表：`GET /measPredict/models`
- 数据检查：`POST /measPredict/checkData`
- 健康检查：`GET /healthz`

### Codex MCP 接入

- 本仓只提供 PDM FastAPI 以及 `compose.mcp-smoke.yml` API 冒烟目标，不再维护 MCP Server 或训练 Skill 的副本。
- 平台 MCP 由私有 `ifactory-platform` 总仓的 `components/digital-mcp` 维护；训练 Skill 由该总仓的 `.agents/skills` 维护。
- 本地安全冒烟：`docker compose -f compose.mcp-smoke.yml up -d --build`，API 仅监听 `127.0.0.1:10022`，不挂载真实数据库配置。
- Codex 注册、工具契约和训练审批流程以私有 `ifactory-platform` 总仓为准。

### 数据源

- PostgreSQL（默认）：按 `configs/postgres_config.json` 里 SQL 查询拉取数据。
- CSV：在 `configs/model_registry.yaml` 对应条目设置 `source: csv`、`data_path: data/<文件>`。API/MCP 只接受 `data/` 下的相对路径，并拒绝 `..`、绝对路径和解析后越界的符号链接；格式需包含 `value` 列，最好有 `collect_time/collecttime/timestamp`。

### 训练状态回写（可选）

- 需 `configs/sqlserver_config.json` 填好连接和列名，并在表中预置 `ModelInfoId`/`IsActive` 等条件匹配行。
- 状态码默认：1=Training，2=Failed，3=Completed；描述写入英文短语；附件列可写预测图路径。

## 如何新增场景（设备/参数组合）

1) 在 `configs/model_registry.yaml` 添加条目，示例：
   ```yaml
   models:
     EQP-001:
       SENSOR_A:
         model_type: informer         # informer/testmodel/autoformer
         freq: "15min"
         days_back: 180
         source: db                   # db 或 csv
         data_path: data/xxx.csv      # 仅当 source=csv 时需要
         train_params:
           seq_len: 672
           label_len: 192
           pred_len: 288
           batch_size: 32
           epochs: 50
           lr: 0.0005
           ...
   ```
2) 数据源准备：
   - DB：确认 `configs/postgres_config.json` 的 SQL 能返回该设备/参数。
   - CSV：设定 `source: csv`、`data_path`，CSV 至少有 `value` 列，建议有 `collect_time`（或 `collecttime/timestamp` 会被映射）。
3) 运行：
   - CLI：`valeo-pdm train -e EQP-001 -m SENSOR_A`
   - API：训练 `POST /measPredict/train`，传 `EquipmentCode/MeasCode/ModelInfoID/ParamArr/DataSource`，可选 `ModelType` 覆盖
   - 预测使用 `POST /measPredict/predict`；`ModelInfoID` 不传时使用固定默认 checkpoint

## 如何新增模型类型（自定义 model_type）

1) 编写训练器：新增 `src/valeo_pdm/transformer/train_<name>.py`，实现 `do_training(**kwargs)`，返回至少包含 `best_path`、`best_val`、`test_loss`、`plots`。
2) 编写预测器：在 `src/valeo_pdm/transformer/predict.py` 增加预测函数，并在 `MODEL_PREDICT_FUNCS["<name>"] = your_predict_func` 注册。
(`MODEL_PREDICT_FUNCS` 定义在 router.py (lines 20-24)，映射了 `informer/testmodel/autoformer` 到对应的预测函数。)
3) 注册 Trainer：在 `src/valeo_pdm/training/registry.py` 的 `get_trainer` 中为新 `model_type` 返回你的 `do_training`。
4) 配置注册表：在 `configs/model_registry.yaml` 中将目标设备/参数的 `model_type` 设为新类型，并补充相应 `train_params`。
5) 请求指定：训练/预测接口的 `ModelType`，或 CLI 的 `--model-type`，可覆盖 registry 中的默认模型类型；产物会写入包含模型名的目录，便于区分实验。

## 代码结构

```
.
├─ README.md                # 使用说明
├─ configs/                 # 运行/连接/注册配置
│  ├─ model_registry.yaml   # 设备-参数-模型映射与超参
│  ├─ postgres_config*.json # PostgreSQL 连接/查询
│  └─ sqlserver_config*.json# SQL Server 训练状态回写配置
├─ artifacts/               # 训练产物（checkpoints、图片、保存的参数）
├─ data/                    # CSV 数据示例/挂载目录
├─ src/valeo_pdm/
│  ├─ cli.py                # 命令行入口
│  ├─ api/                  # FastAPI 服务
│  │  ├─ app.py             # FastAPI 应用入口
│  │  └─ router.py          # 训练/预测接口逻辑、参数解析
│  ├─ training/             # 训练调度
│  │  ├─ registry.py        # 模型类型 -> Trainer 映射
│  │  └─ from_config.py     # 读取配置并调用 Trainer，保存参数
│  ├─ transformer/          # 模型、训练、预测实现
│  │  ├─ train_informer.py  # informer 训练
│  │  ├─ train_autoformer.py# autoformer 训练（仅 CSV）
│  │  ├─ train_testmodel.py # testmodel（GRU 简化版）训练
│  │  ├─ predict.py         # 预测 API 逻辑（informer/testmodel/autoformer）
│  │  ├─ models/            # 模型结构
│  │  └─ data.py            # 数据预处理、窗口切分工具
│  ├─ db/                   # 数据源 & 状态回写
│  │  ├─ data_reader.py     # PostgreSQL/SQL Server 数据读取
│  │  └─ train_status.py    # SQL Server 训练状态回写
│  └─ paths.py              # 路径辅助
└─ docker-compose.yml, Dockerfile # 容器部署
```

## 日常操作

- 列表配置：`valeo-pdm list-models`
- 快速训练（DB）：`valeo-pdm train -e <设备> -m <参数>`
- 快速训练（CSV）：在 registry 设置 `source: csv`、`data_path` 后同上
- API 训练 JSON 示例：
  ```json
  {
    "EquipmentCode": "V-SZ-ISD-102",
    "MeasCode": "CCD-Score1",
    "ModelInfoID": "Run_001",
    "ModelType": "informer",
    "ParamArr": [{ "FieldName": "epochs", "CurValue": 3 }],
    "DataSource": "postgres",
    "ExecutionMode": "platform"
  }
  ```
- API 预测 JSON 示例：
  ```json
  {
    "EquipmentCode": "V-SZ-ISD-102",
    "MeasCode": "CCD-Score1",
    "ModelInfoID": "Run_001",
    "ModelType": "informer",
    "HistoryData": []
  }
  ```

## 提示
- `testmodel` 是简化 GRU 示例，如需真实模型可按“新增模型类型”步骤替换。
