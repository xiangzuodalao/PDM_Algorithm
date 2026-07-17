# Codex Skill 自动化需求调研

调研日期：2026-07-16
调研范围：`IFactoryMom.PdM-Algorithm` 当前工作树
调研方式：只读扫描仓库结构、源码、配置、文档、Docker、Git 跟踪文件和现有命令；未连接数据库、未训练模型、未构建镜像、未输出本地连接信息。

## 1. 结论摘要

这个仓库最值得做成 Codex skill 的并不是通用的“写代码”或“跑命令”，而是需要同时理解项目约定、跨多个文件修改、识别外部副作用并完成验证的工作流。

建议保留 5 个 skill 候选：

1. `pdm-onboard-scenario`：接入新的设备/测点训练场景。
2. `pdm-add-model-type`：接入新的模型类型并补齐训练、预测、API、产物和文档契约。
3. `pdm-data-doctor`：训练前时序数据质量与窗口可行性体检。
4. `pdm-train-smoke-triage`：受控小规模训练、产物检查和失败诊断。
5. `pdm-audit-artifacts`：训练产物审计与候选比较；需先补结构化 run manifest。

第一批推荐做 3 个 MVP：

1. `pdm-onboard-scenario`：频繁、边界清晰、能立即减少重复 YAML 和“改了但未生效”的问题。
2. `pdm-add-model-type`：当前已经存在跨文件模型类型漂移，价值很高；先内置确定性的模型契约检查脚本。
3. `pdm-data-doctor`：在昂贵训练前尽早发现采样、缺失、窗口数量和配置问题。

以下事项更适合普通脚本或 CI，不应单独包装成依赖 LLM 判断的 skill：模型支持集一致性、OpenAPI/README 契约、端口与镜像安全、`src`/构建产物漂移、依赖可复现性。

## 2. 仓库与工作流概览

### 2.1 技术栈

- Python 3.12，`src` 布局，Setuptools 构建，`uv.lock` 锁依赖。
- CLI 入口：`valeo-pdm`。
- FastAPI、Pydantic、Uvicorn，本地 Swagger 静态资源。
- PyTorch 时序预测：Informer、Autoformer，以及代码中的 GRU `testmodel`。
- PostgreSQL：`psycopg2`；SQL Server：`pyodbc`。
- YAML 模型注册表、JSON 数据库/上传配置；真实配置由 `.gitignore` 排除。
- Docker/Compose，包含 ODBC、CUDA Torch、GPU 设备和持久化目录配置。

### 2.2 主链路

```text
pyproject.toml [project.scripts]
  └─ valeo_pdm.cli:main
      ├─ list-models
      │   └─ transformer.config.list_all_models
      │       ├─ configs/model_registry.yaml
      │       └─ SQL Server 动态配置（同名时覆盖 YAML）
      ├─ train / train --all
      │   └─ training.from_config.train_from_config
      │       ├─ 解析最终模型配置
      │       ├─ 读取 PostgreSQL / SQL Server / CSV
      │       ├─ training.registry.get_trainer
      │       ├─ 写 artifacts/checkpoints
      │       └─ 可选：SQL Server 状态回写与图片上传
      └─ serve
          └─ api.app:app
              └─ api.router
                  ├─ POST /measPredict/train
                  ├─ POST /measPredict/predict
                  ├─ GET  /measPredict/models
                  └─ POST /measPredict/checkData

Dockerfile
  └─ uv sync → 安装 CUDA Torch → uvicorn :10021
      └─ docker-compose
          ├─ configs 只读挂载
          ├─ artifacts/data 持久化
          ├─ src 宿主机挂载
          └─ NVIDIA GPU + /healthz
```

### 2.3 现有操作入口

- 列出配置：`uv run valeo-pdm list-models`
- 单场景训练：`uv run valeo-pdm train -e <equipment> -m <measurement>`
- 全量训练：`uv run valeo-pdm train --all`
- 启动服务：`uv run valeo-pdm serve --host 0.0.0.0 --port 8000`
- CSV 重采样：`python -m valeo_pdm.transformer.scripts.resample --in <csv> --freq 15min`
- SQL Server 手工探针：`python -m valeo_pdm.db.test_sqlserver_connection --config <path>`
- 容器启动：`docker compose up -d --build`

## 3. 为什么需要自动化

### 3.1 模型类型需要跨多个位置保持一致，但目前已经漂移

证据：

- `src/valeo_pdm/training/registry.py:10-29` 注册 Trainer。
- `src/valeo_pdm/api/router.py:22-26` 注册预测函数。
- `src/valeo_pdm/api/router.py:289-370` 决定 API 训练支持的默认参数和调用形态。
- `src/valeo_pdm/training/from_config.py:116-389` 又有一套 CLI 训练编排。
- `src/valeo_pdm/transformer/predict.py` 实现预测加载和响应。
- `README.md:151-158` 只描述了部分手工接入步骤。

已发现的具体漂移：

- `testmodel` 有独立 `train_testmodel.py` 和预测函数，但 Trainer registry 将它映射到 Informer；API 参数合并阶段还会拒绝 `testmodel`。
- Trainer registry 引用了 `xlstm` 和 `timellm`，仓库中却没有对应实现模块。
- 新模型还必须同步 checkpoint 命名、API schema、配置示例和验证，README 未覆盖全部契约。

### 3.2 场景配置重复，且“文件配置”不一定是最终生效配置

- `configs/model_registry.yaml` 有多个高度重复的设备/测点超参块。
- `src/valeo_pdm/transformer/config.py:74-140` 查询单个配置时采用 SQL Server 优先、YAML 兜底。
- `src/valeo_pdm/transformer/config.py:143-216` 列表时合并两套来源，并由数据库覆盖同名 YAML。
- 因此只编辑 YAML 不能保证最终生效，人工检查很容易遗漏覆盖关系。

### 3.3 训练前检查不完整，且计算口径存在漂移

- 数据清洗、列名归一、时间解析和重采样散落在 `db/data_reader.py`、两个训练器和 `scripts/resample.py` 中。
- `src/valeo_pdm/api/router.py:544-547` 的数据量检查按 70/10/20 口径估算，而 `src/valeo_pdm/transformer/data.py:45-49` 默认切分是 80/10/10。
- 现有 `/checkData` 主要检查行数，不能完整覆盖采样间隔、缺失、重复、异常、重采样后有效窗口数和资源需求。

### 3.4 训练与外部系统副作用交织

- CLI/API 训练会写 checkpoint、参数和图片。
- 传入特定运行标识时，还可能更新 SQL Server 状态并上传预测图片。
- 当前手工数据库探针和 SQL Server 读取代码会打印完整连接串，不适合原样进入自动化日志。
- Skill 必须把本地写入、数据库读取、数据库写入、HTTP 上传和 GPU 训练分成不同确认级别。

### 3.5 缺少自动门禁

- 未发现正式 `tests/`、`conftest.py`、pytest 测试函数或 CI 配置。
- 唯一 `test_*.py` 是会连接真实 SQL Server 的手工探针，不是隔离测试。
- `build/lib` 被 Git 跟踪，但 Setuptools 的包源是 `src`；两者已有大量差异。
- README、CLI、直接运行入口和 Docker/Compose 使用了不同端口口径。
- README 的 API 路径与代码实际路由不一致；请求必填字段也存在文档遗漏。

## 4. Skill 候选排序

| 排名 | Skill | 价值 | 可行性 | 优先级 | 核心产出 |
|---:|---|---:|---:|---|---|
| 1 | `pdm-onboard-scenario` | 5/5 | 4.5/5 | P0 | 安全接入设备/测点配置，并证明最终生效配置 |
| 2 | `pdm-add-model-type` | 5/5 | 3.5/5 | P0 | 一次完成新模型的跨文件接入与契约验证 |
| 3 | `pdm-data-doctor` | 4.5/5 | 4/5 | P1 | 训练前数据质量、窗口数量和参数可行性报告 |
| 4 | `pdm-train-smoke-triage` | 4.5/5 | 3/5 | P1 | 受控 smoke 训练、产物验收和失败归因 |
| 5 | `pdm-audit-artifacts` | 3.5/5 | 2.5/5 | P2 | 实验产物盘点与候选对比，不自动上线 |

## 5. Skill 需求详述

### 5.1 `pdm-onboard-scenario`

**目标**

把“新增一个设备/测点场景”从复制 YAML、猜参数、运行训练，变成可预览、可验证、默认不触达外部写入的流程。

**典型触发语句**

- “给设备 `<equipment>` 新增测点 `<measurement>` 的预测配置。”
- “把这个 CSV 场景接入 Informer，频率 15 分钟。”
- “复制现有测点配置给新测点，但先检查数据量是否够。”
- “为什么我改了 `model_registry.yaml`，服务里还是旧配置？”

**输入**

- 设备和测点标识。
- 模型类型、采样频率、历史天数、数据源。
- CSV 路径或数据库来源；如允许访问数据库，需单独确认。
- 训练参数或可复用的模板场景。
- 配置落点：MVP 仅支持生成/编辑 YAML；数据库配置只做只读冲突检查。

**输出**

- 目标配置 diff。
- 静态 schema 与模型支持检查结果。
- “YAML 配置 / 数据库覆盖 / 最终生效配置”对比。
- 数据预检摘要和建议训练命令。
- 未执行的高风险步骤清单。

**工作流**

1. 读取当前 registry、模型支持集和目标场景。
2. 判断是否存在 YAML/数据库同名覆盖；数据库访问未获授权时明确标记未知。
3. 从模板生成最小配置，拒绝静默填入不确定的生产标识或路径。
4. 校验模型类型、必填字段、参数类型、窗口关系和 CSV 路径。
5. 展示 diff；获授权后只修改目标 YAML 块。
6. 运行无外部副作用的静态验证。
7. 如用户授权数据库读取，再做最终配置和数据量核验。
8. 输出训练前检查表，不自动开始完整训练。

**建议复用资源**

- `scripts/validate_scenario.py`：确定性校验 YAML、字段、参数类型、窗口关系和模型支持集。
- `references/config-resolution.md`：说明 SQL Server 优先、YAML 兜底、列表合并规则。
- `references/scenario-parameters.md`：记录本项目非通用的字段含义和约束。
- 无需 `assets/`。

**自由度**

中等。配置解析和验证应低自由度、脚本化；参数建议和冲突解释允许基于上下文判断。

**边界与确认点**

- 默认只读；编辑 YAML 前展示 diff。
- 不直接写 SQL Server 配置表。
- 不输出连接串、凭证或原始数据行。
- 数据库读取、训练和任何外部写入分别确认。

**验证与验收**

- YAML 可解析，新增项唯一，未改动无关场景。
- 模型类型在 Trainer、预测和 API 支持集内一致。
- 窗口参数满足基本关系，数据路径存在或数据库检查已明确跳过。
- 能生成确定的 `list-models`/训练命令，但不依赖真实生产写入才能通过。

### 5.2 `pdm-add-model-type`

**目标**

一次完成新模型类型的训练器、预测器、API、配置、checkpoint 和文档接入，并用确定性契约测试防止漏改。

**典型触发语句**

- “给项目新增 `patchtst` 模型类型。”
- “把这个训练实现接到 CLI 和预测 API。”
- “检查 `testmodel` 为什么训练和预测支持不一致。”
- “补齐 `xlstm` 的模型注册和最小测试。”

**输入**

- 模型类型名称和已有实现路径。
- 训练函数签名、预测函数签名、默认超参。
- 支持的数据源和 checkpoint 内容约定。
- 最小合成数据或小型 fixture。
- 是否需要 API 暴露、文档示例和向后兼容。

**输出**

- 所有必要源码和配置 diff。
- 模型契约矩阵：Trainer、predict、API train、checkpoint、registry、docs。
- 静态导入/签名检查和最小 smoke 结果。
- 未执行的 GPU/生产验证清单。

**工作流**

1. 先运行模型支持集一致性检查，记录基线漂移。
2. 明确模型训练/预测契约和数据源限制。
3. 实现或接入 Trainer 与 predictor。
4. 更新统一 registry；避免继续增加分散的字符串分支。
5. 补齐 API schema/defaults、CLI、checkpoint 命名和配置示例。
6. 生成或更新契约测试和最小合成数据 smoke。
7. 更新文档，并从代码/OpenAPI 反向校验示例。
8. 不修改 `build/lib`；构建产物应由 clean build 生成。

**建议复用资源**

- `scripts/check_model_contracts.py`：比较各层模型集合，验证 import、函数签名和 checkpoint 约定。
- `references/model-extension-contract.md`：项目专用训练/预测返回值、目录、数据和 API 约定。
- `references/model-support-matrix.md`：由脚本生成或维护的支持矩阵。
- `assets/model-fixture/`：仅在确有可复用的微型合成数据/配置模板时使用。

**自由度**

中等偏低。契约、文件落点和检查顺序低自由度；模型内部实现保留较高自由度。

**边界与确认点**

- 修改源码属于用户明确请求后的正常实施步骤。
- 默认只运行静态、CPU 或极小规模 smoke；GPU 长训练需确认。
- 不触发状态回写、图片上传或生产数据库写入。
- 不同时维护 `src` 与 `build/lib` 两套源码。

**验证与验收**

- 新模型在 Trainer、predict、API train、配置和文档中的支持集合一致。
- 所有惰性 import 可解析，训练/预测函数满足签名和返回值契约。
- 合成数据最小 smoke 可运行，checkpoint 可被预测器重新加载。
- 旧模型的契约测试仍通过。

### 5.3 `pdm-data-doctor`

**目标**

在训练前只读分析时序数据，给出可复现的数据质量、重采样和窗口可行性结论。

**典型触发语句**

- “检查这个 CSV 是否够训练 7 天预测模型。”
- “设备 `<equipment>` 的测点 `<measurement>` 为什么没有有效样本？”
- “按 15 分钟重采样后还能生成多少训练窗口？”
- “训练 loss 异常前先帮我检查缺失、重复和异常值。”

**输入**

- CSV 路径，或经明确授权的只读数据库来源。
- 时间列、数值列、目标频率。
- `seq_len`、`label_len`、`pred_len`、stride 和实际切分口径。
- 可选的异常阈值和资源预算。

**输出**

- 时间覆盖、推断频率、缺失、重复、乱序、非数值、间断和异常统计。
- 重采样前后数据量和信息损失。
- 按真实切分逻辑计算的 train/val/test 有效窗口数。
- 训练参数和数据补充建议。
- 机器可读 JSON 与简洁 Markdown 摘要。

**工作流**

1. 确定数据源和访问权限；数据库默认不连接。
2. 只读取必要列，绝不在报告中输出原始数据行。
3. 复用与训练代码一致的列名、时间解析、清洗和重采样逻辑。
4. 计算每个阶段的行数变化和原因。
5. 使用实际训练切分函数计算窗口，而不是复制另一套公式。
6. 估算最小可行参数和资源风险。
7. 输出结论，不自动改数据或启动训练。

**建议复用资源**

- `scripts/inspect_timeseries.py`：确定性读取、脱敏统计、重采样模拟和窗口计算。
- `references/data-contract.md`：支持的列名、数据源和训练代码口径。
- `assets/fixtures/`：小型合成 CSV，用于正常、缺失、重复、间断场景测试。

**自由度**

统计和窗口计算低自由度；问题解释和参数建议中等自由度。

**边界与确认点**

- 本地文件默认只读；数据库连接需确认且必须使用只读查询。
- 日志中屏蔽用户名、密码、主机、数据库名和完整 SQL 参数。
- 不修复或覆盖原始数据；如需生成清洗副本，单独展示目标路径并确认。

**验证与验收**

- 合成 fixture 覆盖正常、缺失、重复、乱序、频率不匹配和不足样本。
- 窗口数与训练代码使用同一实现，避免 70/10/20 与 80/10/10 漂移。
- 相同输入产生稳定 JSON 结果；报告无原始数据和凭证。

### 5.4 `pdm-train-smoke-triage`

**目标**

以明确资源预算和副作用边界执行单场景小规模训练，检查产物，并把失败归因到环境、配置、数据、模型或外部集成。

**典型触发语句**

- “对这个场景跑 1 个 epoch 的 smoke test。”
- “这次训练失败了，帮我定位是数据、CUDA 还是配置问题。”
- “训练完成后检查 checkpoint、参数文件和曲线是否齐全。”

**输入**

- 设备、测点、模型类型和运行标识。
- epoch、样本范围、设备、超时等资源预算。
- 是否允许写本地 artifacts。
- 是否允许数据库读取、状态回写或 HTTP 上传；三者分别授权。

**输出**

- 执行前计划、最终配置和副作用清单。
- 命令、关键日志摘要、耗时和资源信息。
- checkpoint/参数/图片验收结果。
- 失败分类、证据和下一步建议。

**工作流**

1. 先运行场景校验和 data doctor。
2. 展示最终配置、命令、资源预算和可能的外部副作用。
3. 默认禁用 SQL 状态回写和图片上传。
4. 只运行一个场景和低 epoch smoke；禁止默认使用 `train --all`。
5. 验收训练返回值与产物目录。
6. 失败时收集脱敏日志，按环境/依赖/数据/配置/模型/外部系统分类。
7. 输出建议，不自动扩大到完整训练。

**建议复用资源**

- `scripts/run_smoke.py`：固定预算、超时、环境信息和本地产物检查。
- `references/training-error-catalog.md`：本项目常见错误、证据和安全诊断动作。
- `references/side-effects.md`：状态回写、上传和 artifact 写入边界。

**自由度**

执行顺序和预算低自由度；日志归因中等自由度。

**边界与确认点**

- 写本地 artifacts 前确认目标目录。
- GPU 长任务、数据库访问、状态写回和上传分别确认。
- 不自动执行全量训练，不清理既有 checkpoint，不切换生产模型。

**验证与验收**

- 先修复 `testmodel` 契约后，用小型合成数据建立 CPU smoke 基线。
- 超时和失败路径能留下脱敏、可定位的结果。
- 成功路径至少验证 checkpoint 可加载、参数可解析、预期文件存在。

### 5.5 `pdm-audit-artifacts`

**目标**

盘点训练 run、检查产物契约、比较候选并生成晋级建议，但不自动部署或替换生产模型。

**典型触发语句**

- “列出这个设备最近的训练 run，并检查哪些产物不完整。”
- “比较两个 ModelInfoID 的参数和指标。”
- “给出可以进入人工验收的模型候选。”

**输入**

- 设备、测点、模型类型和 artifact 根目录。
- 候选 run 或 ModelInfoID。
- 指标方向、最低门槛和人工验收规则。

**输出**

- run inventory、缺失文件和 checkpoint 契约结果。
- 参数与指标对比。
- “可验收 / 信息不足 / 不合格”建议及原因。
- 不执行 promotion 的明确声明。

**前置条件**

当前产物不足以可靠自动排名：CLI 与 API 的目录层级不同，指标没有统一持久化，无“最新模型”解析和回滚/promotion 原语。应先增加结构化 `run-manifest.json`，至少记录：run ID、场景、模型类型、代码版本、参数、数据摘要、指标、文件校验和、开始/结束状态。

**建议复用资源**

- `scripts/audit_artifacts.py`：扫描目录、读取 manifest、校验 checkpoint 和文件哈希。
- `references/artifact-contract.md`：目录、manifest 和 promotion 前置规则。

**自由度**

产物校验低自由度；候选解释中等自由度。

**边界与确认点**

- 只读扫描；不删除、不重命名、不移动 checkpoint。
- 不自动把任何候选标记为生产模型。
- promotion、回滚和部署必须是独立、显式授权的工作流。

**验证与验收**

- 用完整、缺失、损坏和版本不匹配的 artifact fixtures 验证。
- 同一输入得到确定的 inventory 和契约结论。
- 没有 manifest 时只能报告“信息不足”，不能猜测最佳模型。

## 6. 更适合脚本或 CI 的需求

### 6.1 模型支持集一致性检查（P0）

用确定性脚本比较以下集合：Trainer registry、预测映射、API 训练分支、checkpoint 约定、配置与文档。对每种模型做 import、签名和最小 mock contract 检查。

它应成为 `pdm-add-model-type` 的 bundled script，同时在 CI 中独立运行；不需要单独做成 skill。

### 6.2 配置、OpenAPI 与 README 契约检查（P0）

应从 FastAPI OpenAPI/schema 校验或生成 API 示例，检查：

- 实际路由与文档路径。
- 请求必填字段与 README 示例。
- `VALEO_PDM_MODEL_REGISTRY` 覆盖行为。
- `/checkData` 与真实数据切分/窗口公式。

测试必须 mock 数据库，避免 CI 访问外部系统。

### 6.3 Docker/发布预检与 secret build-context 防线（P0）

应在 CI 运行：

- `docker compose config`。
- README、CLI、直接运行入口、Docker 和 Compose 的端口一致性检查。
- 构建上下文 secret 扫描。
- 不含真实配置的镜像健康检查。

特别注意：`.gitignore` 屏蔽真实配置并不等于 `.dockerignore` 已全部屏蔽；`COPY . /app` 前必须单独验证。

### 6.4 `src`/构建产物与依赖可复现性检查（P1）

- `pyproject.toml` 以 `src` 为包源，不应人工同步 `build/lib`。
- clean wheel build/import smoke 应替代提交和双写构建产物。
- 应消除或明确 `pyproject_bak.toml`、`uv_bak.lock`、`tmp_reqs.txt` 的权威性。
- frozen lock 后再次单独安装 Torch 会削弱可复现性，需要明确 CUDA/GPU 兼容策略。

### 6.5 安全数据库 doctor 脚本（P1）

先实现确定性脚本，再决定是否由 skill 编排：

- 校验配置键和驱动。
- 默认只做本地检查。
- 获授权后执行只读 `SELECT 1` 或受限数据统计。
- 对连接串、SQL 参数和标识进行脱敏。
- 将“连通性测试”与“状态写入/附件上传”完全分离。

在修复当前连接串打印问题前，不应直接把现有手工探针放进 skill。

## 7. 不建议无人值守自动化

- `valeo-pdm train --all`：耗时、GPU 成本和失败影响不可控，且当前逐个串行、任一异常可中止。
- 自动选择并替换“生产最新模型”：当前没有可靠 run manifest、promotion、回滚和审计原语。
- 在诊断过程中自动写 SQL Server 状态或上传附件：诊断应默认只读。
- 同时维护 `src` 与 `build/lib` 两份代码：应删除或重新生成构建副本，不应让 skill 双写。
- 仅凭通用指标自动决定模型架构、阈值或生产上线：需要业务 KPI、设备工程知识和人工验收。
- 自动修改真实连接配置或把凭证写入仓库、日志、报告、镜像构建上下文。

## 8. 推荐 MVP 路线

### MVP 1：`pdm-onboard-scenario`

**最小范围**

- 仅支持本地 YAML 场景新增/复制。
- 内置静态 schema、模型支持、窗口关系和路径检查。
- 只读提示可能存在的数据库覆盖；无授权时不连接数据库。
- 输出 diff 和下一步命令，不自动训练。

**验收标准**

- 用一个 DB 场景和一个 CSV 场景 fixture 完成新增。
- 对重复标识、未知模型、错误类型、无效窗口和缺失文件给出稳定错误。
- 不修改无关配置，不泄露本地配置内容。

### MVP 2：`pdm-add-model-type`

**先决条件**

先实现 `scripts/check_model_contracts.py`，让当前 `testmodel`、`xlstm`、`timellm` 漂移能够被机器稳定发现。

**最小范围**

- 支持接入一个已有 Trainer/predictor 的新模型。
- 更新统一支持矩阵、API、配置示例和文档。
- 运行静态 import/签名检查和合成数据 smoke。
- 不运行 GPU 长训练，不修改 `build/lib`。

**验收标准**

- 新模型不漏任何契约点。
- 故意删除一个注册步骤时，检查脚本必定失败并指出位置。
- 旧模型行为不回归。

### MVP 3：`pdm-data-doctor`

**先决条件**

让 doctor 与训练代码共用同一个切分/窗口实现，先消除现有公式漂移。

**最小范围**

- 首版只分析本地 CSV。
- 输出 JSON/Markdown 数据质量报告。
- 覆盖采样频率、缺失、重复、乱序、异常和有效窗口数。
- 数据库只读检查放到后续版本。

**验收标准**

- 合成 fixture 覆盖至少 6 种数据问题。
- 统计结果可复现，且不输出原始数据行。
- 训练窗口数与真实训练代码一致。

## 9. 创建 Skill 前需要确认

1. Skill 放置位置：个人 Codex skills 目录，还是仓库内可共享的位置。
2. 场景配置的最终权威来源：SQL Server、YAML，还是二者并存。
3. 第一阶段是否允许只读连接数据库；是否存在专用只读账号/测试库。
4. 哪些本地写入默认允许：配置 diff、测试 fixture、artifacts、报告。
5. GPU/CUDA 目标矩阵和可接受的 smoke 预算。
6. 新模型的业务验收 KPI，以及谁批准进入生产。
7. `build/lib` 是否可以停止跟踪并由构建生成。
8. 计划使用的 CI 平台。
9. 生产日志、设备/测点标识和数据摘要的脱敏要求。

## 10. 实施方式

选定 MVP 后，按 `skill-creator` 流程实施：

1. 用具体用户语句确认触发范围和反触发范围。
2. 先实现可复用的 `scripts/`、`references/` 和必要 fixture。
3. 使用 `init_skill.py` 初始化 skill，而不是手工拼目录。
4. 保持 `SKILL.md` 精简，把项目契约放入按需加载的 references。
5. 实际运行 bundled scripts，并用 `quick_validate.py` 验证 skill 结构。
6. 用新子代理进行无答案泄露的 forward test，验证 skill 是否能在真实任务中稳定执行。

## 11. 关键未知与限制

- 仓库只有一个初始提交，无法从历史频率判断哪类工作流最常发生；当前排序主要依据代码重复、跨文件契约和故障影响。
- 当前没有隔离测试和 CI，无法给出已有基准成功率。
- 未连接真实数据库、上传服务和 GPU 主机，外部集成结论来自静态代码。
- README 与代码对 API、端口、配置覆盖和模型支持的描述存在差异，应先确定哪个是期望行为，再修复实现或文档。
- artifact 指标和“最新模型”语义不足，`pdm-audit-artifacts` 必须晚于 run manifest 建设。
