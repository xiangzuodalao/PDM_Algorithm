# AGENTS.md

> 本文件面向在本仓库中工作的 AI 编码 agent。完整用法见 `README.md`,深度架构调查见 `ARCHITECTURE_FINDINGS.md`。**本仓库正在被重构,以下以 2026-07-16 代码为准;动手前对关键结论自行复核。**

## 项目概述

IFactoryMom.PdM-Algorithm 是 Valeo 产线预测性维护(PdM)的时序预测系统。PyTorch 训练/推理 Informer 与 Autoformer(及 SimpleGRU),FastAPI 暴露训练/预测接口,数据源 PostgreSQL / SQL Server / CSV。Python 3.12 + uv。

## 启动链与验证

API 启动链(`import valeo_pdm.api.app`)由 `tests/test_api_startup.py` 守护:覆盖 `SimpleGRUForecast` 可导入、FastAPI app 可导入、`/healthz` 返回 200(冒烟测试 loopback-only 断网)。改 `predict.py`/`train_testmodel.py`/`router.py` 的 import 后跑该测试。

> 注:`testmodel` 的**训练**由 registry(`registry.py:11`)路由到 `train_informer.do_training`,而 `train_testmodel.py` 提供 `SimpleGRUForecast` 供 `predict.py:16` 的 testmodel **预测**使用--两者别混淆。

## 环境搭建(关键陷阱)

- **Python >= 3.12**,包管理用 **uv**。
- **torch 不在 `pyproject.toml` 依赖里**,需单独从 PyTorch 索引装,如:
  ```bash
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```
  不要把 torch 加进 `pyproject.toml` dependencies(会破坏 `uv sync`)。
- 安装项目:`uv venv --python 3.12 && uv pip install -e .`(先确保 torch 已装)。
- **密钥配置 gitignore**:`configs/postgres_config.json`、`configs/sqlserver_config.json`、`configs/app_config.json` 只保留 `*.example.json` 模板,**勿提交真实连接串**;新增配置项同步更新 `.example.json`。
- `artifacts/`、`data/` 同样 gitignore。

## 常用命令

```bash
# CLI
uv run valeo-pdm list-models                 # -l / ls / list
uv run valeo-pdm train -e <设备> -m <参数>    # --all / --model-type / --model-info-id / --sqlserver-config
uv run valeo-pdm serve --host 0.0.0.0 --port 8000 --reload
uv run uvicorn valeo_pdm.api.app:app --port 8000

# 测试(pyproject 默认 addopts 排除 integration_db)
uv run pytest
uv run pytest tests/test_data_pipeline.py -v
uv run pytest -m integration_db              # 需授权环境变量,见 tests/integration

# Lint / 格式化(ruff,line-length 100,py312)
uv run ruff check .
uv run ruff format .
```

入口:CLI `valeo-pdm` → `src/valeo_pdm/cli.py:main`(也可 `python -m valeo_pdm`);API → `src/valeo_pdm/api/app.py`。

API 端点(前缀 `/measPredict`,**已无旧的 `/transformer` 段**,定义在 `api/router.py`):
- `POST /measPredict/train`(`router.py:261`)、`POST /measPredict/predict`(`router.py:479`)
- `GET /measPredict/models`(`router.py:552`)、`POST /measPredict/checkData`(`router.py:573`)
- `GET /healthz`(`app.py:27`)
- `API_TRAIN_MODEL_TYPES = frozenset({"informer","autoformer"})`(`router.py:40`)-- **testmodel 不可经 API 训练**(但 `MODEL_PREDICT_FUNCS` 含 testmodel 预测)

## 代码规范

- **ruff**:line-length 100,`target-version = "py312"`。提交前 `ruff check`。
- 文件普遍 `from __future__ import annotations`,类型注解用 `dict[str, Any]` / `list[str]` 等内置泛型,新代码保持一致。
- 注释/文档用中文。
- `[tool.pytest.ini_options]`(`pyproject.toml:184`)注册 `integration_db` marker 并默认排除;默认测试由 `tests/conftest.py` 的 `isolate_default_tests_from_network` fixture 断网。

## 代码布局

```
src/valeo_pdm/
├─ cli.py                 # CLI 入口
├─ paths.py               # 路径辅助
├─ api/                   # FastAPI(app.py / router.py)
├─ training/
│  ├─ registry.py         # TRAINER_IMPORTS 常量表 + get_trainer()
│  ├─ entrypoints.py      # informer/autoformer 薄包装,延迟 import do_training(使探针无需 torch)
│  └─ from_config.py      # 配置驱动训练编排 + list_available_configs + _format_metrics_desc
├─ transformer/
│  ├─ config.py           # 配置加载统一入口(YAML/DB)
│  ├─ config_resolution.py# 优先级合并:DB>YAML>defaults(get_model_block/apply_defaults/resolve_model_config)
│  ├─ data.py             # 统一清洗 clean_and_resample_timeseries + 窗口校验/切分/归一化
│  ├─ artifacts.py        # checkpoint 路径解析
│  ├─ predict.py          # 推理 + 残差注入
│  ├─ train_informer.py   # Informer 训练(testmodel 训练也走这里 do_training)
│  ├─ train_autoformer.py # Autoformer 训练(仅 CSV)
│  ├─ train_testmodel.py  # SimpleGRUForecast 定义(供 testmodel 预测用)
│  └─ models/             # informer.py / autoformer.py
└─ db/                    # data_reader.py(PG/SQLServer)、train_status.py(状态回写)
configs/                  # model_registry.yaml + *_config.json(.example 入库)
tests/                    # pytest 套件 + conftest 网络隔离 + fixtures + integration/
```

**数据流**:原始数据(PG/SQLServer/CSV)→ `clean_and_resample_timeseries`(列名规范化→丢坏行→按 freq 桶均值聚合,**不插值**)→ 滑窗(seq_len+pred_len)→ 80/10/10 顺序切分 → 训练集 mean/std 归一化 → DataLoader → Informer/Autoformer(MSE+Adam+ReduceLROnPlateau+EarlyStopping,`set_seed(42)`)→ checkpoint(.pt+mean/std+args)+ loss_curve.png + forecast.png → 推理取最新 seq_len 点 → 反归一化 → (可选)残差注入 → JSON 响应。

## 给 Agent 的注意事项(踩坑清单)

1. **`testmodel` 路由特殊**:registry(`registry.py:11`)把 `testmodel` **训练**指向 `train_informer.do_training`(不是 `train_testmodel.py`);`train_testmodel.py` 提供 `SimpleGRUForecast`,被 `predict.py:16` 用于 testmodel **预测**。改任一侧别混淆。
2. **配置三层合并**:`model_registry.yaml`(基线)← SQL Server 表 `mom_bas_ai_model_config`(整块覆盖)← API `ParamArr`(字段覆盖)。合并逻辑在 `transformer/config_resolution.py`,只有 `model_type/freq/days_back/source` 四个字段能从顶层 `defaults` 继承。
3. **预测不可复现**:`predict.py:71` `np.random.normal` 残差注入,`predict.py` 无 `set_seed`,与训练侧 `set_seed(42)` 矛盾。需可复现时要处理。
4. **训练管线仍重复**:`build_dataloaders/train_one_epoch/evaluate` 在 `train_informer.py` 与 `train_autoformer.py` 各一份;`set_seed` 三处重复;可视化重复。改一处注意同步另一处,或抽公共基类。
5. **xlstm/timellm 注册但无实现**:`registry.py:13-14` 注册,`train_xlstm.py`/`time_llm/` 文件不存在,调用会 `ModuleNotFoundError`。`pyproject` 已装 `xlstm` 依赖但无训练代码。
6. **Autoformer 仅支持 CSV**(`train_autoformer.py:135`)。
7. **清洗不插值**:`clean_and_resample_timeseries` 只丢坏行 + 桶均值聚合;旧 `data_reader.py:preprocess_timeseries` 的线性插值未被调用。数据稀疏会直接丢窗口。
8. **硬编码 SQL 表名**:`data_reader.py:92`(`mom_bas_ai_model_train_data`)、`config.py:118`(`mom_bas_ai_model_config`)、`train_status.py:118`。改表结构要改代码。
9. **有完整测试套件**(旧文档“无测试”已失效):改 `data.py`/`config_resolution.py`/registry/onboard skill 后跑对应 `tests/test_*.py`;API 启动链由 `tests/test_api_startup.py` 守护;默认断网,真实库测试走 `integration_db` marker 并需授权环境变量。
10. **API 端点已去 `/transformer` 段**:现为 `/measPredict/train` 等,别再用旧路径。

## 新增模型类型 / 新增场景

按 `README.md` 的「如何新增模型类型」「如何新增场景」操作。新增模型类型要点:① 实现 `do_training(**kwargs)`(放 `transformer/train_<name>.py` 或 `training/entrypoints.py` 薄包装,返回 `best_path/best_val/test_loss/plots`);② 在 `transformer/predict.py` 加预测函数并在 `api/router.py:MODEL_PREDICT_FUNCS` 注册,若要支持 API 训练还需加入 `API_TRAIN_MODEL_TYPES`;③ 在 `training/registry.py:TRAINER_IMPORTS` 注册 `model_type -> (module, attribute)`;④ `configs/model_registry.yaml` 设 `model_type` 与 `train_params`。
