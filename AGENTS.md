# AGENTS

## 工作循环

```powershell
uv sync                 # setup：创建 .venv 并安装锁定依赖
uv run musicflow doctor # 环境自检与语料检查
uv run pytest           # test
uv run ruff check .     # lint（格式化用 ruff format .）
```

`build` 对纯 Python 项目是 no-op，`uv sync` 就是可运行环境的产出。不要用 `pip install`、全局 `npm`、
不带版本号的 `npx`/`uvx` 来补依赖：需要的依赖写进 `pyproject.toml` 并提交 `uv.lock`。

## 代码约定

- 代码内除注释外一律英文：标识符、字符串字面量、日志、断言消息。
- 包内模块是库：只发日志记录，不 `print`、不配置 logging、不读环境变量；出口只有 `musicflow.cli`。
- 公共接口在 `musicflow/model.py`（数据结构）与各子包 `__init__.py`（`__all__`）里显式导出，其余默认私有。
- 不做向后兼容：接口、产物字段直接用新定义替换，不留别名、兼容层、旧路由。
- 内部不变量用 `assert`（违约即 bug）；外部输入用 `musicflow.errors` 里的类型化异常。
- 日志记录边界事件（文件、数量、耗时、结论），不是逐行旁白。
- 注释只写代码本身读不出来的信息（为什么、约束来源），不写复述。

## 模块边界

见 [doc/design.md](doc/design.md) 第四节的接口表。跨模块只经 `model.py` 的数据结构和这些函数通信；
不要在别人的模块里加自己的逻辑，也不要为一个模块的需求去改另一个模块的产物字段（先改契约）。

流水线只缓存昂贵步骤：`symbolic`、`pdf`、`align`、`flow`（见 design.md 第三节）。`route` 与小节绑定
在内存里算，不建目录、不加缓存键；改任何一个步骤的行为都必须递增 `pipeline.PIPELINE_VERSION`。渲染
只有分块续渲，不是缓存步骤。

## 语料

`test_cases/` 是纯输入、只读：不加清单、标注、中间产物、清理脚本。需要中间产物时写到 `build/`。
`build/` 已在 `.gitignore` 中。

## 测试

- `uv run pytest` 跑唯一一套单元测试：全部使用合成输入或内存中构造的数据，不依赖 `test_cases/`，整套 10 s
  以内跑完。这是编辑循环，每改一处就跑一次。
- 只写能验证语义的测试：模型编解码与不变量、路线展开与覆盖文件校验、合成布局上的小节绑定、合成数组上的
  对齐特征/DTW/snap、内联 SVG 几何、不经 ffmpeg 的合帧，以及用假步骤验证的流水线缓存/报告/batch/CLI。
  不要为覆盖率凑断言。
- 语料级验证只在 `tools/acceptance.py`：它逐 case 复用一次分析、渲染两种模式，判定 `report.json` 的 checks、
  对齐证据门槛与 MIDI 真值误差。提交前与验收跑它，不在编辑循环里反复跑。
- 精度类断言用可量化判据（覆盖率、残差中位数、p50/p90、时长差、帧数），不用"看起来对"。

## 文档

- 需求以 [doc/PLAN.md](doc/PLAN.md) 为准，设计契约以 [doc/design.md](doc/design.md) 为准。
- 仓库 Markdown：代码块必须标语言；表格单元格里不要出现竖线字符，公式不要写进表格。
