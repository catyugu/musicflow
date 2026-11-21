# 设计：数据契约、模块边界与验收

本文固定 `musicflow` 的基础设施决策：坐标系、数据结构、模块接口、产物格式、错误与缓存契约。
需求见 [PLAN.md](PLAN.md)，语料见 [TEST_DATA.md](TEST_DATA.md)。基础决策一旦在实践中失效，只重开
对应的一条，不做整层重构。

## 一、坐标系与命名约定

| 概念 | 单位 | 说明 |
| --- | --- | --- |
| `q` | 四分音符数 | 谱面位置。`Score` 用书写位置（文档顺序），`Route`/`Alignment` 用演奏位置（反复展开后） |
| 小节 | 整数，1 起 | MusicXML 文档顺序的小节序号 |
| 时间 | 秒 | 录音时间轴。原音频是唯一主时钟，任何步骤都不改音频时长或采样率 |
| PDF 几何 | PDF point | `StaffLayout`、`System.bbox`、`System.staves` |
| 排谱几何 | Verovio SVG unit | `FlowLayout.note_x`、`FlowLayout.strip_*`；瓦片图像 1 px = 1 unit |
| 帧窗口 | 视频像素 | 渲染阶段的裁切与合成 |

音符标识：`mf-NNNNNN`，六位十进制，按 MusicXML 文档顺序给**每个** `<note>` 元素编号（含休止、
装饰音、cue）。编号由 `musicflow.scorexml.assign_note_ids` 写入 `<note id="...">`，Verovio 会把它带到
SVG（`id="mf-000123"`）与 timemap 中，这是流动模式把"源音符 → 横坐标"接起来的唯一桥梁。

同一小节的不同演奏次数用 `pass_index`（1 起）区分。对齐与渲染都以 `PerformanceNote` 为准，不直接
用书写音符。

## 二、数据结构

全部位于 `musicflow/model.py`，都是冻结 slots dataclass，核心不变量写进了 `__post_init__` 的
`assert`：内部实现违约就是 bug，不当作数据错误处理。

产物类型的 JSON 格式由字段声明唯一决定。`Payload` 用 `dataclasses.fields` 与
`typing.get_type_hints` 实现一个反射编解码器：`to_dict()` 只写声明过的字段（嵌套产物递归，`tuple`
变数组），`from_dict()` 严格按类型解码——未知键、缺失键、类型不符（`int` 字段收到小数、`float`
字段收到字符串、`Literal` 以外的取值）都报错，因此不存在半填充的旧结构。`VideoSpec` 是配置，只有
`to_dict()`。

每个字段只归一个类型所有：`PerformanceNote` 只记演奏位置，音高、时值、声部与谱表一律通过
`Score.note_by_id()` 查书写音符。

| 类型 | 语义 | 关键不变量 |
| --- | --- | --- |
| `Rect` | 轴对齐矩形 | `x1 >= x0`、`y1 >= y0` |
| `PartInfo` | 声部与谱表数 | `staves >= 1` |
| `MeasureInfo` | 书写小节起点与长度 | `start_q` 严格递增 |
| `Repeat` | 书面反复/跳转指令 | 小节号在谱面范围内 |
| `NoteEvent` | 书写音符 | `duration_q > 0`、`pitch` 为 MIDI 音高 |
| `Score` | 书写谱面 | 音符按 onset 有序、小节存在、`total_q` 为正；音高/时值/声部/谱表的唯一来源 |
| `MeasurePass` | 演奏中的一次小节出现 | `start_q` 接续前序 pass 的长度累加 |
| `PerformanceNote` | 演奏中的音符位置 | 按 onset 有序，回指 `note_id` 与 `pass_index`，不含音高与时值 |
| `Route` | 演奏路线 | `total_q` 为正且等于各 pass 长度之和，每个音符落在已声明的 `(measure, pass_index)` 上 |
| `Anchor` / `Alignment` | 谱位-时间对应 | 两者都严格递增，至少两个锚点 |
| `System` / `StaffLayout` | PDF 谱行 | `index` 从 1 连续编号，`page` 存在 |
| `Tile` / `FlowLayout` | 排谱长条与瓦片 | 至少一个瓦片、至少一个小节区间 |
| `VideoSpec` | 画布与配色 | 画布非空、阅读线比例在 (0,1) 内 |

`Alignment.time_at(q)` 是唯一的谱位→时间入口，分段线性、两端钳制。渲染层不得自己再积一次时间，
也不得累计帧间位移：每帧状态由绝对源时间算出。

## 三、管线与产物

一次 `analyze` 只为昂贵步骤在 workspace 下建目录，`render` 只读它们。步骤目录里 `payload.json` 存
`to_dict()` 结果，该步骤写下的二进制副产物（`written.musicxml`、长条瓦片、`--debug-images` 的诊断位图）
与 `payload.json` 同目录。`route` 与小节绑定不在表内：它们从已加载的数据在内存里算出
（`symbolic.build_route`、`pdf.bind_measures`），没有目录，也没有缓存键。

| 步骤 | 输入 | 产物 | 说明 |
| --- | --- | --- | --- |
| `symbolic` | 符号谱 | `Score` + `written.musicxml` | `written.musicxml` 是注入 id 后的书写谱，供 Verovio 使用 |
| `pdf` | PDF + `Score.staff_count` | `StaffLayout`（未绑小节） | 谱行检测；只有 `--debug-images` 才写 `debug/` 诊断位图 |
| `align` | 符号谱 + 演奏路线 + 音频 | `Alignment` + 对照证据 | 写 `report.json` 与 `alignment.png` |
| `flow` | `written.musicxml` + 目标高度 + 纸色 | `FlowLayout` + 瓦片 PNG | 仅 MusicXML/MXL |

缓存键（`workdir.key`）= 该步骤真正读的文件内容摘要 + 影响该步骤的参数 + 工具版本。`analyze` 每次
运行只对每个输入文件算一次 sha256（`workdir.digest_file`），同一批摘要既喂给步骤键也写进报告；键里
只放该步骤读的依赖：

| 步骤 | 文件摘要 | 参数 |
| --- | --- | --- |
| `symbolic` | 符号谱 | `provider`、`partitura`/`lxml` 版本、`PIPELINE_VERSION` |
| `pdf` | PDF | `expected_staves`、`debug_images`、`pymupdf`/`opencv-python-headless` 版本 |
| `align` | 音频（给了 `--anchors` 时再加它） | `Score` 载荷 hash、`Route` 载荷 hash、`librosa`/`numpy` 版本 |
| `flow` | `written.musicxml` | `Score` 载荷 hash、目标内容高度、纸色、`verovio`/`resvg-py` 版本 |

`workdir.Step.run(key, compute, validate=...)` 负责命中与落盘，步骤内部不判断缓存。`validate` 拿到已
存载荷并回答"它指名的二进制文件是否都还在"，不在就重算：`symbolic` 查 `written.musicxml`，`flow` 查
每个瓦片，`align` 查 `report.json`。`pipeline.PIPELINE_VERSION` 在所有键的参数里；任何步骤的行为改变
都必须递增它（当前 6），否则旧产物会被当作仍然有效。

渲染不是缓存步骤：`render-switch/` 放页面位图与分块 MP4，`render-flow/` 放分块 MP4，`flow/` 只放
排谱产物。分块续渲（`render/encoder.py`）是唯一的渲染缓存：分块名带参数摘要，复用时用 ffprobe 复核
帧数。`encoder` 每完成一次编码把耗时（总时长、帧合成、喂 ffmpeg、封装）与帧数、分块数写进模块级
`LAST_STATS` 并打一条 INFO 日志，验收工具读它填 `acceptance.json`。

### 报告产物

`analyze` 写一个 `<workspace>/report.json`，把 provenance、摘要、检查与产物路径放在同一个文件里：

| 键 | 内容 |
| --- | --- |
| `inputs` | 每个输入文件的绝对路径、字节数、sha256 |
| `tools` | 各阶段依赖的版本；`pipeline` 是 `PIPELINE_VERSION` |
| `spec` | `VideoSpec` |
| `score` | provider、标题、谱表数、声部、小节数、书写音符数、`total_q`、`unconfirmed` |
| `route` | 来源、pass 数、音符数、`total_q`、`unconfirmed` |
| `layout` | 页数、谱行数、已绑定谱行数、`unconfirmed`；本次没算布局时为 `null` |
| `alignment` | 方法、锚点数、录音时长、隐含 BPM、相似度、覆盖率、残差中位数、`unconfirmed` |
| `flow` | 长条尺寸、瓦片数、scale、音符覆盖率；本次没排谱时为 `null` |
| `checks` | `checks()` 的结果，见下 |
| `artifacts` | `align_report`、`alignment_figure`、`debug_images`（仅 `--debug-images`）、`videos`（按 mode 记录已渲染的成片） |
| `timings` | 每个实际运行的阶段耗时，加 `total` |

`checks(analysis)` 只检查真正产出过的证据：算了布局才有 `staffs_per_system` 与 `measure_count`；
`tempo` 与 `alignment_confidence` 恒有；有 flow 布局才有 `flow_note_coverage`。

详细的对照证据留在 `align/report.json` 与 `align/alignment.png`；合并报告只引用它们的路径并给摘要
数字，不复制统计量。`cli analyze` 打印的摘要与 `--json` 输出都来自这个文件。

## 四、模块接口

包内每个模块都是库：不配置 logging、不 print、不读环境变量。唯一的入口是 `musicflow.cli`。

| 模块 | 入口 | 语义 |
| --- | --- | --- |
| `pipeline` | `analyze` | 按 `with_pdf`/`with_flow`/`debug_images` 运行缓存的昂贵步骤，返回 `Analysis` 并写 `report.json` |
| `pipeline` | `render` | 从 `Analysis` 渲染一个模式；没给 `analysis` 时先跑该模式需要的阶段，并断言它带齐该模式要的数据 |
| `pipeline` | `batch` | 单位是案例：一个 workspace、一次分析、每个模式一条记录；`jobs` 只在案例之间并行 |
| `symbolic` | `load_score` | 读 MusicXML / MXL / MIDI 为 `Score`，MXL 就地解包 |
| `symbolic` | `build_route` | 按书面反复展开演奏顺序，override 文件优先 |
| `pdf` | `detect_layout` | 以光栅检测谱行、谱表与小节线；`debug_dir` 非空才写诊断位图，`None` 不建目录 |
| `pdf` | `bind_measures` | 严格绑定小节范围：印刷小节号优先，无文本层时小节线计数必须恰好铺满全谱，否则 `AnalysisError` 要求 `--staffs` 修正 |
| `align` | `align` | 产出谱位到录音时间的锚点与对照证据 |
| `flow` | `build_layout` | 把书写谱排成一整条 system 并栅格化成瓦片 |
| `render` | `render_switch` | 用原 PDF 位图按谱行切谱，分块写进 `work_dir` |
| `render` | `render_flow` | 从 `tiles_dir` 读瓦片，把长条沿阅读线平移，分块写进 `work_dir` |

`preview` 是秒数（只渲前 N 秒），用于快速迭代。`switch` 消费原 PDF 位图与录音，`flow` 消费 `flow`
步骤写下的瓦片与录音、不读 PDF；两条路径都在调用处显式命名输入，不从别的字段推断（missing →
`InputError`）。

`flow` 的瓦片写在 `flow` 步骤目录，`Tile.path` 相对该目录，所以 `render_flow` 的 `tiles_dir` 是
`<workspace>/flow/`；它自己的分块落进 `<workspace>/render-flow/`。排谱产物与渲染缓存因此不再共用目录。

## 五、视频规格

| 项 | 默认值 | 说明 |
| --- | --- | --- |
| 画布 | 1920×1080 | `--size WxH` 可改；宽高必须为偶数（yuv420p） |
| 帧率 | 30 | `--fps` |
| 底色 | `#FAF8F1` | 纸色 |
| 阅读线 | 竖线，x = 0.35×宽，颜色 `#287D9F`，3 px | `switch` 不用竖线 |
| 编码 | `libx264`、`crf 18`、`yuv420p`、`+faststart` | 音频 `aac 192k`，时长以音频为准 |
| 帧状态 | 由绝对源时间计算 | 不累计上帧位移，缓动不得改变事件时间 |

`switch`：切谱时刻 = 该谱行首个发声音符的录音时间；画面是原 PDF 位图按谱行裁切后等比缩放、留白居中，
不做重排、不加滤镜。

`flow`：整个曲子排成一条连续系统（`breaks: none`），多声部自上而下，横坐标单调递增。镜头位置
`x(t)` 以（音符时间, 音符横坐标）为锚点做单调三次插值（PCHIP）：只平滑空间运动，不改变任何音符
时间；遇到反复导致的横坐标回跳，该段用线性插值，不做跨跳平滑。

## 六、对齐与诚实边界

- 对齐输出锚点序列，不是每个音符独立测时。锚点来源写进 `kind`：`start`、`onset`（局部音头证据）、
  `manual`（已审约束）、`end`。
- 音头指标取真实起始点；细对齐用 `ATTACK_DECAY` 的衰减窗（1 帧满权，随后两帧 0.5、0.25）加权，平滑
  代价下限 `FINE_MIN_SMOOTH_PENALTY = 0.5`，避免换音高被当成换时间。
- 无法可靠确定的片段写进 `Alignment.unconfirmed`，并在报告里给出残差分布；禁止把低置信度结果当成
  成功静默输出。
- 谱面与录音实质不符（音符数、段落、时长比例）时，`analyze` 必须在报告里列出，且不得自动改写谱面。
- `align/report.json` 是对照证据：`method`、`audio_duration`、`score_total_q`、`implied_bpm`、
  `similarity_mean`、`attack_count`/`snapped_attacks`/`coverage`、`residual_ms`（中位数/p90/最大值）、
  `onset_lag_ms`、`length_ratio`、`relocated_attacks`、`dtw_path_length`、`dropped_anchors`、
  `reviewed_anchors`、`chosen_hypothesis`、逐小节 pass 的时间与 `unconfirmed`。
- `align/alignment.png` 两栏：左栏谱位→时间折线（锚点按 `kind` 标形状，人工锚点另色），右栏每个被接受
  音头的残差（ms）与 ±250 ms 带；两栏都标出覆盖率与残差中位数。

## 七、错误、日志与退出码

| 退出码 | 含义 | 异常类型 |
| --- | --- | --- |
| 0 | 成功 | — |
| 1 | `batch` 至少一个案例失败（逐案例状态见 stdout） | — |
| 2 | 命令行用法错误 | `UsageError` |
| 3 | 输入缺失/损坏/自相矛盾 | `InputError` |
| 4 | 输入合法但模式不支持（如 MIDI-only + flow） | `UnsupportedInputError` |
| 5 | 必需分析步骤无法给出可信结果（含小节绑定无法确定） | `AnalysisError` |
| 6 | 合帧或编码失败 | `RenderError` |
| 7 | 依赖的外部程序缺失（ffmpeg/ffprobe） | `MissingToolError` |

预期失败用异常携带退出码，在 `cli.main` 转成一行消息；未预期失败带着 traceback 抛出。日志只配置一次
（`cli._configure_logging`），级别 `INFO`（`-v` 到 `DEBUG`，`-q` 到 `WARNING`），写 stderr；数据性
输出（报告路径、摘要）写 stdout。

## 八、测试策略

- 模块级测试只测语义与不变量，不追求分支覆盖。
- 端到端以 `test_cases/case1..4` 为准：每个案例只分析一次，两种模式都从这次分析渲染。V1 谱行检测与
  排谱（case4 是单谱表谱面）、V2 两种符号输入的 switch、V3 flow、V4 拒绝与缓存。
- 同步精度用可量化判据：合成音频（case1、case3）的对齐残差中位数应 ≤ 20 ms；真实录音（case2、case4）应
  ≤ 150 ms，并且报告里残差分布可视检。
- 渲染验收抽样帧，检查阅读线位置、谱面完整、无空白帧、音频时长一致；`tools/acceptance.py` 逐案例复用
  一次分析，把成本拆成分析、排谱、合帧、编码四项写进 `acceptance.json`。

## 九、待定

以下事项在实践中确认，不由实现者自行发挥：

1. `switch` 是否需要固定留白比例（当前按画布高度 90% 缩放、垂直居中）。
2. MIDI-only 的小节划分：按时间签名推断。MIDI 若是"已展开反复"的演奏，其小节数与印刷谱不同
   （case3：453 对 314），谱行无法按小节网格对齐——此时不做按比例映射，绑定直接报 `AnalysisError`
   （退出码 5）要求 `--staffs` 修正；这类输入要么改用 MusicXML，要么人工给出每行的小节范围。
3. 长条瓦片尺寸受内部常量 `MAX_STRIP_WIDTH`（Verovio `pageWidth` 上限 100000）约束，超出时降 scale 重排。
