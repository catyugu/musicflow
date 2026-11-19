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

全部位于 `musicflow/model.py`，都是冻结 dataclass，`to_dict()/from_dict()` 即 JSON 产物格式。
核心不变量写进了 `__post_init__` 的 `assert`：内部实现违约就是 bug，不当作数据错误处理。

| 类型 | 语义 | 关键不变量 |
| --- | --- | --- |
| `Rect` | 轴对齐矩形 | `x1 >= x0`、`y1 >= y0` |
| `PartInfo` | 声部与谱表数 | `staves >= 1` |
| `MeasureInfo` | 书写小节起点与长度 | `start_q` 严格递增 |
| `Repeat` | 书面反复/跳转指令 | 小节号在谱面范围内 |
| `NoteEvent` | 书写音符 | `duration_q > 0`、`pitch` 为 MIDI 音高 |
| `Score` | 书写谱面 | 音符按 onset 有序、小节存在、`total_q` 为正 |
| `MeasurePass` | 演奏中的一次小节出现 | 与 `Route` 一起检查有序、连续 |
| `PerformanceNote` | 演奏中的音符 | 按 onset 有序，回指 `note_id` 与 `pass_index` |
| `Route` | 演奏路线 | pass 起点递增、`total_q` 为正 |
| `Anchor` / `Alignment` | 谱位-时间对应 | 两者都严格递增，至少两个锚点 |
| `System` / `StaffLayout` | PDF 谱行 | `index` 从 1 连续编号，`page` 存在 |
| `Tile` / `FlowLayout` | 排谱长条与瓦片 | 至少一个瓦片、至少一个小节区间 |
| `VideoSpec` | 画布与配色 | 画布非空、阅读线比例在 (0,1) 内 |

`Alignment.time_at(q)` 是唯一的谱位→时间入口，分段线性、两端钳制。渲染层不得自己再积一次时间，
也不得累计帧间位移：每帧状态由绝对源时间算出。

## 三、管线与产物

一次 `analyze` 在 workspace 下产生下列步骤目录，`render` 只读它们。步骤目录 `payload.json` 存
`to_dict()` 结果，二进制副产物（页面位图、长条瓦片）与 `payload.json` 同目录。

| 步骤 | 输入 | 产物 | 说明 |
| --- | --- | --- | --- |
| `symbolic` | 符号谱 | `Score` + `written.musicxml` | `written.musicxml` 是注入 id 后的书写谱，供 Verovio 使用 |
| `route` | `symbolic` | `Route` | 反复展开；默认书写顺序，MIDI 不做推测性展开 |
| `staffs` | PDF + `symbolic.staff_count` | `StaffLayout`（未绑小节） | 谱行检测；同时写诊断位图 |
| `staffs-bound` | `staffs` + `symbolic` | `StaffLayout`（含 `first/last_measure`） | 小节线计数 → 小节区间 |
| `align` | `symbolic` + `route` + 音频 | `Alignment` + 报告 | 谱位→时间锚点、残差报告、待确认项 |
| `flow` | `symbolic.written.musicxml` + 目标高度 | `FlowLayout` + 瓦片 PNG | 仅 MusicXML/MXL |
| `render-switch` / `render-flow` | 上述 + `VideoSpec` | MP4 分块与成片 | 分块可续渲 |

缓存键 = 输入文件 sha256 + 影响该步骤的参数 + 工具版本。`workdir.Step.run(key, compute)` 负责命中与
落盘，步骤内部不判断缓存。

## 四、模块接口

包内每个模块都是库：不配置 logging、不 print、不读环境变量。唯一的入口是 `musicflow.cli`。

```python
# musicflow.symbolic
def load_score(path: Path, *, written_path: Path | None = None) -> Score
def build_route(score: Score, *, override: Path | None = None) -> Route

# musicflow.pdf
def detect_layout(pdf: Path, *, expected_staves: int | None, debug_dir: Path) -> StaffLayout
def bind_measures(layout: StaffLayout, score: Score) -> StaffLayout

# musicflow.align
def align(
    score: Score,
    route: Route,
    audio: Path,
    *,
    work_dir: Path,
    constraints: Path | None = None,
) -> Alignment

# musicflow.flow
def build_layout(
    score: Score,
    written_musicxml: Path,
    *,
    work_dir: Path,
    target_content_height: int,
    max_strip_width: float,
) -> FlowLayout

# musicflow.render
def render_switch(
    *, score: Score, route: Route, layout: StaffLayout, alignment: Alignment,
    spec: VideoSpec, out: Path, work_dir: Path, preview: float | None,
) -> Path
def render_flow(
    *, score: Score, route: Route, alignment: Alignment, flow: FlowLayout,
    spec: VideoSpec, out: Path, work_dir: Path, preview: float | None,
) -> Path
```

`preview` 是秒数（只渲前 N 秒），用于快速迭代；分块编码必须在 `work_dir` 里可续。

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
  `dtw`（密集 DTW 路径）、`manual`（已审约束）、`end`。
- 无法可靠确定的片段写进 `Alignment.unconfirmed`，并在报告里给出残差分布；禁止把低置信度结果当成
  成功静默输出。
- 谱面与录音实质不符（音符数、段落、时长比例）时，`analyze` 必须在报告里列出，且不得自动改写谱面。

## 七、错误、日志与退出码

| 退出码 | 含义 | 异常类型 |
| --- | --- | --- |
| 0 | 成功 | — |
| 2 | 命令行用法错误 | `UsageError` |
| 3 | 输入缺失/损坏/自相矛盾 | `InputError` |
| 4 | 输入合法但模式不支持（如 MIDI-only + flow） | `UnsupportedInputError` |
| 5 | 必需分析步骤无法给出可信结果 | `AnalysisError` |
| 6 | 合帧或编码失败 | `RenderError` |

预期失败用异常携带退出码，在 `cli.main` 转成一行消息；未预期失败带着 traceback 抛出。日志只配置一次
（`cli._configure_logging`），级别 `INFO`（`-v` 到 `DEBUG`，`-q` 到 `WARNING`），写 stderr；数据性
输出（报告路径、摘要）写 stdout。

## 八、测试策略

- 模块级测试只测语义与不变量，不追求分支覆盖。
- 端到端以 `test_cases/case1..3` 为准：V1 谱行检测与排谱、V2 两种符号输入的 switch、V3 flow、V4 拒绝与
  缓存。
- 同步精度用可量化判据：合成音频（case1、case3）的对齐残差中位数应 ≤ 20 ms；真实录音（case2）应
  ≤ 150 ms，并且报告里残差分布可视检。
- 渲染验收抽样帧，检查阅读线位置、谱面完整、无空白帧、音频时长一致。

## 九、待定

以下事项在实践中确认，不由实现者自行发挥：

1. `switch` 是否需要固定留白比例（当前按画布高度 90% 缩放、垂直居中）。
2. MIDI-only 的小节划分：按时间签名推断，若与 PDF 小节线数不符则按比例映射并标记 `unconfirmed`。
3. 长条瓦片尺寸与 `max_strip_width` 上限（Verovio `pageWidth` 上限 100000）。
