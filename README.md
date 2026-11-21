# musicflow

将同一首曲子的 PDF 乐谱、MusicXML/MXL 乐谱和演奏录音合成为谱面与录音同步的 MP4 视频。

- `switch` 从原始 PDF 裁切谱行，按演奏顺序切换显示。
- `flow` 用 Verovio 生成连续谱面，让谱面滚动经过固定的阅读线。

两种模式共用同一条演奏路线，并以原录音为时间基准。

## 安装与运行

使用 `uv` 安装 Python 3.12，并确保能通过 `PATH` 调用 FFmpeg。运行验收测试还需要 ffprobe。
Verovio 和 resvg 随 Python 依赖一并安装，视频编码在 CPU 上完成。

在仓库根目录运行以下 PowerShell 命令。

```powershell
uv sync
$score = @("--pdf", "test_cases/case1/case1.pdf", "--audio", "test_cases/case1/case1.mp3", "--symbolic", "test_cases/case1/case1.mxl")

uv run musicflow analyze @score --with-flow --json
uv run musicflow render @score --mode switch --out build/case1.switch.mp4
uv run musicflow render @score --mode flow --preview 10 --out build/case1.preview.mp4
uv run musicflow batch test_cases --out build/corpus
```

视频默认分辨率为 `1920x1080`，帧率为每秒 `30` 帧，阅读线位于距画面左缘 `0.35` 倍画面宽度处。
可分别用 `--size`、`--fps` 和 `--line-x` 调整。`batch` 对每个案例只做一次分析，供 `--modes` 选定的
渲染模式共用。`--jobs` 设置同时处理的案例数上限。

## 人工校正与输出

| 参数 | JSON 文件格式与用途 |
| --- | --- |
| `--route` | `{"spans": [[first_measure, last_measure], ...]}` 按演奏顺序列出小节范围。 |
| `--staffs` | `{"systems": [[system_index, first_measure, last_measure], ...]}` 指定 PDF 谱行对应的小节范围。 |
| `--anchors` | `{"anchors": [[q, seconds], ...]}` 指定演奏谱位对应的录音时间。谱位以四分音符为单位，录音时间以秒为单位。 |

小节和谱行均从 1 开始编号。`report.json` 记录各项检查结果、阶段耗时和待确认的问题。
`analyze --debug-images` 额外生成标注了检测结果的 PDF 调试图。报告和可复用的符号谱、PDF 检测、
对齐与连续排版结果默认保存在 `build/` 下以 PDF 文件名（去掉扩展名）命名的目录中，可用 `--workdir`
指定其他目录。

## 测试与验收

```powershell
uv run pytest
uv run ruff check .
uv run pytest tools/acceptance.py -s --log-cli-level=INFO
```

验收测试将四个案例分别以两种模式渲染为 1080p、每秒 30 帧的视频。加上 `-k case4` 可单独运行
第四个案例。验收产物保存在 `build/acceptance/`。

[设计](doc/design.md) · [需求](doc/PLAN.md) · [测试数据与许可](doc/TEST_DATA.md) · [验收记录](doc/ACCEPTANCE.md)
