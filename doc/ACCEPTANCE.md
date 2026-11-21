# 验收记录

```bash
uv run pytest
uv run ruff check .
uv run pytest tools/acceptance.py -s --log-cli-level=INFO
```

语料为四个案例，每例共享一次分析，生成 `switch` 和 `flow` 两部 1920×1080、30 fps 的完整视频。
测试对比 `fa38ba2` 的 PDF 谱行与小节绑定，检查对齐覆盖率、音头残差及 case1 和 case3 的 MIDI 时间真值。
视频检查包括帧数、时长、音轨波形相关性、抽样帧墨迹和 `flow` 阅读线。

基线为 [fa38ba2 的验收记录](https://github.com/catyugu/musicflow/blob/fa38ba2/doc/ACCEPTANCE.md)。

## 本次结果

Linux CPU（Intel Xeon Platinum 8370C），Python 3.12，FFmpeg 7.1.5。44 个单元测试和 Ruff 检查通过。
语料验收覆盖 4 个分析用例和 8 部整片。三例在首轮通过，扫描件 case2 在坐标序列化修复后复测通过。
表格采用每例已完成的最近测量。整理 PR 时最终串行复跑仍在进行。
`src/` 与 `tools/` 从 8,468 行降至 6,322 行，净减 25.3%。

| 案例 | 谱行数 | 覆盖率 | 残差中位数 | MIDI 误差的第 50／90 百分位数 |
| --- | ---: | ---: | ---: | ---: |
| case1 | 42 | 98.97% | 6.45 ms | 3.65 ms / 42.48 ms |
| case3 | 39 | 99.27% | 8.37 ms | 14.12 ms / 39.67 ms |
| case4 | 21 | 98.93% | 6.18 ms | — |
| case2 | 21 | 99.36% | 6.35 ms | — |

| 案例 | 帧数（每模式） | switch 渲染 ms/帧 | flow 渲染 ms/帧 | flow 合成 ms/帧 |
| --- | ---: | ---: | ---: | ---: |
| case1 | 6981 | 7.00 | 15.02 | 2.07 |
| case3 | 15547 | 5.64 | 13.38 | 1.84 |
| case4 | 4169 | 7.72 | 17.89 | 2.62 |
| case2 | 8232 | 5.84 | 11.25 | 1.78 |

渲染耗时涵盖送帧、合成、编码与封装。合成耗时只统计生成画面的时间。
计算 MIDI 误差的百分位数时，先扣除时移的中位数，再取绝对误差。
