# 测试数据（V2 / V3）

`test_cases/` 是纯输入目录：每个 case 目录里只有五类文件，没有清单、标注或中间产物。**一个 case 对应一首原始乐曲**，同一 case 的五类文件描述同一首曲子。

```text
test_cases/
  caseN/
    caseN.mp3        录音
    caseN.pdf        原始谱面
    caseN.mid        MIDI
    caseN.musicxml   未压缩 MusicXML
    caseN.mxl        压缩 MusicXML
```

## 内容与规模

| case | 作品 | 编制 | 谱面 PDF | 录音 | MIDI | MusicXML |
| --- | --- | --- | --- | --- | --- | --- |
| case1 | MuseScore 用户作品《真的很流行の小曲》 | 钢琴（1 件） | 11 页，矢量 | 232.7 s，44.1 kHz 立体声，192 kbit/s | 229.6 s，2500 音符 | 2 个 part，296 小节，2500 音符 |
| case2 | Schubert, Winterreise D.911 No.5 "Der Lindenbaum" | 声乐 + 钢琴（2 件，3 行谱表） | 4 页，96 dpi 扫描 | 274.5 s，22.05 kHz 单声道，128 kbit/s | 163.9 s，1638 音符 | 2 个 part，164 小节，1641 音符 |
| case3 | Schubert, String Quartet D.703 "Quartettsatz" | 弦乐四重奏（4 件，含中音谱号） | 8 页，矢量 | 518.3 s，44.1 kHz 立体声，128 kbit/s | 515.5 s，7664 音符 | 4 个 part，1256 小节，4158 音符 |

## 与时间同步相关的特征

| case | 录音来源 | 录音/谱面时长 | 反复 | 变速 | 谱面与录音的音符差 |
| --- | --- | --- | --- | --- | --- |
| case1 | 记谱软件导出，与谱面严格同步 | 1.01 | 无反复记号 | 有速度标记 | 0 |
| case2 | 真实录音（Musopen，Public Domain Mark 1.0） | 1.67 | 无反复记号 | 段落速度与性格变化 | 3 |
| case3 | 由谱面导出 MIDI 后渲染 | 1.01 | 1 条反复指令（第 1–139 小节，四行谱各印一次），MIDI 已展开 | 12 个 tempo 事件，147–165 BPM | 3506（1779 为反复展开的第一遍，1727 为 93 处 tremolo 在 MIDI 中被拆成重复音） |

case2 的谱面 MIDI 按记谱软件的默认速度导出，录音时长是它的 1.67 倍。case3 同一小节在录音中出现两次，因为书面反复被展开；case1 与 case3 的录音/谱面比值接近 1。

case3 的反复展开量：书面 314 小节、4158 个发声音符，展开后 453 个演奏小节、5937 个发声音符，`total_q` 从 942 变为 1359（比值 1.4426）。MIDI 的最后 tick 为 651360，即 1359 个四分音符，与展开结果一致。

## 来源与许可

| case | 来源 | 许可 |
| --- | --- | --- |
| case1 | 曲作者 catyugu 本人提供（原创作品） | 未附加开源许可，随本仓库公开 |
| case2 | [Schubert Winterreise Dataset 2.1](https://zenodo.org/records/10839767)（doi:10.5281/zenodo.10839767） | CC BY 3.0；HU33 录音为 Public Domain Mark 1.0 |
| case3 | [OpenScore String Quartet Corpus 3.0.0](https://github.com/OpenScore/StringQuartets)（doi:10.5281/zenodo.21862554） | CC0 1.0 |

引文：Weiß 等，*Schubert Winterreise Dataset*，ACM JOCCH 14(2), 2021；Gotham & Jonas，*The OpenScore String Quartet Corpus*, 2026。

数据集内的 SC06 录音虽为 CC BY 3.0，但禁止以修改形式再分发，因此没有进入语料。

## 已知限制

- case1 的 PDF 是记谱软件的矢量导出：这是一首用户原创曲，没有可扫描的出版谱。
- case2 的 PDF 图像只有 650×932 像素（约 96 dpi），是数据集本身的分辨率上限。
- case3 的 PDF 是 OpenScore 的数字排版，不是扫描件。Schubert D.703 在 Archive.org 只有一个 1840 年代的合订本，内容是按分谱印刷的四个声部，没有全谱扫描。
- `case3.mxl` 的 `META-INF/container.xml` 不带 XML 命名空间（与 `case1.mxl` 一致），`case3.musicxml` 是该容器 rootfile 的明文版本。
- `case2.mid` 没有时间签名元事件，按 4/4 推断出 62 小节（书面谱是 6/8、82 小节、246 个四分音符）；对齐不受影响，但 MIDI-only 的谱行绑定只能按比例映射。
- `case2.musicxml` 是"演奏展开"后的转写：partitura 会把部分和弦音的时值截到下一个实际起音，且有一处连音线（`mf-001797`）因起点稍晚而没有合并。
- `case3.musicxml` 中 93 处 `<tremolo>` 在 MIDI 里被实现为重复音，因此 MIDI 的音符数比书写谱多，不能直接用音符数比较两者。
