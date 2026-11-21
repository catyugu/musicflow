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
| case4 | Bach, Cello Suite No.1 BWV 1007 前奏曲 | 大提琴独奏（1 件，1 行谱表） | 2 页，矢量 | 139.0 s，48 kHz 立体声，192 kbit/s | 102.9 s，704 音符 | 1 个 part，42 小节，704 音符 |

case4 是唯一的单谱表案例：谱面每行只有一行谱表，没有大谱表可供找小节线，是单谱表路径的回归目标。

## 与时间同步相关的特征

| case | 录音来源 | 录音/谱面时长 | 反复 | 变速 | 谱面与录音的音符差 |
| --- | --- | --- | --- | --- | --- |
| case1 | 记谱软件导出，与谱面严格同步 | 1.01 | 无反复记号 | 有速度标记 | 0 |
| case2 | 真实录音（Musopen，Public Domain Mark 1.0） | 1.67 | 无反复记号 | 段落速度与性格变化 | 3 |
| case3 | 由谱面导出 MIDI 后渲染 | 1.01 | 1 条反复指令（第 1–139 小节，四行谱各印一次），MIDI 已展开 | 12 个 tempo 事件，147–165 BPM | 3506（1779 为反复展开的第一遍，1727 为 93 处 tremolo 在 MIDI 中被拆成重复音） |
| case4 | 真实录音（classicals.de，Philip Milman 项目署名，CC BY） | 1.35 | 无反复记号，前奏曲一气呵成 | 自由速度，无明显恒速段 | 0（MIDI 由同一 MusicXML 导出） |

case2 的谱面 MIDI 按记谱软件的默认速度导出，录音时长是它的 1.67 倍；case4 同理（1.35 倍）。case3 同一小节在录音中出现两次，因为书面反复被展开；case1 与 case3 的录音/谱面比值接近 1。

case3 的反复展开量：书面 314 小节、4158 个发声音符，展开后 453 个演奏小节、5937 个发声音符，`total_q` 从 942 变为 1359（比值 1.4426）。MIDI 的最后 tick 为 651360，即 1359 个四分音符，与展开结果一致。

## 来源与许可

| case | 来源 | 许可 |
| --- | --- | --- |
| case1 | 曲作者 catyugu 本人提供（原创作品） | 未附加开源许可，随本仓库公开 |
| case2 | [Schubert Winterreise Dataset 2.1](https://zenodo.org/records/10839767)（doi:10.5281/zenodo.10839767） | CC BY 3.0；HU33 录音为 Public Domain Mark 1.0 |
| case3 | [OpenScore String Quartet Corpus 3.0.0](https://github.com/OpenScore/StringQuartets)（doi:10.5281/zenodo.21862554） | CC0 1.0 |
| case4 | [Tobis Notenarchiv](https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/chamber-music/suites-for-solo-cello/) 的 BWV 1007 排版（取第 1–2 页与其第 1–42 小节的 MusicXML）；录音 [classicals.de](https://www.classicals.de/bach-cello-suites) | 谱面与 MusicXML CC BY-NC 4.0；录音 CC BY（文件 ID3 与下载页链接写 3.0，页面许可栏写 4.0，以 3.0 保守执行；署名 Philip Milman 与 classicals.de） |

引文：Weiß 等，*Schubert Winterreise Dataset*，ACM JOCCH 14(2), 2021；Gotham & Jonas，*The OpenScore String Quartet Corpus*, 2026。

数据集内的 SC06 录音虽为 CC BY 3.0，但禁止以修改形式再分发，因此没有进入语料。

## 已知限制

- case1 的 PDF 是记谱软件的矢量导出：这是一首用户原创曲，没有可扫描的出版谱。
- case2 的 PDF 图像只有 650×932 像素（约 96 dpi），是数据集本身的分辨率上限。
- case3 的 PDF 是 OpenScore 的数字排版，不是扫描件。Schubert D.703 在 Archive.org 只有一个 1840 年代的合订本，内容是按分谱印刷的四个声部，没有全谱扫描。
- `case3.mxl` 的 `META-INF/container.xml` 不带 XML 命名空间（与 `case1.mxl` 一致），`case3.musicxml` 是该容器 rootfile 的明文版本。
- `case4` 的 PDF 与 MusicXML 是整部组曲（6 页、220 个编号小节）的前 2 页与前 42 小节，即第一乐章前奏曲；其余乐章不在语料内，所以本案例只覆盖单谱表、无反复的连续演奏。
- `case4.mid` 由 `case4.musicxml` 导出，按记谱速度只有 102.9 s，不是 139.0 s 的录音本身；因此它不能当演奏真值，验收工具的 MIDI 真值判定按"不是同一演奏"记录原因后跳过它（case2 因小节网格不同同样跳过）。
- `case4` 的前奏曲全是十六分音符的分解和弦，符干与和弦很多；单谱表小节线检测的判据是"笔画止于谱表线，还是越过去接上符头/符尾"，阈值 0.35 个谱表间距。本例实测阈值两侧都有余量：被留下的小节线越界量最大 0.08 个间距（矢量页 0.00），被拒的符干最小 0.54 个间距，页内外的连音线之类独立笔画不受影响；阈值本身由单元套件里的合成页覆盖。
- `case2.mid` 没有时间签名元事件，按 4/4 推断出 62 小节（书面谱是 6/8、82 小节、246 个四分音符）；对齐不受影响，但小节网格与印刷谱对不上，谱行绑定不会按比例摊派：`analyze` 以 `AnalysisError`（退出码 5）拒绝，需要 `--staffs` 给出每个谱行的页范围。
- `case2.musicxml` 是"演奏展开"后的转写：partitura 会把部分和弦音的时值截到下一个实际起音，且有一处连音线（`mf-001797`）因起点稍晚而没有合并。
- `case3.musicxml` 中 93 处 `<tremolo>` 在 MIDI 里被实现为重复音，因此 MIDI 的音符数比书写谱多，不能直接用音符数比较两者。
