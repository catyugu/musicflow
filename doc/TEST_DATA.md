# 测试数据

`test_cases/` 存放测试输入。每个案例目录对应一首曲子，包含以下五类文件。

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

| 案例 | 作品 | 编制 | 谱面 PDF | 录音 | MIDI | MusicXML |
| --- | --- | --- | --- | --- | --- | --- |
| case1 | MuseScore 用户作品《真的很流行の小曲》 | 钢琴（1 件） | 11 页，矢量 | 232.7 s，44.1 kHz 立体声，192 kbit/s | 229.6 s，2500 个音符 | 2 个分谱，296 小节，2500 个音符 |
| case2 | 舒伯特《冬之旅》D. 911 第 5 首《菩提树》 | 人声与钢琴（3 行谱表） | 4 页，96 dpi 扫描 | 274.5 s，22.05 kHz 单声道，128 kbit/s | 163.9 s，1638 个音符 | 2 个分谱，164 小节，1641 个音符 |
| case3 | 舒伯特《弦乐四重奏乐章》D. 703 | 弦乐四重奏（4 件，含中音谱号） | 8 页，矢量 | 518.3 s，44.1 kHz 立体声，128 kbit/s | 515.5 s，7664 个音符 | 4 个分谱，1256 小节，4158 个音符 |
| case4 | 巴赫《第一号无伴奏大提琴组曲》BWV 1007 前奏曲 | 大提琴独奏（1 件，1 行谱表） | 2 页，矢量 | 139.0 s，48 kHz 立体声，192 kbit/s | 102.9 s，704 个音符 | 1 个分谱，42 小节，704 个音符 |

case4 是唯一的单谱表案例：谱面每行只有一行谱表，没有大谱表可供找小节线，是单谱表路径的回归目标。

## 与时间同步相关的特征

| 案例 | 录音来源 | 录音与谱面时长之比 | 反复 | 变速 | 谱面与录音的音符差 |
| --- | --- | --- | --- | --- | --- |
| case1 | 记谱软件导出，与谱面严格同步 | 1.01 | 无反复记号 | 有速度标记 | 0 |
| case2 | 真实录音（Musopen，Public Domain Mark 1.0） | 1.67 | 无反复记号 | 段落速度与性格变化 | 3 |
| case3 | 由谱面导出 MIDI 后渲染 | 1.01 | 1 条反复指令（第 1–139 小节，四行谱各印一次），MIDI 已展开 | 12 个速度事件，147–165 BPM | 3506（1779 来自反复展开的第一遍，1727 来自 MIDI 将 93 处震音拆成重复音） |
| case4 | 真实录音（classicals.de，Philip Milman 项目署名，CC BY） | 1.35 | 无反复记号，前奏曲一气呵成 | 自由速度，无明显恒速段 | 0（MIDI 由同一 MusicXML 导出） |

case2 的谱面 MIDI 按记谱软件的默认速度导出，录音时长是它的 1.67 倍；case4 同理（1.35 倍）。case3 同一小节在录音中出现两次，因为书面反复被展开；case1 与 case3 的录音/谱面比值接近 1。

case3 的书面谱有 314 小节和 4158 个发声音符。展开反复后得到 453 个演奏小节和 5937 个发声音符，`total_q` 从 942 增至 1359（比值为 1.4426）。MIDI 终点位于第 651360 个计时单位，折合 1359 个四分音符，与展开结果一致。

## 来源与许可

| 案例 | 来源 | 许可 |
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
- `case3.mxl` 和 `case1.mxl` 的 `META-INF/container.xml` 均省略 XML 命名空间。`case3.musicxml` 是 `case3.mxl` 中 `rootfile` 指向的未压缩谱面。
- `case4` 的 PDF 与 MusicXML 是整部组曲（6 页、220 个编号小节）的前 2 页与前 42 小节，即第一乐章前奏曲；其余乐章不在语料内，所以本案例只覆盖单谱表、无反复的连续演奏。
- MIDI 仅用于验收。case1 和 case3 的 MIDI 对应录音演奏，可作为音头时间真值。case2 和 case4 的 MIDI 按记谱速度导出。
- `case2.musicxml` 是按演奏顺序展开后的转写。partitura 会把部分和弦音的时值截到下一个实际起音。有一处连音线（`mf-001797`）的起点稍晚，导致对应音符未合并。
- `case3.musicxml` 中的 93 处震音（`<tremolo>`）在 MIDI 中展开为多个重复音，比较音符数量时须计入这部分差异。
