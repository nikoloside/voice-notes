# voice-notes

[English](README.md) · **中文** · [日本語](README.ja.md)

![license](https://img.shields.io/badge/license-MIT-blue.svg)
![python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![100% local](https://img.shields.io/badge/100%25-本地%20%2F%20离线-brightgreen.svg)
![whisper](https://img.shields.io/badge/ASR-faster--whisper-orange.svg)
![MCP](https://img.shields.io/badge/MCP-server-8A2BE2.svg)
![i18n](https://img.shields.io/badge/界面-中文%20%7C%20日本語%20%7C%20English-ff69b4.svg)
![platform](https://img.shields.io/badge/平台-macOS%20%7C%20Linux-lightgrey.svg)

![demo](docs/demo-zh.gif)

> 首页 → 某场会议的三层总结 → 跨会议知识图谱。
> _（占位演示数据。[▶ 高清 MP4](docs/demo-zh.mp4)。）_

从 [`type-by-voice`](https://github.com/kotaro-nakata/type-by-voice) 中拆分出来的独立本地语音笔记工具。

它录制或导入音频，用 `faster-whisper` **本地**转写，并生成：

- `transcript.md`
- `notes.md`
- `summary.md`
- `audio.wav`
- `meta.json`

它不做全局按键即说的听写、托盘 UI，也不会把文字粘贴到当前应用——这些仍由
[`type-by-voice`](https://github.com/kotaro-nakata/type-by-voice) 负责。

## 安装

**一键安装。** 它会创建 Python venv、下载本地 Whisper 转写模型、安装
[Ollama](https://ollama.com) + 一个小的本地摘要模型、并写好配置——于是**转写和
摘要都在本地跑，无需云端，开箱即用**。

macOS / Linux：

```bash
git clone https://github.com/nikoloside/voice-notes
cd voice-notes
./install.sh
```

Windows（PowerShell）：

```powershell
git clone https://github.com/nikoloside/voice-notes
cd voice-notes
powershell -ExecutionPolicy Bypass -File install.ps1
```

然后启动 Web 界面：

```bash
./record-notes          # macOS / Linux
.\record-notes.ps1      # Windows
```

配置较低的机器可用更小的模型：
`VOICE_NOTES_WHISPER_MODEL=small ./install.sh`（以及 `VOICE_NOTES_LLM=llama3.2:3b`）。

<details><summary>手动安装（Python 3.11+）</summary>

```bash
cd voice-notes
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./record-notes
```

Layer 2/3 摘要需要 LLM——本地跑 Ollama，或把 `openai_url` 指向一个 OpenAI 兼容
服务（见下文）。
</details>

默认页面是 <http://127.0.0.1:8765>。可以开始/停止录音、上传音频文件、导入
macOS 语音备忘录，并下载生成的文件。界面右上角可切换 **中文 / 日本語 / English**。
生成的笔记就是本地文件夹里的纯 Markdown——你可以放进自己的仓库、分享出去，或用
[MCP 服务](#mcp-服务让-claude-读你的笔记)让 Claude 读取。

三层严格按顺序执行 —— Layer 1 → Layer 2 → Layer 3：

- **Layer 1**：原始分块转写，每段音频完成后实时写入。转写全部完成前不做其它事。
- **Layer 2**：Layer 1 完成后，把**整段**转写一次性交给本地 LLM，生成按主题分组的
  详细全文总结（决策、数字、人名、待办、待确认问题）。转写过长超出模型上下文时，
  内部会切成几个大块再合并——这一步在输出中不可见。
- **Layer 3**：把 Layer 2 的全文总结再精简成一页纸：一句话结论 / 核心要点 /
  Action Todo / Checkpoints。

`summary.md` = Layer 3 一页纸，后面接 Layer 2 全文总结。`notes.md` 展示全部三层。

Layer 2/3 需要 LLM。后端解析顺序（`[summary] backend = "auto"`）：

1. 若设置了 `openai_url` 且可达，使用 OpenAI 兼容服务——例如通过 Tailscale
   连到另一台机器上的 LM Studio（`openai_url = "http://100.x.x.x:1234/v1"`；
   `openai_model` 留空则用服务器返回的第一个模型）。
2. 否则用本地 Ollama（`ollama_url`、`ollama_model`）。
3. 否则回落到基于规则的抽取（质量差很多）。

上传/导入的音频，列表页和会话页会显示转换进度：已转写时长、音频总时长、完成百分比。
实时录音时，进度对比已转写部分与当前已录制时长。

上传/导入的音频按块记录在 `chunks.json`。若转换中断，打开会话点 **Resume conversion**；
已完成的块会复用，只补转缺失的块。整条流水线是分块的：

```text
块 N 音频 -> 原始转写 -> 核心笔记 -> value/todo/checkpoints -> checkpoint
块 N+1 音频 -> 原始转写 -> 核心笔记 -> value/todo/checkpoints -> checkpoint
...
最终转写 -> 最终 summary.md
```

所有转写块完成后才生成最终的 `summary.md`。

## 命令

```bash
./record-notes                    # 启动本地 UI
./record-notes --record           # 在终端录音；回车停止
./record-notes --import file.m4a   # 导入一个音频文件并等待
./record-notes --import file.m4a --language zh
./record-notes --list-devices      # 列出麦克风设备
```

常用覆盖项：

```bash
./record-notes --port 8770 --no-browser
./record-notes --data-dir ~/.local/share/voice-notes/sessions
```

## 知识图谱

每场会议结束后会跑一次知识图谱抽取：同一个 LLM 从笔记里抽出实体
（人物 / 项目 / 机构 / 概念 / 决策 / 待办）及其关系，缓存在会话目录的
`entities.json`。

`/api/graph` 把所有会话的 `entities.json` 聚合成一张图——同名实体跨会议合并，
于是反复出现的项目或人物会成为**枢纽**，连到所有提到它的会议。图谱以通用的
`{nodes, edges}` 契约提供，由内置的 `graph.html` 查看器渲染（任何遵循同一契约的
数据源都能复用它）。

在首页点 **🕸️ 知识图谱**（或访问 `/graph`）打开交互式力导向图：点节点看详情、
双击会议节点打开它、用搜索框聚焦。

为已有会话（在此功能之前生成的）回填实体：

```bash
.venv/bin/python -c "import voice_notes as v, tomllib; \
c=tomllib.load(open('$HOME/.config/voice-notes/config.toml','rb'))['summary']; \
s=v.Summarizer(**{k:c[k] for k in ('backend','ollama_model','ollama_url','openai_url','openai_model','openai_api_key')}); \
import pathlib; [v.build_entities(p,s,force=True) for p in sorted((pathlib.Path.home()/'.local/share/voice-notes/sessions').glob('2026*'))]"
```

## MCP 服务（让 Claude 读你的笔记）

`voice-notes-mcp` 是一个只读的 [MCP](https://modelcontextprotocol.io) 服务（stdio），
把 voice-notes 已生成的笔记暴露给任意 MCP 客户端——Claude Code、Claude Desktop 等。
无需任何服务在运行，它直接读磁盘上的会话目录。工具：

- `list_notes(limit=50)` —— 最新会话的 id、标题、日期、时长、状态
- `read_note(session_id, part="summary")` —— `part` 可为 `summary`（一页纸 + 全文）、
  `one_page`、`full`、`notes`（三层全含）、`transcript`
- `search_notes(query, limit=20)` —— 跨摘要/笔记的关键词搜索
- `knowledge_graph(top=20)` —— 跨会议图谱总览：统计、实体类型，以及被提到最多的
  实体及其所在会议
- `find_entity(name)` —— 查某个人物/项目/概念：说明、被哪些会议提到、关联实体

加到 **Claude Code**：

```bash
claude mcp add voice-notes --scope user -- /绝对路径/voice-notes/voice-notes-mcp
```

加到 **Claude Desktop**（`~/Library/Application Support/Claude/claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "voice-notes": {
      "command": "/绝对路径/voice-notes/voice-notes-mcp"
    }
  }
}
```

它读取与主程序相同的数据目录；用 `--data-dir DIR` 或环境变量
`VOICE_NOTES_DATA_DIR` 覆盖。然后就可以问 Claude：“列出我的语音笔记”、
“读一下昨天那场会议的一页纸总结”、“在我的笔记里搜某个关键词”。

## 配置

首次运行会创建：

```text
~/.config/voice-notes/config.toml
```

默认会话数据存放在：

```text
~/.local/share/voice-notes/sessions/<session-id>/
```

每个会话目录包含 `meta.json`、`transcript.md`、`summary.md`、`notes.md`、
导入音频的 `chunks.json`，通常还有 `audio.wav`。

默认转写语言是中文（`zh`）。对刻意混合语言的音频，用网页上的语言选择器或
`--language auto`。Auto 是分块级别的：每个音频块自行检测语言；置信度低时回落到
上一块的语言，避免短中文块被误判为英文。

## 精度说明

本项目跑的是本地 Whisper，转写质量主要取决于模型、音频质量、块长，以及模型能否
获得足够上下文。它不像豆包那样用云端输入法，所以云端的中文 ASR/后处理可能仍然更好。

默认值偏向中文记笔记的准确度：

- 默认强制中文，而不是自动语言检测。
- 音频导入用更长的块，让中文口语有更多上下文。
- 每个块会带上最近的前文作为上下文。
- `beam_size` 默认 `8`。

想要更高精度、可接受更慢速度，编辑 `~/.config/voice-notes/config.toml`：

```toml
[model]
name = "large-v3"
language = "zh"
# 或使用分块级检测：
# language = "auto"
# auto_language_threshold = 0.45

[transcription]
chunk_seconds = 45.0
beam_size = 8
condition_on_previous_text = true
```

想要更快的预览，用 `large-v3-turbo` 并调小 `chunk_seconds`，代价是更多识别错误。

## 本地摘要

摘要是本地的：

- 若 Ollama 在运行，`voice-notes` 使用配置的本地模型。
- 否则回落到内置的抽取式摘要。

本项目不使用任何云端 API。

## 系统声音

Linux 上扬声器采集用 `pactl` 和 `parec` 读取默认 sink 的 monitor。macOS 上需要
BlackHole 之类的回环设备加一个多输出设备。若系统采集不可用，录音会继续以仅麦克风方式进行。

对导入的文件，尽量安装 `ffmpeg`。macOS 也可回退到 `afconvert`。
