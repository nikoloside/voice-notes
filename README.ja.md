# voice-notes

[English](README.md) · [中文](README.zh.md) · **日本語**

![license](https://img.shields.io/badge/license-MIT-blue.svg)
![python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![100% local](https://img.shields.io/badge/100%25-ローカル%20%2F%20オフライン-brightgreen.svg)
![whisper](https://img.shields.io/badge/ASR-faster--whisper-orange.svg)
![MCP](https://img.shields.io/badge/MCP-server-8A2BE2.svg)
![i18n](https://img.shields.io/badge/UI-中文%20%7C%20日本語%20%7C%20English-ff69b4.svg)
![platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)

![demo](docs/demo-ja.gif)

> ホーム → あるミーティングの3層要約 → ミーティング横断のナレッジグラフ。
> _（プレースホルダーのデモデータ。[▶ 高画質 MP4](docs/demo-ja.mp4)。）_

[`type-by-voice`](https://github.com/kotaro-nakata/type-by-voice) から切り出した、
スタンドアロンのローカル音声メモツールです。

音声を録音または取り込み、`faster-whisper` で**ローカル**に文字起こしし、
次を書き出します：

- `transcript.md`
- `notes.md`
- `summary.md`
- `audio.wav`
- `meta.json`

グローバルなプッシュトゥトーク入力、トレイ UI、アクティブなアプリへの貼り付けは
行いません。それらは
[`type-by-voice`](https://github.com/kotaro-nakata/type-by-voice) の担当です。

## セットアップ

**ワンコマンドでインストール。** Python venv を作成し、ローカルの Whisper
文字起こしモデルをダウンロードし、[Ollama](https://ollama.com) と小さなローカル
要約モデルを導入し、設定を書き込みます —— つまり**文字起こしも要約も完全にローカルで
動き、クラウド不要、そのまま使えます**。

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

その後、Web UI を起動：

```bash
./record-notes          # macOS / Linux
.\record-notes.ps1      # Windows
```

非力なマシンではより小さいモデルを使えます：
`VOICE_NOTES_WHISPER_MODEL=small ./install.sh`（および `VOICE_NOTES_LLM=llama3.2:3b`）。

<details><summary>手動セットアップ（Python 3.11+）</summary>

```bash
cd voice-notes
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./record-notes
```

Layer 2/3 の要約には LLM が必要です —— ローカルで Ollama を動かすか、`openai_url`
を OpenAI 互換サーバーに向けてください（後述）。
</details>

既定のページは <http://127.0.0.1:8765>。録音の開始/停止、音声ファイルのアップロード、
macOS ボイスメモの取り込み、生成ファイルのダウンロードができます。右上で
**中文 / 日本語 / English** を切り替えられます。生成されたメモはローカルフォルダの
プレーンな Markdown です —— 自分のリポジトリに入れる、共有する、[MCP サーバー](#mcp-サーバーclaude-からメモを読む)経由で Claude から読む、いずれも可能です。

3つのレイヤーは厳密に順番に実行されます —— Layer 1 → Layer 2 → Layer 3：

- **Layer 1**：生のチャンク文字起こし。各音声セグメントが終わるたびにリアルタイムで
  書き込まれます。全文が終わるまで他の処理は走りません。
- **Layer 2**：Layer 1 完了後、**全文**をローカル LLM に一括で渡し、トピックごとに
  まとめた詳細な全文要約（決定事項・数値・人名・ToDo・未確定の論点）を生成します。
  モデルのコンテキストに収まらない長い文字起こしは、内部でいくつかの大きなブロックに
  分割してマージします（出力上は見えません）。
- **Layer 3**：Layer 2 の全文要約をさらに1ページに凝縮：一言結論 / 要点 /
  Action Todo / Checkpoints。

`summary.md` = Layer 3 の1ページ + その後に Layer 2 の全文要約。`notes.md` は
3レイヤーすべてを表示します。

Layer 2/3 には LLM が必要です。バックエンドの解決順（`[summary] backend = "auto"`）：

1. `openai_url` が設定され到達可能なら OpenAI 互換サーバー —— 例えば Tailscale
   経由で別マシン上の LM Studio（`openai_url = "http://100.x.x.x:1234/v1"`；
   `openai_model` が空ならサーバーが返す最初のモデルを使用）。
2. それ以外はローカル Ollama（`ollama_url`、`ollama_model`）。
3. それ以外はルールベースの抽出（品質は大きく劣ります）。

アップロード/取り込みした音声では、一覧とセッションページに変換の進捗（文字起こし済み
時間・音声全体の長さ・完了率）が表示されます。ライブ録音では、文字起こし済み部分と
現在の録音時間を比較します。

アップロード/取り込みした音声はチャンク単位で `chunks.json` に記録されます。変換が
中断したらセッションを開いて **Resume conversion** を押すと、完了済みチャンクは再利用され、
欠けているチャンクのみ再文字起こしします。パイプラインはチャンク指向です：

```text
チャンク N 音声 -> 生文字起こし -> コアメモ -> value/todo/checkpoints -> checkpoint
チャンク N+1 音声 -> 生文字起こし -> コアメモ -> value/todo/checkpoints -> checkpoint
...
最終文字起こし -> 最終 summary.md
```

すべてのチャンクが完了した後に最終の `summary.md` が生成されます。

## コマンド

```bash
./record-notes                    # ローカル UI を起動
./record-notes --record           # ターミナルから録音；Enter で停止
./record-notes --import file.m4a   # 音声ファイルを1つ取り込んで待機
./record-notes --import file.m4a --language ja
./record-notes --list-devices      # マイクデバイス一覧
```

便利なオーバーライド：

```bash
./record-notes --port 8770 --no-browser
./record-notes --data-dir ~/.local/share/voice-notes/sessions
```

## ナレッジグラフ

各ミーティングの終了後、ナレッジグラフ抽出を1回実行します：同じ LLM がメモから
エンティティ（人物 / プロジェクト / 組織 / 概念 / 決定 / ToDo）とその関係を抽出し、
セッションの `entities.json` にキャッシュします。

`/api/graph` はすべてのセッションの `entities.json` を1つのグラフに集約します ——
同名のエンティティはミーティングを跨いでマージされるので、繰り返し出てくる
プロジェクトや人物は、それに言及したすべてのミーティングをつなぐ**ハブ**になります。
グラフは汎用の `{nodes, edges}` コントラクトで提供され、同梱の `graph.html` ビューアで
描画されます（同じコントラクトを話すデータソースなら再利用できます）。

ホームで **🕸️ ナレッジグラフ**（または `/graph`）を開くと、インタラクティブな
力学配置ビューになります：ノードをクリックで詳細、セッションノードをダブルクリックで
開く、検索でフォーカス。

既存セッション（この機能より前に作られたもの）にエンティティをバックフィル：

```bash
.venv/bin/python -c "import voice_notes as v, tomllib; \
c=tomllib.load(open('$HOME/.config/voice-notes/config.toml','rb'))['summary']; \
s=v.Summarizer(**{k:c[k] for k in ('backend','ollama_model','ollama_url','openai_url','openai_model','openai_api_key')}); \
import pathlib; [v.build_entities(p,s,force=True) for p in sorted((pathlib.Path.home()/'.local/share/voice-notes/sessions').glob('2026*'))]"
```

## MCP サーバー（Claude からメモを読む）

`voice-notes-mcp` は読み取り専用の [MCP](https://modelcontextprotocol.io) サーバー
（stdio）で、voice-notes が生成済みのメモを任意の MCP クライアント（Claude Code、
Claude Desktop など）に公開します。何も起動している必要はなく、ディスク上のセッション
フォルダを直接読みます。ツール：

- `list_notes(limit=50)` —— 最新セッションの id・タイトル・日付・長さ・状態
- `read_note(session_id, part="summary")` —— `part` は `summary`（1ページ + 全文）、
  `one_page`、`full`、`notes`（3レイヤー全部）、`transcript`
- `search_notes(query, limit=20)` —— 要約/メモ横断のキーワード検索
- `knowledge_graph(top=20)` —— ミーティング横断グラフの概要：件数、エンティティ種別、
  最も言及されたエンティティとその登場ミーティング
- `find_entity(name)` —— 人物/プロジェクト/概念を検索：説明・言及ミーティング・関連エンティティ

**Claude Code** に追加：

```bash
claude mcp add voice-notes --scope user -- /絶対パス/voice-notes/voice-notes-mcp
```

**Claude Desktop** に追加（`~/Library/Application Support/Claude/claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "voice-notes": {
      "command": "/絶対パス/voice-notes/voice-notes-mcp"
    }
  }
}
```

アプリと同じデータディレクトリを読みます；`--data-dir DIR` または環境変数
`VOICE_NOTES_DATA_DIR` で上書きできます。あとは Claude に「音声メモを一覧して」
「昨日のミーティングの1ページ要約を読んで」「メモをキーワードで検索して」などと
頼めます。

## 設定

初回起動で作成されます：

```text
~/.config/voice-notes/config.toml
```

既定のセッションデータの保存先：

```text
~/.local/share/voice-notes/sessions/<session-id>/
```

各セッションフォルダには `meta.json`、`transcript.md`、`summary.md`、`notes.md`、
取り込み音声の `chunks.json`、通常は `audio.wav` が入ります。

既定の文字起こし言語は中国語（`zh`）です。意図的に多言語が混ざる音声には、Web ページの
言語セレクタか `--language auto` を使ってください。Auto はチャンク単位で、各チャンクが
自身の言語を判定します。信頼度が低い場合は前のチャンクの言語にフォールバックし、短い
中国語チャンクが英語と誤認されるのを防ぎます。

## 精度に関するメモ

本プロジェクトはローカル Whisper を使うため、文字起こし品質は主にモデル・音声品質・
チャンク長・モデルが十分な文脈を得られるかに左右されます。Doubao のようなクラウド入力
メソッドは使っていないので、クラウド側の中国語 ASR/後処理の方が優れている場合もあります。

既定値は中国語のメモ精度を優先しています：

- 自動言語判定ではなく、既定で中国語を強制。
- 音声取り込みは長めのチャンクを使い、中国語の口語により多くの文脈を与える。
- 各チャンクに直近の文字起こしを文脈として渡す。
- `beam_size` は既定で `8`。

より高い精度が欲しく、速度低下を許容できるなら
`~/.config/voice-notes/config.toml` を編集：

```toml
[model]
name = "large-v3"
language = "zh"
# またはチャンク単位の検出：
# language = "auto"
# auto_language_threshold = 0.45

[transcription]
chunk_seconds = 45.0
beam_size = 8
condition_on_previous_text = true
```

より速いプレビューが欲しければ `large-v3-turbo` を使い `chunk_seconds` を下げます
（認識ミスは増えます）。

## ローカル要約

要約はローカルで行われます：

- Ollama が動いていれば、`voice-notes` は設定済みのローカルモデルを使います。
- そうでなければ、内蔵の抽出式要約にフォールバックします。

本プロジェクトはクラウド API を一切使いません。

## システム音声

Linux のスピーカー取得は `pactl` と `parec` で既定 sink の monitor を読みます。macOS では
BlackHole のようなループバックデバイスと複数出力装置が必要です。システム取得が使えない
場合、録音はマイクのみで続行します。

取り込みファイルには可能なら `ffmpeg` をインストールしてください。macOS は `afconvert`
にフォールバックすることもできます。
