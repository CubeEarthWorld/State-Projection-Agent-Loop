# 射影ループ(State-Projection Loop)設計仕様書 v1.0

対象読者:本パッケージの実装設計を担当するLLM。
この文書は自己完結しており、これのみを根拠に実装設計を行えること。
本文中の「MUST / MUST NOT / SHOULD / MAY」はRFC 2119に準じた強制力を持つ。
「裁量」と明記された箇所は設計担当LLMが自由に決定してよい。

---

## 1. 目的と製品像

Pythonパッケージとして提供する、汎用LLMエージェントループの実装である。
利用者はPythonでツール(関数)を定義・登録し、システムプロンプト(カーネル)を書くだけで、
任意の用途のエージェントを構築できる。

想定用途(いずれも本コアの変更なしに、ツール登録と設定のみで成立すること):

- ゲームマスタ(なりきり、フラグ管理、環境操作、クリア条件の認識と修正)
- ウェブバックエンドのサポートAI(ルーティング、資料参照、レポート生成)
- コーディングエージェント(Claude Code相当)
- デスクワークエージェント(PC・ブラウザ操作)
- Agent swarm(エージェントがエージェントを起動し指揮する)
- アプリ内キャラクターエージェント

非機能要求:

- 数千件のツールを登録可能で、1ターンあたりのコンテキスト消費が小さいこと
- 軽量・高速・シンプル・高拡張性
- デフォルト設定は最小構成で動き、機能はすべて足し算で有効化される(引き算で消すのではない)

---

## 2. 設計原理

### 2.1 公理

- 公理1:LLMの推論品質とコストは、コンテキストの量と質に支配される。注意は有限資源である。
- 公理2:1ステップの意思決定に必要な情報は、全ツールN件のうち高々k件(k≪N)と、
  会話の直近部分+それ以前の要約である。
- 公理3:LLMの本質的な仕事は意思決定である。データ運搬・信頼性制御・予算管理は
  決定論的なコード(ランタイム)の仕事であり、LLMに委ねてはならない。

### 2.2 一文原理

> 真実はコンテキストの外に置き、コンテキストは毎ターンそこから射影される最小の使い捨てビューとする。

現行の一般的なエージェントループは「追記型トランスクリプト=真実の保存場所=モデル入力」を
同一物にしているため、ツール定義の全量先読み(O(N)占有)、履歴再課金のO(N²)コスト、
中間データの素通り、ツール数増加による選択精度の急落、目標ドリフト等が構造的に発生する。
本設計はこの三位一体を分解することで、それらを根本から除去する。

### 2.3 全体構造:3名詞・4動詞

- 名詞:**ツール台帳(Registry)**、**射影パイプライン(Projection)**、**ランタイム(Runtime)**
- 動詞:**射影(render)→ 決定(decide)→ 実行(execute)→ 反映(commit)**

ループの概念形(言語非依存の擬似コード。実装形式は裁量):

```
session = init(config, seed)                 # シード状態・カーネル・台帳を構成
loop:
    prompt   = projection.render(session)    # 射影:セクション列からプロンプトを組み立てる
    decision = llm(prompt)                   # 決定:テキスト応答 or ツールコール列
    if decision is plain_text and mode == "chat":
        emit(decision); await user_input; continue
    results  = runtime.execute(decision.calls)   # 実行:検証・並列・再試行・予算強制
    session.commit(decision, results)        # 反映:ハンドル化・会話への追記・ログ
    if runtime.budget_exceeded or decision.is_done: break
```

---

## 3. 射影パイプライン

### 3.1 セクション

射影は「セクション」の順序付きリストであり、利用者が追加・削除・並べ替えできる。
各セクションは以下のインターフェースを満たす(表現形式は裁量。Protocol/ABCを推奨):

```python
class Section(Protocol):
    name: str
    cache_class: Literal["fixed", "append", "volatile"]
    def render(self, turn: TurnContext) -> list[Message] | str: ...
```

- `fixed`   : セッション中不変。プレフィックスキャッシュの土台。
- `append`  : 末尾追記のみ。キャッシュ前方を壊さない。
- `volatile`: 毎ターン変化しうる。**必ず射影の末尾(最新メッセージ側)に置く**(不変条件I3)。

### 3.2 デフォルト構成(これが「シンプルな底」)

```
[1] kernel        (fixed)    システムプロンプト + 常駐ツールSpec + ツール目次
[2] summary       (稀に更新)  ウィンドウから溢れた会話の折り畳み(溢れるまで空)
[3] conversation  (append)   直近の会話・ツールコール・観測の原文
[4] candidates    (volatile) 自動注入されたツールカード
```

何も設定しなければ、これだけで通常のチャットエージェントとして動作すること。
状態ビュー・予算表示・カスタムセクションは利用者が任意に追加登録する(§7)。

### 3.3 トークン予算

射影の合計トークンは `projection.window_tokens` を超えてはならず、超過時の削減順序は
candidates縮小 → conversationの古い側をsummaryへ折り畳み、とする。強制はランタイムの責務。

---

## 4. ツール定義(JSON)

### 4.1 スキーマ

ツールのメタデータはJSON(Python上ではdict)で宣言し、実行体(ハンドラ)はPython関数として
登録する。デコレータ登録と、docstring/型ヒントからのメタデータ自動生成を併設すること(裁量)。

```json
{
  "name": "web_search",
  "category": "web/search",

  "card": {
    "summary": "ウェブを検索し上位結果(タイトル・URL・抜粋)を返す",
    "signature": "web_search(query: str, max_results: int = 5) -> list[SearchResult]",
    "tags": ["web", "検索", "調査"]
  },

  "spec": {
    "description": "検索エンジン経由でウェブ全体を検索する。結果は関連度順。",
    "parameters": {
      "type": "object",
      "properties": {
        "query":       { "type": "string",  "description": "検索クエリ。1〜6語が最適" },
        "max_results": { "type": "integer", "default": 5, "minimum": 1, "maximum": 20 }
      },
      "required": ["query"]
    },
    "returns": { "type": "array", "items": { "$ref": "#/defs/SearchResult" } },
    "usage_notes": "変化しうる事実・最新情報の確認に使う。本文全文が必要なら web_fetch と組み合わせる。",
    "examples": [
      { "call": { "query": "USD JPY 為替 今日" }, "note": "単純な事実確認は1回で足りる" }
    ]
  },

  "discovery": {
    "pinned": false,
    "require_spec": false,
    "embedding_text": "調べて 検索して 最新情報 ニュース 現在の 価格 いつ 誰が",
    "no_embed": false
  },

  "execution": {
    "handler": "myapp.tools.web_search",
    "timeout_s": 20,
    "retries": 2,
    "parallel_safe": true,
    "output_policy": { "max_inline_tokens": 800, "overflow": "handle", "preview": "head" }
  }
}
```

### 4.2 フィールド規約

| フィールド | 必須 | 意味 |
|---|---|---|
| `name` | MUST | 一意。ツールコール時の識別子 |
| `category` | SHOULD | `/`区切りの階層。目次(§5 第1層)の生成元 |
| `card.summary` | MUST | 1行説明。カードは合計約30トークンを目標とする |
| `card.signature` | MUST | 型シグネチャ。**カードだけでツールを呼べる**ための最小情報 |
| `card.tags` | MAY | 検索補助 |
| `spec.parameters` | MUST | JSON Schema。ランタイム検証(§6)の根拠 |
| `spec.usage_notes` | SHOULD | 「いつ・どの順で・何と組み合わせて使うか」。JSON Schemaが表現できない使用知識を自然言語で書く層 |
| `spec.examples` | MAY | 呼び出し例 |
| `discovery.pinned` | MAY (default false) | trueならフルSpecをカーネルに常駐(第0層) |
| `discovery.require_spec` | MAY (default false) | trueなら初回呼び出し前にSpecの取り込みを強制(誤用が危険なツール向け) |
| `discovery.embedding_text` | MAY | 埋め込み対象テキスト。**説明文でなく想定される呼び出し文脈・言い回しを書く**(ユーザーの語彙と説明文の語彙は意味空間でずれるため)。省略時は summary+tags から自動生成 |
| `discovery.no_embed` | MAY (default false) | trueなら候補層(第2層)の対象外。目次・能動検索・ピン留め経由では到達可能なまま |
| `execution.output_policy` | MAY | 閾値超の結果をハンドル化する規則(§8.3) |

### 4.3 カードの導出

`card` を省略した登録に対しては `spec` から自動導出できること(summary=descriptionの先頭文、
signature=parametersから合成)。利用者の記述量を最小化するため。

---

## 5. ツール認識の4層

「モデルはどうやってツールの存在を知るか」への回答。コストの異なる4層を重ね、
通常経路では追加のLLM呼び出しをゼロにする。

| 層 | 名称 | 常駐/自動/要求時 | コスト | 内容 |
|---|---|---|---|---|
| 第0層 | 常駐(pinned) | 常駐 | 設定次第 | 利用者指定ツールのフルSpecをカーネルに固定 |
| 第1層 | 目次(TOC) | 常駐 | 約50〜100tk | カテゴリ名と件数のみ(例:`web/search(2) file(12) game/flags(24)`) |
| 第2層 | 自動候補 | 毎ターン自動 | 約300tk | ランタイムが類似度検索を自動実行し、上位k件のカードをcandidatesセクションへ注入 |
| 第3層 | 能動検索 `find_tools` | 要求時のみ | +ループ1周 | モデル自身が自然文クエリを書いて検索する。フォールバック |

### 5.1 第2層(自動候補)の規定

- クエリはランタイムが合成する:`config.discovery.query_sources` の既定は
  `["last_user_message", "last_model_thought", "goal_if_exists"]`。
- 検索エンジンは第3層と共有(ベクトル+語彙の混合。§9)。LLMは関与しない純粋計算。
- 注入位置はcandidatesセクション=射影末尾。プレフィックスキャッシュを壊さない(不変条件I3)。
- 常駐ツールと重複するカードは除去する。

### 5.2 第3層(能動検索)の規定

`find_tools` は通常のツールコールである。「LLMの追加起動」ではなく、ループが1周増えるだけ。
検索処理自体はLLMを含まない。第0〜2層により、この経路は例外時にしか通らない設計とする。

### 5.3 到達可能性の保証

いかなる設定(ベクトル無効、embedding空)でも、登録済み全ツールは
目次+能動検索+ピン留めのいずれかで到達可能でなければならない(不変条件I10)。

---

## 6. 呼び出し:遅延Specとランタイム検証

カードには型シグネチャが含まれるため、**モデルはカードから直接ツールを呼んでよい**。
「カード閲覧→Spec展開→呼び出し」の中間1周を既定では発生させない。

1. モデルがツールコールを出力する。
2. ランタイムが引数を `spec.parameters`(JSON Schema)で検証する。
3. 合格 → 実行。
4. 不合格 → 実行せず、**フルSpecを添えた検証エラー**を観測として返す。モデルは次周で再試行する
   (自己修復)。同一ツールへの連続検証失敗は `config.limits.max_validation_retries`(既定2)で打ち切る。
5. `require_spec: true` のツールは、Specが会話内に一度も現れていない状態での呼び出しを
   手順4と同じ経路で差し戻す(引数が偶然スキーマを通る誤用への対策)。

---

## 7. 状態はツールである(コア外・オプション同梱)

goal・flags・変数などの構造化状態は**コアの必須部品ではない**。パッケージは以下を
「同梱ツール群+同梱セクション」として提供し、利用者が登録するかどうかを選ぶ。

- 同梱ツール(例):`state_set(path, value)` / `state_get(path)` / `state_delete(path)` /
  `set_goal(text)` / `set_flag(name, value)`。すべて§4のJSON形式で定義された普通のツール。
- 同梱セクション:`state_view`(状態の要点を射影に載せる。cache_class=volatileまたは
  低頻度更新。載せる内容・整形は利用者がテンプレートで指定可能)。
- 初期状態はセッション開始時に `seed` としてコードから注入できる。
- 編集主体は三者:利用者(API経由)、LLM(ツール経由)、セッション開始時のシード。

ゲームマスタ用途はこれをフル装備し(flags=ゲームフラグ、goal=クリア条件、
state_viewを常時射影して目標ドリフトを構造的に防ぐ)、単純なサポートボットは一切使わない。
コアは両者で同一である。

---

## 8. ランタイム

### 8.1 責務

LLMは「何をするか」のみ出力し、以下はすべてランタイムが決定論的に行う(不変条件I5):

- 引数の型検証(§6)
- 並列実行(`parallel_safe: true` のコール群は同時実行。asyncio前提でよい:裁量)
- 再試行・タイムアウト(ツール定義の `execution.*` に従う)
- 予算強制:`max_steps / max_tokens / max_cost / max_seconds`。超過時はループを停止し、
  その旨を観測としてモデルに伝えた上で終了処理を1周だけ許す
- 構造化ログ(全射影・全決定・全実行結果を機械可読で記録。再現とデバッグのため)

### 8.2 観測のラベリング

ツール結果は必ず「観測(ツール結果)」として構造上区別されたロールで射影する。
非信頼データ(ツールが外部から取得したテキスト)を指示として扱わないための
攻撃面縮小策であり、完全な防御ではないことを文書化する。

### 8.3 参照渡し(ハンドル)

- ツール結果が `output_policy.max_inline_tokens` を超える場合、結果は値ストアに保存され、
  射影には **ハンドル(例 `$h7`)+型+サイズ+プレビュー** のみが載る。
- ツールは引数としてハンドルを受け取れる(値はランタイムが解決)。
  大きなデータがモデルのコンテキストを素通りしない。
- モデルは常駐メタツール `peek(handle, query?, range?)` で中身を部分閲覧できる。
- 値ストアのバックエンド(メモリ/ディスク)は裁量。セッション終了で破棄してよい(§12)。

---

## 9. 検索エンジン(第2層・第3層共用)

- ベクトル機能は **オプション**。`config.discovery.vector = "auto" | "on" | "off"`。
  - auto/on:登録時に `embedding_text`(なければ自動生成テキスト)を自動で埋め込み計算。
  - off:第2層は語彙一致(BM25等)に退化するか、利用者設定でスキップ。
- 埋め込みが空(`no_embed: true` または未計算)のツールは第2層の対象外。
- スコアリングはベクトル類似度+語彙一致+タグ一致の混合(重みは裁量)。
- 埋め込みモデルのインターフェースは差し替え可能に抽象化する(裁量)。
- 発展(v1では任意):使用ログ・共起(同一タスクで一緒に使われたツール)をスコアに還流する口を
  インターフェースとして残す。

---

## 10. 要約(compaction)と要約の契約

### 10.1 トリガと実行

- conversationが `window_tokens × trigger_ratio`(既定0.8)を超えたら、古い側から折り畳み、
  summaryセクションに積む。
- 要約の実行モデルは設定可能(`compaction.model = "same" | モデル指定`。安価モデル可)。
- summaryはプレフィックス側にあるため更新時にキャッシュを一部無効化する。発生が
  「溢れた時だけ」であることで償却する。これは既知のトレードオフとして受容する。

### 10.2 要約の契約 v1(要約器プロンプトに固定で組み込むこと)

エージェントの連続性(仮想的な自我)は毎ターンの射影から再構築される。切断リスクは
折り畳み時に「行動の理由」が落ちることに集中するため、要約は以下をMUSTとする:

1. 一人称で書く(「私は〜と判断し〜した」)。
2. 時系列を保存する。
3. 各項目に「行動・観測の要点・決定と**その理由**・未完了の意図」を含める。
4. ユーザーの明示的な指示・制約・確定事項は逐語で保持する。
5. 生データ本体はハンドル参照(`$hN`)に置換して捨てる(原本はpeekで常に回帰可能)。
6. 長さ上限(既定:折り畳み対象の1/10、設定可)。

---

## 11. サブエージェント(spawn)

- `spawn(task: str, model?, kernel?, tool_scope?, budget?) -> Handle` は同梱の普通のツール
  (§4形式)。swarmが必要な利用者だけが登録・ピン留めする。
- サブエージェントは本仕様と同一のループを独立コンテキストで回す。親との共有は
  **task文字列(入力)と結果ハンドル(出力)のみ**(不変条件I9)。伝言ゲームと前提衝突を
  構造で回避するための意図的制約である。
- `tool_scope` は台帳のサブセット指定(カテゴリ or ツール名リスト)。
- 待機は同期(結果を待つ)を既定とし、非同期(ハンドル先行返却+完了ポーリング)は裁量で追加可。

---

## 12. 常駐メタツール定義

常駐が必須なのは以下の2つのみ。`spawn` と `done` は用途に応じたオプション同梱。

```json
{
  "name": "find_tools",
  "category": "meta",
  "card": {
    "summary": "ツール台帳を自然文で検索し、該当ツールのカード一覧を返す",
    "signature": "find_tools(query: str, category: str | null = null, k: int = 8) -> list[ToolCard]",
    "tags": ["meta", "検索"]
  },
  "spec": {
    "parameters": {
      "type": "object",
      "properties": {
        "query":    { "type": "string", "description": "やりたいことを自然文で" },
        "category": { "type": ["string", "null"], "description": "目次のカテゴリで絞り込み" },
        "k":        { "type": "integer", "default": 8 }
      },
      "required": ["query"]
    },
    "usage_notes": "自動候補に必要なツールが見当たらない時に使う。目次のカテゴリ名で絞れる。"
  },
  "discovery": { "pinned": true, "no_embed": true }
}
```

```json
{
  "name": "peek",
  "category": "meta",
  "card": {
    "summary": "ハンドル($hN)の中身を部分閲覧する",
    "signature": "peek(handle: str, query: str | null = null, range: str | null = null) -> str",
    "tags": ["meta", "参照"]
  },
  "spec": {
    "parameters": {
      "type": "object",
      "properties": {
        "handle": { "type": "string" },
        "query":  { "type": ["string", "null"], "description": "中身から探したい内容" },
        "range":  { "type": ["string", "null"], "description": "行範囲やキーパス等の指定" }
      },
      "required": ["handle"]
    },
    "usage_notes": "要約プレビューで足りない時のみ使う。全量展開は避け、queryかrangeで絞る。"
  },
  "discovery": { "pinned": true, "no_embed": true }
}
```

`done` について:chatモードではツールコールなしのテキスト応答がターン終了を意味するため不要。
jobモード(単発タスク実行)では `done(result)` を常駐させ、これを終了条件とする。

---

## 13. 設定デフォルト(config)

```json
{
  "mode": "chat",
  "projection": {
    "sections": ["kernel", "summary", "conversation", "candidates"],
    "window_tokens": 30000
  },
  "discovery": {
    "vector": "auto",
    "k": 8,
    "toc": true,
    "query_sources": ["last_user_message", "last_model_thought", "goal_if_exists"]
  },
  "compaction": { "trigger_ratio": 0.8, "model": "same", "contract": "v1" },
  "budget": { "max_steps": 50, "max_tokens": null, "max_cost": null, "max_seconds": null },
  "handles": { "inline_threshold_tokens": 800 },
  "limits": { "max_validation_retries": 2 }
}
```

トークン経済の目標値(1,000ツール登録時):カーネル(常駐除く)≤2k、目次≤100、
候補≤400、会話以外のターン毎オーバーヘッド合計≤3k。全Spec先読み(概算15万tk)に対し
2桁の削減を受け入れ基準とする。

---

## 14. 不変条件(実装レビュー時のチェックリスト)

- I1. ピン留め以外のツールSpecを一括でコンテキストに載せてはならない。
- I2. 射影の合計トークンは window_tokens を超えない(ランタイムが強制)。
- I3. volatileセクション(候補等)は射影末尾のみ。プレフィックスキャッシュを壊さない。
- I4. カーネルはセッション中不変。
- I5. 再試行・並列・タイムアウト・予算はランタイムの責務。LLMに委ねない。
- I6. ツール結果は「観測」として構造上区別されたロールで射影する。
- I7. `output_policy` 閾値超の結果は必ずハンドル化し、プレビュー+参照のみ射影する。
- I8. 要約は契約v1に従う(特に「決定の理由」と「未完了の意図」の保持)。
- I9. サブエージェントと親の共有はtask文字列と結果ハンドルのみ。
- I10. いかなる設定でも全登録ツールが目次・能動検索・ピン留めのいずれかで到達可能。
- I11. デフォルト設定のみで(状態・ベクトル・spawn無しで)通常のチャットエージェントが成立する。

---

## 15. 非目標(non-goals)

- セッションを跨ぐ記憶の引き継ぎ(値ストア・状態はセッション終了で破棄してよい)。
- 特定LLMベンダーへの依存(モデル呼び出しはアダプタで抽象化)。
- ツールの自動生成・自動コード実行環境(利用者がハンドラを書く)。
- UI・ホスティング。

---

## 16. 既知の限界(文書化して受容するもの)

1. 第2層の候補品質はクエリ表現に依存する。query_sourcesに直近のモデル思考を含めることで
   緩和するが、第1層・第3層は撤去できない。
2. カテゴリ数が数百に達すると目次が肥大する。その場合は目次自体を階層化し、
   上位層のみ常駐+下位はfind_toolsのcategory絞り込みで参照する。
3. 遅延Spec方式は「引数が偶然スキーマを通る誤用」を防げない。危険なツールは
   `require_spec: true` またはピン留めで対処する。
4. summary更新はプレフィックスキャッシュを一部無効化する(低頻度により償却)。
5. 観測ラベリングはプロンプトインジェクションの緩和であり、防止ではない。

---

## 17. 設計担当LLMの裁量範囲(未決事項)

以下は本仕様の不変条件に反しない限り自由に決定してよい:

- 内部メッセージ表現、Section/Registry/RuntimeのクラスAPI詳細、デコレータ構文
- 埋め込みモデルの選定と抽象化インターフェース、混合スコアの重み
- 値ストアのバックエンド、ハンドル命名規則
- 並列実行機構(asyncio推奨)、構造化ログのフォーマット
- 目次の整形、カード自動導出の具体規則
- テスト戦略。ただし受け入れテストとして最低限:
  (a) 1,000ツール登録・既定設定で1ターンのツール関連オーバーヘッドが3kトークン以下、
  (b) ベクトル無効時に全ツールへ到達可能、
  (c) 検証失敗→Spec添付→再試行の自己修復経路が動作、
  (d) 既定設定のみでチャットエージェントが成立、を含めること。
