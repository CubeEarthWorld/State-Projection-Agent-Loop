# チェックリストによる計画管理

Python版とDart版は同じ仕様・JSON形式を実装しています。Sessionを作るだけで、
`planning.checklist.manage` が常駐ツールとして登録されます。追加インストールは不要です。
LLMプロバイダへ送る関数名は既存の命名規約により `planning__checklist__manage` になります。

## 保存と寿命

- **会話と同じ保存方式**です。既定は `InMemoryLedger` によるメモリのみの保存です。
  プロセスを終了すると、会話とチェックリストは共に失われます。
- `persistence.ledger_directory`（Dartでも設定JSONのキーは同じ）を指定した場合だけ、
  会話と同じJSONL台帳・スナップショットへ保存します。専用ファイルやDBは作りません。
- 会話履歴の圧縮、コンテキスト非表示、タスク完了では削除しません。完了済みも残ります。
  再起動時は既存の `resume_from_ledger` / `resumeFromLedger` にrun IDを渡して復元します。
  新規Sessionが別の会話の計画を暗黙に読み込むことはありません。
- ツールによる変更は `checklists_changed` イベントへ記録してから作業状態へ反映します。
  最新スナップショット以降の変更も復元時に再生します。削除も記録するため、
  古いスナップショットから削除済みの計画が復活することを防ぎます。
- `branch` は現在の計画を独立コピーします。既存APIと同様、`at_message` / `atMessage`
  はコピーする会話履歴の範囲を指定し、計画を過去へ戻す指定ではありません。
  `rewind` は明示的な巻き戻しとして、計画も対象ターン開始時のチェックポイントへ戻します。
- ホスト側で記録付きの操作をする場合も `Session.invoke`（Pythonの非同期では `ainvoke`）
  を使ってください。`session.checklists.execute` はセットアップ・オフライン処理向けの
  直接操作であり、それ自体は台帳へ記録しません。

## データ

チェックリスト: `id`（26文字のULID）、`name`、`revision`（1から増加）、
`include_in_context`（bool、既定true）、`context_mode`（既定summary）、順序付き `items`。

項目: `id`（ULID）、`text`、`status`（既定pending）、`notes`（既定空文字）。
状態は `pending / in_progress / blocked / completed / cancelled`。
チェックリスト内で同時に `in_progress` にできる項目は一つです。
項目の切り替えは二つの編集、または `update.items` による一括更新で行えます。
完了は検証後に記録し、停止理由や検証結果はnotesへ記録する運用を推奨します。
このツール自体が作業やテストの成否を検証するわけではありません。

全体の `status` と `progress` は読み取り時に計算し、直接編集はしません。
progressには全件数、状態別件数、残件数、完了率fractionを含みます。
fractionは `completed / (total - cancelled)`、分母が0なら0です。
空の計画はpending、全件取消はcancelled、未完了・未取消が0ならcompleted、
それ以外はin_progressの存在、blockedの存在の順で決め、残りはpendingです。
一部の項目を完了しただけでは残りを自動完了しません。

名前は200文字、項目本文500文字、メモ2000文字、1計画200項目、1Session100計画まで。
名前の重複は許可し、操作対象は常にULIDで指定します。
不正な型、未知のフィールド、不正ULID、ID重複、上限超過は拒否します。

## 統合ツール

`action` ごとに次の引数だけを渡します。全編集・削除は最新の `expected_revision` が必須です。
競合時は `get` で読み直して変更内容を再検討してください。全更新は検証後に一括反映され、
途中で失敗しても既存の計画は変わりません。

| action | 引数 | 動作 |
|---|---|---|
| create | name、任意のitems/include_in_context/context_mode | 新規作成。計画・新規項目のULIDを発行 |
| list | 任意のmode | 全計画を一覧。既定summary |
| get | id、任意のmode | 指定計画を取得。既定full |
| update | id、expected_revision、変更するname/items/include_in_context/context_mode | 指定フィールドだけ更新。itemsは**順序を含め全置換** |
| delete | id、expected_revision | 計画を削除 |
| add_item | id、expected_revision、item | 項目を末尾に追加。item.text必須 |
| update_item | id、expected_revision、item_id、item | text/status/notesだけを部分更新。項目IDは不変 |
| delete_item | id、expected_revision、item_id | 指定項目を削除 |
| export | 任意のid | 指定計画または全計画をversion 1のJSON文書へ書き出し |
| import | document | export文書をコピーとして取り込み。既存IDとの衝突は全体を拒否 |

`update.items` では維持したい項目のIDを含めてください。ID省略の項目は新規項目になります。
返却値・書き出したJSONの変更が、保存中の計画へ勝手に反映されることはありません。
読み取りと編集を同じツールにまとめているため、ツール全体が内部状態へのwriteとして扱われ、
呼び出し順に実行されます。`auto_safe` と `auto_workspace_dev` はこの内部操作を許可しますが、
利用者が指定したdenyや承認ポリシーを迂回しません。自動再試行は行いません。

## コンテキスト

既定の `checklists` セクションが最新状態を毎ターン投影します。

- `name`: 名前と操作用ULIDのみ。
- `summary`: 名前、ULID、リビジョン、全体状態、進捗。
- `full`: summaryに項目とメモを追加。
- `include_in_context=false`: この自動投影から除外。保存・明示的なget/listは有効。

既定では最大6000文字程度に抑え、大きなfull表示はsummaryに縮小します。
入りきらない計画には省略通知を出し、さらに会話ウィンドウが足りない場合は
表示を縮小・省略します。元データは削除せず、getでいつでも確認できます。
この設定はアクセス制御や履歴の消去ではありません。過去のツール結果やexportには
非表示の計画が含まれ得ます。カスタムsectionsを指定する場合は `checklists` を含めてください。
計画テキストは状態データとして投影し、追加のシステム命令として扱わないよう案内します。
タスク完了前の見直しを促しますが、保留・取消・ブロックがある業務も扱えるよう、
未完了項目を理由にfinishを一律禁止することはありません。

## Python

```python
from state_projection_loop import Session, ScriptedLLM

session = Session(ScriptedLLM([]))  # 実運用では利用するLLMAdapterを渡す
plan = session.invoke(
    "planning.checklist.manage", action="create", name="機能実装",
    context_mode="full", items=[{"text": "実装"}, {"text": "テスト"}],
)
plan = session.invoke(
    "planning.checklist.manage", action="update_item",
    id=plan["id"], expected_revision=plan["revision"],
    item_id=plan["items"][0]["id"], item={"status": "in_progress"},
)
document = session.invoke("planning.checklist.manage", action="export", id=plan["id"])
other = Session(ScriptedLLM([]))
other.invoke("planning.checklist.manage", action="import", document=document)
```

## Dart

```dart
import 'package:state_projection_loop/state_projection_loop.dart';

Future<void> main() async {
  final session = Session(ScriptedLLM([]));
  var plan = await session.invoke('planning.checklist.manage', {
    'action': 'create', 'name': '機能実装', 'context_mode': 'full',
    'items': [{'text': '実装'}, {'text': 'テスト'}],
  }) as Map;
  plan = await session.invoke('planning.checklist.manage', {
    'action': 'update_item', 'id': plan['id'],
    'expected_revision': plan['revision'],
    'item_id': (plan['items'] as List).first['id'],
    'item': {'status': 'in_progress'},
  }) as Map;
  final document = await session.invoke('planning.checklist.manage', {
    'action': 'export', 'id': plan['id'],
  });
  final other = Session(ScriptedLLM([]));
  await other.invoke('planning.checklist.manage', {
    'action': 'import', 'document': document,
  });
}
```

## 子エージェントへの受け渡し

既存の `install_spawn` / `installSpawn` で有効化する `meta.agent.spawn` に
`checklist_ids: [ULID, ...]` を指定すると、その計画だけを独立コピーします。
指定した場合の戻り値は `{"result": 子の結果, "checklists": version 1の文書}`。
省略時の従来の戻り値は変更しません。空配列を明示すると、子が新たに作った計画も
戻り値で取得できます。親の会話や他の計画は共有せず、親へ自動マージもしません。
既存の親計画へ反映する場合はgetで最新リビジョンを確認し、返却内容を検討してupdateします。
同じULIDの計画をimportして親を上書きすることはできません。

## 参考と検証

設計の参考: [OpenCodeのtodowrite](https://opencode.ai/docs/tools/#todowrite)、
[Hermesの計画・オーケストレーション用ツール](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools/)。
このパッケージでは、複数の名前付き計画、ULID、投影設定、リビジョン競合検出、
両言語共通の受け渡し形式を組み合わせています。

両言語で共通の `checklists_v1.json` fixtureを使い、JSON往復、進捗、CRUD、
不正更新時の原子性、圧縮・表示予算、保存方式、再起動、削除の復元、
branch/rewind、子エージェントへの独立コピーをテストしています。
