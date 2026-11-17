# Streamlitアプリの公開手順（講師実演用）— ローカル → GitHub → Streamlit Community Cloud

## 1. ローカルで動かす

```bash
cd 08_streamlit_cloud
pip install -r requirements.txt
streamlit run app.py
```

- ブラウザが `http://localhost:8501` で開く。終了は黒い画面で `Ctrl + C`
- APIキー：**アプリのサイドバー「3. APIキー（任意）」に直接入力**（そのセッションの中だけで使われ、保存されません）／環境変数（`OPENAI_API_KEY` など）／`.streamlit/secrets.toml`（`secrets.toml.example` をコピーして記入）のいずれか。未設定なら模擬モード
- `secrets.toml` は **Gitに入れない**（`.gitignore` 済み）

## 2. GitHub アカウントとリポジトリ

1. https://github.com/ で **Sign up**（メールアドレス・パスワード・ユーザー名。無料プランで十分）。メールの確認コードを入力
2. 右上「＋」→ **New repository**
   - Repository name：例 `pos-copilot-app`
   - **Public**（無料で公開する場合。Private でも Community Cloud から使えるが、権限設定が増える）
   - 「Add a README file」にチェック → **Create repository**
3. ファイルを置く（**gitコマンドを使わない方法**）：リポジトリ画面の **Add file → Upload files** に、このフォルダの中身をドラッグ＆ドロップ
   - 置くもの：`app.py`、`analysis_tools.py`、`copilot.py`、`llm_providers.py`、`charts.py`、`requirements.txt`、`data/pos_sales_clean.csv`、`.gitignore`
   - **置かないもの**：`.streamlit/secrets.toml`（APIキー）。`secrets.toml.example` は置いてもよい
   - フォルダ `data/` ごとドラッグすると階層が保たれる。画面下の **Commit changes** で確定
4. （gitを使う場合）`git clone` → ファイルをコピー → `git add . && git commit -m "add app" && git push`

## 3. Streamlit Community Cloud で公開

1. https://share.streamlit.io/ を開き、**Continue with GitHub** でサインイン（GitHubアカウントと連携。初回はアクセス許可の画面が出る）
2. 右上 **Create app** → 「Yup, I have an app」
3. 入力：**Repository**（`ユーザー名/pos-copilot-app`）、**Branch**（`main`）、**Main file path**（`app.py`）、**App URL**（任意のサブドメイン。例 `pos-copilot-demo` → `pos-copilot-demo.streamlit.app`）
4. **Advanced settings**：Python version（3.12 など、サポート中の版）、**Secrets** に次を貼り付けて Save
   ```toml
   OPENAI_API_KEY = "sk-..."
   # ANTHROPIC_API_KEY = "sk-ant-..."
   # GEMINI_API_KEY = "AIza..."
   ```
5. **Deploy** → 数分でURLが発行される（右側にログ。依存ライブラリの導入に時間がかかることがある）

## 4. 更新・停止

- **更新**：GitHubのファイルを編集して Commit（またはpush）すると、自動で再デプロイされる（1分あたり5回まで）
- **Secrets の変更**：アプリ右下「Manage app」→ Settings → Secrets
- **再起動／削除**：Manage app → Reboot / Delete
- 一定期間アクセスがないアプリはスリープする（訪問すると再起動。起動に数十秒かかる）

## 5. 注意点（実演で必ず触れること）

| 項目 | 内容 |
|---|---|
| APIキー | リポジトリに書かない。Secrets に入れる。公開リポジトリにキーを入れると数分で悪用される |
| 画面のキー入力欄 | ローカルで各自が試すための機能。**公開URLでは利用者に自分のキーを入力させない**（入力値はそのセッション限りでサーバーに保存されないが、公開アプリでキーを求めるのは避ける）。公開時は Secrets のキーが使われ、入力欄は「設定済み」表示になる |
| 公開範囲 | Public リポジトリ＋公開URLは **誰でもアクセスできる**。URLを知っていれば使えるので、自社データや高額なAPIキーを紐づけない。講座では講座のサンプルデータ＋利用上限を設定したキーで実演 |
| API費用 | 公開URLから誰かが質問すると講師のキーで課金される。実演後はアプリを削除するか、キーを無効化する |
| アップロードCSV | 利用者がアップロードしたCSVはアプリのメモリ上で処理される（Community Cloud のサーバーは米国）。社内データを扱う用途には向かない |
| リソース | 無料枠のため、メモリ・CPUに上限がある。大きなCSVや同時アクセスには弱い |
| 本番利用 | 認証・権限・監視・ログ・費用管理は別途必要。Community Cloud は学習・デモ用と割り切る |

## 6. よくあるエラー

| 症状 | 対処 |
|---|---|
| `ModuleNotFoundError` | `requirements.txt` にライブラリが無い／ファイルがリポジトリの**ルート**にない（Main file path と一致させる） |
| `FileNotFoundError: data/pos_sales_clean.csv` | `data/` フォルダごとアップロードされていない |
| 日本語が □□□ | `japanize-matplotlib` が requirements.txt にあるか確認 |
| 起動が終わらない | ログを見る。依存ライブラリの版の衝突が多い。`requirements.txt` を最小にする |
| APIエラー | Secrets のキー名（`OPENAI_API_KEY`）の綴り、クォート、モデル名（`POS_COPILOT_MODEL`）を確認 |
