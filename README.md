# Fun Annotator

4コマ漫画のためのAI支援アノテーションツール

## 🚀 クイックスタート

### 1. リポジトリのクローン
```bash
git clone https://github.com/YOUR_USERNAME/fun_annotator.git
cd fun_annotator
```

### 2. AIモデルの取得

このプロジェクトは複数のAIモデル（合計約2.2GB）を使用します。GitHubの制限により、モデルファイルはHugging Faceで配布しています。

#### 自動ダウンロード（推奨）
```bash
# 必要なライブラリをインストール
uv pip install huggingface_hub

# 全モデルを自動ダウンロード
python download_models.py --all

# 個別ダウンロード例
python download_models.py --balloon  # 吹き出し検出器のみ
python download_models.py --face     # 顔検出器のみ
```

#### 手動ダウンロード
```bash
# Hugging Face CLIを使用
huggingface-cli download YOUR_USERNAME/fun-annotator-models --local-dir ./models
```

### 3. 依存関係のインストール
```bash
# Python依存関係
uv pip install -r requirements.txt

# Node.js依存関係 
npm install
```

### 4. 環境変数の設定
```bash
cp .env.example .env
# .envファイルを編集してAPIキーを設定
```

### 5. アプリケーションの起動
```bash
# バックエンド（API サーバー）
uvicorn main:app --reload

# フロントエンド（React アプリ）
npm start

# WebUI アノテーター
python tail_shape_annotation_tool_web.py
```

## 🤖 使用されているAIモデル

| モデル | 用途 | サイズ | フレームワーク |
|--------|------|--------|----------------|
| YOLO11x | 吹き出し検出 | 109MB | YOLO11 |
| YOLO11 | しっぽ検出 | 18MB | YOLO11 |
| YOLO11l | 顔検出 | 49MB | YOLO11 |
| YOLO11l | マルチクラス検出 | 49MB | YOLO11 |
| DINOv2 | キャラクター分類 | 349MB | PyTorch |
| OpenCV Cascade | アニメ顔検出 | <1MB | OpenCV |

## 📁 プロジェクト構成

```
fun_annotator/
├── src/                     # Reactフロントエンド
├── main.py                  # FastAPI バックエンド
├── download_models.py       # モデルダウンロードスクリプト
├── tail_shape_annotation_tool_web.py  # WebUI アノテーター
└── models/                  # AIモデル（ダウンロード後作成）
```

## 🔧 開発者向け情報

詳細な技術情報は以下のファイルを参照してください：

- `CLAUDE.md` - プロジェクト設定とAIモデル情報
- `ANNOTATOR_MODEL_PATHS.md` - モデルパス詳細情報
- `requirements.txt` - Python依存関係
- `package.json` - Node.js依存関係
