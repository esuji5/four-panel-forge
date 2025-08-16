import json
import os
from collections import defaultdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
from pathlib import Path
from difflib import SequenceMatcher
import copy
import sys

import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm_utils import (
    build_koma_id,
    fetch_anthropic_4koma_response,
    fetch_anthropic_discussion_response,
    fetch_gemini_4koma_response,
    fetch_gemini_discussion_response,
    fetch_gemini_response,
    fetch_gemini_response_base64,
    fetch_gemini_response_multimodal,
    LLMManager,
    LLMConfig,
)
from balloon_detection_integration import get_balloon_detector, integrate_balloon_detection, draw_tail_shape_results_on_image

# デバッグ用のレスポンスモデル
class DebugResponse(BaseModel):
    message: str
    pipeline_available: bool
    visualize_method_available: bool
    test_result: str
import base64
import io
from PIL import Image
import cv2
import numpy as np

# forceAPI用の直接API呼び出し関数
def fetch_gemini_response_base64_direct_api(prompt, base64_image, original_image_path=None):
    """forceAPI用: Gemini API直接呼び出し"""
    from datetime import datetime
    import json
    import base64 as b64
    import requests
    
    model = "gemini-2.5-flash-preview-05-20"
    log_with_time(f"start fetch_gemini_response_base64_direct_api with {model}", level="INFO")
    
    # 設定から API キーを取得
    llm_manager = LLMManager()
    config = llm_manager.config
    
    # デバッグ: APIキーの確認
    api_key = config.google_api_key
    env_api_key = os.getenv("GOOGLE_API_KEY")
    log_with_time(f"🔑 設定APIキー: {api_key[:20]}..." if api_key else "🔑 設定APIキーなし", level="DEBUG")
    log_with_time(f"🔑 環境変数APIキー: {env_api_key[:20]}..." if env_api_key else "🔑 環境変数APIキーなし", level="DEBUG")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    query_dir = f"tmp/api_queries/gemini_direct_api_{timestamp}"
    os.makedirs(query_dir, exist_ok=True)

    # APIキーを取得（フォールバック方式） - 一時的に直接指定
    final_api_key = "AIzaSyC9DUiCaiNINeoOiI1YDIqVWrSIFfVVBBs"
    log_with_time(f"🔑 最終使用APIキー: {final_api_key[:20]}...", level="DEBUG")
    log_with_time(f"🔑 完全なAPIキー: {final_api_key}", level="DEBUG")

    # 画像データの詳細をログ出力
    log_with_time(f"🖼️ 受信したbase64_image長さ: {len(base64_image) if base64_image else 0}", level="DEBUG")
    log_with_time(f"🖼️ base64_image先頭50文字: {base64_image[:50] if base64_image else 'None'}", level="DEBUG")
    
    # base64_imageがdata:image/jpeg;base64,で始まる場合は削除
    if base64_image.startswith("data:image"):
        log_with_time("🖼️ data:image プレフィックスを除去", level="DEBUG")
        base64_image = base64_image.split(",")[1]
        log_with_time(f"🖼️ プレフィックス除去後の長さ: {len(base64_image)}", level="DEBUG")

    # プロンプトを保存（空の場合の警告付き）
    if not prompt or prompt.strip() == "":
        log_with_time("⚠️ 警告: プロンプトが空です", level="WARNING")
        log_with_time(f"🔍 受信したプロンプト引数: '{prompt}'", level="DEBUG")
        prompt_to_save = "# WARNING: Empty prompt received\n"
    else:
        prompt_to_save = prompt
        log_with_time(f"✅ プロンプト保存: 長さ={len(prompt_to_save)}, 最初の200文字={prompt_to_save[:200]}", level="DEBUG")
        
    with open(f"{query_dir}/prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt_to_save)

    # 画像を保存（Base64デコードして保存）
    try:
        image_data = b64.b64decode(base64_image)
        log_with_time(f"🖼️ 画像デコード成功: {len(image_data)} bytes", level="DEBUG")
        with open(f"{query_dir}/image.jpg", "wb") as f:
            f.write(image_data)
        log_with_time(f"🖼️ 画像ファイル保存完了: {query_dir}/image.jpg", level="DEBUG")
    except Exception as e:
        log_with_time(f"❌ 画像デコードエラー: {e}", level="ERROR")
        raise

    # クエリ情報を保存
    query_info = {
        "model": model,
        "timestamp": timestamp,
        "prompt_length": len(prompt),
        "prompt_preview": prompt[:100] if prompt else "EMPTY",
        "prompt_is_empty": not prompt or prompt.strip() == "",
        "base64_length": len(base64_image),
        "image_size": len(image_data),
        "original_image_path": original_image_path,
        "force_api": True,
    }
    with open(f"{query_dir}/query_info.json", "w", encoding="utf-8") as f:
        json.dump(query_info, f, ensure_ascii=False, indent=2)

    log_with_time(f"Query saved to: {query_dir}", level="INFO")

    # CombinedKomaPromptTesterと同じ方式でHTTPリクエストを送信
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={final_api_key}"
    
    headers = {
        "Content-Type": "application/json",
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                ],
            },
        ],
    }

    log_with_time(f"🌐 HTTP Request to: {url[:80]}...", level="DEBUG")
    log_with_time(f"🌐 Full URL: {url}", level="DEBUG")
    log_with_time(f"🌐 Headers: {headers}", level="DEBUG")
    log_with_time(f"🌐 Payload keys: {list(payload.keys())}", level="DEBUG")
    log_with_time(f"🌐 Payload parts count: {len(payload['contents'][0]['parts'])}", level="DEBUG")
    log_with_time(f"🌐 Payload text part length: {len(payload['contents'][0]['parts'][0]['text'])}", level="DEBUG")
    log_with_time(f"🌐 Payload text part preview: {payload['contents'][0]['parts'][0]['text'][:100]}...", level="DEBUG")
    log_with_time(f"🌐 Payload image data length: {len(payload['contents'][0]['parts'][1]['inline_data']['data'])}", level="DEBUG")
    log_with_time(f"🌐 Payload image mime_type: {payload['contents'][0]['parts'][1]['inline_data']['mime_type']}", level="DEBUG")
    
    response = requests.post(url, headers=headers, json=payload)
    
    if not response.ok:
        log_with_time(f"❌ HTTP Error: {response.status_code} - {response.text}", level="ERROR")
        raise Exception(f"Gemini API error: {response.status_code} - {response.text}")

    response_data = response.json()
    log_with_time("✅ HTTP Request successful", level="DEBUG")
    
    # レスポンスを保存
    with open(f"{query_dir}/response.json", "w", encoding="utf-8") as f:
        json.dump(response_data, f, ensure_ascii=False, indent=2)
    
    # レスポンステキストも保存
    if "candidates" in response_data and len(response_data["candidates"]) > 0:
        response_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
        with open(f"{query_dir}/response.txt", "w", encoding="utf-8") as f:
            f.write(response_text)
        log_with_time(f"💾 レスポンス保存完了: {query_dir}/response.json, response.txt", level="DEBUG")
    else:
        log_with_time("⚠️ レスポンスにcandidatesが含まれていません", level="WARNING")
    
    # SDKのレスポンス形式に合わせて返す
    class MockResponse:
        def __init__(self, data):
            self.text = data["candidates"][0]["content"]["parts"][0]["text"]
    
    return MockResponse(response_data)

# タイムスタンプ付きログ関数
def log_with_time(message: str, level: str = "INFO"):
    """タイムスタンプ付きでログを出力"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    prefix = {
        "INFO": "ℹ️",
        "DEBUG": "🔍",
        "ERROR": "❌",
        "WARNING": "⚠️",
        "SUCCESS": "✅"
    }.get(level, "📝")
    
    log_message = f"[{timestamp}] {prefix} {message}"
    print(log_message)
    sys.stdout.flush()  # 即座に出力を反映

# YOLO+DINOv2検出システムのインポート
try:
    from yuyu_yolo_dinov2_pipeline import YuyuYOLODINOv2Pipeline
    yolo_dinov2_pipeline = None
    current_detection_mode = "multiclass"  # "face_recognition" or "multiclass"
except ImportError as e:
    log_with_time(f"YOLO+DINOv2パイプラインのインポートエラー: {e}", "WARNING")
    yolo_dinov2_pipeline = None
    current_detection_mode = "multiclass"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じてドメインを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO+DINOv2パイプラインの初期化
@app.on_event("startup")
async def startup_event():
    global yolo_dinov2_pipeline, current_detection_mode
    try:
        log_with_time("YOLO+DINOv2パイプライン初期化中...", level="INFO")
        yolo_dinov2_pipeline = YuyuYOLODINOv2Pipeline()
        # デフォルトでマルチクラス検出モードに設定
        yolo_dinov2_pipeline.set_mode(True)  # True = マルチクラス検出モード
        current_detection_mode = "multiclass"
        log_with_time("YOLO+DINOv2パイプライン初期化完了（マルチクラス検出モード）", level="SUCCESS")
    except Exception as e:
        log_with_time(f"YOLO+DINOv2パイプライン初期化エラー: {e}", level="ERROR")
        yolo_dinov2_pipeline = None


def get_yuyu_pipeline():
    """YOLO+DINOv2パイプラインのインスタンスを取得"""
    global yolo_dinov2_pipeline
    return yolo_dinov2_pipeline


class ImageData(BaseModel):
    dataName: str
    currentIndex: int
    komaPath: str
    imageData: dict
    summary: str




class CombinedImageData(BaseModel):
    komaPath: str  # Base64画像データまたはファイルパス
    originalImagePath: str = None  # 元の画像パス（結合前）
    prompt: str = None  # 従来の文字列プロンプト（後方互換性のため）
    promptMessages: list = None  # 配列形式のプロンプト（マルチモーダル対応）
    mode: str = "combined-four-panel"
    forceAPI: bool = False  # API強制使用フラグ
    useMultiModal: bool = False  # マルチモーダル使用フラグ
    enableBalloonDetection: bool = True  # 吹き出し検出フラグ
    imagePathList: dict = None  # 4コマ個別画像パス（吹き出し検出用）
    enableVisualization: bool = False  # 画像可視化フラグ（尻尾形状分類結果を画像上に表示）
    detectionResults: dict = None  # AI検出結果（改善版プロンプト用）
    detectOnly: bool = False  # 検出・分類・可視化のみ実行（Gemini呼び出しなし）
    detectionMode: str = None  # 検出モード ("face_recognition" or "multiclass")


class FeedbackRequest(BaseModel):
    dataName: str
    currentIndex: int
    imageData: dict
    imagePathList: dict


class DiscussionRequest(BaseModel):
    dataName: str
    currentIndex: int
    imageData: dict
    imagePathList: dict
    summary: str
    question: str


class YoloDinov2DetectionRequest(BaseModel):
    """YOLO+DINOv2検出リクエスト"""
    komaPath: str  # 画像パスまたはBase64データ
    mode: str = "single"  # "single" または "four-panel"
    detectionThreshold: float = 0.25
    classificationThreshold: float = 0.5
    visualize: bool = False  # 可視化フラグ
    detectionMode: str = None  # 検出モード ("face_recognition" or "multiclass")



class AIProposal(BaseModel):
    """AI提案データの保存用モデル"""
    image_path: str
    timestamp: str
    model: str
    proposal: dict
    confidence_scores: Optional[Dict[str, float]] = None
    processing_time: Optional[float] = None
    api_cost: Optional[float] = None


class RevisionEntry(BaseModel):
    """修正履歴のエントリ"""
    timestamp: str
    revision_type: str  # "ai_proposal", "human_edit", "auto_save"
    changes: List[Dict[str, Any]]  # 各変更の詳細
    editor: str  # "ai", "human"
    confidence: Optional[float] = None
    notes: Optional[str] = None


class RevisionHistory(BaseModel):
    """修正履歴全体"""
    image_path: str
    created_at: str
    last_updated: str
    total_revisions: int
    ai_proposals: List[AIProposal]
    revisions: List[RevisionEntry]
    current_data: dict


class DiffAnalysis(BaseModel):
    """差分分析結果"""
    field_path: str  # e.g., "characters.0.name"
    old_value: Any
    new_value: Any
    change_type: str  # "added", "modified", "deleted"
    similarity: Optional[float] = None


class DetectionModeRequest(BaseModel):
    """検出モード切り替えリクエスト"""
    mode: str  # "face_recognition" or "multiclass"


class DetectionModeResponse(BaseModel):
    """検出モード切り替えレスポンス"""
    success: bool
    current_mode: str
    message: str


@app.post("/api/discussion-claude")
async def get_discussion(request: DiscussionRequest):
    # ここでLLMにリクエストを送信し、フィードバックを取得する処理を実装
    log_with_time(f"Discussion request (Claude): {request}", level="DEBUG")
    prompt = (
        "今までのデータを基にこの漫画についてディスカッションしましょう。質問はこちらです："
        + request.question
    )
    # import pdb;pdb.set_trace()
    summary = json.loads(request.summary)
    response = fetch_anthropic_discussion_response(prompt, summary, request.imageData)
    # with open(f'public/saved_json/discussion_{request.dataName}_{request.currentIndex}.json', 'a+') as f:
    #     f.write(json.dumps(response.json()))

    # 既存の内容をリストとして取得
    existing_data = []
    try:
        with open(
            f"public/saved_json/discussion_{request.dataName}_{request.currentIndex}.json",
            "r",
        ) as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        pass  # ファイルが存在しない場合は新規作成
    existing_data.append({"user": request.question})
    existing_data.append(response.json())
    # 更新されたリストをファイルに書き込む
    with open(
        f"public/saved_json/discussion_{request.dataName}_{request.currentIndex}.json",
        "w",
    ) as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    return JSONResponse(content=response.json())


@app.post("/api/delete-json")
async def delete_json_files(data: dict):
    """JSONファイルと関連ファイルを削除"""
    try:
        data_name = data.get("dataName", "yuyu10")
        current_index = data.get("currentIndex", 0)
        
        files_to_delete = [
            f"public/saved_json/imageData_{data_name}_{current_index}.json",
            f"public/saved_json/feedback_{data_name}_{current_index}.json",
            f"public/saved_json/discussion_{data_name}_{current_index}.json",
            f"public/saved_json/chat_history_{data_name}_{current_index}.json",
            f"public/ai_proposals/proposal_{data_name}_{current_index}_*.json",
            f"public/revision_history/revision_{data_name}_{current_index}_*.json"
        ]
        
        deleted_files = []
        for file_pattern in files_to_delete:
            if "*" in file_pattern:
                # ワイルドカードパターンの場合
                import glob
                matching_files = glob.glob(file_pattern)
                for file_path in matching_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        deleted_files.append(file_path)
                        log_with_time(f"Deleted file: {file_path}", level="INFO")
            else:
                # 通常のファイルパスの場合
                if os.path.exists(file_pattern):
                    os.remove(file_pattern)
                    deleted_files.append(file_pattern)
                    log_with_time(f"Deleted file: {file_pattern}", level="INFO")
        
        return JSONResponse(content={
            "status": "success",
            "deleted_files": deleted_files,
            "message": f"Deleted {len(deleted_files)} files"
        })
        
    except Exception as e:
        log_with_time(f"Error deleting JSON files: {e}", level="ERROR")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )


@app.post("/api/discussion")
async def get_discussion_gemini(request: DiscussionRequest):
    # Geminiでディスカッション
    log_with_time(f"Discussion request (Gemini): {request}", level="DEBUG")
    prompt = (
        "今までのデータを基にこの漫画についてディスカッションしましょう。質問はこちらです："
        + request.question
    )
    summary = json.loads(request.summary) if isinstance(request.summary, str) else request.summary
    response = fetch_gemini_discussion_response(prompt, summary, request.imageData)
    
    # Geminiのレスポンスを処理
    response_data = {
        "role": "assistant",
        "content": [{"type": "text", "text": response.text}],
        "model": "gemini-2.5-flash-preview-05-20",
    }
    
    # 既存の内容をリストとして取得
    existing_data = []
    try:
        with open(
            f"public/saved_json/discussion_{request.dataName}_{request.currentIndex}.json",
            "r",
        ) as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        pass  # ファイルが存在しない場合は新規作成
    
    existing_data.append({"user": request.question})
    existing_data.append(response_data)
    
    # 更新されたリストをファイルに書き込む
    with open(
        f"public/saved_json/discussion_{request.dataName}_{request.currentIndex}.json",
        "w",
    ) as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    return JSONResponse(content=response_data)


# 各エンドポイントで画像データの検証を行う関数
def validate_image_data(data: dict) -> dict:
    """画像データの検証と正規化"""
    validated_data = {}
    for img_key, img_value in data.items():
        if isinstance(img_value, dict) and "komaPath" in img_value:
            validated_data[img_key] = img_value
        else:
            # 適切な形式でない場合はスキップまたはエラー処理
            log_with_time(f"Invalid image data format for key {img_key}", level="WARNING")
    return validated_data


@app.post("/api/feedback-claude")
async def get_feedback(request: FeedbackRequest):
    # ここでLLMにリクエストを送信し、フィードバックを取得する処理を実装
    log_with_time(f"Feedback request (Claude): {request}", level="DEBUG")
    prompt = "画像群とimageDataから4コマ全体でどんな話かをまとめてください"
    # import pdb;pdb.set_trace()
    response = fetch_anthropic_4koma_response(
        prompt, list(request.imagePathList.values()), request.imageData
    )
    # print(response.content)
    with open(
        f"public/saved_json/feedback_{request.dataName}_{request.currentIndex}.json",
        "w",
    ) as f:
        f.write(json.dumps(response.json()))
    return JSONResponse(content=response.json())


@app.post("/api/feedback")
async def get_feedback_gemini(request: FeedbackRequest):
    # Geminiで4コマ全体のフィードバックを取得
    log_with_time(f"Feedback request (Gemini): {request}", level="DEBUG")
    prompt = "画像群とimageDataから4コマ全体でどんな話かをまとめてください"
    response = fetch_gemini_4koma_response(
        prompt, list(request.imagePathList.values()), request.imageData
    )

    try:
        # Geminiのレスポンスを処理
        response_data = {
            "role": "assistant",
            "content": [{"type": "text", "text": response.text}],
            "model": "gemini-2.5-flash-preview-05-20",
        }

        # ファイルに保存
        with open(
            f"public/saved_json/feedback_{request.dataName}_{request.currentIndex}.json",
            "w",
        ) as f:
            f.write(json.dumps(response_data))

        return JSONResponse(content=response_data)

    except Exception as e:
        log_with_time(f"Error processing Gemini feedback: {e}", level="ERROR")
        return JSONResponse(
            content={"error": f"Failed to get feedback with Gemini: {str(e)}"},
            status_code=500,
        )





@app.post("/api/analyze-image-improved/")
async def analyze_image_improved(data: CombinedImageData):
    """改善版プロンプトを使用した画像分析エンドポイント（AI検出結果活用版）"""
    try:
        # リクエストデータの詳細ログ
        log_with_time("🔄 改善版API呼び出し開始", level="INFO")
        log_with_time(f"📥 受信データ: mode={data.mode}", level="DEBUG")
        log_with_time(f"📥 detectionResults有無: {bool(hasattr(data, 'detectionResults') and data.detectionResults)}", level="DEBUG")
        
        # APIを使用
        llm_manager = LLMManager()
        config = llm_manager.config
        
        # プロンプトのインポート
        from prompts import IMPROVED_PROMPT_SINGLE, IMPROVED_PROMPT_FOUR_PANEL
        
        # 元のプロンプトをログ出力
        log_with_time(f"元のプロンプト: 長さ={len(data.prompt) if data.prompt else 0}, 内容={data.prompt[:100] if data.prompt else 'None'}", level="DEBUG")
        
        # プロンプトの処理ロジック
        # フロントエンドから詳細なプロンプト（AI検出結果付き）が送信されているかチェック
        has_enhanced_prompt = (
            data.prompt and 
            len(data.prompt) > 1000 and 
            ("AI検出結果" in data.prompt or "detected_characters" in data.prompt)
        )
        
        if has_enhanced_prompt:
            # フロントエンドで生成された拡張プロンプトをそのまま使用
            log_with_time("✅ フロントエンド生成の拡張プロンプトを使用", level="INFO")
            log_with_time(f"拡張プロンプト詳細: 長さ={len(data.prompt)}, AI検出結果含む={('detected_characters' in data.prompt)}", level="DEBUG")
        else:
            # フロントエンドからの詳細プロンプトがない場合のみ改善版プロンプトを使用
            log_with_time("⚠️ 詳細プロンプトなし - 改善版プロンプトにフォールバック", level="WARNING")
            
            # モードに応じて改善版プロンプトを使用
            if data.mode == "combined-four-panel" or data.mode == "four-panel":
                improved_prompt = IMPROVED_PROMPT_FOUR_PANEL
            else:
                improved_prompt = IMPROVED_PROMPT_SINGLE
            
            # 改善版プロンプトを設定
            data.prompt = improved_prompt
            log_with_time(f"改善版プロンプト設定: 長さ={len(data.prompt)}, 最初の100文字={data.prompt[:100]}", level="DEBUG")
            
            # AI検出結果がある場合はプロンプトに追加
            if hasattr(data, 'detectionResults') and data.detectionResults:
                detection_results_str = f"\n\n## AI検出結果:\n{json.dumps(data.detectionResults, ensure_ascii=False, indent=2)}"
                data.prompt += detection_results_str
                log_with_time(f"AI検出結果追加: 検出結果長さ={len(detection_results_str)}, 総プロンプト長さ={len(data.prompt)}", level="DEBUG")
            else:
                log_with_time("AI検出結果なし - プロンプトにはベースのみ使用", level="DEBUG")
        
        # 改善版API専用ロジック（APIのみ使用）
        log_with_time("🔧 改善版API - API使用", level="DEBUG")
        
        if data.mode == "combined-four-panel" or data.mode == "four-panel":
            if data.komaPath.startswith("data:image"):
                original_path = data.originalImagePath if hasattr(data, 'originalImagePath') else None
                log_with_time("🌐 改善版API(4コマ) - API呼び出し", level="INFO")
                response = fetch_gemini_response_base64(data.prompt, data.komaPath, original_image_path=original_path)
            else:
                image_path = "./public" + data.komaPath
                log_with_time("🌐 改善版API(4コマ, ファイル) - API呼び出し", level="INFO")
                response = fetch_gemini_response(data.prompt, image_path)
        else:
            if data.useMultiModal and data.promptMessages:
                response = fetch_gemini_response_multimodal(data.promptMessages)
            else:
                if data.komaPath.startswith("data:image"):
                    original_path = data.originalImagePath if hasattr(data, 'originalImagePath') else None
                    log_with_time(f"改善版API - Gemini APIに渡すプロンプト長さ: {len(data.prompt)}", level="DEBUG")
                    log_with_time(f"改善版API - プロンプト最初の200文字: {data.prompt[:200]}", level="DEBUG")
                    response = fetch_gemini_response_base64(data.prompt, data.komaPath, original_image_path=original_path)
                else:
                    image_path = "./public" + data.komaPath
                    response = fetch_gemini_response(data.prompt, image_path)

        # レスポンス処理（既存と同じ）
        content = response.text
        
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_content = content[json_start:json_end].strip()
        else:
            json_content = content

        try:
            content_data = json.loads(json_content)
        except json.JSONDecodeError as e:
            import re
            json_content = re.sub(r"'([^']*)':", r'"\1":', json_content)
            json_content = re.sub(r":\s*'([^']*)'", r': "\1"', json_content)
            json_content = re.sub(r',\s*}', '}', json_content)
            json_content = re.sub(r',\s*]', ']', json_content)
            try:
                content_data = json.loads(json_content)
            except:
                content_data = {"error": "Failed to parse JSON response", "raw_content": content[:1000]}

        # 吹き出し検出（既存と同じ）
        enable_balloon_detection = getattr(data, 'enableBalloonDetection', True)
        if enable_balloon_detection:
            balloon_detector = get_balloon_detector()
            
            if data.mode == "four-panel":
                for i, panel_key in enumerate(["panel1", "panel2", "panel3", "panel4"], 1):
                    if panel_key in content_data and isinstance(content_data[panel_key], dict):
                        image_key = f"image{i}"
                        if hasattr(data, 'imagePathList') and image_key in data.imagePathList:
                            image_path = "./public" + data.imagePathList[image_key]
                            balloon_detections = balloon_detector.detect_from_path(image_path)
                        else:
                            balloon_detections = []
                        
                        if balloon_detections:
                            content_data[panel_key] = integrate_balloon_detection(
                                content_data[panel_key], 
                                balloon_detections
                            )
            
            elif data.mode == "combined-four-panel":
                if data.komaPath.startswith("data:image"):
                    balloon_detections = balloon_detector.detect_from_base64(data.komaPath)
                else:
                    image_path = "./public" + data.komaPath
                    balloon_detections = balloon_detector.detect_from_path(image_path)
                
                if balloon_detections:
                    panel_balloons = {"panel1": [], "panel2": [], "panel3": [], "panel4": []}
                    
                    for detection in balloon_detections:
                        x_coord = detection["coordinate"][0]
                        y_coord = detection["coordinate"][1]
                        
                        if x_coord >= 0.5 and y_coord < 0.5:
                            panel_balloons["panel1"].append(detection)
                        elif x_coord < 0.5 and y_coord < 0.5:
                            panel_balloons["panel2"].append(detection)
                        elif x_coord >= 0.5 and y_coord >= 0.5:
                            panel_balloons["panel3"].append(detection)
                        else:
                            panel_balloons["panel4"].append(detection)
                    
                    for panel_key, detections in panel_balloons.items():
                        if panel_key in content_data and detections:
                            content_data[panel_key] = integrate_balloon_detection(
                                content_data[panel_key],
                                detections
                            )

        # レスポンス構築（API使用）
        model_name = "gemini-2.5-flash-preview-05-20"
        ret_data = {
            "model": model_name,
            "content_data": content_data,
            "improved_prompt": True  # 改善版プロンプトを使用したことを示すフラグ
        }

        return JSONResponse(content=ret_data)

    except Exception as e:
        log_with_time(f"Error in improved analysis: {e}", level="ERROR")
        return JSONResponse(
            content={"error": f"Failed to analyze image with improved prompt: {str(e)}"},
            status_code=500,
        )


@app.post("/api/analyze-image/")
async def analyze_image_gemini(data: CombinedImageData):
    """Geminiでの画像分析エンドポイント（APIのみ）"""
    try:
        # 検出・分類・可視化のみモード
        if getattr(data, 'detectOnly', False):
            log_with_time("🎯 検出・分類・可視化のみモード開始", level="INFO")
            
            # detectionModeが指定されている場合は検出モードを切り替え
            detection_mode = getattr(data, 'detectionMode', None)
            if detection_mode:
                try:
                    pipeline = get_yuyu_pipeline()
                    if pipeline:
                        multiclass_mode = detection_mode == "multiclass"
                        pipeline.set_mode(multiclass_mode=multiclass_mode)
                        log_with_time(f"🔄 検出モード切り替え: {detection_mode} (multiclass={multiclass_mode})", level="INFO")
                    else:
                        log_with_time("⚠️ パイプラインが利用できないため検出モード切り替えをスキップ", level="WARNING")
                except Exception as e:
                    log_with_time(f"❌ 検出モード切り替えエラー: {str(e)}", level="ERROR")
            
            # 画像パスを取得
            if data.komaPath.startswith("data:image"):
                # Base64の場合は一時ファイルに保存
                import tempfile
                import base64 as b64
                
                header, image_data = data.komaPath.split(",", 1)
                image_bytes = b64.b64decode(image_data)
                
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                    tmp_file.write(image_bytes)
                    image_path = tmp_file.name
            else:
                image_path = "./public" + data.komaPath
            
            try:
                # 画像を読み込み
                import cv2
                image = cv2.imread(image_path)
                if image is None or image.size == 0:
                    raise ValueError(f"画像ファイルが読み込めません: {image_path}")
                
                # 吹き出し検出を実行
                balloon_detector = get_balloon_detector()
                balloon_detections = balloon_detector.detect_balloons(
                    image=image,
                    confidence_threshold=0.25,
                    detect_tails=True
                )
                
                # 人物検出を実行（マルチクラス対応）
                character_detections = []
                if yolo_dinov2_pipeline is not None:
                    try:
                        # 一時的な画像ファイルとして保存して process_image を使用
                        import tempfile
                        import os
                        
                        temp_image_path = None
                        try:
                            # 一時ファイルに画像を保存
                            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                                temp_image_path = temp_file.name
                                cv2.imwrite(temp_image_path, image)
                            
                            # 統一されたprocess_imageメソッドを使用（現在のモードに応じて処理される）
                            log_with_time(f"人物検出実行: マルチクラス={yolo_dinov2_pipeline.multiclass_mode}", level="INFO")
                            processing_result = yolo_dinov2_pipeline.process_image(temp_image_path)
                            
                            # 結果を変換
                            h, w = image.shape[:2]
                            
                            for i, result in enumerate(processing_result.results):
                                x1, y1, x2, y2 = result.detection.bbox
                                identification = result.identification
                                
                                # バウンディングボックスの中心座標を計算（正規化）
                                center_x = (x1 + x2) / 2 / w
                                center_y = (y1 + y2) / 2 / h
                                
                                character_detections.append({
                                    "characterId": f"char_{i + 1}",
                                    "characterName": identification.character_name,
                                    "confidence": identification.confidence,
                                    "classificationConfidence": identification.confidence,
                                    "detectionConfidence": result.detection.confidence,
                                    "boundingBox": {
                                        "x1": float(x1),
                                        "y1": float(y1),
                                        "x2": float(x2),
                                        "y2": float(y2)
                                    },
                                    "coordinate": [center_x, center_y]
                                })
                        
                        finally:
                            # 一時ファイルを削除
                            if temp_image_path and os.path.exists(temp_image_path):
                                os.unlink(temp_image_path)
                                
                    except Exception as e:
                        log_with_time(f"人物検出エラー: {e}", level="WARNING")
                
                # 可視化を実行（必要に応じて）
                visualization_image_base64 = None
                if getattr(data, 'enableVisualization', True):
                    from balloon_detection_integration import draw_tail_shape_results_on_image
                    import base64 as b64
                    
                    # 可視化画像を生成（人物検出結果も含む）
                    visualization_image = draw_tail_shape_results_on_image(
                        image, balloon_detections, character_detections
                    )
                    
                    # Base64に変換
                    _, buffer = cv2.imencode('.png', visualization_image)
                    visualization_image_base64 = b64.b64encode(buffer).decode('utf-8')
                
                # 検出結果を整理
                detection_results = {
                    "balloon_detections": balloon_detections,
                    "detection_count": len(balloon_detections),
                    "character_detections": character_detections,
                    "character_count": len(character_detections)
                }
                
                # 検出結果のみを返す
                ret_data = {
                    "model": "detection-only",
                    "detection_results": detection_results,
                    "mode": "detection_only"
                }
                
                # 可視化画像がある場合は追加
                if visualization_image_base64:
                    ret_data["visualization_image"] = f"data:image/png;base64,{visualization_image_base64}"
                
                log_with_time("✅ 検出・分類・可視化のみモード完了", level="INFO")
                return JSONResponse(content=ret_data)
                
            finally:
                # 一時ファイルを削除
                if data.komaPath.startswith("data:image") and 'image_path' in locals():
                    import os
                    try:
                        os.unlink(image_path)
                    except:
                        pass
        
        # 通常モード（Gemini API使用）
        llm_manager = LLMManager()
        config = llm_manager.config
        
        log_with_time("🌐 Gemini API使用", level="INFO")

        if data.mode == "four-panel":
            # 4コマ形式の場合
            if data.komaPath.startswith("data:image"):
                # Base64形式の画像
                original_path = data.originalImagePath if hasattr(data, 'originalImagePath') else None
                response = fetch_gemini_response_base64(data.prompt, data.komaPath, original_image_path=original_path)
            else:
                # ファイルパスの場合
                image_path = "./public" + data.komaPath
                response = fetch_gemini_response(data.prompt, image_path)
        else:
            # combined-four-panel形式（デフォルト）
            if data.useMultiModal and data.promptMessages:
                # マルチモーダル形式
                log_with_time("🔄 マルチモーダル形式でAPI呼び出し", level="INFO")
                response = fetch_gemini_response_multimodal(data.promptMessages)
            else:
                # 従来の形式
                if data.komaPath.startswith("data:image"):
                    # Base64画像
                    original_path = data.originalImagePath if hasattr(data, 'originalImagePath') else None
                    log_with_time("🖼️ Base64画像でAPI呼び出し", level="INFO")
                    response = fetch_gemini_response_base64(data.prompt, data.komaPath, original_image_path=original_path)
                else:
                    image_path = "./public" + data.komaPath
                    response = fetch_gemini_response(data.prompt, image_path)

        # Geminiのレスポンスから直接テキストを取得
        content = response.text
        log_with_time(f"🔍 レスポンステキスト: {content[:200]}...", level="DEBUG")

        # JSONを抽出（```json ... ``` の形式を想定）
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_content = content[json_start:json_end].strip()
        else:
            # JSONブロックがない場合は全体をJSONとして扱う
            json_content = content

        # JSONをパース
        try:
            content_data = json.loads(json_content)
        except json.JSONDecodeError as e:
            log_with_time(f"JSON parse error: {e}", level="ERROR")
            log_with_time(f"Content to parse: {json_content[:500]}...", level="DEBUG")
            # JSONパースエラーの場合は、正規表現で修正を試みる
            import re
            # シングルクォートをダブルクォートに変換
            json_content = re.sub(r"'([^']*)':", r'"\1":', json_content)
            json_content = re.sub(r":\s*'([^']*)'", r': "\1"', json_content)
            # 末尾のカンマを削除
            json_content = re.sub(r',\s*}', '}', json_content)
            json_content = re.sub(r',\s*]', ']', json_content)
            try:
                content_data = json.loads(json_content)
            except:
                # それでもパースできない場合はエラーレスポンスを返す
                content_data = {"error": "Failed to parse JSON response", "raw_content": content[:1000]}

        # 4コマ形式の場合は各パネルから不要なフィールドを削除
        if isinstance(content_data, dict):
            if "panel1" in content_data:
                # 4コマ形式の場合
                for panel_key in ["panel1", "panel2", "panel3", "panel4"]:
                    if panel_key in content_data and isinstance(content_data[panel_key], dict):
                        content_data[panel_key].pop("charactersNum", None)
                        content_data[panel_key].pop("serifsNum", None)
            else:
                # 単一パネル形式の場合
                content_data.pop("charactersNum", None)
                content_data.pop("serifsNum", None)

        log_with_time(f"Processed content_data (Gemini): {content_data}", level="DEBUG")
        
        # 吹き出し検出を実行（オプション）
        enable_balloon_detection = getattr(data, 'enableBalloonDetection', True)
        if enable_balloon_detection:
            log_with_time("🎈 吹き出し検出を開始します", level="INFO")
            balloon_detector = get_balloon_detector()
            
            if data.mode == "four-panel":
                # 4コマ形式の場合、各パネルで吹き出し検出
                for i, panel_key in enumerate(["panel1", "panel2", "panel3", "panel4"], 1):
                    if panel_key in content_data and isinstance(content_data[panel_key], dict):
                        # 対応する画像を取得
                        image_key = f"image{i}"
                        if hasattr(data, 'imagePathList') and image_key in data.imagePathList:
                            image_path = "./public" + data.imagePathList[image_key]
                            balloon_detections = balloon_detector.detect_from_path(image_path)
                        elif data.komaPath.startswith("data:image"):
                            # Base64画像の場合（結合画像なので個別パネルの検出はスキップ）
                            balloon_detections = []
                        else:
                            balloon_detections = []
                        
                        # 吹き出し検出結果を統合
                        if balloon_detections:
                            content_data[panel_key] = integrate_balloon_detection(
                                content_data[panel_key], 
                                balloon_detections
                            )
                            log_with_time(f"✅ {panel_key}: {len(balloon_detections)}個の吹き出しを検出", level="INFO")
            
            elif data.mode == "combined-four-panel":
                # 結合4コマ形式の場合
                if data.komaPath.startswith("data:image"):
                    # Base64画像全体で吹き出し検出
                    balloon_detections = balloon_detector.detect_from_base64(data.komaPath)
                else:
                    image_path = "./public" + data.komaPath
                    balloon_detections = balloon_detector.detect_from_path(image_path)
                
                if balloon_detections:
                    # 各パネルに吹き出しを割り当て（座標ベース）
                    panel_balloons = {"panel1": [], "panel2": [], "panel3": [], "panel4": []}
                    
                    for detection in balloon_detections:
                        # 座標から所属パネルを判定（2x2レイアウトを想定）
                        x_coord = detection["coordinate"][0]
                        y_coord = detection["coordinate"][1]
                        
                        if x_coord >= 0.5 and y_coord < 0.5:
                            panel_balloons["panel1"].append(detection)
                        elif x_coord < 0.5 and y_coord < 0.5:
                            panel_balloons["panel2"].append(detection)
                        elif x_coord >= 0.5 and y_coord >= 0.5:
                            panel_balloons["panel3"].append(detection)
                        else:
                            panel_balloons["panel4"].append(detection)
                    
                    # 各パネルに吹き出し情報を統合
                    for panel_key, detections in panel_balloons.items():
                        if panel_key in content_data and detections:
                            content_data[panel_key] = integrate_balloon_detection(
                                content_data[panel_key],
                                detections
                            )
                            log_with_time(f"✅ {panel_key}: {len(detections)}個の吹き出しを検出", level="INFO")

        # 画像可視化処理（有効な場合のみ）
        visualization_image_base64 = None
        if getattr(data, 'enableVisualization', False) and data.enableBalloonDetection:
            try:
                import cv2
                import base64
                import io
                from PIL import Image
                import numpy as np
                
                # 元画像を読み込み
                if data.komaPath.startswith("data:image"):
                    # Base64画像の場合
                    header, encoded = data.komaPath.split(",", 1)
                    image_data = base64.b64decode(encoded)
                    pil_image = Image.open(io.BytesIO(image_data))
                    image_np = np.array(pil_image)
                    if len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA -> RGB
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                else:
                    # ファイルパスの場合
                    image_path = "./public" + data.komaPath
                    image_np = cv2.imread(image_path)
                    if image_np is not None:
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                
                # 吹き出し検出結果があれば可視化
                if image_np is not None and 'balloon_detections' in locals() and balloon_detections:
                    # 尻尾形状分類結果を画像上に描画
                    visualized_image = draw_tail_shape_results_on_image(image_np, balloon_detections)
                    
                    # PILに変換してBase64にエンコード
                    pil_result = Image.fromarray(visualized_image)
                    buffer = io.BytesIO()
                    pil_result.save(buffer, format='PNG')
                    buffer.seek(0)
                    visualization_image_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    log_with_time(f"✅ 画像可視化完了: {len(balloon_detections)}個の検出結果を描画", level="INFO")
                
            except Exception as e:
                log_with_time(f"❌ 画像可視化エラー: {e}", level="ERROR")
                import traceback
                traceback.print_exc()

        # レスポンスデータを構築（API使用）
        model_name = "gemini-2.5-flash-preview-05-20"
        ret_data = {
            "model": model_name,
            "content_data": content_data,
        }
        
        # 可視化画像がある場合は追加
        if visualization_image_base64:
            ret_data["visualization_image"] = f"data:image/png;base64,{visualization_image_base64}"

        return JSONResponse(content=ret_data)

    except Exception as e:
        log_with_time(f"Error processing Gemini response: {e}", level="ERROR")
        return JSONResponse(
            content={"error": f"Failed to analyze image with Gemini: {str(e)}"},
            status_code=500,
        )




@app.get("/api/load-json")
async def load_json(dataName: str, currentIndex: int):
    """保存済みのフォームJSONデータを読み込み"""
    try:
        json_path = f"public/saved_json/imageData_{dataName}_{currentIndex}.json"
        log_with_time(f"Load JSON request: {json_path}", level="DEBUG")
        
        if not os.path.exists(json_path):
            log_with_time(f"JSON file not found: {json_path}", level="INFO")
            return JSONResponse(content=None, status_code=200)
        
        with open(json_path, "r", encoding="utf-8") as f:
            image_data = json.load(f)
        
        log_with_time(f"JSON loaded successfully: {json_path}", level="INFO")
        return JSONResponse(content=image_data)
    except Exception as e:
        log_with_time(f"❌ JSON読み込みエラー: {str(e)}", level="ERROR")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.post("/api/save-json")
async def save_json(data: ImageData):
    log_with_time(f"Save JSON request: {data}", level="DEBUG")
    json_path = f"public/saved_json/imageData_{data.dataName}_{data.currentIndex}.json"
    with open(json_path, "w") as f:
        f.write(json.dumps(data.imageData))
    return {"result": "saved!"}


@app.post("/api/save-csv")
async def save_csv(request: Request):
    try:
        # リクエストボディを取得
        raw_data = await request.json()
        log_with_time(f"📥 Save CSV リクエスト受信: {raw_data}", level="DEBUG")
        
        # Pydanticモデルでバリデーション
        data = ImageData(**raw_data)
        log_with_time(f"✅ ImageDataバリデーション成功: {data.dataName}", level="DEBUG")
    except Exception as e:
        log_with_time(f"❌ Save CSV バリデーションエラー: {str(e)}", level="ERROR")
        log_with_time(f"❌ リクエストデータ: {raw_data if 'raw_data' in locals() else 'データ取得失敗'}", level="ERROR")
        raise HTTPException(status_code=422, detail=f"バリデーションエラー: {str(e)}")
    
    log_with_time(f"Save CSV request: {data}", level="DEBUG")
    
    # フロントエンドから送信されたdataNameを直接使用
    data_name = data.dataName
    
    # JSON形式で保存（ユーザーの要望に従い）
    json_data = {
        "dataName": data_name,
        "currentIndex": data.currentIndex,
        "komaPath": data.komaPath,
        "imageData": data.imageData.dict() if hasattr(data.imageData, 'dict') else dict(data.imageData),
        "summary": data.summary,
        "timestamp": datetime.now().isoformat()
    }
    
    json_filename = f"imageData_{data_name}_{data.currentIndex}.json"
    json_path = f"public/saved_json/{json_filename}"
    
    # ディレクトリが存在しない場合は作成
    os.makedirs("public/saved_json", exist_ok=True)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    log_with_time(f"✅ JSON保存完了: {json_path}")
    return {"message": f"Data saved successfully as JSON: {json_filename}"}
    # except Exception as e:
    # raise HTTPException(status_code=500, detail=str(e))


# ================== Human in the Loop 機能 ==================

def calculate_confidence_scores(data: dict, model: str = "gemini") -> Dict[str, float]:
    """AIの提案に対する信頼度スコアを計算"""
    confidence_scores = {}
    
    # モデル別の基本信頼度
    base_confidence = {
        "gemini": 0.85,
        "gpt-4": 0.88,
        "claude": 0.86
    }.get(model, 0.80)
    
    # キャラクター認識の信頼度
    for i, char in enumerate(data.get("characters", [])):
        char_key = f"characters.{i}"
        
        # 名前の信頼度（既知のキャラクターかどうか）
        known_characters = ["縁", "唯", "相川", "ふみ", "みほ", "千穂", "岡野", "佳"]
        if char.get("character") in known_characters:
            confidence_scores[f"{char_key}.character"] = base_confidence * 1.1
        else:
            confidence_scores[f"{char_key}.character"] = base_confidence * 0.7
        
        # セリフの信頼度（長さベース）
        serif_length = len(char.get("serif", ""))
        if serif_length > 0:
            confidence_scores[f"{char_key}.serif"] = min(base_confidence * (1 + serif_length / 50), 0.95)
        
        # 表情の信頼度
        confidence_scores[f"{char_key}.expression"] = base_confidence * 0.9
        
        # 位置の信頼度
        confidence_scores[f"{char_key}.position"] = base_confidence * 0.95
    
    # シーンデータの信頼度
    if "sceneData" in data:
        confidence_scores["sceneData.scene"] = base_confidence * 0.9
        confidence_scores["sceneData.location"] = base_confidence * 0.85
        confidence_scores["sceneData.backgroundEffects"] = base_confidence * 0.8
    
    return confidence_scores


def calculate_field_similarity(old_value: Any, new_value: Any) -> float:
    """2つの値の類似度を計算"""
    if old_value == new_value:
        return 1.0
    
    if isinstance(old_value, str) and isinstance(new_value, str):
        # 文字列の類似度
        return SequenceMatcher(None, old_value, new_value).ratio()
    
    if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
        # 数値の類似度
        if old_value == 0 and new_value == 0:
            return 1.0
        max_val = max(abs(old_value), abs(new_value))
        if max_val == 0:
            return 1.0
        return 1.0 - abs(old_value - new_value) / max_val
    
    return 0.0


def extract_differences(old_data: dict, new_data: dict, path: str = "") -> List[DiffAnalysis]:
    """2つのデータの差分を抽出"""
    differences = []
    
    # old_dataのキーをチェック
    for key in old_data:
        current_path = f"{path}.{key}" if path else key
        
        if key not in new_data:
            # 削除された項目
            differences.append(DiffAnalysis(
                field_path=current_path,
                old_value=old_data[key],
                new_value=None,
                change_type="deleted",
                similarity=0.0
            ))
        elif isinstance(old_data[key], dict) and isinstance(new_data[key], dict):
            # 再帰的に差分を抽出
            differences.extend(extract_differences(old_data[key], new_data[key], current_path))
        elif isinstance(old_data[key], list) and isinstance(new_data[key], list):
            # リストの差分
            for i, (old_item, new_item) in enumerate(zip(old_data[key], new_data[key])):
                if isinstance(old_item, dict) and isinstance(new_item, dict):
                    differences.extend(extract_differences(old_item, new_item, f"{current_path}.{i}"))
                elif old_item != new_item:
                    differences.append(DiffAnalysis(
                        field_path=f"{current_path}.{i}",
                        old_value=old_item,
                        new_value=new_item,
                        change_type="modified",
                        similarity=calculate_field_similarity(old_item, new_item)
                    ))
            
            # リストの長さが異なる場合
            if len(old_data[key]) < len(new_data[key]):
                for i in range(len(old_data[key]), len(new_data[key])):
                    differences.append(DiffAnalysis(
                        field_path=f"{current_path}.{i}",
                        old_value=None,
                        new_value=new_data[key][i],
                        change_type="added",
                        similarity=0.0
                    ))
            elif len(old_data[key]) > len(new_data[key]):
                for i in range(len(new_data[key]), len(old_data[key])):
                    differences.append(DiffAnalysis(
                        field_path=f"{current_path}.{i}",
                        old_value=old_data[key][i],
                        new_value=None,
                        change_type="deleted",
                        similarity=0.0
                    ))
        elif old_data[key] != new_data[key]:
            # 値が変更された
            differences.append(DiffAnalysis(
                field_path=current_path,
                old_value=old_data[key],
                new_value=new_data[key],
                change_type="modified",
                similarity=calculate_field_similarity(old_data[key], new_data[key])
            ))
    
    # new_dataの新しいキーをチェック
    for key in new_data:
        if key not in old_data:
            current_path = f"{path}.{key}" if path else key
            differences.append(DiffAnalysis(
                field_path=current_path,
                old_value=None,
                new_value=new_data[key],
                change_type="added",
                similarity=0.0
            ))
    
    return differences


def get_history_file_path(image_path: str) -> str:
    """画像パスから履歴ファイルパスを生成"""
    # 画像名を取得して分かりやすいファイル名にする
    # 例: yuyu10/pages_corrected/combined_koma/four_panel_062.jpg -> four_panel_062_history.json
    image_name = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
    
    # 画像名だけでファイル名を作成（最もわかりやすい）
    # ファイル名の重複を避けるため、パスに含まれる特徴的な部分を追加
    path_parts = image_path.split('/')
    
    # yuyu10などのデータセット名を含める場合
    dataset_name = ""
    if len(path_parts) > 0 and path_parts[0]:
        dataset_name = path_parts[0]
    
    # 基本形式: {画像名}_history.json
    # 重複する可能性がある場合のみデータセット名を追加
    base_name = f"{image_name}_history.json"
    
    # より複雑なケースでは、短いハッシュを使用
    if len(image_path) > 80:  # 80文字以上の場合のみハッシュ使用
        path_hash = hashlib.md5(image_path.encode()).hexdigest()[:6]
        return f"public/revision_history/{image_name}_{path_hash}_history.json"
    
    return f"public/revision_history/{base_name}"


def get_ai_proposal_file_path(image_path: str, timestamp: str) -> str:
    """AI提案ファイルパスを生成"""
    # 画像名を取得して分かりやすいファイル名にする
    image_name = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
    # タイムスタンプをファイル名に適した形式に変換（日時のみ）
    timestamp_clean = timestamp.replace(':', '-').replace('.', '-')[:19]
    
    # AI提案ファイルの場合は、画像名 + タイムスタンプで十分識別可能
    return f"public/ai_proposals/{image_name}_{timestamp_clean}.json"


@app.post("/api/save-detection-debug")
async def save_detection_debug(data: dict):
    """AI検出デバッグ情報を保存"""
    try:
        debug_dir = data.get("debugDir", "tmp/ai_detection_debug/unknown")
        step = data.get("step", "unknown")
        timestamp = data.get("timestamp", datetime.now().isoformat())
        
        # ディレクトリを作成
        os.makedirs(debug_dir, exist_ok=True)
        
        # デバッグ情報をファイルに保存
        debug_file = f"{debug_dir}/{step}.json"
        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "step": step,
                "imagePathList": data.get("imagePathList", {}),
                "detectionResults": data.get("detectionResults", []),
                "enhancedPrompt": data.get("enhancedPrompt", ""),
                "promptLength": data.get("promptLength", 0),
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "step_description": {
                        "1_detection_results": "AI検出結果（キャラクター・吹き出し）",
                        "2_enhanced_prompt": "拡張プロンプト（基本版）",
                        "3_improved_prompt": "改良プロンプト（詳細版）"
                    }.get(step, "不明なステップ")
                }
            }, f, ensure_ascii=False, indent=2)
        
        log_with_time(f"💾 AI検出デバッグ情報保存: {debug_file}", level="INFO")
        
        return JSONResponse(content={
            "status": "success",
            "debugFile": debug_file,
            "step": step
        })
        
    except Exception as e:
        log_with_time(f"❌ AI検出デバッグ情報保存エラー: {str(e)}", level="ERROR")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@app.post("/api/save-ai-proposal")
async def save_ai_proposal(data: Dict[str, Any]):
    """AI提案を保存"""
    try:
        timestamp = datetime.now().isoformat()
        image_path = data.get("imagePath")
        model = data.get("model", "gemini")
        proposal_data = data.get("proposalData", {})
        processing_time = data.get("processingTime")
        api_cost = data.get("apiCost")
        
        # 信頼度スコアを計算
        confidence_scores = calculate_confidence_scores(proposal_data, model)
        
        # AI提案データを作成
        ai_proposal = AIProposal(
            image_path=image_path,
            timestamp=timestamp,
            model=model,
            proposal=proposal_data,
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            api_cost=api_cost
        )
        
        # ディレクトリを作成
        Path("public/ai_proposals").mkdir(parents=True, exist_ok=True)
        Path("public/revision_history").mkdir(parents=True, exist_ok=True)
        
        # AI提案を個別ファイルとして保存
        proposal_path = get_ai_proposal_file_path(image_path, timestamp)
        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(ai_proposal.dict(), f, ensure_ascii=False, indent=2)
        
        # 履歴ファイルを更新または作成
        history_path = get_history_file_path(image_path)
        
        if os.path.exists(history_path):
            # 既存の履歴を読み込む
            with open(history_path, "r", encoding="utf-8") as f:
                history_data = json.load(f)
            history = RevisionHistory(**history_data)
        else:
            # 新しい履歴を作成
            history = RevisionHistory(
                image_path=image_path,
                created_at=timestamp,
                last_updated=timestamp,
                total_revisions=0,
                ai_proposals=[],
                revisions=[],
                current_data=proposal_data
            )
        
        # AI提案を履歴に追加
        history.ai_proposals.append(ai_proposal)
        
        # AI提案を修正履歴にも追加
        revision_entry = RevisionEntry(
            timestamp=timestamp,
            revision_type="ai_proposal",
            changes=[],  # 初回なので差分なし
            editor="ai",
            confidence=sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0,
            notes=f"Initial AI proposal using {model}"
        )
        history.revisions.append(revision_entry)
        history.total_revisions += 1
        history.last_updated = timestamp
        history.current_data = proposal_data
        
        # 履歴を保存
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history.dict(), f, ensure_ascii=False, indent=2)
        
        return JSONResponse(content={
            "status": "success",
            "proposalId": f"{image_path}_{timestamp}",
            "confidenceScores": confidence_scores,
            "historyPath": history_path
        })
        
    except Exception as e:
        log_with_time(f"Error saving AI proposal: {e}", level="ERROR")
        return JSONResponse(
            content={"error": f"Failed to save AI proposal: {str(e)}"},
            status_code=500
        )


@app.post("/api/save-human-revision")
async def save_human_revision(data: Dict[str, Any]):
    """人間による修正を保存し、差分を記録"""
    try:
        timestamp = datetime.now().isoformat()
        image_path = data.get("imagePath")
        # フロントエンドからはrevisionDataで送られてくる場合もある
        new_data = data.get("imageData") or data.get("revisionData", {})
        notes = data.get("notes", "")
        
        # ディレクトリを作成
        Path("public/revision_history").mkdir(parents=True, exist_ok=True)
        
        history_path = get_history_file_path(image_path)
        
        if not os.path.exists(history_path):
            # 履歴が存在しない場合は新規作成
            log_with_time(f"Creating new history for {image_path}", level="INFO")
            history = RevisionHistory(
                image_path=image_path,
                created_at=timestamp,
                last_updated=timestamp,
                total_revisions=0,
                ai_proposals=[],
                revisions=[],
                current_data={}  # 空の状態から開始
            )
        else:
            # 既存の履歴を読み込む
            with open(history_path, "r", encoding="utf-8") as f:
                history_data = json.load(f)
            history = RevisionHistory(**history_data)
        
        # 差分を計算
        old_data = history.current_data
        differences = extract_differences(old_data, new_data)
        
        # 差分をDictに変換
        changes = [diff.dict() for diff in differences]
        
        # 修正エントリを作成
        revision_entry = RevisionEntry(
            timestamp=timestamp,
            revision_type="human_edit",
            changes=changes,
            editor="human",
            notes=notes
        )
        
        # 履歴を更新
        history.revisions.append(revision_entry)
        history.total_revisions += 1
        history.last_updated = timestamp
        history.current_data = new_data
        
        # 履歴を保存
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history.dict(), f, ensure_ascii=False, indent=2)
        
        # 既存のJSONファイルも更新（後方互換性のため）
        json_path = f"public/saved_json/imageData_{data.get('dataName')}_{data.get('currentIndex')}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        
        return JSONResponse(content={
            "status": "success",
            "revisionId": f"{image_path}_{timestamp}",
            "changesCount": len(changes),
            "changes": changes[:10]  # 最初の10件の変更を返す
        })
        
    except Exception as e:
        log_with_time(f"Error saving human revision: {e}", level="ERROR")
        return JSONResponse(
            content={"error": f"Failed to save human revision: {str(e)}"},
            status_code=500
        )


@app.get("/api/get-revision-history/{image_path:path}")
async def get_revision_history(image_path: str):
    """特定の画像の修正履歴を取得"""
    try:
        history_path = get_history_file_path(image_path)
        
        if not os.path.exists(history_path):
            return JSONResponse(content={"revisions": [], "aiProposals": []})
        
        with open(history_path, "r", encoding="utf-8") as f:
            history_data = json.load(f)
        
        return JSONResponse(content=history_data)
        
    except Exception as e:
        log_with_time(f"Error getting revision history: {e}", level="ERROR")
        return JSONResponse(
            content={"error": f"Failed to get revision history: {str(e)}"},
            status_code=500
        )


@app.post("/api/export-learning-data")
async def export_learning_data(data: Dict[str, Any]):
    """人間のフィードバックから学習データを生成"""
    try:
        export_type = data.get("exportType", "all")  # "all", "corrections_only", "high_confidence"
        min_revisions = data.get("minRevisions", 2)  # 最小修正回数
        
        learning_data = []
        history_dir = Path("public/revision_history")
        
        if not history_dir.exists():
            return JSONResponse(content={"error": "No revision history found"}, status_code=404)
        
        # すべての履歴ファイルを処理
        for history_file in history_dir.glob("*_history.json"):
            with open(history_file, "r", encoding="utf-8") as f:
                history = RevisionHistory(**json.load(f))
            
            # フィルタリング条件
            if history.total_revisions < min_revisions:
                continue
            
            # AI提案と人間の最終データの差分を学習データとして抽出
            if history.ai_proposals and history.revisions:
                for ai_proposal in history.ai_proposals:
                    # 人間による修正を探す
                    human_edits = [r for r in history.revisions if r.editor == "human" and r.timestamp > ai_proposal.timestamp]
                    
                    if human_edits:
                        learning_entry = {
                            "image_path": history.image_path,
                            "ai_proposal": ai_proposal.proposal,
                            "human_correction": history.current_data,
                            "confidence_scores": ai_proposal.confidence_scores,
                            "total_edits": len(human_edits),
                            "significant_changes": []
                        }
                        
                        # 重要な変更を抽出
                        for edit in human_edits:
                            significant = [c for c in edit.changes if c.get("similarity", 1.0) < 0.5]
                            learning_entry["significant_changes"].extend(significant)
                        
                        # エクスポートタイプによるフィルタリング
                        if export_type == "corrections_only" and not learning_entry["significant_changes"]:
                            continue
                        
                        if export_type == "high_confidence":
                            avg_confidence = sum(ai_proposal.confidence_scores.values()) / len(ai_proposal.confidence_scores)
                            if avg_confidence < 0.8:
                                continue
                        
                        learning_data.append(learning_entry)
        
        # 学習データを保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"public/learning_data/export_{timestamp}.json"
        Path("public/learning_data").mkdir(parents=True, exist_ok=True)
        
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump({
                "export_info": {
                    "timestamp": timestamp,
                    "export_type": export_type,
                    "min_revisions": min_revisions,
                    "total_entries": len(learning_data)
                },
                "data": learning_data
            }, f, ensure_ascii=False, indent=2)
        
        # CSVでもエクスポート（簡易版）
        csv_path = f"public/learning_data/export_{timestamp}.csv"
        csv_rows = []
        
        for entry in learning_data:
            for change in entry.get("significant_changes", []):
                csv_rows.append({
                    "image_path": entry["image_path"],
                    "field": change.get("field_path"),
                    "ai_value": change.get("old_value"),
                    "human_value": change.get("new_value"),
                    "change_type": change.get("change_type"),
                    "similarity": change.get("similarity")
                })
        
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            df.to_csv(csv_path, index=False, encoding="utf-8")
        
        return JSONResponse(content={
            "status": "success",
            "exportPath": export_path,
            "csvPath": csv_path if csv_rows else None,
            "totalEntries": len(learning_data),
            "summary": {
                "totalImages": len(set(e["image_path"] for e in learning_data)),
                "totalChanges": sum(len(e["significant_changes"]) for e in learning_data)
            }
        })
        
    except Exception as e:
        log_with_time(f"Error exporting learning data: {e}", level="ERROR")
        return JSONResponse(
            content={"error": f"Failed to export learning data: {str(e)}"},
            status_code=500
        )


# ================== YOLO+DINOv2 検出エンドポイント ==================

@app.post("/api/detect-characters-yolo-dinov2")
async def detect_characters_yolo_dinov2(request: YoloDinov2DetectionRequest):
    """YOLO+DINOv2でキャラクター検出・識別を実行"""
    try:
        if yolo_dinov2_pipeline is None:
            return JSONResponse(
                content={"error": "YOLO+DINOv2パイプラインが初期化されていません"},
                status_code=503
            )
        
        # detectionModeが指定されている場合は検出モードを切り替え
        if request.detectionMode:
            try:
                multiclass_mode = request.detectionMode == "multiclass"
                yolo_dinov2_pipeline.set_mode(multiclass_mode=multiclass_mode)
                log_with_time(f"🔄 検出モード切り替え: {request.detectionMode} (multiclass={multiclass_mode})", level="INFO")
            except Exception as e:
                log_with_time(f"❌ 検出モード切り替えエラー: {str(e)}", level="ERROR")
        
        # キャラクター名のマッピング
        CHARACTER_NAME_MAP = {
            "yuzuko": "野々原ゆずこ",
            "yukari": "日向縁",
            "yui": "櫟井唯",
            "yoriko": "松本頼子",
            "chiho": "相川千穂",
            "kei": "岡野佳",
            "fumi": "長谷川ふみ",
            "unknown": "不明"
        }
        
        # 画像の準備
        if request.komaPath.startswith("data:image"):
            # Base64データの場合
            import base64
            import io
            from PIL import Image
            import numpy as np
            
            # Base64デコード
            image_data = request.komaPath.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            # OpenCV形式に変換
            image_cv = np.array(image.convert('RGB'))[:, :, ::-1].copy()
        else:
            # ファイルパスの場合
            import cv2
            image_path = "./public/" + request.komaPath
            image_cv = cv2.imread(image_path)
            if image_cv is None:
                return JSONResponse(
                    content={"error": f"画像読み込みエラー: {image_path}"},
                    status_code=400
                )
        
        # 一時的な画像ファイルとして保存して処理
        import tempfile
        import os
        
        temp_image_path = None
        try:
            # 一時ファイルに画像を保存
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_image_path = temp_file.name
                cv2.imwrite(temp_image_path, image_cv)
            
            # 閾値の一時的な変更
            original_detection_threshold = yolo_dinov2_pipeline.detection_conf_threshold
            original_classification_threshold = yolo_dinov2_pipeline.classification_conf_threshold
            
            yolo_dinov2_pipeline.detection_conf_threshold = request.detectionThreshold
            yolo_dinov2_pipeline.classification_conf_threshold = request.classificationThreshold
            
            # 統一されたprocess_imageメソッドを使用（現在のモードに応じて処理される）
            log_with_time(f"YOLO+DINOv2検出実行: モード={current_detection_mode}, マルチクラス={yolo_dinov2_pipeline.multiclass_mode}, 検出閾値={request.detectionThreshold}", level="INFO")
            processing_result = yolo_dinov2_pipeline.process_image(temp_image_path)
            
            # 閾値を元に戻す
            yolo_dinov2_pipeline.detection_conf_threshold = original_detection_threshold
            yolo_dinov2_pipeline.classification_conf_threshold = original_classification_threshold
            
            # 結果をフォーマット
            characters = []
            h, w = image_cv.shape[:2]
            
            for result in processing_result.results:
                x1, y1, x2, y2 = result.detection.bbox
                identification = result.identification
                
                # バウンディングボックスの中心座標を計算（正規化）
                center_x = (x1 + x2) / 2 / w
                center_y = (y1 + y2) / 2 / h
                
                # 位置を判定（左、中央、右）
                if center_x < 0.33:
                    position = "左"
                elif center_x < 0.67:
                    position = "中央"
                else:
                    position = "右"
                
                # 表情を推定（信頼度ベース）
                if identification.confidence > 0.8:
                    expression = "通常"
                elif identification.confidence > 0.6:
                    expression = "微笑"
                else:
                    expression = "不明"
                
                # バウンディングボックスのサイズを画像比率で計算
                bbox_width = (x2 - x1) / w
                bbox_height = (y2 - y1) / h
                character_size_str = f"{bbox_width:.2f}, {bbox_height:.2f}"
                
                # 英語名を日本語名にマッピング
                japanese_name = CHARACTER_NAME_MAP.get(identification.character_name, identification.character_name)
                
                # 人物認識（分類）信頼度を小数点2桁で文字列化
                classification_confidence_str = f"{identification.confidence:.2f}"
                
                characters.append({
                    "character": japanese_name,
                    "faceDirection": "",  # 空欄
                    "position": classification_confidence_str,  # 人物認識信頼度を表示
                    "shotType": "",  # 空欄
                    "characterSize": character_size_str,
                    "expression": "",  # 空欄
                    "clothing": "",  # 空欄
                    "isVisible": True,
                    "coordinate": [center_x, center_y],
                    "detectionConfidence": result.detection.confidence,
                    "classificationConfidence": identification.confidence,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })
        
        finally:
            # 一時ファイルを削除
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
        
        # キャラクターを右から左の順序でソート（X座標の降順）
        characters.sort(key=lambda c: -c.get("coordinate", [0, 0])[0])
        
        # 検出されなかった場合は空のキャラクターを1つ追加
        if not characters:
            characters.append({
                "character": "",
                "faceDirection": "",
                "position": "",
                "shotType": "",
                "characterSize": "0.00, 0.00",
                "expression": "",
                "clothing": "",
                "isVisible": True,
                "coordinate": [0, 0],
                "detectionConfidence": 0.0,
                "classificationConfidence": 0.0
            })
        
        # 検出結果を可視化（オプション）
        visualization = None
        visualization_error = None
        
        print(f"🔍 request.visualize = {request.visualize}")
        if request.visualize:
            import io
            import base64
            from PIL import Image
            
            # 直接的な可視化処理（既に読み込まれている画像データを使用）
            try:
                print(f"可視化開始: image_cv shape={image_cv.shape}")
                print(f"processing_result.results数: {len(processing_result.results)}")
                
                # 既存のimage_cvをRGB形式に変換
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                from matplotlib import font_manager
                import cv2
                
                # RGB形式に変換
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                
                # プロット作成
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax.imshow(image_rgb)
                
                # 日本語フォント設定を試行
                try:
                    font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
                    if os.path.exists(font_path):
                        prop = font_manager.FontProperties(fname=font_path)
                        plt.rcParams['font.family'] = prop.get_name()
                except:
                    pass
                
                # キャラクター別色設定
                CHARACTER_COLORS = {
                    'yuzuko': (255, 105, 180),  # Hot Pink
                    'yukari': (30, 144, 255),   # Dodger Blue
                    'yui': (50, 205, 50),       # Lime Green
                    'yoriko': (255, 165, 0),    # Orange
                    'chiho': (138, 43, 226),    # Blue Violet
                    'kei': (220, 20, 60),       # Crimson
                    'fumi': (0, 206, 209),      # Dark Turquoise
                    'unknown': (128, 128, 128)  # Gray
                }
                
                # 重複除去処理（IoU 0.3閾値）
                filtered_results = []
                for i, result in enumerate(processing_result.results):
                    is_duplicate = False
                    for j, other_result in enumerate(filtered_results):
                        # IoU計算
                        x1_1, y1_1, x2_1, y2_1 = result.detection.bbox
                        x1_2, y1_2, x2_2, y2_2 = other_result.detection.bbox
                        
                        # 交差領域
                        intersection_x1 = max(x1_1, x1_2)
                        intersection_y1 = max(y1_1, y1_2)
                        intersection_x2 = min(x2_1, x2_2)
                        intersection_y2 = min(y2_1, y2_2)
                        
                        if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                            intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                            union_area = area1 + area2 - intersection_area
                            iou = intersection_area / union_area if union_area > 0 else 0
                            
                            if iou > 0.3:  # 重複と判定
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        filtered_results.append(result)
                
                print(f"重複除去: {len(processing_result.results)} -> {len(filtered_results)}人")
                
                # 重複除去済み結果を描画
                for i, face_result in enumerate(filtered_results):
                    x1, y1, x2, y2 = face_result.detection.bbox
                    character_en = face_result.identification.character_name
                    char_conf = face_result.identification.confidence
                    
                    # 英語名を日本語名に変換
                    character_jp = CHARACTER_NAME_MAP.get(character_en, character_en)
                    
                    # 色の選択
                    color_rgb = CHARACTER_COLORS.get(character_en, CHARACTER_COLORS['unknown'])
                    color = tuple(c/255.0 for c in color_rgb)
                    
                    # バウンディングボックス（太線）
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=4, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # ラベルを内側の適切な位置に配置
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    text_x = x1 + bbox_width * 0.02  # 左端から少し内側
                    text_y = y1 + bbox_height * 0.95  # 下端近く
                    
                    label = f'{character_jp} ({char_conf:.1%})'
                    
                    ax.text(
                        text_x, text_y, label,
                        color='white', fontsize=12, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='white', linewidth=1),
                        verticalalignment='bottom'
                    )
                
                ax.axis('off')
                mode_str = "マルチクラス検出" if yolo_dinov2_pipeline.multiclass_mode else "顔検出+認識"
                character_names = [CHARACTER_NAME_MAP.get(r.identification.character_name, r.identification.character_name) for r in filtered_results]
                character_list = ", ".join(character_names) if character_names else "なし"
                title = f'{mode_str} | 検出: {len(filtered_results)}人 | キャラクター: {character_list}'
                ax.set_title(title, fontsize=14, weight='bold')
                
                plt.tight_layout()
                
                # BufferIOを使用してPNG形式で画像を取得（Mac互換）
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                
                # PILで読み込んでBase64エンコード
                pil_image = Image.open(buf)
                final_buf = io.BytesIO()
                pil_image.save(final_buf, format='PNG')
                final_buf.seek(0)
                visualization_base64 = base64.b64encode(final_buf.read()).decode('utf-8')
                
                plt.close()
                buf.close()
                final_buf.close()
                
                # 可視化画像を結果に追加
                visualization = f"data:image/png;base64,{visualization_base64}"
                print("✅ 直接可視化処理完了（重複除去・ラベル内側配置適用）")
            except Exception as e:
                import traceback
                visualization_error = f"エラー: {str(e)} | トレースバック: {traceback.format_exc()[:500]}"
                print(f"❌ 可視化処理エラー: {e}")
                print(f"詳細なエラー: {traceback.format_exc()}")
                visualization = None
        
        # 結果を返す
        mode_info = "マルチクラス検出" if yolo_dinov2_pipeline.multiclass_mode else "顔検出+認識"
        model_description = "YOLO11l (8クラス直接検出)" if yolo_dinov2_pipeline.multiclass_mode else "YOLO11l + DINOv2"
        
        result = {
            "characters": characters,
            "charactersNum": len([c for c in characters if c["character"]]),
            "detectionStats": {
                "totalFaces": len(processing_result.results),
                "identifiedFaces": len([c for c in characters if c["character"]]),
                "processingTime": processing_result.processing_time
            },
            "model": model_description,
            "detectionMode": current_detection_mode,
            "modeDescription": mode_info,
            "modelInfo": {
                "detection": "YOLO11l (anime face)" if not yolo_dinov2_pipeline.multiclass_mode else "YOLO11l (multiclass)",
                "classification": "DINOv2 (8 classes)" if not yolo_dinov2_pipeline.multiclass_mode else "直接8クラス検出"
            },
            "debug": {
                "visualize_requested": request.visualize,
                "visualization_created": visualization is not None,
                "visualization_error": visualization_error if 'visualization_error' in locals() else None
            }
        }
        
        if visualization:
            result["visualization"] = visualization
        
        return JSONResponse(content=result)
        
    except Exception as e:
        log_with_time(f"YOLO+DINOv2検出エラー: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"検出処理エラー: {str(e)}"},
            status_code=500
        )


# ================== 検出モード切り替えエンドポイント ==================

@app.post("/api/detection-mode")
async def set_detection_mode(request: DetectionModeRequest) -> DetectionModeResponse:
    """検出モードの切り替え"""
    global current_detection_mode, yolo_dinov2_pipeline
    
    try:
        if request.mode not in ["face_recognition", "multiclass"]:
            return DetectionModeResponse(
                success=False,
                current_mode=current_detection_mode,
                message="無効なモードです。'face_recognition' または 'multiclass' を指定してください。"
            )
        
        if yolo_dinov2_pipeline is None:
            return DetectionModeResponse(
                success=False,
                current_mode=current_detection_mode,
                message="YOLO+DINOv2パイプラインが初期化されていません。"
            )
        
        # マルチクラスモードの切り替え
        multiclass_mode = (request.mode == "multiclass")
        yolo_dinov2_pipeline.set_mode(multiclass_mode)
        
        current_detection_mode = request.mode
        mode_name = "マルチクラス検出" if multiclass_mode else "顔検出+認識"
        
        log_with_time(f"検出モードを{mode_name}に切り替えました", level="INFO")
        
        return DetectionModeResponse(
            success=True,
            current_mode=current_detection_mode,
            message=f"検出モードを{mode_name}に切り替えました。"
        )
        
    except Exception as e:
        log_with_time(f"検出モード切り替えエラー: {e}", level="ERROR")
        return DetectionModeResponse(
            success=False,
            current_mode=current_detection_mode,
            message=f"モード切り替えに失敗しました: {str(e)}"
        )


@app.get("/api/detection-mode")
async def get_detection_mode() -> DetectionModeResponse:
    """現在の検出モードを取得"""
    mode_name = "マルチクラス検出" if current_detection_mode == "multiclass" else "顔検出+認識"
    
    return DetectionModeResponse(
        success=True,
        current_mode=current_detection_mode,
        message=f"現在のモード: {mode_name}"
    )


@app.post("/api/detect-balloons")
async def detect_balloons(request: Dict[str, Any]):
    """吹き出し検出専用エンドポイント"""
    try:
        balloon_detector = get_balloon_detector()
        
        # 画像パスまたはBase64データから検出
        if "imagePath" in request:
            image_path = "./public" + request["imagePath"]
            detections = balloon_detector.detect_from_path(
                image_path,
                confidence_threshold=request.get("confidenceThreshold", 0.25),
                max_det=request.get("maxDet", 300)
            )
        elif "imageBase64" in request:
            detections = balloon_detector.detect_from_base64(
                request["imageBase64"],
                confidence_threshold=request.get("confidenceThreshold", 0.25),
                max_det=request.get("maxDet", 300)
            )
        else:
            return JSONResponse(
                content={"error": "imagePath or imageBase64 is required"},
                status_code=400
            )
        
        return JSONResponse(content={
            "detections": detections,
            "count": len(detections)
        })
        
    except Exception as e:
        log_with_time(f"吹き出し検出エラー: {e}", level="ERROR")
        return JSONResponse(
            content={"error": f"吹き出し検出に失敗しました: {str(e)}"},
            status_code=500
        )


@app.post("/api/crop-classify-balloons")
async def crop_classify_balloons(request: Dict[str, Any]):
    """吹き出し検出・切り出し・分類・可視化エンドポイント"""
    try:
        balloon_detector = get_balloon_detector()
        
        # 必須パラメータのチェック
        if "imagePath" not in request:
            return JSONResponse(
                content={"error": "imagePath is required"},
                status_code=400
            )
        
        image_path = "./public" + request["imagePath"]
        
        # オプションパラメータ
        confidence_threshold = request.get("confidenceThreshold", 0.25)
        max_det = request.get("maxDet", 300)
        save_crops = request.get("saveCrops", True)
        output_dir = request.get("outputDir", "./tmp/balloon_crops")
        
        # 吹き出し検出・切り出し・分類・可視化を実行
        result = balloon_detector.crop_and_classify_balloons(
            image_path=image_path,
            confidence_threshold=confidence_threshold,
            max_det=max_det,
            save_crops=save_crops,
            output_dir=output_dir
        )
        
        # エラーがある場合
        if "error" in result:
            return JSONResponse(
                content={"error": result["error"]},
                status_code=500
            )
        
        # 可視化画像がある場合は、公開可能なパスに変換
        if result.get("visualization_path"):
            # tmp/から始まるパスを公開パスに変換
            vis_path = result["visualization_path"]
            if vis_path.startswith("./tmp/"):
                # public/tmp/に移動してフロントエンドからアクセス可能にする
                public_tmp_dir = Path("./public/tmp")
                public_tmp_dir.mkdir(parents=True, exist_ok=True)
                
                # ファイル名を取得
                vis_filename = Path(vis_path).name
                public_vis_path = public_tmp_dir / vis_filename
                
                # ファイルをコピー
                import shutil
                shutil.copy2(vis_path, public_vis_path)
                
                # 公開パスに更新
                result["visualization_path"] = f"/tmp/{vis_filename}"
        
        return JSONResponse(content={
            "success": True,
            "result": result
        })
        
    except Exception as e:
        log_with_time(f"吹き出し切り出し・分類エラー: {e}", level="ERROR")
        return JSONResponse(
            content={"error": f"吹き出し切り出し・分類に失敗しました: {str(e)}"},
            status_code=500
        )


@app.post("/api/analyze-tail-shape")
async def analyze_tail_shape(request: Dict[str, Any]):
    """しっぽの形状分析エンドポイント"""
    try:
        from tail_shape_analyzer import TailShapeAnalyzer
        
        # 必須パラメータのチェック
        if "imageBase64" not in request:
            return JSONResponse(
                content={"error": "imageBase64 is required"},
                status_code=400
            )
        
        if "tailBBox" not in request or "balloonBBox" not in request:
            return JSONResponse(
                content={"error": "tailBBox and balloonBBox are required"},
                status_code=400
            )
        
        # アナライザーの初期化
        analyzer = TailShapeAnalyzer()
        
        # しっぽの方向を分析
        result = analyzer.analyze_tail_direction(
            image_base64=request["imageBase64"],
            tail_bbox=request["tailBBox"],
            balloon_bbox=request["balloonBBox"]
        )
        
        log_with_time(f"しっぽ形状分析完了: 先端点={result['tip_point']}", level="INFO")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        log_with_time(f"しっぽ形状分析エラー: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"しっぽ形状分析に失敗しました: {str(e)}"},
            status_code=500
        )

@app.post("/api/visualize-detection")
async def visualize_detection(request: Dict[str, Any]):
    """画像に検出結果を可視化して返す"""
    try:
        from balloon_detection_integration import (
            BalloonDetector,
            draw_tail_shape_results_on_image,
            classify_tail_shapes_in_detections
        )
        import cv2
        import base64
        from io import BytesIO
        from PIL import Image
        
        # リクエストパラメータ
        file_path = request.get("filePath")
        balloon_detection = request.get("balloonDetection", True)
        character_detection = request.get("characterDetection", False)
        tail_detection = request.get("tailDetection", True)
        
        if not file_path:
            return JSONResponse(
                content={"error": "filePath is required"},
                status_code=400
            )
        
        # 画像を読み込み
        image = cv2.imread(file_path)
        if image is None:
            return JSONResponse(
                content={"error": f"画像ファイルが見つかりません: {file_path}"},
                status_code=404
            )
        
        balloon_detections = []
        character_detections = []
        
        # 吹き出し検出
        if balloon_detection:
            detector = BalloonDetector()
            balloon_detections = detector.detect_balloons(
                image,
                confidence_threshold=0.25,
                max_det=300,
                detect_tails=tail_detection
            )
            
            # しっぽ形状分類
            if tail_detection:
                balloon_detections = classify_tail_shapes_in_detections(image, balloon_detections)
        
        # キャラクター検出
        if character_detection:
            # キャラクター検出を実行（YOLOまたはDinoV2を使用）
            try:
                from yuyu_yolo_dinov2_pipeline import detect_characters_yolo_dinov2
                yolo_model_path = "/Users/esuji/work/fun_annotator/yolo11x_face_detection_model/yolo11l_480_12_14.pt"
                dinov2_model_path = "/Users/esuji/work/fun_annotator/yuyu_face_recognize_model/yuyu_dinov2_final.pth"
                
                char_results = detect_characters_yolo_dinov2(
                    image,
                    yolo_model_path=yolo_model_path,
                    dinov2_model_path=dinov2_model_path,
                    confidence_threshold=0.25
                )
                character_detections = char_results["characters"]
            except Exception as e:
                log_with_time(f"キャラクター検出エラー: {e}", level="WARNING")
        
        # 結果を画像に描画
        result_image = draw_tail_shape_results_on_image(
            image,
            balloon_detections,
            character_detections
        )
        
        # 画像をBase64エンコード
        result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', result_bgr)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse(content={
            "success": True,
            "imageBase64": f"data:image/png;base64,{image_base64}",
            "detections": {
                "balloons": balloon_detections,
                "characters": character_detections
            }
        })
        
    except Exception as e:
        log_with_time(f"可視化エラー: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"可視化処理に失敗しました: {str(e)}"},
            status_code=500
        )

@app.get("/api/debug-visualization")
async def debug_visualization():
    """可視化処理のデバッグ情報を取得"""
    try:
        pipeline_available = yolo_dinov2_pipeline is not None
        visualize_method_available = hasattr(yolo_dinov2_pipeline, "visualize_results") if pipeline_available else False
        
        test_result = "未実行"
        if pipeline_available and visualize_method_available:
            try:
                # シンプルなテスト画像で可視化テスト
                test_image = "public/test_detection_image.jpg"
                if os.path.exists(test_image):
                    result = yolo_dinov2_pipeline.process_image(test_image)
                    visualized = yolo_dinov2_pipeline.visualize_results(
                        image_path=test_image,
                        result=result,
                        show=False,
                        use_character_colors=True
                    )
                    test_result = f"成功: {visualized.shape}"
                else:
                    test_result = "テスト画像が見つからない"
            except Exception as e:
                import traceback
                test_result = f"エラー: {str(e)} | {traceback.format_exc()[:200]}"
        
        return {
            "message": "デバッグ情報取得成功",
            "pipeline_available": pipeline_available,
            "visualize_method_available": visualize_method_available,
            "test_result": test_result
        }
    except Exception as e:
        import traceback
        return {
            "message": f"デバッグエラー: {str(e)}",
            "pipeline_available": False,
            "visualize_method_available": False,
            "test_result": f"エラー詳細: {traceback.format_exc()[:200]}"
        }

