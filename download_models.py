#!/usr/bin/env python3
"""
Fun Annotator Models Downloader
Hugging FaceからAIモデルを自動ダウンロードするスクリプト

使用方法:
    python download_models.py [--all] [--balloon] [--tail] [--face] [--character]
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

# Hugging Faceリポジトリ設定
REPO_ID = "YOUR_USERNAME/fun-annotator-models"  # 実際のユーザー名に置き換えてください

# モデル設定
MODELS_CONFIG = {
    "balloon_detection": {
        "filename": "balloon_detection/yolo11x_best.pt",
        "local_path": "yolo11x_balloon_model/best.pt",
        "description": "吹き出し検出器（YOLO11x）"
    },
    "tail_detection": {
        "filename": "tail_detection/yolo11_best.pt", 
        "local_path": "balloon_tail_detector/train_20250726_022600/weights/best.pt",
        "description": "しっぽ検出器（YOLO11）"
    },
    "face_detection": {
        "filename": "face_detection/yolo11l_face_only.pt",
        "local_path": "yolo11x_face_detection_model/yolo11l_480_12_14.pt",
        "description": "顔検出器（YOLO11l）"
    },
    "multiclass_detection": {
        "filename": "face_detection/yolo11l_multiclass.pt",
        "local_path": "yolo11x_face_detection_model/yolo11l_480_12_14_multi.pt", 
        "description": "マルチクラス検出器（YOLO11l）"
    },
    "anime_face_cascade": {
        "filename": "face_detection/lbpcascade_animeface.xml",
        "local_path": "lbpcascade_animeface.xml",
        "description": "アニメ顔検出器（OpenCV Cascade）"
    },
    "character_classification": {
        "filename": "character_classification/yuyu_dinov2_final.pth",
        "local_path": "yuyu_face_recognize_model/yuyu_dinov2_final.pth",
        "description": "ゆゆ式キャラクター分類器（DINOv2）"
    }
}

def ensure_directory(file_path):
    """ディレクトリが存在しない場合は作成"""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def download_model(model_key, config):
    """個別モデルをダウンロード"""
    print(f"📥 {config['description']} をダウンロード中...")
    
    try:
        # ローカルディレクトリを作成
        ensure_directory(config['local_path'])
        
        # Hugging Faceからダウンロード
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=config['filename'],
            local_dir=".",
            local_dir_use_symlinks=False
        )
        
        # 指定されたパスに移動
        if downloaded_path != config['local_path']:
            os.makedirs(os.path.dirname(config['local_path']), exist_ok=True)
            if os.path.exists(config['local_path']):
                os.remove(config['local_path'])
            os.rename(downloaded_path, config['local_path'])
        
        print(f"✅ {config['description']} → {config['local_path']}")
        return True
        
    except Exception as e:
        print(f"❌ {config['description']} のダウンロードに失敗: {e}")
        return False

def download_all_models():
    """全モデルを一括ダウンロード"""
    print("🚀 全モデルをダウンロード中...")
    
    success_count = 0
    total_count = len(MODELS_CONFIG)
    
    for model_key, config in MODELS_CONFIG.items():
        if download_model(model_key, config):
            success_count += 1
    
    print(f"\n📊 ダウンロード結果: {success_count}/{total_count} 成功")
    
    if success_count == total_count:
        print("🎉 全モデルのダウンロードが完了しました！")
    else:
        print("⚠️  一部のモデルダウンロードに失敗しました")

def main():
    parser = argparse.ArgumentParser(description="Fun Annotator AI Models Downloader")
    parser.add_argument("--all", action="store_true", help="全モデルをダウンロード")
    parser.add_argument("--balloon", action="store_true", help="吹き出し検出器のみ")
    parser.add_argument("--tail", action="store_true", help="しっぽ検出器のみ")
    parser.add_argument("--face", action="store_true", help="顔検出器のみ")
    parser.add_argument("--character", action="store_true", help="キャラクター分類器のみ")
    
    args = parser.parse_args()
    
    print("🤖 Fun Annotator Models Downloader")
    print("=" * 50)
    
    try:
        if args.all or not any([args.balloon, args.tail, args.face, args.character]):
            download_all_models()
        else:
            if args.balloon:
                download_model("balloon_detection", MODELS_CONFIG["balloon_detection"])
                download_model("anime_face_cascade", MODELS_CONFIG["anime_face_cascade"])
            if args.tail:
                download_model("tail_detection", MODELS_CONFIG["tail_detection"])
            if args.face:
                download_model("face_detection", MODELS_CONFIG["face_detection"])
                download_model("multiclass_detection", MODELS_CONFIG["multiclass_detection"])
                download_model("anime_face_cascade", MODELS_CONFIG["anime_face_cascade"])
            if args.character:
                download_model("character_classification", MODELS_CONFIG["character_classification"])
        
    except KeyboardInterrupt:
        print("\n⏹️  ダウンロードがキャンセルされました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")

if __name__ == "__main__":
    main()