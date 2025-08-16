#!/usr/bin/env python3
"""
吹き出し検出の統合モジュール
Fun Annotatorシステムに吹き出し検出機能を追加
"""

import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# 吹き出しクラスの定義
# YOLOモデル(yolo11x_balloon_model)のdataset.yamlと同期
BALLOON_CLASSES = {
    0: "speech_bubble",  # 通常の吹き出し
    1: "thought_bubble",  # 思考の吹き出し
    2: "exclamation_bubble",  # 感嘆符の吹き出し
    3: "combined_bubble",  # 結合型吹き出し
    4: "offserif_bubble",  # オフセリフ（画面外の声）
    5: "inner_voice_bubble",  # 内なる声（つぶやき）
    6: "narration_box",  # ナレーションボックス
    7: "chractor_bubble_yuzuko",  # ゆずこ専用吹き出し
    8: "chractor_bubble_yukari",  # ゆかり専用吹き出し
    9: "chractor_bubble_yui",  # 唯専用吹き出し
    10: "chractor_bubble_yoriko",  # よりこ専用吹き出し
    11: "chractor_bubble_chiho",  # 千穂専用吹き出し
    12: "chractor_bubble_kei",  # 恵専用吹き出し
    13: "chractor_bubble_fumi",  # 史専用吹き出し
}

# 尻尾形状分類のカテゴリ
# 尻尾形状分類の全カテゴリ（モデル用）
TAIL_SHAPE_CATEGORIES = [
    "しっぽじゃない",
    "オフセリフ",
    "思考",
    "真上",
    "真下",
    "上左30度以上",
    "上左少し",
    "上左やや",
    "上右やや",
    "上右少し",
    "上右30度以上",
    "下左30度以上",
    "下左少し",
    "下左やや",
    "下右やや",
    "下右少し",
    "下右30度以上"
]

# 除外する分類（表示しない）
EXCLUDED_TAIL_CATEGORIES = ["しっぽじゃない"]


class DINOv2Classifier(nn.Module):
    """DINOv2を使った尻尾形状分類器"""
    
    def __init__(self, num_classes=17, model_name='dinov2_vits14', freeze_backbone=True):
        super().__init__()
        
        # DINOv2モデルをロード
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        
        # バックボーンのパラメータを固定するか
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 分類ヘッド
        embed_dim = self.backbone.embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # DINOv2で特徴抽出
        features = self.backbone(x)
        # 分類
        logits = self.classifier(features)
        return logits


class TailShapeClassifier:
    """尻尾形状分類器"""
    
    def __init__(self, model_path: str = None, device: str = 'auto'):
        """
        初期化
        
        Args:
            model_path: 学習済みモデルのパス
            device: 使用デバイス
        """
        # デバイス設定
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # デフォルトモデルパス
        if model_path is None:
            # 最新のresultsディレクトリを探す
            results_dirs = list(Path(".").glob("tail_shape_dinov2_results_*"))
            if results_dirs:
                latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
                model_path = str(latest_dir / "models" / "best_model.pth")
            else:
                logger.warning("尻尾形状分類モデルが見つかりません")
                self.model = None
                self.is_loaded = False
                return
        
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
        # カテゴリ
        self.categories = TAIL_SHAPE_CATEGORIES
        
        # 前処理設定
        self.transform = transforms.Compose([
            transforms.Resize((98, 98), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # モデルをロード
        self._load_model()
    
    def _load_model(self):
        """モデルをロード"""
        try:
            if not Path(self.model_path).exists():
                logger.warning(f"尻尾形状分類モデルが見つかりません: {self.model_path}")
                return
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 設定を取得
            config = checkpoint.get('config', {})
            model_name = config.get('model_name', 'dinov2_vits14')
            
            # モデルを作成
            self.model = DINOv2Classifier(
                num_classes=len(self.categories),
                model_name=model_name,
                freeze_backbone=True
            )
            
            # 重みをロード
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"✅ 尻尾形状分類モデル読み込み完了: {self.model_path}")
            
        except Exception as e:
            logger.error(f"❌ 尻尾形状分類モデル読み込みエラー: {e}")
            self.model = None
            self.is_loaded = False
    
    def classify_tail_shape(self, tail_image: np.ndarray) -> Dict[str, Any]:
        """
        尻尾形状を分類
        
        Args:
            tail_image: 尻尾画像（numpy array）
            
        Returns:
            分類結果
        """
        if not self.is_loaded or self.model is None:
            return {
                'error': 'Model not loaded',
                'predicted_category': 'unknown',
                'confidence': 0.0,
                'top3_predictions': []
            }
        
        try:
            # PIL Imageに変換
            if len(tail_image.shape) == 3:
                # RGB
                pil_image = Image.fromarray(tail_image)
            else:
                # グレースケール
                pil_image = Image.fromarray(tail_image).convert('RGB')
            
            # 前処理
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # 予測
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_idx = predicted.item()
            confidence_score = confidence.item()
            
            # 上位3つの予測を取得
            top3_probs, top3_indices = torch.topk(probabilities[0], 3)
            top3_predictions = []
            for prob, idx in zip(top3_probs, top3_indices):
                top3_predictions.append({
                    'category': self.categories[idx],
                    'confidence': prob.item()
                })
            
            return {
                'predicted_category': self.categories[predicted_idx],
                'confidence': confidence_score,
                'top3_predictions': top3_predictions
            }
            
        except Exception as e:
            logger.error(f"尻尾形状分類エラー: {e}")
            return {
                'error': str(e),
                'predicted_category': 'unknown',
                'confidence': 0.0,
                'top3_predictions': []
            }



class BalloonDetector:
    """吹き出し検出器"""

    def __init__(
        self,
        model_path: str = "/Users/esuji/work/fun_annotator/yolo11x_balloon_model/best.pt",
    ):
        """
        初期化

        Args:
            model_path: 吹き出し検出モデルのパス
        """
        self.model_path = model_path
        self.model = None
        self._current_characters = None  # 位置ベース話者推定用のキャラクター情報
        self._load_model()

    def _load_model(self):
        """モデルの読み込み"""
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                logger.info(f"✅ 吹き出し検出モデル読み込み完了: {self.model_path}")
            else:
                logger.warning(
                    f"⚠️ 吹き出し検出モデルが見つかりません: {self.model_path}"
                )
        except Exception as e:
            logger.error(f"❌ モデル読み込みエラー: {e}")

    def _calculate_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """
        2つのバウンディングボックスのIoU（Intersection over Union）を計算
        
        Args:
            bbox1, bbox2: {"x1": float, "y1": float, "x2": float, "y2": float}
            
        Returns:
            IoU値 (0.0-1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1["x1"], bbox1["y1"], bbox1["x2"], bbox1["y2"]
        x1_2, y1_2, x2_2, y2_2 = bbox2["x1"], bbox2["y1"], bbox2["x2"], bbox2["y2"]
        
        # 重複部分の座標を計算
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        # 重複がない場合
        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return 0.0
        
        # 重複面積
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 各バウンディングボックスの面積
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Union面積
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def _get_balloon_priority(self, balloon_type: str) -> int:
        """
        吹き出しタイプの優先度を取得（数値が小さいほど高優先度）
        
        Args:
            balloon_type: 吹き出しタイプ
            
        Returns:
            優先度（数値）
        """
        priority_map = {
            "speech_bubble": 1,                 # 最優先
            "thought_bubble": 2,
            "exclamation_bubble": 3,
            "combined_bubble": 4,
            "narration_box": 5,
            "inner_voice_bubble": 6,
            "chractor_bubble_yuzuko": 7,
            "chractor_bubble_yukari": 7,
            "chractor_bubble_yui": 7,
            "chractor_bubble_yoriko": 7,
            "chractor_bubble_chiho": 7,
            "chractor_bubble_kei": 7,
            "chractor_bubble_fumi": 7,
            "offserif_bubble": 10,              # 最低優先度
        }
        return priority_map.get(balloon_type, 8)  # デフォルトは中程度

    def _remove_overlapping_balloons(self, detections: List[Dict[str, Any]], iou_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        重複する吹き出し検出を除去（優先度ベース）
        
        Args:
            detections: 検出結果のリスト
            iou_threshold: IoU閾値（これ以上なら重複とみなす）
            
        Returns:
            重複除去後の検出結果リスト
        """
        if len(detections) <= 1:
            return detections
        
        # 信頼度でソート（高い順）
        sorted_detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        
        # 重複除去
        keep_detections = []
        for current in sorted_detections:
            is_duplicate = False
            current_bbox = current["boundingBox"]
            current_type = current["type"]
            current_priority = self._get_balloon_priority(current_type)
            
            for kept in keep_detections:
                kept_bbox = kept["boundingBox"]
                kept_type = kept["type"]
                kept_priority = self._get_balloon_priority(kept_type)
                
                iou = self._calculate_iou(current_bbox, kept_bbox)
                
                if iou > iou_threshold:
                    # 重複検出: 優先度で決定
                    if current_priority < kept_priority:
                        # 現在の方が高優先度 → 既存を除去して現在を採用
                        keep_detections.remove(kept)
                        logger.info(f"🔄 重複除去: {kept_type}({kept['confidence']:.3f}) → {current_type}({current['confidence']:.3f}) (IoU: {iou:.3f})")
                        break
                    else:
                        # 既存の方が高優先度 → 現在を除外
                        is_duplicate = True
                        logger.info(f"❌ 重複除去: {current_type}({current['confidence']:.3f}) ← {kept_type}({kept['confidence']:.3f}) (IoU: {iou:.3f})")
                        break
            
            if not is_duplicate:
                keep_detections.append(current)
        
        if len(keep_detections) != len(detections):
            logger.info(f"🎯 吹き出し重複除去完了: {len(detections)} -> {len(keep_detections)}個")
        
        return keep_detections

    def detect_balloons(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.25,
        max_det: int = 300,
        detect_tails: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        画像から吹き出しを検出

        Args:
            image: 検出対象の画像（numpy array）
            confidence_threshold: 検出閾値
            max_det: 最大検出数
            detect_tails: しっぽ検出を行うかどうか

        Returns:
            検出結果のリスト
        """
        if self.model is None:
            logger.warning("モデルが読み込まれていません")
            return []

        try:
            # YOLO検出実行
            results = self.model(image, conf=confidence_threshold, max_det=max_det)

            # しっぽ検出器のインスタンス取得（必要な場合のみ）
            tail_detector = get_tail_detector() if detect_tails else None

            detections = []
            for r in results:
                if r.boxes is not None:
                    for i, box in enumerate(r.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        # 吹き出しタイプを判定
                        balloon_type = BALLOON_CLASSES.get(cls, "unknown")

                        # クラスごとに異なる信頼度閾値を適用
                        # キャラクター専用吹き出しとオフセリフは通常の閾値
                        # その他の一般的な吹き出しは低い閾値で検出
                        class_specific_threshold = confidence_threshold
                        if balloon_type in [
                            "speech_bubble",
                            "thought_bubble",
                            "exclamation_bubble",
                            "combined_bubble",
                            "narration_box",
                        ]:
                            # 一般的な吹き出しは閾値を下げて積極的に検出
                            class_specific_threshold = min(
                                0.1, confidence_threshold * 0.4
                            )
                        elif (
                            balloon_type.startswith("chractor_bubble_")
                            or balloon_type == "offserif_bubble"
                        ):
                            # キャラクター専用とオフセリフは通常の閾値
                            class_specific_threshold = confidence_threshold

                        # 閾値チェック
                        if conf < class_specific_threshold:
                            continue

                        # 中心座標を計算（正規化）
                        center_x = (x1 + x2) / 2 / image.shape[1]
                        center_y = (y1 + y2) / 2 / image.shape[0]

                        detection = {
                            "dialogueId": f"balloon_{i + 1}",
                            "type": balloon_type,
                            "boundingBox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "width": float(x2 - x1),
                                "height": float(y2 - y1),
                            },
                            "coordinate": [float(center_x), float(center_y)],
                            "confidence": float(conf),
                            "classId": cls,
                            "readingOrderIndex": i + 1,  # 暫定的に検出順を読み順とする
                        }

                        # キャラクター専用吹き出しの場合、話者を推定（char_X形式に変換）
                        if balloon_type.startswith("chractor_bubble_"):
                            character_name = balloon_type.replace(
                                "chractor_bubble_", ""
                            )
                            # キャラクター名をchar_X形式にマッピング
                            character_id_map = {
                                "yuzuko": "char_A",
                                "yukari": "char_B",
                                "yui": "char_C",
                                "yoriko": "char_D",
                                "chiho": "char_E",
                                "kei": "char_F",
                                "fumi": "char_G",
                            }
                            detection["speakerCharacterId"] = character_id_map.get(
                                character_name, None
                            )

                        # 全ての吹き出しタイプについてログ出力
                        print(f"🎯 吹き出し処理中: {balloon_type} (ID: {detection.get('dialogueId', 'unknown')})")
                        
                        # しっぽ検出を実行（対象の吹き出しタイプの場合のみ）
                        print(f"🔍 しっぽ検出条件チェック: {balloon_type}")
                        print(f"  - detect_tails: {detect_tails}")
                        print(f"  - tail_detector: {tail_detector is not None}")
                        print(f"  - balloon_type not in offserif_bubble: {balloon_type not in ['offserif_bubble']}")
                        print(f"  - not startswith chractor_bubble_: {not balloon_type.startswith('chractor_bubble_')}")
                        
                        if (
                            detect_tails
                            and tail_detector
                            and balloon_type
                            not in ["offserif_bubble"]
                            and not balloon_type.startswith("chractor_bubble_")
                        ):
                            # 吹き出し部分の画像を切り出し
                            x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
                            x2_int, y2_int = (
                                min(image.shape[1], int(x2)),
                                min(image.shape[0], int(y2)),
                            )
                            balloon_image = image[y1_int:y2_int, x1_int:x2_int]

                            if balloon_image.size > 0:
                                print(f"🔍 {balloon_type}でしっぽ検出実行中... 画像サイズ: {balloon_image.shape}")
                                
                                # しっぽ検出（全吹き出し種類で段階的閾値調整）
                                tail_detections = None
                                
                                # 吹き出し種類に応じた段階的閾値設定
                                if balloon_type == 'exclamation_bubble':
                                    # 感嘆吹き出し：最も段階的（しっぽが少ないため）
                                    thresholds = [0.25, 0.15, 0.1, 0.05, 0.01]
                                    threshold_labels = ["標準", "低", "非常に低", "極低", "最低"]
                                    balloon_type_label = "感嘆吹き出し"
                                else:
                                    # 通常の吹き出し：やや保守的な段階的調整
                                    thresholds = [0.25, 0.2, 0.15, 0.1, 0.05]
                                    threshold_labels = ["標準", "やや低", "低", "非常に低", "極低"]
                                    balloon_type_label = "通常吹き出し"
                                
                                print(f"🔄 {balloon_type_label}で段階的しっぽ検出を開始")
                                
                                for i, threshold in enumerate(thresholds):
                                    print(f"📊 段階{i+1}: 信頼度閾値 {threshold} ({threshold_labels[i]})")
                                    
                                    tail_detections = tail_detector.detect_tails(
                                        balloon_image,
                                        balloon_type,
                                        confidence_threshold=threshold,
                                    )
                                    
                                    if tail_detections:
                                        print(f"🎯 段階{i+1}でしっぽ検出成功: {len(tail_detections)}個 (閾値: {threshold})")
                                        break
                                    else:
                                        print(f"❌ 段階{i+1}ではしっぽ未検出 (閾値: {threshold})")
                                
                                print(f"🎯 最終しっぽ検出結果: {len(tail_detections) if tail_detections else 0}個")

                                # しっぽ情報を追加（元画像の座標に変換）
                                if tail_detections:
                                    detection["tails"] = []
                                    
                                    for tail in tail_detections:
                                        # しっぽの座標を元画像の座標系に変換
                                        tail_global = {
                                            **tail,
                                            "globalBoundingBox": {
                                                "x1": x1_int
                                                + tail["boundingBox"]["x1"],
                                                "y1": y1_int
                                                + tail["boundingBox"]["y1"],
                                                "x2": x1_int
                                                + tail["boundingBox"]["x2"],
                                                "y2": y1_int
                                                + tail["boundingBox"]["y2"],
                                            },
                                            "globalPosition": [
                                                (
                                                    x1_int
                                                    + (
                                                        tail["boundingBox"]["x1"]
                                                        + tail["boundingBox"]["x2"]
                                                    )
                                                    / 2
                                                )
                                                / image.shape[1],
                                                (
                                                    y1_int
                                                    + (
                                                        tail["boundingBox"]["y1"]
                                                        + tail["boundingBox"]["y2"]
                                                    )
                                                    / 2
                                                )
                                                / image.shape[0],
                                            ],
                                        }
                                        
                                        detection["tails"].append(tail_global)
                                else:
                                    print(f"❌ {balloon_type}: しっぽが検出されませんでした")
                                    
                                    # 全ての吹き出しでしっぽが検出されない場合、位置関係による話者推定を実行
                                    if True:  # 全ての吹き出し種類で位置ベース推定を実行
                                        print(f"🎯 {balloon_type_label}でしっぽ未検出のため位置関係による話者推定を実行")
                                        
                                        # キャラクター検出結果があれば位置ベース推定を実行
                                        if hasattr(self, '_current_characters') and self._current_characters:
                                            speaker_result = self._estimate_speaker_by_proximity(detection, self._current_characters)
                                            if speaker_result:
                                                speaker_id = speaker_result["speaker_id"]
                                                confidence = speaker_result["confidence"]
                                                distance = speaker_result["distance"]
                                                confidence_level = speaker_result["confidence_level"]
                                                stage = speaker_result["threshold_stage"]
                                                
                                                print(f"📍 位置関係により話者推定: {speaker_id} (段階{stage}, {confidence_level})")
                                                
                                                # ダミーのしっぽ情報を作成（位置関係ベース、信頼度反映）
                                                dummy_tail = {
                                                    "boundingBox": detection["boundingBox"],
                                                    "globalBoundingBox": detection["boundingBox"],
                                                    "direction": "position_based",
                                                    "category": f"位置ベース推定({confidence_level})",
                                                    "confidence": confidence,
                                                    "distance": distance,
                                                    "position_based": True,
                                                    "estimated_speaker": speaker_id,
                                                    "threshold_stage": stage
                                                }
                                                detection["tails"].append(dummy_tail)
                                                detection["position_based_speaker"] = True
                                            else:
                                                print(f"🚫 位置関係による話者推定に失敗")
                                        else:
                                            print(f"⚠️ キャラクター情報がないため位置ベース推定をスキップ")
                            else:
                                print(f"⚠️ {balloon_type}: 吹き出し画像が空です")
                        else:
                            print(f"🚫 {balloon_type}: しっぽ検出条件を満たさないためスキップ")

                        detections.append(detection)

            # 重複する吹き出し検出を除去（優先度ベース）
            if len(detections) > 1:
                detections = self._remove_overlapping_balloons(detections, iou_threshold=0.3)

            # 右上から左下への読み順でソート（日本の漫画の一般的な読み方）
            # より詳細な読み順ロジック：
            # 1. まず大まかな行（Y座標を基準）でグループ化
            # 2. 各行内で右から左にソート

            # Y座標の閾値（画像高さの10%以内なら同じ行とみなす）
            y_threshold = 0.1

            # 行ごとにグループ化
            rows = []
            for detection in detections:
                y = detection["coordinate"][1]
                placed = False
                for row in rows:
                    if any(abs(y - d["coordinate"][1]) < y_threshold for d in row):
                        row.append(detection)
                        placed = True
                        break
                if not placed:
                    rows.append([detection])

            # 各行を上から下に、行内は右から左にソート
            rows.sort(key=lambda row: min(d["coordinate"][1] for d in row))
            sorted_detections = []
            for row in rows:
                row.sort(key=lambda d: -d["coordinate"][0])  # 右から左
                sorted_detections.extend(row)

            detections = sorted_detections

            # readingOrderIndexを更新
            for i, detection in enumerate(detections):
                detection["readingOrderIndex"] = i + 1

            # 尻尾形状分類を実行
            detections = classify_tail_shapes_in_detections(image, detections)

            return detections

        except Exception as e:
            logger.error(f"吹き出し検出エラー: {e}")
            return []

    def detect_from_base64(
        self,
        base64_image: str,
        confidence_threshold: float = 0.25,
        max_det: int = 300,
        detect_tails: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Base64画像から吹き出しを検出

        Args:
            base64_image: Base64エンコードされた画像
            confidence_threshold: 検出閾値
            max_det: 最大検出数
            detect_tails: しっぽ検出を行うかどうか

        Returns:
            検出結果のリスト
        """
        try:
            # Base64デコード
            if base64_image.startswith("data:image"):
                base64_image = base64_image.split(",")[1]

            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))

            # numpy配列に変換
            image_np = np.array(image)
            if image_np.shape[2] == 4:  # RGBA -> RGB
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            elif len(image_np.shape) == 2:  # Grayscale -> RGB
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

            return self.detect_balloons(
                image_np, confidence_threshold, max_det, detect_tails
            )

        except Exception as e:
            logger.error(f"Base64画像の処理エラー: {e}")
            return []

    def detect_from_path(
        self,
        image_path: str,
        confidence_threshold: float = 0.25,
        max_det: int = 300,
        detect_tails: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        画像パスから吹き出しを検出

        Args:
            image_path: 画像のパス
            confidence_threshold: 検出閾値
            max_det: 最大検出数
            detect_tails: しっぽ検出を行うかどうか

        Returns:
            検出結果のリスト
        """
        try:
            # 画像読み込み
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"画像を読み込めません: {image_path}")
                return []

            # BGRからRGBに変換
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return self.detect_balloons(
                image, confidence_threshold, max_det, detect_tails
            )

        except Exception as e:
            logger.error(f"画像読み込みエラー: {e}")
            return []

    def crop_and_classify_balloons(
        self,
        image_path: str,
        confidence_threshold: float = 0.25,
        max_det: int = 300,
        save_crops: bool = True,
        output_dir: str = "./tmp/balloon_crops"
    ) -> Dict[str, Any]:
        """
        吹き出しを検出・切り出し・分類し、結果を可視化
        
        Args:
            image_path: 画像のパス
            confidence_threshold: 検出閾値
            max_det: 最大検出数
            save_crops: 切り出し画像を保存するかどうか
            output_dir: 切り出し画像の保存ディレクトリ
        
        Returns:
            検出・分類結果と可視化画像のパス
        """
        try:
            # 出力ディレクトリを作成
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 元画像を読み込み
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"画像を読み込めません: {image_path}")
                return {"error": "画像読み込み失敗"}
            
            # BGRからRGBに変換
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 吹き出し検出実行
            detections = self.detect_balloons(image_rgb, confidence_threshold, max_det)
            
            if not detections:
                return {
                    "detections": [],
                    "cropped_images": [],
                    "classification_results": [],
                    "visualization_path": None
                }
            
            # 結果を格納するリスト
            cropped_images = []
            classification_results = []
            
            # 各吹き出しを切り出し・分類
            for i, detection in enumerate(detections):
                balloon_id = detection.get("dialogueId", f"balloon_{i}")
                bbox = detection.get("boundingBox", {})
                
                if not bbox:
                    continue
                
                # バウンディングボックスから切り出し
                x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                x2, y2 = int(bbox["x2"]), int(bbox["y2"])
                
                # 切り出し画像を取得
                cropped_balloon = image_rgb[y1:y2, x1:x2]
                
                if cropped_balloon.size == 0:
                    continue
                
                # 切り出し画像情報を追加
                crop_info = {
                    "balloon_id": balloon_id,
                    "bbox": bbox,
                    "type": detection.get("type", "unknown"),
                    "confidence": detection.get("confidence", 0.0),
                    "size": {
                        "width": x2 - x1,
                        "height": y2 - y1
                    }
                }
                
                # 切り出し画像を保存（オプション）
                if save_crops:
                    crop_filename = f"{balloon_id}_{detection.get('type', 'unknown')}.png"
                    crop_path = output_path / crop_filename
                    
                    # PIL Imageで保存
                    pil_image = Image.fromarray(cropped_balloon)
                    pil_image.save(crop_path)
                    crop_info["crop_path"] = str(crop_path)
                
                cropped_images.append(crop_info)
                
                # 簡単な分類（サイズベース）
                classification = self._classify_balloon_by_features(cropped_balloon, detection)
                classification_results.append({
                    "balloon_id": balloon_id,
                    "classification": classification
                })
            
            # 可視化画像を生成
            visualization_path = self._create_visualization(
                image_rgb, detections, classification_results, output_path
            )
            
            return {
                "detections": detections,
                "cropped_images": cropped_images,
                "classification_results": classification_results,
                "visualization_path": visualization_path,
                "total_balloons": len(detections)
            }
            
        except Exception as e:
            logger.error(f"吹き出し切り出し・分類エラー: {e}")
            return {"error": str(e)}
    
    def _classify_balloon_by_features(self, balloon_image: np.ndarray, detection: Dict) -> Dict:
        """
        吹き出し画像の特徴による分類
        
        Args:
            balloon_image: 切り出した吹き出し画像
            detection: 検出結果
        
        Returns:
            分類結果
        """
        try:
            height, width = balloon_image.shape[:2]
            area = height * width
            aspect_ratio = width / height if height > 0 else 0
            
            # 基本的な特徴量を計算
            features = {
                "width": width,
                "height": height,
                "area": area,
                "aspect_ratio": aspect_ratio,
                "type": detection.get("type", "unknown"),
                "confidence": detection.get("confidence", 0.0)
            }
            
            # サイズベースでの簡単な分類
            if area < 1000:
                size_category = "small"
            elif area < 5000:
                size_category = "medium"
            else:
                size_category = "large"
            
            # アスペクト比による形状分類
            if aspect_ratio > 2.0:
                shape_category = "wide"
            elif aspect_ratio < 0.5:
                shape_category = "tall"
            else:
                shape_category = "normal"
            
            return {
                "features": features,
                "size_category": size_category,
                "shape_category": shape_category,
                "area_pixels": area
            }
            
        except Exception as e:
            logger.error(f"特徴分類エラー: {e}")
            return {"error": str(e)}
    
    def _create_visualization(
        self, 
        image: np.ndarray, 
        detections: List[Dict], 
        classifications: List[Dict],
        output_path: Path
    ) -> str:
        """
        検出結果と分類結果の可視化画像を作成
        
        Args:
            image: 元画像
            detections: 吹き出し検出結果
            classifications: 分類結果
            output_path: 出力ディレクトリパス
        
        Returns:
            可視化画像のパス
        """
        try:
            # OpenCVで描画するため、RGBからBGRに変換
            vis_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 分類結果をIDでマッピング
            classification_map = {c["balloon_id"]: c for c in classifications}
            
            for i, detection in enumerate(detections):
                balloon_id = detection.get("dialogueId", f"balloon_{i}")
                bbox = detection.get("boundingBox", {})
                
                if not bbox:
                    continue
                
                x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                x2, y2 = int(bbox["x2"]), int(bbox["y2"])
                
                # バウンディングボックスを描画
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # ラベルテキストを準備
                balloon_type = detection.get("type", "unknown")
                confidence = detection.get("confidence", 0.0)
                label = f"{balloon_type} ({confidence:.2f})"
                
                # 分類結果があれば追加
                if balloon_id in classification_map:
                    classification = classification_map[balloon_id]["classification"]
                    size_cat = classification.get("size_category", "")
                    shape_cat = classification.get("shape_category", "")
                    label += f"\n{size_cat}/{shape_cat}"
                
                # ラベルを描画
                cv2.putText(vis_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(vis_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # 可視化画像を保存
            timestamp = int(time.time())
            vis_filename = f"balloon_visualization_{timestamp}.png"
            vis_path = output_path / vis_filename
            
            cv2.imwrite(str(vis_path), vis_image)
            return str(vis_path)
            
        except Exception as e:
            logger.error(f"可視化作成エラー: {e}")
            return None

    def _estimate_speaker_by_proximity(self, balloon_detection: Dict, characters: List[Dict]) -> Optional[str]:
        """
        位置関係による話者推定（しっぽがない感嘆吹き出し用）
        段階的閾値調整で確実に話者を見つける
        
        Args:
            balloon_detection: 吹き出し検出結果
            characters: キャラクター検出結果
            
        Returns:
            推定された話者ID（char_A, char_B等）またはNone
        """
        if not characters:
            return None
            
        balloon_center = balloon_detection.get("coordinate")
        if not balloon_center:
            return None
            
        # 各キャラクターとの距離を計算
        character_distances = []
        for char in characters:
            if not char.get("coordinate"):
                continue
                
            char_pos = char["coordinate"]
            # 吹き出しの中心とキャラクターの中心の距離を計算
            distance = ((balloon_center[0] - char_pos[0])**2 + 
                       (balloon_center[1] - char_pos[1])**2)**0.5
            
            character_distances.append({
                "character": char,
                "distance": distance,
                "speaker_id": char.get("character") or char.get("characterId")
            })
            
            print(f"📏 {char.get('character', 'unknown')}との距離: {distance:.2f}")
        
        if not character_distances:
            print(f"🚫 有効なキャラクターが見つかりません")
            return None
        
        # 距離でソート（最も近いキャラクターから）
        character_distances.sort(key=lambda x: x["distance"])
        closest = character_distances[0]
        
        # 段階的閾値調整で話者を決定
        # 閾値の段階: [0.25, 0.4, 0.6, 0.8, 1.2] (最後は画像全体)
        thresholds = [0.25, 0.4, 0.6, 0.8, 1.2]
        confidence_labels = ["非常に高い", "高い", "中程度", "低い", "非常に低い"]
        confidence_values = [0.9, 0.7, 0.5, 0.3, 0.1]  # 信頼度の数値
        
        for i, threshold in enumerate(thresholds):
            if closest["distance"] <= threshold:
                confidence_level = confidence_labels[i]
                confidence_value = confidence_values[i]
                print(f"🎯 段階{i+1}で話者発見: {closest['speaker_id']} (距離: {closest['distance']:.2f}, 信頼度: {confidence_level})")
                
                # 結果に信頼度情報を含めて返す
                return {
                    "speaker_id": closest["speaker_id"],
                    "confidence": confidence_value,
                    "distance": closest["distance"],
                    "confidence_level": confidence_level,
                    "threshold_stage": i + 1
                }
        
        # どの閾値でも見つからない場合（理論上は起こらない）
        print(f"🚫 すべての閾値で話者が見つかりません (最小距離: {closest['distance']:.2f})")
        return None

    def set_character_information(self, characters: Optional[List[Dict]]):
        """
        キャラクター情報を設定（位置ベース話者推定用）
        
        Args:
            characters: キャラクター検出結果
        """
        self._current_characters = characters
        if characters:
            print(f"💾 キャラクター情報を設定: {len(characters)}人")
        else:
            print(f"🧹 キャラクター情報をクリア")


def integrate_balloon_detection(
    panel_data: Dict,
    balloon_detections: List[Dict],
    characters: Optional[List[Dict]] = None,
) -> Dict:
    """
    吹き出し検出結果をパネルデータに統合

    Args:
        panel_data: 既存のパネルデータ
        balloon_detections: 吹き出し検出結果
        characters: キャラクター検出結果（しっぽからの話者推定用）

    Returns:
        統合されたパネルデータ
    """
    # serifsがない場合は初期化
    if "serifs" not in panel_data or panel_data["serifs"] is None:
        panel_data["serifs"] = []

    # 既存のセリフを保持しつつ、吹き出し情報を更新
    existing_serifs = {s.get("dialogueId"): s for s in panel_data["serifs"]}
    detection_map = {d["dialogueId"]: d for d in balloon_detections}
    processed_ids = set()

    updated_serifs = []

    # 1. 既存のセリフを更新（検出結果がある場合のみ）
    for serif in panel_data["serifs"]:
        dialogue_id = serif.get("dialogueId")
        processed_ids.add(dialogue_id)

        if dialogue_id in detection_map:
            detection = detection_map[dialogue_id]
            # 既存のセリフをコピーして更新
            updated_serif = serif.copy()
            # 吹き出し検出結果で更新（タイプ、座標、読み順のみ）
            updated_serif["type"] = detection["type"]
            updated_serif["boundingBox"] = detection["boundingBox"]
            updated_serif["coordinate"] = detection["coordinate"]
            updated_serif["readingOrderIndex"] = detection["readingOrderIndex"]

            # しっぽ情報があれば追加
            if "tails" in detection:
                updated_serif["tails"] = detection["tails"]

            # 話者推定（オフセリフの場合はスキップ）
            if detection["type"] == "offserif_bubble":
                # オフセリフの場合は話者不明のまま（話者推定をスキップ）
                print(f"🔕 オフセリフのため話者推定をスキップ: {dialogue_id}")
                updated_serif["speakerCharacterId"] = None
            elif (
                detection["type"].startswith("chractor_bubble_")
                and "speakerCharacterId" in detection
            ):
                # キャラクター専用吹き出しの場合は自動設定
                updated_serif["speakerCharacterId"] = detection["speakerCharacterId"]
            elif characters and "tails" in detection and detection["tails"]:
                # 16クラス尻尾形状から話者を推定（改善版）
                speaker_id = _estimate_speaker_from_tails_16class(detection, characters)
                if speaker_id:
                    updated_serif["speakerCharacterId"] = speaker_id
            # それ以外は既存の話者を保持（更新しない）

            updated_serifs.append(updated_serif)
        else:
            # 検出されなかった既存セリフもそのまま保持
            updated_serifs.append(serif)

    # 2. 新規検出された吹き出しを追加
    for detection in balloon_detections:
        dialogue_id = detection["dialogueId"]
        if dialogue_id not in processed_ids:
            # 話者推定（オフセリフの場合はスキップ）
            speaker_id = None
            if detection["type"] == "offserif_bubble":
                # オフセリフの場合は話者不明のまま（話者推定をスキップ）
                print(f"🔕 オフセリフのため話者推定をスキップ: {dialogue_id}")
                speaker_id = None
            elif detection["type"].startswith("chractor_bubble_"):
                speaker_id = detection.get("speakerCharacterId", None)
            elif characters and "tails" in detection and detection["tails"]:
                speaker_id = _estimate_speaker_from_tails_16class(detection, characters)

            # 新規セリフ
            serif = {
                "dialogueId": dialogue_id,
                "text": "",  # テキストは空で初期化
                "type": detection["type"],
                "speakerCharacterId": speaker_id,
                "boundingBox": detection["boundingBox"],
                "readingOrderIndex": detection["readingOrderIndex"],
                "coordinate": detection["coordinate"],
            }

            # しっぽ情報があれば追加
            if "tails" in detection:
                serif["tails"] = detection["tails"]

            updated_serifs.append(serif)

    # 読み順でソート
    updated_serifs.sort(key=lambda s: s["readingOrderIndex"])

    panel_data["serifs"] = updated_serifs
    panel_data["serifsNum"] = len(updated_serifs)

    return panel_data


def _estimate_speaker_from_tails(
    balloon_detection: Dict, characters: List[Dict]
) -> Optional[str]:
    """
    しっぽの方向から話者を推定

    Args:
        balloon_detection: 吹き出し検出結果（しっぽ情報含む）
        characters: キャラクター検出結果

    Returns:
        推定された話者ID（char_A, char_B等）またはNone
    """
    # オフセリフの場合は話者推定をスキップ
    if balloon_detection.get("type") == "offserif_bubble":
        print(f"🔕 オフセリフのため従来話者推定をスキップ")
        return None
    
    if not balloon_detection.get("tails") or not characters:
        return None

    # 最初のしっぽの情報を使用（通常は1つ）
    tail = balloon_detection["tails"][0]
    tail_direction = tail.get("direction", "")

    # しっぽの方向から推定される相対位置を計算
    balloon_center = balloon_detection["coordinate"]

    # 方向に基づいて話者の推定位置を計算
    estimated_speaker_pos = None
    if "bottom" in tail_direction:
        # しっぽが下向き -> 話者は下にいる
        estimated_speaker_pos = [balloon_center[0], balloon_center[1] + 0.2]
    elif "top" in tail_direction:
        # しっぽが上向き -> 話者は上にいる
        estimated_speaker_pos = [balloon_center[0], balloon_center[1] - 0.2]
    elif "left" in tail_direction:
        # しっぽが左向き -> 話者は左にいる
        estimated_speaker_pos = [balloon_center[0] - 0.2, balloon_center[1]]
    elif "right" in tail_direction:
        # しっぽが右向き -> 話者は右にいる
        estimated_speaker_pos = [balloon_center[0] + 0.2, balloon_center[1]]
    else:
        # 中央または不明な場合は最も近いキャラクターを選択
        estimated_speaker_pos = balloon_center

    # 最も近いキャラクターを見つける
    min_distance = float("inf")
    closest_character = None

    for char in characters:
        if char.get("coordinate"):
            char_pos = char["coordinate"]
            distance = (
                (char_pos[0] - estimated_speaker_pos[0]) ** 2
                + (char_pos[1] - estimated_speaker_pos[1]) ** 2
            ) ** 0.5

            # しっぽの方向と整合性があるかチェック
            direction_match = True
            if "bottom" in tail_direction and char_pos[1] < balloon_center[1]:
                direction_match = False
            elif "top" in tail_direction and char_pos[1] > balloon_center[1]:
                direction_match = False
            elif "left" in tail_direction and char_pos[0] > balloon_center[0]:
                direction_match = False
            elif "right" in tail_direction and char_pos[0] < balloon_center[0]:
                direction_match = False

            if direction_match and distance < min_distance:
                min_distance = distance
                closest_character = char

    # 距離が妥当な範囲内（画面の40%以内）であれば話者と判定
    if closest_character and min_distance < 0.4:
        return closest_character.get("character") or closest_character.get(
            "characterId"
        )

    return None


def _estimate_speaker_from_tails_16class(
    balloon_detection: Dict, characters: List[Dict]
) -> Optional[str]:
    """
    16クラス尻尾形状分類結果から話者を推定（改善版）
    
    Args:
        balloon_detection: 吹き出し検出結果（しっぽ情報含む）
        characters: キャラクター検出結果
        
    Returns:
        推定された話者ID（char_A, char_B等）またはNone
    """
    # オフセリフの場合は話者推定をスキップ
    if balloon_detection.get("type") == "offserif_bubble":
        print(f"🔕 オフセリフのため16クラス話者推定をスキップ")
        return None
    
    if not balloon_detection.get("tails") or not characters:
        return None
    
    # しっぽの情報を取得
    tail = balloon_detection["tails"][0]
    tail_shape = tail.get("shape_category", "") or tail.get("direction", "")
    
    if not tail_shape or tail_shape in ["しっぽじゃない", "オフセリフ", "unknown"]:
        # オフセリフ分類の場合は明示的に話者推定をスキップ
        if tail_shape == "オフセリフ":
            print(f"🔕 しっぽ分類が「オフセリフ」のため話者推定をスキップ")
            return None
        # その他の分類できない場合は従来の方法にフォールバック
        return _estimate_speaker_from_tails(balloon_detection, characters)
    
    balloon_center = balloon_detection["coordinate"]
    
    # 16クラス分類から方向と強度を抽出
    direction_info = _parse_tail_direction_16class(tail_shape)
    if not direction_info:
        return _estimate_speaker_from_tails(balloon_detection, characters)
    
    direction = direction_info["direction"]  # "left", "right", "up", "down"
    intensity = direction_info["intensity"]  # "strong", "medium", "weak"
    vertical = direction_info.get("vertical", "")  # "up", "down", ""
    
    # 方向ベクトルを計算
    direction_vector = _calculate_direction_vector(direction, intensity, vertical)
    if not direction_vector:
        return _estimate_speaker_from_tails(balloon_detection, characters)
    
    # 話者候補をスコアリング
    best_character = None
    best_score = 0
    
    for char in characters:
        if not char.get("coordinate"):
            continue
            
        char_pos = char["coordinate"]
        
        # 方向一致度スコア
        direction_score = _calculate_direction_score(
            balloon_center, char_pos, direction_vector, intensity
        )
        
        # 距離スコア
        distance = np.sqrt(
            (char_pos[0] - balloon_center[0])**2 + 
            (char_pos[1] - balloon_center[1])**2
        )
        distance_score = max(0, 1.0 - distance / 0.5)
        
        # レイ交差スコア
        intersection_score = _calculate_intersection_score(
            balloon_center, direction_vector, char, char_pos
        )
        
        # 総合スコア（方向重視）
        total_score = (
            direction_score * 0.5 +
            distance_score * 0.2 +
            intersection_score * 0.3
        )
        
        logger.debug(f"キャラクター {char.get('character', 'unknown')}: "
                    f"方向スコア={direction_score:.2f}, 距離スコア={distance_score:.2f}, "
                    f"交差スコア={intersection_score:.2f}, 総合={total_score:.2f}")
        
        if total_score > best_score and total_score > 0.4:  # 閾値上げ
            best_score = total_score
            best_character = char
    
    if best_character:
        return best_character.get("character") or best_character.get("characterId")
    
    return None


def _parse_tail_direction_16class(tail_shape: str) -> Optional[Dict[str, str]]:
    """16クラス分類結果から方向と強度を解析"""
    
    # 上下と左右の組み合わせを詳細にチェック
    if "下左" in tail_shape or ("下" in tail_shape and "左" in tail_shape):
        direction = "left"
        vertical = "down"
    elif "下右" in tail_shape or ("下" in tail_shape and "右" in tail_shape):
        direction = "right"
        vertical = "down"
    elif "上左" in tail_shape or ("上" in tail_shape and "左" in tail_shape):
        direction = "left"
        vertical = "up"
    elif "上右" in tail_shape or ("上" in tail_shape and "右" in tail_shape):
        direction = "right"
        vertical = "up"
    elif "左" in tail_shape:
        direction = "left"
        vertical = ""
    elif "右" in tail_shape:
        direction = "right"
        vertical = ""
    elif "真上" in tail_shape or ("上" in tail_shape and "下" not in tail_shape):
        direction = "up"
        vertical = "up"
    elif "真下" in tail_shape or "下" in tail_shape:
        direction = "down"
        vertical = "down"
    else:
        return None
    
    # 強度判定
    if "30度以上" in tail_shape:
        intensity = "strong"
    elif "少し" in tail_shape:
        intensity = "medium"
    elif "やや" in tail_shape:
        intensity = "weak"
    else:
        intensity = "medium"  # デフォルト
    
    return {
        "direction": direction,
        "intensity": intensity,
        "vertical": vertical
    }


def _calculate_direction_vector(direction: str, intensity: str, vertical: str) -> Optional[List[float]]:
    """方向と強度から方向ベクトルを計算"""
    
    # 基本角度（度）
    base_angles = {
        "strong": 60,   # 30度以上
        "medium": 20,   # 少し
        "weak": 5       # やや
    }
    
    angle = base_angles.get(intensity, 20)
    
    if direction == "left":
        if vertical == "up":
            # 上左方向
            rad = np.radians(180 - angle)
        elif vertical == "down":
            # 下左方向
            rad = np.radians(180 + angle)
        else:
            # 真左
            rad = np.radians(180)
    elif direction == "right":
        if vertical == "up":
            # 上右方向
            rad = np.radians(angle)
        elif vertical == "down":
            # 下右方向
            rad = np.radians(-angle)
        else:
            # 真右
            rad = np.radians(0)
    elif direction == "up":
        rad = np.radians(90)
    elif direction == "down":
        rad = np.radians(-90)
    else:
        return None
    
    # 方向ベクトル（右を0度として反時計回り）
    return [np.cos(rad), -np.sin(rad)]  # 画像座標系では上が負


def _calculate_direction_score(balloon_center: List[float], char_pos: List[float], 
                              direction_vector: List[float], intensity: str) -> float:
    """方向一致度スコアを計算"""
    
    # 吹き出しからキャラクターへのベクトル
    char_vector = [
        char_pos[0] - balloon_center[0],
        char_pos[1] - balloon_center[1]
    ]
    
    if char_vector[0] == 0 and char_vector[1] == 0:
        return 0
    
    # ベクトルを正規化
    char_mag = np.sqrt(char_vector[0]**2 + char_vector[1]**2)
    char_normalized = [char_vector[0] / char_mag, char_vector[1] / char_mag]
    
    # 内積（コサイン類似度）
    dot_product = (
        direction_vector[0] * char_normalized[0] + 
        direction_vector[1] * char_normalized[1]
    )
    
    # 角度許容範囲（強度によって調整）
    tolerance = {
        "strong": 0.7,   # 約45度まで
        "medium": 0.5,   # 約60度まで
        "weak": 0.3      # 約73度まで
    }
    
    min_similarity = tolerance.get(intensity, 0.5)
    
    # スコア計算
    if dot_product >= min_similarity:
        return dot_product
    else:
        return max(0, dot_product * 0.5)  # 部分点


def _calculate_intersection_score(balloon_center: List[float], direction_vector: List[float],
                                char: Dict, char_pos: List[float]) -> float:
    """レイ交差スコアを計算"""
    
    # 複数距離でレイキャスト
    max_distance = 0.6  # 正規化座標での最大距離
    steps = 50
    
    for i in range(1, steps + 1):
        t = (i / steps) * max_distance
        ray_point = [
            balloon_center[0] + direction_vector[0] * t,
            balloon_center[1] + direction_vector[1] * t
        ]
        
        # キャラクターのバウンディングボックスと交差チェック
        if char.get('boundingBox'):
            bbox = char['boundingBox']
            if _point_in_bbox(ray_point, bbox):
                return 1.0
        else:
            # 座標ベースの近接チェック
            distance_to_char = np.sqrt(
                (ray_point[0] - char_pos[0])**2 + 
                (ray_point[1] - char_pos[1])**2
            )
            if distance_to_char < 0.05:  # 近接判定
                return 0.8
    
    return 0


def _point_in_bbox(point: List[float], bbox: Dict) -> bool:
    """点がバウンディングボックス内にあるかチェック"""
    return (bbox.get('x1', 0) <= point[0] <= bbox.get('x2', 1) and
            bbox.get('y1', 0) <= point[1] <= bbox.get('y2', 1))


# しっぽタイプの定義
TAIL_CLASSES = {
    0: "balloon_tail",  # 通常の吹き出しのしっぽ
    1: "thought_tail",  # 思考の吹き出しのしっぽ（点線状）
}


class BalloonTailDetector:
    """吹き出しのしっぽ検出器"""

    def __init__(
        self,
        model_path: str = "/Users/esuji/work/fun_annotator/balloon_tail_detector/train_20250726_022600/weights/best.pt",
    ):
        """
        初期化

        Args:
            model_path: しっぽ検出モデルのパス
        """
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """モデルの読み込み"""
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                logger.info(f"✅ しっぽ検出モデル読み込み完了: {self.model_path}")
            else:
                logger.warning(f"⚠️ しっぽ検出モデルが見つかりません: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ モデル読み込みエラー: {e}")

    def detect_tails(
        self,
        balloon_image: np.ndarray,
        balloon_type: str,
        confidence_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        吹き出し画像からしっぽを検出

        Args:
            balloon_image: 吹き出し部分の画像（numpy array）
            balloon_type: 吹き出しタイプ
            confidence_threshold: 検出閾値

        Returns:
            しっぽ検出結果のリスト
        """
        if self.model is None:
            logger.warning("しっぽ検出モデルが読み込まれていません")
            return []

        # オフセリフまたはキャラクター専用吹き出しの場合はスキップ
        if balloon_type == "offserif_bubble" or balloon_type.startswith(
            "chractor_bubble_"
        ):
            return []

        try:
            # YOLO検出実行
            results = self.model(balloon_image, conf=confidence_threshold)

            detections = []
            for r in results:
                if r.boxes is not None:
                    for i, box in enumerate(r.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        # しっぽタイプを判定
                        tail_type = TAIL_CLASSES.get(cls, "unknown")

                        # 思考吹き出しの場合は thought_tail のみ検出
                        if (
                            balloon_type == "thought_bubble"
                            and tail_type != "thought_tail"
                        ):
                            continue
                        # その他の吹き出しの場合は balloon_tail のみ検出
                        elif (
                            balloon_type != "thought_bubble"
                            and tail_type != "balloon_tail"
                        ):
                            continue

                        # 相対座標を計算（吹き出し画像内での位置）
                        center_x = (x1 + x2) / 2 / balloon_image.shape[1]
                        center_y = (y1 + y2) / 2 / balloon_image.shape[0]

                        detection = {
                            "tailId": f"tail_{i + 1}",
                            "type": tail_type,
                            "boundingBox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "width": float(x2 - x1),
                                "height": float(y2 - y1),
                            },
                            "relativePosition": [float(center_x), float(center_y)],
                            "confidence": float(conf),
                            "direction": self._estimate_tail_direction(
                                center_x, center_y
                            ),
                        }

                        detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"しっぽ検出エラー: {e}")
            return []

    def _estimate_tail_direction(self, x: float, y: float) -> str:
        """
        しっぽの位置から指し示す方向を推定

        Args:
            x: 相対X座標 (0.0-1.0)
            y: 相対Y座標 (0.0-1.0)

        Returns:
            方向 (top-left, top-right, bottom-left, bottom-right, center)
        """
        # 吹き出しを3x3のグリッドに分割して方向を判定
        if y < 0.33:  # 上部
            if x < 0.33:
                return "top-left"
            elif x > 0.67:
                return "top-right"
            else:
                return "top"
        elif y > 0.67:  # 下部
            if x < 0.33:
                return "bottom-left"
            elif x > 0.67:
                return "bottom-right"
            else:
                return "bottom"
        else:  # 中央
            if x < 0.33:
                return "left"
            elif x > 0.67:
                return "right"
            else:
                return "center"


# グローバルインスタンス（シングルトン）
_balloon_detector = None
_tail_detector = None
_tail_shape_classifier = None


def get_balloon_detector() -> BalloonDetector:
    """吹き出し検出器のインスタンスを取得"""
    global _balloon_detector
    if _balloon_detector is None:
        _balloon_detector = BalloonDetector()
    return _balloon_detector


def get_tail_detector() -> BalloonTailDetector:
    """しっぽ検出器のインスタンスを取得"""
    global _tail_detector
    if _tail_detector is None:
        _tail_detector = BalloonTailDetector()
    return _tail_detector


def get_tail_shape_classifier() -> TailShapeClassifier:
    """尻尾形状分類器のインスタンスを取得"""
    global _tail_shape_classifier
    if _tail_shape_classifier is None:
        _tail_shape_classifier = TailShapeClassifier()
    return _tail_shape_classifier


def extract_tail_image(image: np.ndarray, tail_bbox: Dict[str, float], padding: int = 5) -> np.ndarray:
    """
    尻尾部分の画像を切り出し
    
    Args:
        image: 元画像
        tail_bbox: 尻尾のバウンディングボックス
        padding: パディング（ピクセル）
    
    Returns:
        切り出した尻尾画像
    """
    try:
        h, w = image.shape[:2]
        
        # 座標を取得（globalBoundingBoxの場合は既に絶対座標、boundingBoxの場合は相対座標）
        x1 = tail_bbox.get('x1', 0)
        y1 = tail_bbox.get('y1', 0) 
        x2 = tail_bbox.get('x2', 0)
        y2 = tail_bbox.get('y2', 0)
        
        # 相対座標（0-1範囲）の場合は絶対座標に変換
        if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
        else:
            # 既に絶対座標の場合
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
        
        # パディングを追加
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # 切り出し
        tail_image = image[y1:y2, x1:x2]
        
        return tail_image
        
    except Exception as e:
        logger.error(f"尻尾画像切り出しエラー: {e}")
        return np.array([])


def classify_tail_shapes_in_detections(
    image: np.ndarray, 
    balloon_detections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    検出された吹き出しの尻尾形状を分類
    
    Args:
        image: 元画像
        balloon_detections: 吹き出し検出結果
    
    Returns:
        尻尾形状分類結果を含む検出結果
    """
    classifier = get_tail_shape_classifier()
    
    if not classifier.is_loaded:
        logger.warning("尻尾形状分類器がロードされていません")
        return balloon_detections
    
    logger.info(f"✅ 尻尾形状分類器が利用可能です ({len(balloon_detections)}個の検出結果を処理)")
    
    # 検出結果をコピー
    enhanced_detections = []
    
    for detection in balloon_detections:
        enhanced_detection = detection.copy()
        
        # 尻尾情報がある場合のみ分類（"tails"配列を処理）
        if 'tails' in detection and detection['tails']:
            logger.info(f"📍 尻尾あり検出: {detection.get('dialogueId', 'unknown')} ({len(detection['tails'])}個の尻尾)")
            
            # 最も信頼度が高い尻尾を使用
            best_tail = max(detection['tails'], key=lambda t: t.get('confidence', 0.0))
            
            try:
                # 尻尾画像を切り出し（globalBoundingBoxまたはboundingBoxを使用）
                tail_bbox = best_tail.get('globalBoundingBox', best_tail.get('boundingBox', {}))
                tail_image = extract_tail_image(image, tail_bbox)
                
                if tail_image.size > 0:
                    # 尻尾形状を分類
                    classification_result = classifier.classify_tail_shape(tail_image)
                    
                    # 分類結果を追加
                    enhanced_detection['tail_shape_classification'] = classification_result
                    
                    # 分類結果をtail情報にも反映
                    if 'tails' in enhanced_detection:
                        for tail in enhanced_detection['tails']:
                            # 代替案の選択ロジック
                            category = classification_result.get('predicted_category', 'unknown')
                            confidence = classification_result.get('confidence', 0.0)
                            
                            if category in EXCLUDED_TAIL_CATEGORIES:
                                top3_predictions = classification_result.get('top3_predictions', [])
                                for prediction in top3_predictions:
                                    alt_category = prediction.get('category', 'unknown')
                                    alt_confidence = prediction.get('confidence', 0.0)
                                    if alt_category not in EXCLUDED_TAIL_CATEGORIES:
                                        category = alt_category
                                        confidence = alt_confidence
                                        break
                            
                            tail['shape_category'] = category
                            tail['shape_confidence'] = confidence
                    
                    logger.info(f"🎯 尻尾形状分類完了: {classification_result['predicted_category']} "
                              f"(信頼度: {classification_result['confidence']:.2%})")
                else:
                    logger.warning("⚠️ 尻尾画像の切り出しに失敗")
                    
            except Exception as e:
                logger.error(f"❌ 尻尾形状分類エラー: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.debug(f"📍 尻尾なし検出: {detection.get('dialogueId', 'unknown')}")
        
        # 偽陽性しっぽをフィルタリング
        if 'tails' in enhanced_detection and enhanced_detection['tails']:
            filtered_tails = []
            for tail in enhanced_detection['tails']:
                tail_category = tail.get('shape_category', '')
                tail_confidence = tail.get('shape_confidence', 0.0)
                
                # 「しっぽじゃない」で高信頼度の場合は除外
                if tail_category == "しっぽじゃない" and tail_confidence > 0.8:
                    print(f"🚫 偽陽性しっぽを除外: {tail_category} ({tail_confidence:.1%})")
                    continue
                
                filtered_tails.append(tail)
            
            # フィルタリング後のしっぽ配列を更新
            enhanced_detection['tails'] = filtered_tails
            
            # しっぽがすべて除外された場合はtailsキーを削除
            if not filtered_tails:
                del enhanced_detection['tails']
                print(f"📝 {enhanced_detection.get('dialogueId', 'unknown')}: 全しっぽが偽陽性として除外されました")
        
        enhanced_detections.append(enhanced_detection)
    
    return enhanced_detections


def draw_tail_shape_results_on_image(
    image: np.ndarray, 
    balloon_detections: List[Dict[str, Any]],
    character_detections: Optional[List[Dict[str, Any]]] = None
) -> np.ndarray:
    """
    画像上に尻尾形状分類結果と人物検出結果をオーバーレイ描画
    
    Args:
        image: 元画像
        balloon_detections: 尻尾形状分類結果を含む検出結果
        character_detections: 人物検出結果（オプション）
    
    Returns:
        結果が描画された画像
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # OpenCV画像をPIL画像に変換
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image)
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(pil_image)
    
    # 日本語フォントの設定（全体的に小さくする）
    try:
        # macOSの日本語フォント
        font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 16)  # 24→16
        font_small = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 12)  # 20→12
    except:
        try:
            # Linuxの日本語フォント
            font = ImageFont.truetype("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf", 12)
        except:
            # フォールバック
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # キャラクター色定義（より濃い色合い）
    CHARACTER_COLORS = {
        '野々原ゆずこ': (255, 105, 180),  # yuzuko - ホットピンク（より濃く）
        '日向縁': (147, 112, 219),        # yukari - ミディアムパープル（より濃く）
        '櫟井唯': (255, 215, 0),          # yui - ゴールド（より濃く）
        '松本頼子': (255, 140, 105),      # yoriko - サーモン（より濃く）
        '相川千穂': (152, 251, 152),      # chiho - ペールグリーン（より濃く）
        '岡野佳': (222, 184, 135),        # kei - バーリーウッド（より濃く）
        '長谷川ふみ': (211, 211, 211),    # fumi - ライトグレー（より濃く）
        'お母さん': (255, 182, 193),      # ライトピンク（より濃く）
        '先生': (173, 216, 230),          # ライトブルー（より濃く）
        '生徒A': (221, 160, 221),        # プラム（より濃く）
        '生徒B': (175, 238, 238),        # パールターコイズ（より濃く）
        'その他': (245, 222, 179)         # ウィート（より濃く）
    }
    
    # 英語名から日本語名への変換
    ENGLISH_TO_JAPANESE = {
        'yuzuko': '野々原ゆずこ',
        'yukari': '日向縁',
        'yui': '櫟井唯',
        'yoriko': '松本頼子',
        'chiho': '相川千穂',
        'kei': '岡野佳',
        'fumi': '長谷川ふみ',
        'unknown': '不明'
    }
    
    # 人物検出結果を描画
    if character_detections:
        for char in character_detections:
            bbox = char.get('boundingBox', {})
            x1 = int(bbox.get('x1', 0))
            y1 = int(bbox.get('y1', 0))
            x2 = int(bbox.get('x2', 100))
            y2 = int(bbox.get('y2', 100))
            
            # キャラクター名を取得して日本語名に変換
            char_name = char.get('characterName', 'Unknown')
            japanese_name = ENGLISH_TO_JAPANESE.get(char_name.lower(), char_name)
            
            # キャラクター毎の色を取得（デフォルトは青色）
            char_color = CHARACTER_COLORS.get(japanese_name, (0, 0, 255))
            
            # 人物の枠を描画（キャラクター毎の色）
            draw.rectangle([(x1, y1), (x2, y2)], outline=char_color, width=3)
            
            # キャラクター名と信頼度を表示
            confidence = char.get('confidence', 0.0)
            label = f"{japanese_name} ({confidence:.1%})"
            
            # テキストを内側下部に配置（左下角から5px内側）
            text_x = x1 + 5
            text_y = y2 - 25  # 下から25px上に配置
            
            # テキストの背景を描画（バージョン互換性対応）
            try:
                # 新しいPillowバージョン
                text_bbox = draw.textbbox((text_x, text_y), label, font=font_small)
                draw.rectangle(text_bbox, fill=char_color)
            except AttributeError:
                # 古いPillowバージョン
                text_size = draw.textsize(label, font=font_small)
                draw.rectangle((text_x, text_y, text_x + text_size[0], text_y + text_size[1]), fill=char_color)
            
            draw.text((text_x, text_y), label, font=font_small, fill=(0, 0, 0))  # 黒文字
    
    # 吹き出し検出結果を描画
    for detection in balloon_detections:
        # 吹き出しの位置を取得
        bbox = detection.get('boundingBox', {})
        x1 = int(bbox.get('x1', 0))
        y1 = int(bbox.get('y1', 0))
        x2 = int(bbox.get('x2', 100))
        y2 = int(bbox.get('y2', 100))
        
        # 吹き出しの枠を描画（オレンジ色）
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 165, 0), width=2)
        
        # 吹き出しタイプを内側上部に表示（より小さなフォント）
        balloon_type = detection.get('balloon_type', detection.get('type', 'unknown'))
        if balloon_type and balloon_type != 'unknown':
            # 吹き出しタイプのラベルを作成（略語化）
            type_labels = {
                'speech_bubble': 'Speech',
                'thought_bubble': 'Thought', 
                'exclamation_bubble': 'Exclamation',
                'scream_bubble': 'Scream',
                'whisper_bubble': 'Whisper',
                'narration': 'Narration',
                'onomatopoeia': 'Onoma'
            }
            type_label = type_labels.get(balloon_type, balloon_type.replace('_', ' ').title())
            
            # 内側上部の位置を計算（左上角から3px内側）
            type_x = x1 + 3
            type_y = y1 + 3
            
            # さらに小さいフォントを作成
            try:
                font_tiny = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 10)  # より小さく
            except:
                try:
                    font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf", 10)
                except:
                    font_tiny = font_small
            
            # テキストの背景を描画（バージョン互換性対応）
            try:
                # 新しいPillowバージョン
                type_text_bbox = draw.textbbox((type_x, type_y), type_label, font=font_tiny)
                draw.rectangle(type_text_bbox, fill=(255, 255, 0, 200))  # 黄色背景
            except AttributeError:
                # 古いPillowバージョン
                type_text_size = draw.textsize(type_label, font=font_tiny)
                draw.rectangle((type_x, type_y, type_x + type_text_size[0], type_y + type_text_size[1]), fill=(255, 255, 0, 200))
            
            # タイプテキストを描画（黒色）
            draw.text((type_x, type_y), type_label, font=font_tiny, fill=(0, 0, 0))
        
        # 尻尾形状分類結果の表示は削除（緑地に黒文字の表示をなくす）
            
            # しっぽ領域を描画
            print(f"🎨 可視化: {balloon_type} - しっぽあり: {'tails' in detection}")
            if 'tails' in detection:
                print(f"🎨 しっぽ数: {len(detection['tails'])}")
                for tail in detection['tails']:
                    # しっぽのバウンディングボックスを取得（絶対座標）
                    tail_bbox = tail.get('globalBoundingBox', tail.get('boundingBox', {}))
                    if tail_bbox:
                        tx1 = int(tail_bbox.get('x1', 0))
                        ty1 = int(tail_bbox.get('y1', 0))
                        tx2 = int(tail_bbox.get('x2', 0))
                        ty2 = int(tail_bbox.get('y2', 0))
                        
                        # 相対座標（0-1範囲）の場合は絶対座標に変換
                        if max(tx1, ty1, tx2, ty2) <= 1.0:
                            tx1 = int(tx1 * img_width)
                            ty1 = int(ty1 * img_height)
                            tx2 = int(tx2 * img_width)
                            ty2 = int(ty2 * img_height)
                        
                        # しっぽの矩形を描画（紫色、半透明）
                        draw.rectangle(
                            [tx1, ty1, tx2, ty2],
                            outline=(128, 0, 128),
                            width=2
                        )
                        
                        # しっぽの中心点を描画
                        center_x = (tx1 + tx2) // 2
                        center_y = (ty1 + ty2) // 2
                        draw.ellipse(
                            [center_x - 3, center_y - 3, center_x + 3, center_y + 3],
                            fill=(128, 0, 128)
                        )
                        
                        # しっぽの方向ラベルを描画（信頼度付き）
                        if 'shape_category' in tail and tail['shape_category']:
                            tail_label = tail['shape_category']
                            # 信頼度を追加（パーセント表示）
                            if 'shape_confidence' in tail and tail['shape_confidence'] is not None:
                                confidence_percent = int(tail['shape_confidence'] * 100)
                                tail_label = f"{tail_label} ({confidence_percent}%)"
                        elif 'direction' in tail:
                            tail_label = f"→{tail['direction']}"
                        else:
                            tail_label = "tail"
                        
                        # テキストの背景を描画（バージョン互換性対応）
                        try:
                            # 新しいPillowバージョン
                            text_bbox = draw.textbbox((tx1, ty1 - 20), tail_label, font=font_small)
                            draw.rectangle(text_bbox, fill=(128, 0, 128, 180))
                        except AttributeError:
                            # 古いPillowバージョン
                            text_size = draw.textsize(tail_label, font=font_small)
                            draw.rectangle((tx1, ty1 - 20, tx1 + text_size[0], ty1 - 20 + text_size[1]), fill=(128, 0, 128, 180))
                        
                        draw.text((tx1, ty1 - 20), tail_label, font=font_small, fill=(255, 255, 255))
    
    # PIL画像をOpenCV画像に変換
    result_array = np.array(pil_image)
    if len(result_array.shape) == 3:
        result_image = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
    else:
        result_image = result_array
    
    return result_image
