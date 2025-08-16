#!/usr/bin/env python3
"""
YOLO11xアニメ顔検出 + DINOv2キャラクター識別統合パイプライン
高精度な顔検出と識別を実現する完全統合システム
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager

# 自作モジュールのインポート（既存の分類器を再利用）
from yuyu_character_classifier import DINOv2CharacterClassifier

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """顔検出結果"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    detection_id: int = 0
    
    @property
    def center(self) -> Tuple[float, float]:
        """中心座標"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """面積"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def width(self) -> float:
        """幅"""
        x1, _, x2, _ = self.bbox
        return x2 - x1
    
    @property
    def height(self) -> float:
        """高さ"""
        _, y1, _, y2 = self.bbox
        return y2 - y1


@dataclass
class CharacterIdentification:
    """キャラクター識別結果"""
    character_name: str
    confidence: float
    all_probabilities: Dict[str, float]
    top_k_predictions: List[Dict[str, Union[str, float]]]


@dataclass
class FaceCharacterResult:
    """顔検出＋キャラクター識別の統合結果"""
    detection: FaceDetection
    identification: CharacterIdentification
    face_image: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        result = {
            'bbox': list(self.detection.bbox),
            'detection_confidence': self.detection.confidence,
            'detection_id': self.detection.detection_id,
            'character': self.identification.character_name,
            'character_confidence': self.identification.confidence,
            'all_probabilities': self.identification.all_probabilities,
            'top_k_predictions': self.identification.top_k_predictions
        }
        return result


@dataclass
class ImageProcessingResult:
    """画像処理結果"""
    image_path: str
    image_size: Tuple[int, int]  # width, height
    results: List[FaceCharacterResult]
    processing_time: float
    timestamp: str
    
    def get_character_counts(self) -> Dict[str, int]:
        """キャラクター別カウント"""
        counts = {}
        for result in self.results:
            char = result.identification.character_name
            counts[char] = counts.get(char, 0) + 1
        return counts
    
    def get_high_confidence_results(self, threshold: float = 0.8) -> List[FaceCharacterResult]:
        """高信頼度結果のフィルタリング"""
        return [r for r in self.results if r.identification.confidence >= threshold]
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'image_path': self.image_path,
            'image_size': list(self.image_size),
            'total_faces': len(self.results),
            'character_counts': self.get_character_counts(),
            'results': [r.to_dict() for r in self.results],
            'processing_time': self.processing_time,
            'timestamp': self.timestamp
        }


class YuyuYOLODINOv2Pipeline:
    """YOLO11x + DINOv2統合パイプライン"""
    
    # キャラクター別カラーパレット
    CHARACTER_COLORS = {
        'yuzuko': (255, 182, 193),   # ライトピンク
        'yukari': (173, 216, 230),   # ライトブルー
        'yui': (144, 238, 144),      # ライトグリーン
        'yoriko': (255, 218, 185),   # ピーチ
        'chiho': (221, 160, 221),    # プラム
        'kei': (255, 222, 173),      # ナバホホワイト
        'fumi': (176, 196, 222),     # ライトスチールブルー
        'unknown': (192, 192, 192)   # シルバー
    }
    
    def __init__(
        self,
        yolo_model_path: str = "/Users/esuji/work/fun_annotator/yolo11x_face_detection_model/yolo11l_480_12_14.pt",
        dinov2_model_path: str = "/Users/esuji/work/fun_annotator/yuyu_face_recognize_model/yuyu_dinov2_final.pth",
        multiclass_model_path: str = "/Users/esuji/work/fun_annotator/yolo11x_face_detection_model/yolo11l_480_12_14_multi.pt",
        device: str = 'auto',
        detection_conf_threshold: float = 0.5,
        detection_iou_threshold: float = 0.45,
        classification_conf_threshold: float = 0.5,
        face_margin: float = 0.15,
        multiclass_mode: bool = False
    ):
        """
        初期化
        
        Args:
            yolo_model_path: YOLO11xモデルのパス
            dinov2_model_path: DINOv2分類モデルのパス
            multiclass_model_path: マルチクラスYOLOモデルのパス
            device: 実行デバイス ('auto', 'cpu', 'cuda')
            detection_conf_threshold: 検出信頼度閾値
            detection_iou_threshold: NMS IoU閾値
            classification_conf_threshold: 分類信頼度閾値
            face_margin: 顔切り出し時のマージン比率
            multiclass_mode: マルチクラスモードを使用するか
        """
        self.detection_conf_threshold = detection_conf_threshold
        self.detection_iou_threshold = detection_iou_threshold
        self.classification_conf_threshold = classification_conf_threshold
        self.face_margin = face_margin
        self.multiclass_mode = multiclass_mode
        self.multiclass_model_path = multiclass_model_path
        
        # デバイス設定
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"使用デバイス: {self.device}")
        
        # モデル初期化
        self._load_models(yolo_model_path, dinov2_model_path)
        
        # マルチクラスクラス名定義
        self.multiclass_names = ['yuzuko', 'yukari', 'yui', 'yoriko', 'chiho', 'kei', 'fumi', 'other']
        
        # 統計情報
        self.stats = {
            'total_images': 0,
            'total_faces': 0,
            'total_identifications': 0,
            'character_counts': {},
            'processing_times': []
        }
        
        logger.info("パイプライン初期化完了")
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        IoU（Intersection over Union）を計算
        
        Args:
            bbox1, bbox2: (x1, y1, x2, y2) 形式のバウンディングボックス
            
        Returns:
            IoU値 (0.0-1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
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
    
    def _remove_duplicate_detections(self, results, iou_threshold=0.5):
        """
        重複する検出結果を除去
        
        Args:
            results: FaceCharacterResultのリスト
            iou_threshold: IoU閾値（これ以上なら重複とみなす）
            
        Returns:
            重複除去後のresultsリスト
        """
        if len(results) <= 1:
            return results
        
        # 信頼度でソート（高い順）
        sorted_results = sorted(results, key=lambda r: r.identification.confidence, reverse=True)
        
        # 重複除去
        keep_results = []
        for current in sorted_results:
            is_duplicate = False
            current_bbox = current.detection.bbox
            
            for kept in keep_results:
                kept_bbox = kept.detection.bbox
                iou = self._calculate_iou(current_bbox, kept_bbox)
                
                if iou > iou_threshold:
                    is_duplicate = True
                    logger.debug(f"重複除去: {current.identification.character_name}({current.identification.confidence:.3f}) <- IoU={iou:.3f}")
                    break
            
            if not is_duplicate:
                keep_results.append(current)
        
        logger.info(f"重複除去: {len(results)} -> {len(keep_results)}人")
        return keep_results
    
    def _filter_yukari_kei_duplicates(self, results):
        """
        日向縁と岡野佳の重複検出をフィルタリング
        
        見た目が似ているため同じ人物を異なるキャラクターとして検出する問題を解決
        同じコマに両方が検出された場合、より高い信頼度の方を採用
        
        Args:
            results: FaceCharacterResultのリスト
            
        Returns:
            フィルタリング後のresultsリスト
        """
        if len(results) <= 1:
            return results
        
        # 両方の名前形式（日本語・英語）に対応
        yukari_names = ['日向縁', 'yukari']
        kei_names = ['岡野佳', 'kei']
        
        # 日向縁と岡野佳の検出結果を抽出
        yukari_detections = [r for r in results if r.identification.character_name in yukari_names]
        kei_detections = [r for r in results if r.identification.character_name in kei_names]
        other_detections = [r for r in results if r.identification.character_name not in yukari_names + kei_names]
        
        # 両方が検出されている場合のみフィルタリング実行
        if yukari_detections and kei_detections:
            logger.info(f"🔍 日向縁・岡野佳重複検出フィルタリング開始")
            yukari_names_str = [r.identification.character_name for r in yukari_detections]
            kei_names_str = [r.identification.character_name for r in kei_detections]
            logger.info(f"  ゆかり系: {len(yukari_detections)}件 ({yukari_names_str}, 信頼度: {[f'{r.identification.confidence:.3f}' for r in yukari_detections]})")
            logger.info(f"  恵系: {len(kei_detections)}件 ({kei_names_str}, 信頼度: {[f'{r.identification.confidence:.3f}' for r in kei_detections]})")
            
            # 両グループから最高信頼度のものを選出
            best_yukari = max(yukari_detections, key=lambda r: r.identification.confidence)
            best_kei = max(kei_detections, key=lambda r: r.identification.confidence)
            
            # まず位置関係を確認（重複検出か実際に2人いるかを判定）
            iou = self._calculate_iou(best_yukari.detection.bbox, best_kei.detection.bbox)
            logger.info(f"位置重複度 (IoU): {iou:.3f}")
            
            # 信頼度差を計算
            confidence_diff = abs(best_yukari.identification.confidence - best_kei.identification.confidence)
            
            # 信頼度差の判定閾値
            CONFIDENCE_THRESHOLD = 0.15  # 15%以上の差があれば明確に区別可能
            MIN_CONFIDENCE = 0.7  # 最低信頼度70%
            IoU_THRESHOLD = 0.3  # 30%以上重複している場合は同一人物の可能性が高い
            
            if iou <= IoU_THRESHOLD:
                # 位置が離れている場合は実際に2人いる可能性が高い
                logger.info(f"👥 位置が離れている (IoU: {iou:.3f}) ため両方採用 - 実際に2人存在")
                filtered_results = results
            elif confidence_diff >= CONFIDENCE_THRESHOLD:
                # 信頼度に明確な差がある場合、高い方を採用
                if best_yukari.identification.confidence > best_kei.identification.confidence:
                    chosen = best_yukari
                    rejected = best_kei
                else:
                    chosen = best_kei
                    rejected = best_yukari
                
                logger.info(f"✅ 信頼度差 {confidence_diff:.3f} により {chosen.identification.character_name}({chosen.identification.confidence:.3f}) を採用")
                logger.info(f"❌ {rejected.identification.character_name}({rejected.identification.confidence:.3f}) を除外")
                
                # 採用されたキャラクター以外の同キャラクター検出を除外
                if chosen.identification.character_name in yukari_names:
                    filtered_results = other_detections + yukari_detections
                else:
                    filtered_results = other_detections + kei_detections
            else:
                # 位置が重複しているが信頼度差が小さい場合、より高い信頼度の方を採用
                logger.info(f"⚖️ 位置重複 (IoU: {iou:.3f}) かつ信頼度差が小さい ({confidence_diff:.3f}) ため高信頼度を採用")
                if best_yukari.identification.confidence >= best_kei.identification.confidence:
                    chosen = best_yukari
                    rejected = best_kei
                    filtered_results = other_detections + yukari_detections
                else:
                    chosen = best_kei
                    rejected = best_yukari
                    filtered_results = other_detections + kei_detections
                    
                logger.info(f"✅ 高信頼度により {chosen.identification.character_name}({chosen.identification.confidence:.3f}) を採用")
                logger.info(f"❌ {rejected.identification.character_name}({rejected.identification.confidence:.3f}) を除外")
                    
            if len(filtered_results) != len(results):
                logger.info(f"🎯 日向縁・岡野佳フィルタリング完了: {len(results)} -> {len(filtered_results)}人")
            
            return filtered_results
        else:
            # 重複がない場合はそのまま返す
            return results
    
    def _filter_same_character_duplicates(self, results):
        """
        同じキャラクターが複数検出された場合の重複除去
        
        同じコマに同じキャラクターが複数検出された場合、最も信頼度の高いものを採用
        
        Args:
            results: FaceCharacterResultのリスト
            
        Returns:
            フィルタリング後のresultsリスト
        """
        if len(results) <= 1:
            return results
        
        # キャラクター名でグループ化
        character_groups = {}
        for result in results:
            char_name = result.identification.character_name
            if char_name not in character_groups:
                character_groups[char_name] = []
            character_groups[char_name].append(result)
        
        # 各キャラクターごとに重複チェック
        filtered_results = []
        for char_name, char_results in character_groups.items():
            if len(char_results) == 1:
                # 1人だけの場合はそのまま追加
                filtered_results.extend(char_results)
            else:
                # 複数検出の場合は信頼度でフィルタリング
                logger.info(f"🔄 同一キャラクター重複検出: {char_name} ({len(char_results)}件)")
                
                # 信頼度でソート（高い順）
                sorted_results = sorted(char_results, key=lambda r: r.identification.confidence, reverse=True)
                
                # 最高信頼度の検出を取得
                best_result = sorted_results[0]
                best_confidence = best_result.identification.confidence
                
                # 信頼度差による判定
                SAME_CHARACTER_CONFIDENCE_THRESHOLD = 0.10  # 10%以上の差があれば1つに絞る（緩和）
                MIN_KEEP_CONFIDENCE = 0.8  # 80%以上なら複数人の可能性も考慮
                
                keep_results = [best_result]  # 最高信頼度は必ず保持
                
                for other_result in sorted_results[1:]:
                    other_confidence = other_result.identification.confidence
                    confidence_diff = best_confidence - other_confidence
                    
                    # IoU計算で位置重複をチェック
                    iou = self._calculate_iou(best_result.detection.bbox, other_result.detection.bbox)
                    
                    if confidence_diff >= SAME_CHARACTER_CONFIDENCE_THRESHOLD:
                        # 日向縁の場合、信頼度差が大きい低い方を岡野佳に変更
                        if char_name in ['日向縁', 'yukari'] and other_confidence >= 0.6:
                            # 岡野佳として再分類
                            other_result.identification.character_name = '岡野佳' if char_name == '日向縁' else 'kei'
                            logger.info(f"  🔄 信頼度差 {confidence_diff:.3f} により岡野佳に再分類: {other_confidence:.3f}")
                            keep_results.append(other_result)
                        else:
                            # その他のキャラクターは除外
                            logger.info(f"  ❌ 信頼度差 {confidence_diff:.3f} により除外: {other_confidence:.3f}")
                    elif iou > 0.4:  # 40%以上重複している場合は同一人物
                        logger.info(f"  ❌ 位置重複 (IoU: {iou:.3f}) により除外: {other_confidence:.3f}")
                    elif other_confidence >= MIN_KEEP_CONFIDENCE and confidence_diff < 0.1:
                        # 両方とも高信頼度で差が小さい場合は両方保持（実際に2人いる可能性）
                        keep_results.append(other_result)
                        logger.info(f"  ✅ 高信頼度・低差分により保持: {other_confidence:.3f} (差分: {confidence_diff:.3f})")
                    else:
                        logger.info(f"  ❌ 除外: {other_confidence:.3f} (差分: {confidence_diff:.3f}, IoU: {iou:.3f})")
                
                filtered_results.extend(keep_results)
                
                if len(keep_results) != len(char_results):
                    logger.info(f"  🎯 {char_name} フィルタリング: {len(char_results)} -> {len(keep_results)}人")
        
        if len(filtered_results) != len(results):
            logger.info(f"🎯 同一キャラクター重複除去完了: {len(results)} -> {len(filtered_results)}人")
        
        return filtered_results
    
    def _load_models(self, yolo_path: str, dinov2_path: str):
        """モデルのロード"""
        # YOLOモデル
        logger.info(f"YOLOモデルロード: {yolo_path}")
        try:
            self.yolo_model = YOLO(yolo_path)
            logger.info("YOLOモデルロード完了")
        except Exception as e:
            logger.error(f"YOLOモデルロード失敗: {e}")
            raise
        
        # マルチクラスYOLOモデル
        logger.info(f"マルチクラスYOLOモデルロード: {self.multiclass_model_path}")
        try:
            if Path(self.multiclass_model_path).exists():
                self.multiclass_yolo_model = YOLO(self.multiclass_model_path)
                logger.info("マルチクラスYOLOモデルロード完了")
            else:
                logger.warning(f"マルチクラスモデルファイルが見つかりません: {self.multiclass_model_path}")
                self.multiclass_yolo_model = None
        except Exception as e:
            logger.error(f"マルチクラスYOLOモデルロード失敗: {e}")
            self.multiclass_yolo_model = None
        
        # DINOv2モデル
        logger.info(f"DINOv2モデルロード: {dinov2_path}")
        try:
            self._load_dinov2_classifier(dinov2_path)
            logger.info("DINOv2モデルロード完了")
        except Exception as e:
            logger.error(f"DINOv2モデルロード失敗: {e}")
            raise
    
    def _load_dinov2_classifier(self, model_path: str):
        """DINOv2分類器のロード"""
        # チェックポイントロード
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # モデル情報取得
        model_config = checkpoint.get('model_config', {})
        num_classes = model_config.get('num_classes', 8)
        model_name = model_config.get('model_name', 'facebook/dinov2-base')
        
        # クラス名設定
        if 'class_to_idx' in checkpoint:
            self.class_to_idx = checkpoint['class_to_idx']
            self.class_names = [''] * len(self.class_to_idx)
            for class_name, idx in self.class_to_idx.items():
                self.class_names[idx] = class_name
        else:
            self.class_names = ['yuzuko', 'yukari', 'yui', 'yoriko', 'chiho', 'kei', 'fumi', 'unknown']
        
        logger.info(f"クラス: {self.class_names}")
        
        # モデル初期化
        self.dinov2_model = DINOv2CharacterClassifier(num_classes=num_classes, model_name=model_name)
        
        # 重みロード
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        self.dinov2_model.load_state_dict(state_dict)
        self.dinov2_model.to(self.device)
        self.dinov2_model.eval()
        
        # プロセッサー初期化
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # モデル精度情報
        if 'training_info' in checkpoint:
            training_info = checkpoint['training_info']
            best_acc = training_info.get('best_val_acc', 'Unknown')
            if isinstance(best_acc, float):
                logger.info(f"モデル精度: {best_acc:.4f}")
    
    def set_mode(self, multiclass_mode: bool):
        """
        検出モードの設定
        
        Args:
            multiclass_mode: Trueならマルチクラスモード、Falseなら顔検出+認識モード
        """
        if multiclass_mode and self.multiclass_yolo_model is None:
            logger.warning("マルチクラスモデルが利用できません。顔検出+認識モードを継続します。")
            return
        
        self.multiclass_mode = multiclass_mode
        mode_str = "マルチクラス検出" if multiclass_mode else "顔検出+認識"
        logger.info(f"検出モードを {mode_str} に設定しました")
    
    def detect_multiclass_characters(self, image: np.ndarray) -> List[FaceCharacterResult]:
        """
        マルチクラスYOLOによる直接的なキャラクター検出
        
        Args:
            image: 入力画像 (BGR)
            
        Returns:
            キャラクター検出結果のリスト
        """
        if self.multiclass_yolo_model is None:
            logger.error("マルチクラスモデルが利用できません")
            return []
        
        results = self.multiclass_yolo_model(
            image,
            conf=self.detection_conf_threshold,
            iou=self.detection_iou_threshold,
            verbose=False
        )
        
        detections = []
        for r in results:
            if r.boxes is not None:
                for i, box in enumerate(r.boxes):
                    # バウンディングボックス
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                    
                    # 信頼度とクラス
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # クラス名取得
                    character_name = self.multiclass_names[cls] if cls < len(self.multiclass_names) else 'unknown'
                    
                    # 検出結果作成
                    face_detection = FaceDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        detection_id=i
                    )
                    
                    # キャラクター識別結果作成（マルチクラスの場合は1.0の確率で該当クラス）
                    all_probabilities = {name: 0.0 for name in self.multiclass_names}
                    all_probabilities[character_name] = 1.0
                    
                    character_identification = CharacterIdentification(
                        character_name=character_name,
                        confidence=conf,  # YOLOの検出信頼度をそのまま使用
                        all_probabilities=all_probabilities,
                        top_k_predictions=[{'character': character_name, 'confidence': conf}]
                    )
                    
                    # 結果統合
                    result = FaceCharacterResult(
                        detection=face_detection,
                        identification=character_identification,
                        face_image=None  # マルチクラスでは顔画像は抽出しない
                    )
                    
                    detections.append(result)
        
        return detections
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        YOLO11xによる顔検出
        
        Args:
            image: 入力画像 (BGR)
            
        Returns:
            顔検出結果のリスト
        """
        results = self.yolo_model(
            image, 
            conf=self.detection_conf_threshold, 
            iou=self.detection_iou_threshold,
            verbose=False
        )
        
        detections = []
        detection_id = 0
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    detection = FaceDetection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(conf),
                        detection_id=detection_id
                    )
                    detections.append(detection)
                    detection_id += 1
        
        return detections
    
    def extract_face_image(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """
        顔領域の抽出（マージン付き）
        
        Args:
            image: 元画像
            detection: 顔検出結果
            
        Returns:
            抽出された顔画像
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        
        # マージン計算
        margin_x = int((x2 - x1) * self.face_margin)
        margin_y = int((y2 - y1) * self.face_margin)
        
        # 境界チェック付きで座標計算
        x1 = max(0, int(x1 - margin_x))
        y1 = max(0, int(y1 - margin_y))
        x2 = min(w, int(x2 + margin_x))
        y2 = min(h, int(y2 + margin_y))
        
        return image[y1:y2, x1:x2]
    
    def identify_character(self, face_image: np.ndarray) -> CharacterIdentification:
        """
        DINOv2によるキャラクター識別
        
        Args:
            face_image: 顔画像 (BGR)
            
        Returns:
            キャラクター識別結果
        """
        # PILイメージに変換
        face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        
        # 前処理とテンソル変換
        inputs = self.processor(images=face_pil, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # 推論
        with torch.no_grad():
            logits = self.dinov2_model(pixel_values)
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            predicted_idx = torch.argmax(logits, dim=1).item()
            confidence = probabilities[predicted_idx].item()
        
        # 結果整理
        predicted_class = self.class_names[predicted_idx]
        
        # 信頼度が閾値以下の場合はunknownとする
        if confidence < self.classification_conf_threshold:
            predicted_class = 'unknown'
        
        # 全クラスの確率
        all_probabilities = {
            self.class_names[i]: float(probabilities[i])
            for i in range(len(self.class_names))
        }
        
        # Top-K予測
        top_k_indices = torch.topk(probabilities, k=min(3, len(self.class_names))).indices
        top_k_predictions = [
            {
                'class': self.class_names[idx.item()],
                'confidence': float(probabilities[idx.item()])
            }
            for idx in top_k_indices
        ]
        
        return CharacterIdentification(
            character_name=predicted_class,
            confidence=float(confidence),
            all_probabilities=all_probabilities,
            top_k_predictions=top_k_predictions
        )
    
    def process_image(self, image_path: Union[str, Path]) -> ImageProcessingResult:
        """
        画像全体の処理（検出＋識別）
        
        Args:
            image_path: 入力画像パス
            
        Returns:
            処理結果
        """
        start_time = datetime.now()
        image_path = Path(image_path)
        
        logger.info(f"画像処理開始: {image_path.name}")
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"画像読み込み失敗: {image_path}")
        
        h, w = image.shape[:2]
        
        # モードに応じて処理分岐
        if self.multiclass_mode:
            # マルチクラスモード：YOLOで直接キャラクター検出
            logger.info("マルチクラスモードで処理中...")
            results = self.detect_multiclass_characters(image)
            
            # 検出された全キャラクターの詳細表示
            detected_chars = []
            for result in results:
                char_name = result.identification.character_name
                confidence = result.identification.confidence
                detected_chars.append(f"{char_name}({confidence:.3f})")
                self.stats['character_counts'][char_name] = self.stats['character_counts'].get(char_name, 0) + 1
                logger.debug(f"キャラクター {result.detection.detection_id}: {char_name} (信頼度: {confidence:.3f})")
            
            logger.info(f"マルチクラス検出完了: {len(results)}人 (フィルター前) - [{', '.join(detected_chars)}]")
        else:
            # 従来モード：顔検出→認識の2段階
            logger.info("顔検出+認識モードで処理中...")
            
            # 顔検出
            detections = self.detect_faces(image)
            logger.info(f"検出顔数: {len(detections)}")
            
            # 各顔を識別
            results = []
            for detection in detections:
                try:
                    # 顔画像抽出
                    face_image = self.extract_face_image(image, detection)
                    
                    if face_image.size == 0:
                        logger.warning(f"顔画像抽出失敗: ID {detection.detection_id}")
                        continue
                    
                    # キャラクター識別
                    identification = self.identify_character(face_image)
                    
                    # 結果作成
                    result = FaceCharacterResult(
                        detection=detection,
                        identification=identification,
                        face_image=face_image
                    )
                    results.append(result)
                    
                    # 統計更新
                    char_name = identification.character_name
                    self.stats['character_counts'][char_name] = self.stats['character_counts'].get(char_name, 0) + 1
                    
                    logger.debug(f"顔 {detection.detection_id}: {char_name} (信頼度: {identification.confidence:.3f})")
                    
                except Exception as e:
                    logger.warning(f"識別エラー (ID: {detection.detection_id}): {e}")
                    continue
            
            logger.info(f"顔検出+認識完了: {len(results)}人識別")
        
        # 重複除去処理
        if len(results) > 1:
            original_count = len(results)
            logger.debug(f"フィルタリング開始: {original_count}人")
            
            # IoUベース重複除去
            after_iou = self._remove_duplicate_detections(results, iou_threshold=0.3)
            if len(after_iou) != len(results):
                logger.debug(f"IoU重複除去: {len(results)}人 → {len(after_iou)}人")
            results = after_iou
            
            # 日向縁・岡野佳の重複検出フィルタリング
            after_yukari_kei = self._filter_yukari_kei_duplicates(results)
            if len(after_yukari_kei) != len(results):
                logger.debug(f"日向縁・岡野佳フィルター: {len(results)}人 → {len(after_yukari_kei)}人")
            results = after_yukari_kei
            
            # 同一キャラクター重複除去
            final_results = self._filter_same_character_duplicates(results)
            if len(final_results) != len(results):
                logger.debug(f"同一キャラクター重複除去: {len(results)}人 → {len(final_results)}人")
            results = final_results
            
            if len(results) != original_count:
                # フィルター後の最終結果詳細表示
                final_chars = [f"{r.identification.character_name}({r.identification.confidence:.3f})" for r in results]
                logger.info(f"重複除去完了: {original_count}人 → {len(results)}人 (フィルター後) - [{', '.join(final_chars)}]")
        
        # 処理時間計算
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 結果作成
        processing_result = ImageProcessingResult(
            image_path=str(image_path),
            image_size=(w, h),
            results=results,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        # 統計更新
        self.stats['total_images'] += 1
        self.stats['total_faces'] += len(results)  # 修正: detectionsではなくresultsを使用
        self.stats['total_identifications'] += len(results)
        self.stats['processing_times'].append(processing_time)
        
        logger.info(f"処理完了: {len(results)}人識別 ({processing_time:.2f}秒)")
        
        return processing_result
    
    def visualize_results(
        self,
        image_path: Union[str, Path],
        result: ImageProcessingResult,
        output_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        use_character_colors: bool = True
    ) -> np.ndarray:
        """
        結果の可視化
        
        Args:
            image_path: 元画像パス
            result: 処理結果
            output_path: 保存先パス
            show: 表示するか
            use_character_colors: キャラクター別色を使用するか
            
        Returns:
            描画済み画像
        """
        # 画像読み込み
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # プロット作成
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # 日本語フォント設定
        try:
            font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
            if Path(font_path).exists():
                prop = font_manager.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = prop.get_name()
        except:
            pass
        
        # 重複除去後の結果を描画
        for i, face_result in enumerate(result.results):
            x1, y1, x2, y2 = face_result.detection.bbox
            character = face_result.identification.character_name
            char_conf = face_result.identification.confidence
            det_conf = face_result.detection.confidence
            
            # 色の選択（キャラクター別）
            if use_character_colors:
                color_rgb = self.CHARACTER_COLORS.get(character, self.CHARACTER_COLORS['unknown'])
                color = tuple(c/255.0 for c in color_rgb)
            else:
                # 信頼度ベースの色
                if char_conf >= 0.8:
                    color = 'green'
                elif char_conf >= 0.6:
                    color = 'yellow'
                else:
                    color = 'red'
            
            # バウンディングボックス（太線）
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=4, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # 簡潔なラベル表示（内側に）
            # マルチクラスモードの場合は検出信頼度、従来モードは識別信頼度を重視
            main_conf = char_conf if not self.multiclass_mode else det_conf
            label = f'{character} ({main_conf:.1%})'
            
            # ラベルを内側の適切な位置に配置
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            text_x = x1 + bbox_width * 0.02  # 左端から少し内側
            text_y = y1 + bbox_height * 0.95  # 下端近く
            
            ax.text(
                text_x, text_y, label,
                color='white', fontsize=12, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='white', linewidth=1),
                verticalalignment='bottom'
            )
        
        ax.axis('off')
        
        # タイトルにモード情報を含める
        mode_str = "マルチクラス検出" if self.multiclass_mode else "顔検出+認識"
        character_list = ", ".join(result.get_character_counts().keys()) if result.get_character_counts() else "なし"
        title = f'{mode_str} | 検出: {len(result.results)}人 | キャラクター: {character_list}'
        ax.set_title(title, fontsize=14, weight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            logger.info(f"可視化画像保存: {output_path}")
        
        if show:
            plt.show()
        
        # numpy配列として返す（Mac互換性を考慮）
        fig.canvas.draw()
        
        # BufferIOを使用してPNG形式で画像を取得
        import io
        from PIL import Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        
        # PILで読み込んでnumpy配列に変換
        pil_image = Image.open(buf)
        img_array = np.array(pil_image)
        
        plt.close()
        buf.close()
        
        # RGB形式に変換（透明度チャンネルを除去）
        if img_array.shape[-1] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # RGB
        
        return img_array
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        show_progress: bool = True
    ) -> List[ImageProcessingResult]:
        """
        バッチ処理
        
        Args:
            image_paths: 画像パスのリスト
            show_progress: プログレスバー表示
            
        Returns:
            処理結果のリスト
        """
        results = []
        
        if show_progress:
            image_paths = tqdm(image_paths, desc="画像処理中")
        
        for image_path in image_paths:
            try:
                result = self.process_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"バッチ処理エラー {image_path}: {e}")
                continue
        
        return results
    
    def process_directory(
        self,
        directory: Union[str, Path],
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
        recursive: bool = False,
        max_images: Optional[int] = None
    ) -> List[ImageProcessingResult]:
        """
        ディレクトリ一括処理
        
        Args:
            directory: ディレクトリパス
            extensions: 対象拡張子
            recursive: 再帰的検索
            max_images: 最大処理画像数
            
        Returns:
            処理結果のリスト
        """
        directory = Path(directory)
        
        # 画像ファイル収集
        image_files = []
        for ext in extensions:
            if recursive:
                image_files.extend(directory.rglob(f"*{ext}"))
                image_files.extend(directory.rglob(f"*{ext.upper()}"))
            else:
                image_files.extend(directory.glob(f"*{ext}"))
                image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        # 重複除去とソート
        image_files = sorted(list(set(image_files)))
        
        if max_images and len(image_files) > max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"処理対象: {len(image_files)}枚")
        
        return self.process_batch(image_files)
    
    def save_results(
        self,
        results: List[ImageProcessingResult],
        output_dir: Union[str, Path],
        save_format: str = 'json'
    ):
        """
        結果の保存
        
        Args:
            results: 処理結果のリスト
            output_dir: 出力ディレクトリ
            save_format: 保存形式 ('json' or 'csv')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if save_format == 'json':
            # JSON保存
            output_data = {
                'pipeline_info': {
                    'timestamp': timestamp,
                    'total_images': len(results),
                    'class_names': self.class_names,
                    'statistics': self.get_statistics()
                },
                'results': [r.to_dict() for r in results]
            }
            
            output_path = output_dir / f"results_{timestamp}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"結果保存: {output_path}")
            
        elif save_format == 'csv':
            # CSV保存
            import pandas as pd
            
            csv_rows = []
            for result in results:
                for face_result in result.results:
                    row = {
                        'image_path': result.image_path,
                        'image_width': result.image_size[0],
                        'image_height': result.image_size[1],
                        'face_id': face_result.detection.detection_id,
                        'x1': face_result.detection.bbox[0],
                        'y1': face_result.detection.bbox[1],
                        'x2': face_result.detection.bbox[2],
                        'y2': face_result.detection.bbox[3],
                        'detection_confidence': face_result.detection.confidence,
                        'character': face_result.identification.character_name,
                        'character_confidence': face_result.identification.confidence,
                        'processing_time': result.processing_time
                    }
                    
                    # 各キャラクターの確率を追加
                    for char, prob in face_result.identification.all_probabilities.items():
                        row[f'prob_{char}'] = prob
                    
                    csv_rows.append(row)
            
            df = pd.DataFrame(csv_rows)
            output_path = output_dir / f"results_{timestamp}.csv"
            df.to_csv(output_path, index=False)
            
            logger.info(f"CSV保存: {output_path}")
    
    def get_statistics(self) -> Dict:
        """統計情報の取得"""
        stats = self.stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['total_processing_time'] = sum(stats['processing_times'])
        else:
            stats['avg_processing_time'] = 0.0
            stats['total_processing_time'] = 0.0
        
        # 成功率
        if stats['total_faces'] > 0:
            stats['identification_rate'] = stats['total_identifications'] / stats['total_faces']
        else:
            stats['identification_rate'] = 0.0
        
        return stats
    
    def print_statistics(self):
        """統計情報の表示"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("YOLO11x + DINOv2 パイプライン統計")
        print("="*60)
        print(f"処理画像数: {stats['total_images']}")
        print(f"検出顔数: {stats['total_faces']}")
        print(f"識別成功数: {stats['total_identifications']}")
        print(f"識別率: {stats['identification_rate']:.1%}")
        print(f"平均処理時間: {stats['avg_processing_time']:.2f}秒/枚")
        print(f"総処理時間: {stats['total_processing_time']:.2f}秒")
        
        if stats['character_counts']:
            print(f"\nキャラクター別検出数:")
            for char, count in sorted(stats['character_counts'].items(), 
                                     key=lambda x: x[1], reverse=True):
                print(f"  {char}: {count}人")
        print("="*60)


def main():
    """デモ実行"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='YOLO11x + DINOv2 統合パイプライン',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--yolo_model',
                        default='/Users/esuji/work/fun_annotator/yolo11x_face_detection_model/yolo11l_480_12_14.pt',
                        help='YOLOモデルパス')
    parser.add_argument('--dinov2_model',
                        default='/Users/esuji/work/fun_annotator/yuyu_dinov2_final.pth',
                        help='DINOv2モデルパス')
    parser.add_argument('--input',
                        required=True,
                        help='入力画像またはディレクトリ')
    parser.add_argument('--output_dir',
                        default='./pipeline_output',
                        help='出力ディレクトリ')
    parser.add_argument('--device',
                        choices=['auto', 'cpu', 'cuda'],
                        default='auto',
                        help='使用デバイス')
    parser.add_argument('--detection_conf',
                        type=float,
                        default=0.25,
                        help='検出信頼度閾値')
    parser.add_argument('--classification_conf',
                        type=float,
                        default=0.5,
                        help='分類信頼度閾値')
    parser.add_argument('--visualize',
                        action='store_true',
                        help='結果を可視化')
    parser.add_argument('--save_format',
                        choices=['json', 'csv'],
                        default='json',
                        help='保存形式')
    parser.add_argument('--max_images',
                        type=int,
                        help='最大処理画像数')
    
    args = parser.parse_args()
    
    try:
        # パイプライン初期化
        pipeline = YuyuYOLODINOv2Pipeline(
            yolo_model_path=args.yolo_model,
            dinov2_model_path=args.dinov2_model,
            device=args.device,
            detection_conf_threshold=args.detection_conf,
            classification_conf_threshold=args.classification_conf
        )
        
        # 入力パス確認
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 単一画像処理
            print(f"\n単一画像処理: {input_path}")
            result = pipeline.process_image(input_path)
            results = [result]
            
            # 結果表示
            print(f"\n処理結果:")
            print(f"  検出顔数: {len(result.results)}")
            print(f"  キャラクター: {result.get_character_counts()}")
            print(f"  処理時間: {result.processing_time:.2f}秒")
            
        elif input_path.is_dir():
            # ディレクトリ処理
            print(f"\nディレクトリ処理: {input_path}")
            results = pipeline.process_directory(
                input_path,
                max_images=args.max_images
            )
            
        else:
            raise ValueError(f"入力パスが見つかりません: {input_path}")
        
        # 結果保存
        output_dir = Path(args.output_dir)
        pipeline.save_results(results, output_dir, save_format=args.save_format)
        
        # 可視化
        if args.visualize:
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            print("\n可視化作成中...")
            for result in tqdm(results[:10], desc="可視化"):  # 最大10枚
                if len(result.results) > 0:
                    image_path = Path(result.image_path)
                    vis_path = vis_dir / f"{image_path.stem}_vis.jpg"
                    pipeline.visualize_results(
                        image_path,
                        result,
                        output_path=vis_path
                    )
        
        # 統計表示
        pipeline.print_statistics()
        
        print(f"\n完了！結果は {output_dir} に保存されました。")
        
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())