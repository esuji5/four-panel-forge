#!/usr/bin/env python3
"""
ゆゆ式キャラクター分類器 - DINOv2ベース
学習済みモデルを使用した高精度キャラクター識別システム
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DINOv2CharacterClassifier(nn.Module):
    """DINOv2ベースのキャラクター分類器"""
    
    def __init__(self, num_classes: int = 8, model_name: str = 'facebook/dinov2-base'):
        super().__init__()
        self.dinov2 = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.dinov2.config.hidden_size, 512),  # 0: Linear(768->512)
            nn.ReLU(),                                       # 1: ReLU
            nn.Dropout(0.3),                                 # 2: Dropout
            nn.Linear(512, 256),                             # 3: Linear(512->256)
            nn.ReLU(),                                       # 4: ReLU
            nn.Dropout(0.3),                                 # 5: Dropout
            nn.Linear(256, num_classes)                      # 6: Linear(256->8)
        )
        
    def forward(self, pixel_values):
        outputs = self.dinov2(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.classifier(pooled_output)
        return logits


class YuyuCharacterClassifier:
    """ゆゆ式キャラクター分類器の統合インターフェース"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        初期化
        
        Args:
            model_path: 学習済みモデルのパス
            device: 'auto', 'cpu', 'cuda'のいずれか
        """
        self.model_path = Path(model_path)
        
        # デバイス設定
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"使用デバイス: {self.device}")
        
        # モデルとプロセッサーの初期化
        self.model = None
        self.processor = None
        self.class_names = []
        self.class_to_idx = {}
        self.model_info = {}
        
        self._load_model()
        
    def _load_model(self):
        """学習済みモデルのロード"""
        try:
            logger.info(f"モデルロード開始: {self.model_path}")
            
            # チェックポイントロード
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # モデル設定の取得
            self.model_info = {
                'model_config': checkpoint.get('model_config', {}),
                'training_info': checkpoint.get('training_info', {}),
                'dataset_info': checkpoint.get('dataset_info', {})
            }
            
            num_classes = self.model_info['model_config'].get('num_classes', 8)
            model_name = self.model_info['model_config'].get('model_name', 'facebook/dinov2-base')
            
            # クラス名の設定
            if 'class_to_idx' in checkpoint:
                self.class_to_idx = checkpoint['class_to_idx']
                # インデックス順にクラス名を配列化
                self.class_names = [''] * len(self.class_to_idx)
                for class_name, idx in self.class_to_idx.items():
                    self.class_names[idx] = class_name
            else:
                # デフォルトクラス名
                self.class_names = ['yuzuko', 'yukari', 'yui', 'yoriko', 'chiho', 'kei', 'fumi', 'unknown']
                self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
            
            logger.info(f"クラス数: {num_classes}")
            logger.info(f"クラス名: {self.class_names}")
            
            # モデル初期化
            self.model = DINOv2CharacterClassifier(num_classes=num_classes, model_name=model_name)
            
            # 重みロード
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # プロセッサー初期化
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            
            # モデル情報表示
            if 'training_info' in checkpoint:
                training_info = checkpoint['training_info']
                best_acc = training_info.get('best_val_acc', 'Unknown')
                if isinstance(best_acc, float):
                    logger.info(f"モデル性能: 最高検証精度 {best_acc:.4f}")
                
            logger.info("モデルロード完了")
            
        except Exception as e:
            logger.error(f"モデルロード失敗: {e}")
            raise
    
    def predict_single(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Dict:
        """
        単一画像の予測
        
        Args:
            image: PIL Image, numpy array, または画像パス
            
        Returns:
            予測結果辞書 {
                'predicted_class': str,
                'confidence': float,
                'probabilities': dict,
                'top_k_predictions': list
            }
        """
        try:
            # 画像の前処理
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError(f"サポートされていない画像形式: {type(image)}")
            
            # 前処理とテンソル変換
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
            
            # 推論
            with torch.no_grad():
                logits = self.model(pixel_values)
                probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
                predicted_idx = torch.argmax(logits, dim=1).item()
                confidence = probabilities[predicted_idx].item()
            
            # 結果整理
            predicted_class = self.class_names[predicted_idx]
            prob_dict = {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(probabilities)
            }
            
            # Top-K予測
            top_k_indices = torch.topk(probabilities, k=min(3, len(self.class_names))).indices
            top_k_predictions = [
                {
                    'class': self.class_names[idx.item()],
                    'confidence': probabilities[idx.item()].item()
                }
                for idx in top_k_indices
            ]
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': prob_dict,
                'top_k_predictions': top_k_predictions
            }
            
        except Exception as e:
            logger.error(f"予測エラー: {e}")
            raise
    
    def predict_batch(self, images: List[Union[Image.Image, np.ndarray]], 
                     batch_size: int = 32) -> List[Dict]:
        """
        バッチ予測
        
        Args:
            images: 画像のリスト
            batch_size: バッチサイズ
            
        Returns:
            予測結果のリスト
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_results = []
            
            for image in batch:
                try:
                    result = self.predict_single(image)
                    batch_results.append(result)
                except Exception as e:
                    logger.warning(f"バッチ予測エラー (画像 {i}): {e}")
                    batch_results.append({
                        'predicted_class': 'error',
                        'confidence': 0.0,
                        'probabilities': {},
                        'top_k_predictions': [],
                        'error': str(e)
                    })
            
            results.extend(batch_results)
        
        return results
    
    def get_model_info(self) -> Dict:
        """モデル情報の取得"""
        return {
            'model_path': str(self.model_path),
            'device': str(self.device),
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'num_classes': len(self.class_names),
            'model_info': self.model_info
        }
    
    def save_predictions(self, predictions: List[Dict], output_path: str, 
                        format: str = 'json'):
        """
        予測結果の保存
        
        Args:
            predictions: 予測結果のリスト
            output_path: 出力パス
            format: 'json' または 'csv'
        """
        output_path = Path(output_path)
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)
        elif format == 'csv':
            import pandas as pd
            
            # CSV用にデータを平坦化
            csv_data = []
            for i, pred in enumerate(predictions):
                row = {
                    'image_id': i,
                    'predicted_class': pred['predicted_class'],
                    'confidence': pred['confidence']
                }
                # 各クラスの確率を追加
                for class_name, prob in pred['probabilities'].items():
                    row[f'prob_{class_name}'] = prob
                
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
        
        logger.info(f"予測結果保存: {output_path}")


def main():
    """デモ実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ゆゆ式キャラクター分類器')
    parser.add_argument('--model_path', 
                        default='/Users/esuji/work/fun_annotator/yuyu_dinov2_final.pth',
                        help='学習済みモデルのパス')
    parser.add_argument('--image_path', 
                        help='予測する画像のパス')
    parser.add_argument('--device', 
                        default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='使用デバイス')
    
    args = parser.parse_args()
    
    # 分類器初期化
    classifier = YuyuCharacterClassifier(
        model_path=args.model_path,
        device=args.device
    )
    
    # モデル情報表示
    model_info = classifier.get_model_info()
    print("\n=== モデル情報 ===")
    print(f"クラス数: {model_info['num_classes']}")
    print(f"クラス名: {model_info['class_names']}")
    print(f"デバイス: {model_info['device']}")
    
    if args.image_path:
        # 単一画像予測
        print(f"\n=== 予測実行: {args.image_path} ===")
        
        try:
            result = classifier.predict_single(args.image_path)
            
            print(f"予測クラス: {result['predicted_class']}")
            print(f"信頼度: {result['confidence']:.4f}")
            
            print("\nTop-3予測:")
            for i, pred in enumerate(result['top_k_predictions'], 1):
                print(f"  {i}. {pred['class']}: {pred['confidence']:.4f}")
            
            print("\n全クラス確率:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
                
        except Exception as e:
            print(f"予測エラー: {e}")
    
    else:
        print("\n画像パスが指定されていません。")
        print("使用例: python yuyu_character_classifier.py --image_path path/to/image.jpg")


if __name__ == "__main__":
    main()