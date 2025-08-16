/**
 * しっぽの形状分析ユーティリティ
 * サーバー側のOpenCV処理を利用した高精度な先端点検出
 */

import axios from 'axios';
import { BoundingBox, Point, Vector } from './speakerIdentification';

export interface TailShapeAnalysisResult {
  tail_center: [number, number];
  tip_point: [number, number];
  direction_vector: [number, number];
  raw_direction: [number, number];
  confidence: number;
}

/**
 * 画像URLからBase64エンコードされた画像データを取得
 */
export async function fetchImageAsBase64(imageUrl: string): Promise<string> {
  try {
    const response = await fetch(imageUrl);
    const blob = await response.blob();
    
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (reader.result && typeof reader.result === 'string') {
          // data:image/jpeg;base64,xxxxx の形式から base64部分のみを抽出
          const base64 = reader.result.split(',')[1];
          resolve(base64);
        } else {
          reject(new Error('Failed to convert image to base64'));
        }
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  } catch (error) {
    console.error('画像の取得に失敗:', error);
    throw error;
  }
}

/**
 * しっぽの形状を分析して正確な先端点を取得
 */
export async function analyzeTailShape(
  imageUrl: string,
  tailBBox: BoundingBox,
  balloonBBox: BoundingBox
): Promise<{ center: Point; vector: Vector; farthestPoint: Point } | null> {
  try {
    console.log('しっぽ形状分析開始...');
    
    // 画像をBase64に変換
    const imageBase64 = await fetchImageAsBase64(imageUrl);
    
    // APIリクエストデータ
    const requestData = {
      imageBase64,
      tailBBox: {
        x1: tailBBox.x1,
        y1: tailBBox.y1,
        x2: tailBBox.x2,
        y2: tailBBox.y2
      },
      balloonBBox: {
        x1: balloonBBox.x1,
        y1: balloonBBox.y1,
        x2: balloonBBox.x2,
        y2: balloonBBox.y2
      }
    };
    
    // サーバーに形状分析を依頼
    const response = await axios.post<TailShapeAnalysisResult>(
      'http://localhost:8000/api/analyze-tail-shape',
      requestData
    );
    
    if (response.data) {
      const result = response.data;
      console.log('形状分析成功:', result);
      
      return {
        center: { x: result.tail_center[0], y: result.tail_center[1] },
        vector: { x: result.direction_vector[0], y: result.direction_vector[1] },
        farthestPoint: { x: result.tip_point[0], y: result.tip_point[1] }
      };
    }
    
    return null;
  } catch (error) {
    console.warn('しっぽ形状分析でエラーが発生しました。従来の方法にフォールバック:', error);
    return null;
  }
}

/**
 * 形状分析が有効かどうかをチェック
 */
export async function isShapeAnalysisAvailable(): Promise<boolean> {
  try {
    // ヘルスチェック的な軽いリクエストで確認
    const response = await axios.post(
      'http://localhost:8000/api/analyze-tail-shape',
      {
        imageBase64: 'test',
        tailBBox: { x1: 0, y1: 0, x2: 1, y2: 1 },
        balloonBBox: { x1: 0, y1: 0, x2: 1, y2: 1 }
      },
      { timeout: 1000 }
    );
    // レスポンスが返ってくれば形状分析は利用可能
    return true;
  } catch (error: any) {
    // エラーレスポンスの内容を確認
    if (error.response?.status === 500 && error.response?.data?.error?.includes('cv2')) {
      console.warn('OpenCVがインストールされていません。形状分析は無効です。');
      return false;
    }
    // その他のエラーでも形状分析は利用不可と判断
    return false;
  }
}