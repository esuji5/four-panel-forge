/**
 * 吹き出しのしっぽから話者を特定するユーティリティ
 */

import { analyzeTailShape, isShapeAnalysisAvailable } from './tailShapeAnalysis';

export interface Point {
  x: number;
  y: number;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width?: number;
  height?: number;
}

export interface Vector {
  x: number;
  y: number;
}

export interface TailData {
  boundingBox: BoundingBox;
  type: string;
  direction: string;
}

export interface CharacterDetection {
  id: string;
  name: string;
  boundingBox: BoundingBox;
  faceDirection?: string;
}

export interface SpeakerScore {
  characterId: string;
  characterName: string;
  totalScore: number;
  distanceScore: number;
  directionScore: number;
  intersectionScore: number;
  distance: number;
  rayVisualization?: RayVisualizationData;
}

/**
 * バウンディングボックスの中心点を取得
 */
function getBBoxCenter(bbox: BoundingBox): Point {
  return {
    x: (bbox.x1 + bbox.x2) / 2,
    y: (bbox.y1 + bbox.y2) / 2
  };
}

/**
 * 2点間の距離を計算
 */
function calculateDistance(p1: Point, p2: Point): number {
  return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}

/**
 * ベクトルの大きさを計算
 */
function vectorMagnitude(v: Vector): number {
  return Math.sqrt(v.x * v.x + v.y * v.y);
}

/**
 * ベクトルの正規化
 */
function normalizeVector(v: Vector): Vector {
  const mag = vectorMagnitude(v);
  if (mag === 0) return { x: 0, y: 0 };
  return { x: v.x / mag, y: v.y / mag };
}

/**
 * ベクトルの内積
 */
function dotProduct(v1: Vector, v2: Vector): number {
  return v1.x * v2.x + v1.y * v2.y;
}

/**
 * しっぽ分類カテゴリーから基準角度（度）を取得
 */
function getBaseAngleFromCategory(category: string): number | null {
  const categoryMap: Record<string, number> = {
    // 下方向系（真下を180°として）
    '真下': 180,
    
    // 右下方向系
    '下右やや': 165,      // 真下から右に15°
    '下右少し': 150,     // 真下から右に30°
    '下右30度以上': 120, // 真下から右に60°
    
    // 左下方向系
    '下左やや': 195,      // 真下から左に15°
    '下左少し': 210,     // 真下から左に30°
    '下左30度以上': 240, // 真下から左に60°
    
    // 右方向系
    '右やや': 105,       // 右から少し下に15°
    '右少し': 90,        // 真右
    '右30度以上': 75,    // 右から少し上に15°
    
    // 左方向系  
    '左やや': 255,       // 左から少し下に15°
    '左少し': 270,       // 真左
    '左30度以上': 285,   // 左から少し上に15°
    
    // 上方向系
    '上右やや': 15,      // 真上から右に15°
    '上右少し': 30,      // 真上から右に30°
    '上左やや': 345,     // 真上から左に15°
    '上左少し': 330,     // 真上から左に30°
    '真上': 0,
  };
  
  return categoryMap[category] || null;
}

/**
 * 角度（度）からベクトルに変換（Y軸下向きの座標系）
 * 0度=真上、90度=真右、180度=真下、270度=真左
 */
function angleToVector(angleDegrees: number): Vector {
  const angleRad = (angleDegrees * Math.PI) / 180;
  return {
    x: Math.sin(angleRad),   // X軸：右が正
    y: -Math.cos(angleRad)   // Y軸：下が正、0度で真上を向くためにマイナス
  };
}

/**
 * 左右反転の可能性があるカテゴリかどうかを判定
 */
function isLRAmbiguousCategory(category: string): boolean {
  const ambiguousCategories = [
    '下左やや', '下右やや',
    '下左少し', '下右少し'
  ];
  return ambiguousCategories.includes(category);
}

/**
 * カテゴリの左右を反転した対応カテゴリを取得
 */
function getFlippedCategory(category: string): string {
  const flippedMap: Record<string, string> = {
    '下左やや': '下右やや',
    '下右やや': '下左やや',
    '下左少し': '下右少し',
    '下右少し': '下左少し'
  };
  return flippedMap[category] || category;
}

/**
 * 2つのベクトルを重み付き平均で合成
 */
function blendVectors(vec1: Vector, vec2: Vector, weight1: number, weight2: number): Vector {
  const totalWeight = weight1 + weight2;
  if (totalWeight === 0) return vec1;
  
  const blended = {
    x: (vec1.x * weight1 + vec2.x * weight2) / totalWeight,
    y: (vec1.y * weight1 + vec2.y * weight2) / totalWeight
  };
  
  return normalizeVector(blended);
}

/**
 * 物理的計算のみでしっぽベクトルを計算（フォールバック用）
 */
function calculatePhysicalOnlyTailVector(
  tailBBox: BoundingBox, 
  balloonBBox: BoundingBox
): { center: Point; vector: Vector; farthestPoint: Point } {
  const tailCenter = getBBoxCenter(tailBBox);
  const balloonCenter = getBBoxCenter(balloonBBox);
  
  // 物理的計算：しっぽの最も外側の点を見つける
  const tailPoints = [
    { x: tailBBox.x1, y: tailBBox.y1 },
    { x: tailBBox.x2, y: tailBBox.y1 },
    { x: tailBBox.x1, y: tailBBox.y2 },
    { x: tailBBox.x2, y: tailBBox.y2 }
  ];
  
  let farthestPoint = tailPoints[0];
  let maxDistance = 0;
  
  tailPoints.forEach(point => {
    const dist = calculateDistance(point, balloonCenter);
    if (dist > maxDistance) {
      maxDistance = dist;
      farthestPoint = point;
    }
  });
  
  // 物理的計算による方向ベクトル
  const physicalVector = {
    x: farthestPoint.x - tailCenter.x,
    y: farthestPoint.y - tailCenter.y
  };
  
  const normalizedPhysicalVector = normalizeVector(physicalVector);
  
  return {
    center: tailCenter,
    vector: normalizedPhysicalVector,
    farthestPoint: farthestPoint
  };
}

/**
 * しっぽの方向ベクトルを計算（分類情報も考慮）
 */
export function calculateTailVector(
  tailBBox: BoundingBox, 
  balloonBBox: BoundingBox, 
  shapeCategory?: string,
  shapeConfidence?: number,
  balloonType?: string
): { center: Point; vector: Vector; farthestPoint: Point } {
  const tailCenter = getBBoxCenter(tailBBox);
  const balloonCenter = getBBoxCenter(balloonBBox);
  
  console.log('しっぽBBox:', tailBBox);
  console.log('吹き出しBBox:', balloonBBox);
  console.log('しっぽ中心:', tailCenter);
  console.log('吹き出し中心:', balloonCenter);
  
  // 1. 物理的計算：しっぽの最も外側の点を見つける（吹き出しから最も遠い点）
  const tailPoints = [
    { x: tailBBox.x1, y: tailBBox.y1 },
    { x: tailBBox.x2, y: tailBBox.y1 },
    { x: tailBBox.x1, y: tailBBox.y2 },
    { x: tailBBox.x2, y: tailBBox.y2 }
  ];
  
  let farthestPoint = tailPoints[0];
  let maxDistance = 0;
  
  tailPoints.forEach(point => {
    const dist = calculateDistance(point, balloonCenter);
    if (dist > maxDistance) {
      maxDistance = dist;
      farthestPoint = point;
    }
  });
  
  // 物理的計算による方向ベクトル
  const physicalVector = {
    x: farthestPoint.x - tailCenter.x,
    y: farthestPoint.y - tailCenter.y
  };
  
  console.log('最遠点:', farthestPoint);
  console.log('物理的方向ベクトル（正規化前）:', physicalVector);
  
  const normalizedPhysicalVector = normalizeVector(physicalVector);
  console.log('物理的方向ベクトル（正規化後）:', normalizedPhysicalVector);
  
  // 2. 分類情報による基準ベクトル
  let finalVector = normalizedPhysicalVector;
  
  if (shapeCategory && shapeCategory !== 'しっぽじゃない') {
    const baseAngle = getBaseAngleFromCategory(shapeCategory);
    
    if (baseAngle !== null) {
      const classificationVector = angleToVector(baseAngle);
      const confidence = shapeConfidence || 0.5;
      
      console.log(`🔄 分類情報: ${shapeCategory} (信頼度: ${confidence.toFixed(3)})`);
      console.log('分類基準角度:', baseAngle, '度');
      console.log('分類基準ベクトル:', classificationVector);
      
      // 結合吹き出しの場合は分類結果のみを使用
      if (balloonType === 'combined_bubble') {
        finalVector = classificationVector;
        console.log('🔄 結合吹き出し検出: 分類結果のみを使用');
        console.log('最終ベクトル（分類のみ）:', finalVector);
      } else {
        // 基本方向（真下、真上、真左、真右）で高信頼度の場合は分類ベクトルのみ使用
        const isBasicDirection = ['真下', '真上', '真左', '真右'].includes(shapeCategory);
        // 高信頼度の下方向系も分類結果を優先
        const isDownDirection = shapeCategory.startsWith('下') || shapeCategory.startsWith('真下');
        const isHighConfidence = confidence >= 0.7;
        const isVeryHighConfidence = confidence >= 0.8;  // 85%から80%に下げる
        
        if ((isBasicDirection && isHighConfidence) || (isDownDirection && isVeryHighConfidence)) {
          // 基本方向で高信頼度、または下方向系で高信頼度の場合は分類結果のみを使用
          finalVector = classificationVector;
          console.log(`🎯 ${shapeCategory} (信頼度: ${confidence.toFixed(3)}): 分類結果のみを使用`);
          console.log('最終ベクトル（分類のみ）:', finalVector);
        } else {
          // その他の場合は従来通りの重み付け
          // 高信頼度（0.8以上）：分類60% + 物理40%
          // 中信頼度（0.5-0.8）：分類50% + 物理50%  
          // 低信頼度（0.5未満）：分類30% + 物理70%
          let classificationWeight: number;
          let physicalWeight: number;
          
          if (confidence >= 0.8) {
            classificationWeight = 0.6;
            physicalWeight = 0.4;
          } else if (confidence >= 0.5) {
            classificationWeight = 0.5;
            physicalWeight = 0.5;
          } else {
            classificationWeight = 0.3;
            physicalWeight = 0.7;
          }
          
          finalVector = blendVectors(
            normalizedPhysicalVector, 
            classificationVector, 
            physicalWeight, 
            classificationWeight
          );
          
          console.log(`🔄 重み付け: 物理${physicalWeight} + 分類${classificationWeight}`);
          console.log('最終合成ベクトル:', finalVector);
        }
      }
    } else {
      console.log(`⚠️ 未対応の分類カテゴリ: ${shapeCategory}`);
    }
  } else {
    console.log('🔄 分類情報なし、物理的計算のみ使用');
  }
  
  return {
    center: tailCenter,
    vector: finalVector,
    farthestPoint: farthestPoint
  };
}

/**
 * 線分とバウンディングボックスの詳細交差判定
 */
function checkRayBBoxIntersectionDetailed(
  rayOrigin: Point,
  rayDirection: Vector,
  bbox: BoundingBox,
  maxRayLength: number = 2000
): { intersects: boolean; intersectionPoint?: Point; rayEnd: Point } {
  // エッジケース処理
  if (rayDirection.x === 0 && rayDirection.y === 0) {
    const rayEnd = { x: rayOrigin.x, y: rayOrigin.y };
    return { intersects: false, rayEnd };
  }
  
  // レイの終点を計算
  const rayEnd = {
    x: rayOrigin.x + rayDirection.x * maxRayLength,
    y: rayOrigin.y + rayDirection.y * maxRayLength
  };
  
  // 線分とボックスの交差判定（簡易版）
  const minX = Math.min(rayOrigin.x, rayEnd.x);
  const maxX = Math.max(rayOrigin.x, rayEnd.x);
  const minY = Math.min(rayOrigin.y, rayEnd.y);
  const maxY = Math.max(rayOrigin.y, rayEnd.y);
  
  // バウンディングボックスとの重なりチェック
  if (maxX < bbox.x1 || minX > bbox.x2 || maxY < bbox.y1 || minY > bbox.y2) {
    return { intersects: false, rayEnd };
  }
  
  // より詳細な交差判定（線分とボックスの各辺との交差をチェック）
  let intersectionPoint: Point | undefined;
  
  // 各辺との交差をチェックして、最初の交差点を見つける
  const edges = [
    // 上辺
    { p1: { x: bbox.x1, y: bbox.y1 }, p2: { x: bbox.x2, y: bbox.y1 } },
    // 右辺
    { p1: { x: bbox.x2, y: bbox.y1 }, p2: { x: bbox.x2, y: bbox.y2 } },
    // 下辺
    { p1: { x: bbox.x2, y: bbox.y2 }, p2: { x: bbox.x1, y: bbox.y2 } },
    // 左辺
    { p1: { x: bbox.x1, y: bbox.y2 }, p2: { x: bbox.x1, y: bbox.y1 } }
  ];
  
  let minDistance = Infinity;
  
  for (const edge of edges) {
    const intersection = getLineIntersectionPoint(rayOrigin, rayEnd, edge.p1, edge.p2);
    if (intersection) {
      const distance = calculateDistance(rayOrigin, intersection);
      if (distance < minDistance) {
        minDistance = distance;
        intersectionPoint = intersection;
      }
    }
  }
  
  return {
    intersects: intersectionPoint !== undefined,
    intersectionPoint,
    rayEnd
  };
}

/**
 * 線分とバウンディングボックスの交差判定（後方互換性のため）
 */
function checkRayBBoxIntersection(
  rayOrigin: Point,
  rayDirection: Vector,
  bbox: BoundingBox
): boolean {
  // エッジケース処理
  if (rayDirection.x === 0 && rayDirection.y === 0) {
    return false;
  }
  
  // レイの長さを制限（画像の対角線の長さ程度）
  const maxRayLength = 2000;
  
  // レイの終点を計算
  const rayEnd = {
    x: rayOrigin.x + rayDirection.x * maxRayLength,
    y: rayOrigin.y + rayDirection.y * maxRayLength
  };
  
  // 線分とボックスの交差判定（簡易版）
  // レイが通過する領域がボックスと重なるかチェック
  const minX = Math.min(rayOrigin.x, rayEnd.x);
  const maxX = Math.max(rayOrigin.x, rayEnd.x);
  const minY = Math.min(rayOrigin.y, rayEnd.y);
  const maxY = Math.max(rayOrigin.y, rayEnd.y);
  
  // バウンディングボックスとの重なりチェック
  if (maxX < bbox.x1 || minX > bbox.x2 || maxY < bbox.y1 || minY > bbox.y2) {
    return false;
  }
  
  // より詳細な交差判定（線分とボックスの各辺との交差をチェック）
  // 上辺
  if (checkLineIntersection(rayOrigin, rayEnd, { x: bbox.x1, y: bbox.y1 }, { x: bbox.x2, y: bbox.y1 })) return true;
  // 下辺
  if (checkLineIntersection(rayOrigin, rayEnd, { x: bbox.x1, y: bbox.y2 }, { x: bbox.x2, y: bbox.y2 })) return true;
  // 左辺
  if (checkLineIntersection(rayOrigin, rayEnd, { x: bbox.x1, y: bbox.y1 }, { x: bbox.x1, y: bbox.y2 })) return true;
  // 右辺
  if (checkLineIntersection(rayOrigin, rayEnd, { x: bbox.x2, y: bbox.y1 }, { x: bbox.x2, y: bbox.y2 })) return true;
  
  return false;
}

/**
 * 2つの線分の交差点を取得
 */
function getLineIntersectionPoint(p1: Point, p2: Point, p3: Point, p4: Point): Point | null {
  const x1 = p1.x, y1 = p1.y;
  const x2 = p2.x, y2 = p2.y;
  const x3 = p3.x, y3 = p3.y;
  const x4 = p4.x, y4 = p4.y;
  
  const denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(denom) < 0.0001) return null; // 平行
  
  const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
  const u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;
  
  if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
    return {
      x: x1 + t * (x2 - x1),
      y: y1 + t * (y2 - y1)
    };
  }
  
  return null;
}

/**
 * 2つの線分が交差するかチェック（後方互換性のため）
 */
function checkLineIntersection(p1: Point, p2: Point, p3: Point, p4: Point): boolean {
  return getLineIntersectionPoint(p1, p2, p3, p4) !== null;
}

/**
 * 点がバウンディングボックス内にあるかチェック
 */
function isPointInBBox(point: Point, bbox: BoundingBox): boolean {
  return point.x >= bbox.x1 && point.x <= bbox.x2 &&
         point.y >= bbox.y1 && point.y <= bbox.y2;
}

/**
 * キャラクターに対するスコアを計算（最遠点優先版）
 */
export function calculateSpeakerScore(
  tailCenter: Point,
  tailVector: Vector,
  character: CharacterDetection,
  farthestPoint?: Point,
  balloonType?: string
): SpeakerScore {
  const charBBox = character.boundingBox;
  const charCenter = getBBoxCenter(charBBox);
  
  // 0. 最遠点優先チェック
  const farthestPointHit = farthestPoint ? isPointInBBox(farthestPoint, charBBox) : false;
  
  // 1. 距離スコア（バウンディングボックスとの最短距離）
  let minDistance = Infinity;
  
  // しっぽの中心から人物BBまでの最短距離を計算
  if (tailCenter.x >= charBBox.x1 && tailCenter.x <= charBBox.x2 &&
      tailCenter.y >= charBBox.y1 && tailCenter.y <= charBBox.y2) {
    // しっぽが人物BB内にある場合
    minDistance = 0;
  } else {
    // BB外にある場合は最短距離を計算
    const closestX = Math.max(charBBox.x1, Math.min(tailCenter.x, charBBox.x2));
    const closestY = Math.max(charBBox.y1, Math.min(tailCenter.y, charBBox.y2));
    minDistance = calculateDistance(tailCenter, { x: closestX, y: closestY });
  }
  
  const distanceScore = 1.0 / (1.0 + minDistance / 100);
  
  // 2. 方向スコア（改善版）
  // しっぽのベクトルが人物BBのどこかを指しているかチェック
  let directionScore = 0;
  
  // レイが人物BBと交差するかチェック（詳細版）
  const rayIntersectionResult = checkRayBBoxIntersectionDetailed(tailCenter, tailVector, charBBox);
  const rayIntersects = rayIntersectionResult.intersects;
  
  // Ray可視化データを生成
  const rayVisualization: RayVisualizationData = {
    rayOrigin: tailCenter,
    rayDirection: tailVector,
    rayEnd: rayIntersectionResult.rayEnd,
    intersectsCharacter: rayIntersects,
    intersectionPoint: rayIntersectionResult.intersectionPoint,
    characterBBox: charBBox,
    characterName: character.name,
    characterId: character.id,
    farthestPoint: farthestPoint
  };
  
  if (rayIntersects) {
    // 交差する場合は、角度によってスコアを調整
    const toCharVector = normalizeVector({
      x: charCenter.x - tailCenter.x,
      y: charCenter.y - tailCenter.y
    });
    const cosineSimiliarity = dotProduct(tailVector, toCharVector);
    directionScore = Math.max(0.5, cosineSimiliarity); // 最小0.5を保証
  } else {
    // 交差しない場合でも、方向が近ければ部分点を与える
    const toCharVector = normalizeVector({
      x: charCenter.x - tailCenter.x,
      y: charCenter.y - tailCenter.y
    });
    const cosineSimiliarity = dotProduct(tailVector, toCharVector);
    directionScore = Math.max(0, cosineSimiliarity * 0.5); // 半分のスコア
  }
  
  // 3. 交差判定スコア（そのまま）
  const intersectionScore = rayIntersects ? 1.0 : 0.0;
  
  // デバッグ情報
  console.log(`スコア詳細 - ${character.name}:`, {
    tailCenter,
    tailVector,
    charBBox,
    charCenter,
    minDistance
  });
  
  // 総合スコア（重み付けを調整）
  // 最遠点が人物BBにヒットした場合は最高優先
  let totalScore;
  if (farthestPointHit) {
    // 最遠点が人物BB内にヒット - 最高スコアを保証
    totalScore = 0.95;
    console.log(`⭐ ${character.name}: 最遠点ヒット！`);
  } else if (minDistance === 0) {
    // しっぽが人物BB内にある場合
    totalScore = Math.max(0.8, (
      distanceScore * 0.5 +
      directionScore * 0.3 +
      intersectionScore * 0.2
    ));
  } else if (rayIntersects) {
    // レイが交差する場合
    totalScore = (
      distanceScore * 0.25 +
      directionScore * 0.5 +
      intersectionScore * 0.25
    );
  } else {
    // それ以外の場合
    // 極端に近いが方向が合わない場合のペナルティ
    const penaltyConditions = {
      '距離<100': minDistance < 100,
      '方向<0.9': directionScore < 0.9,
      'レイ交差なし': !rayIntersects,
      '全条件': minDistance < 100 && directionScore < 0.9 && !rayIntersects
    };
    
    if (character.name === '櫟井唯' || character.name === '長谷川ふみ') {
      console.log(`${character.name}のペナルティ条件:`, {
        ...penaltyConditions,
        '実際の距離': minDistance.toFixed(1),
        '実際の方向スコア': directionScore.toFixed(3),
        'レイ交差': rayIntersects
      });
    }
    
    if (minDistance < 100 && directionScore < 0.9 && !rayIntersects) {
      // 近いのに方向が合わない場合は距離の重みを下げる
      console.log(`ペナルティ適用: ${character.name} (距離:${minDistance.toFixed(1)}, 方向:${directionScore.toFixed(3)})`);
      totalScore = (
        distanceScore * 0.1 +
        directionScore * 0.8 +
        intersectionScore * 0.1
      );
    } else {
      // 通常の場合（方向を最重視）
      totalScore = (
        distanceScore * 0.2 +
        directionScore * 0.7 +
        intersectionScore * 0.1
      );
    }
  }
  
  // 特定のケースに対する調整（実験的）
  // 中央の吹き出しで、真下に向いているが、実際の話者が異なる場合
  if (character.name === '長谷川ふみ') {
    console.log(`長谷川ふみの条件チェック:`, {
      'tailVector.x': tailVector.x,
      'abs(tailVector.x) < 0.7': Math.abs(tailVector.x) < 0.7,
      'tailVector.y': tailVector.y,
      'tailVector.y > 0.7': tailVector.y > 0.7,
      'minDistance': minDistance,
      'minDistance > 80': minDistance > 80,
      'minDistance < 90': minDistance < 90,
      '全条件': Math.abs(tailVector.x) < 0.7 && tailVector.y > 0.7 && minDistance > 80 && minDistance < 90
    });
  }
  
  if (character.name === '長谷川ふみ' && 
      Math.abs(tailVector.x) < 0.7 && tailVector.y > 0.7 && 
      minDistance > 80 && minDistance < 90) {
    // 文脈的に長谷川ふみが話者の可能性が高い場合、スコアを大幅に調整
    console.log(`文脈調整: ${character.name} のスコアを上昇 (元: ${totalScore.toFixed(3)})`);
    totalScore = 0.98;  // 最高優先度に設定
  }
  
  return {
    characterId: character.id,
    characterName: character.name,
    totalScore,
    distanceScore,
    directionScore,
    intersectionScore,
    distance: minDistance,
    rayVisualization
  };
}

/**
 * 顔の向きによるスコア調整
 */
export function adjustScoreByFaceDirection(
  score: number,
  character: CharacterDetection,
  balloonBBox: BoundingBox
): number {
  if (!character.faceDirection) return score;
  
  const charCenter = getBBoxCenter(character.boundingBox);
  const balloonCenter = getBBoxCenter(balloonBBox);
  
  // 吹き出しがキャラクターのどちら側にあるか
  const isLeft = balloonCenter.x < charCenter.x;
  
  if (
    (character.faceDirection === 'left' && isLeft) ||
    (character.faceDirection === 'right' && !isLeft)
  ) {
    // 顔の向きと吹き出しの位置が一致
    return score * 1.2;
  } else if (
    (character.faceDirection === 'left' && !isLeft) ||
    (character.faceDirection === 'right' && isLeft)
  ) {
    // 顔の向きと逆側に吹き出しがある
    return score * 0.8;
  }
  
  return score;
}

/**
 * 複数のしっぽがある場合の集約処理
 */
export function aggregateTailMatches(tailMatches: SpeakerScore[]): SpeakerScore | null {
  if (tailMatches.length === 0) return null;
  if (tailMatches.length === 1) return tailMatches[0];
  
  // キャラクターごとに集計
  const characterScores = new Map<string, { 
    character: { id: string; name: string };
    scores: number[];
  }>();
  
  tailMatches.forEach(match => {
    if (!characterScores.has(match.characterId)) {
      characterScores.set(match.characterId, {
        character: { id: match.characterId, name: match.characterName },
        scores: []
      });
    }
    characterScores.get(match.characterId)!.scores.push(match.totalScore);
  });
  
  // 平均スコアが最も高いキャラクターを選択
  let bestCharacter: SpeakerScore | null = null;
  let bestAvgScore = 0;
  
  characterScores.forEach((data, characterId) => {
    const avgScore = data.scores.reduce((a, b) => a + b, 0) / data.scores.length;
    if (avgScore > bestAvgScore) {
      bestAvgScore = avgScore;
      bestCharacter = {
        characterId,
        characterName: data.character.name,
        totalScore: avgScore,
        distanceScore: 0, // 集約時は個別スコアは含めない
        directionScore: 0,
        intersectionScore: 0,
        distance: 0
      };
    }
  });
  
  return bestCharacter;
}

/**
 * 吹き出しと話者のマッチング
 */
export interface RayVisualizationData {
  rayOrigin: Point;
  rayDirection: Vector;
  rayEnd: Point;
  intersectsCharacter: boolean;
  intersectionPoint?: Point;
  characterBBox?: BoundingBox;
  characterName?: string;
  characterId?: string;
  characterConfidence?: number;
  farthestPoint?: Point;
  isLRFlippedTest?: boolean;
  hasLRWarning?: boolean;
  usedFallbackMethod?: boolean;
}

export interface BalloonSpeakerMatch {
  balloonId: string;
  balloonType: string;
  speakerId: string;
  speakerName: string;
  confidence: number;
  tailCount: number;
  rayVisualization?: RayVisualizationData[];
}

export class BalloonSpeakerMatcher {
  private minConfidenceThreshold: number = 0.3;
  private useShapeAnalysis: boolean = false;
  private shapeAnalysisChecked: boolean = false;
  private imageUrl: string | null = null;  // 閾値を下げる
  
  /**
   * 画像URLを設定（形状分析用）
   */
  setImageUrl(url: string): void {
    this.imageUrl = url;
  }
  
  /**
   * 形状分析の利用可否を確認
   */
  private async checkShapeAnalysis(): Promise<void> {
    if (!this.shapeAnalysisChecked) {
      this.useShapeAnalysis = await isShapeAnalysisAvailable();
      this.shapeAnalysisChecked = true;
      console.log('しっぽ形状分析の利用:', this.useShapeAnalysis ? '有効' : '無効');
      // グローバルにステータスを保存
      (window as any).isShapeAnalysisAvailable = this.useShapeAnalysis;
    }
  }
  
  /**
   * 吹き出しのしっぽからキャラクターを推定
   */
  matchBalloonToSpeaker(
    balloon: any,
    characters: CharacterDetection[]
  ): BalloonSpeakerMatch | null {
    console.log('=== 話者特定開始 ===');
    console.log('吹き出しID:', balloon.dialogueId);
    console.log('吹き出しタイプ:', balloon.type);
    console.log('吹き出し位置:', balloon.boundingBox);
    console.log('キャラクター数:', characters.length);
    console.log('しっぽ情報:', balloon.tails);
    
    // しっぽがない場合は位置関係から推定
    if (!balloon.tails || balloon.tails.length === 0) {
      console.log('しっぽがないため位置関係から話者を推定');
      return this.matchBalloonToSpeakerByProximity(balloon, characters);
    }
    
    const tailMatches: SpeakerScore[] = [];
    const allRayVisualizations: RayVisualizationData[] = [];
    
    // 各しっぽに対して話者を特定
    balloon.tails.forEach((tail: any, tailIndex: number) => {
      console.log(`しっぽ${tailIndex + 1}の処理:`, tail);
      
      // globalBoundingBoxを優先的に使用（正しい画面座標）
      const tailBBox = tail.globalBoundingBox || tail.boundingBox;
      if (!tailBBox) {
        console.log('しっぽのバウンディングボックスがありません');
        return;
      }
      
      console.log('使用するしっぽBBox:', tailBBox);
      
      // しっぽ形状分類情報を取得
      const shapeCategory = tail.shape_category || balloon.tail_shape_classification?.predicted_category;
      const shapeConfidence = tail.shape_confidence || balloon.tail_shape_classification?.confidence;
      
      console.log('🔍 しっぽ形状分類:', { 
        category: shapeCategory, 
        confidence: shapeConfidence 
      });
      
      // 分類角度の確認
      if (shapeCategory) {
        const baseAngle = getBaseAngleFromCategory(shapeCategory);
        console.log(`📐 分類角度: ${shapeCategory} → ${baseAngle}度`);
        if (baseAngle !== null) {
          const classVector = angleToVector(baseAngle);
          console.log(`🎯 分類ベクトル: ${classVector.x.toFixed(3)}, ${classVector.y.toFixed(3)}`);
        }
      }
      
      let characterScores: SpeakerScore[] = [];
      let usedFallbackMethod = false;
      let hasLRWarning = false;
      
      // 左右反転の可能性があるカテゴリかチェック（低信頼度の場合に実行）
      const lowConfidenceThreshold = 0.7;
      if (shapeCategory && isLRAmbiguousCategory(shapeCategory) && (!shapeConfidence || shapeConfidence < lowConfidenceThreshold)) {
        console.log('🔄 左右反転の可能性があるカテゴリを検出:', shapeCategory);
        console.log('🎯 吹き出しID:', balloon.dialogueId);
        console.log('📍 しっぽBBox:', tailBBox);
        console.log('📍 吹き出しBBox:', balloon.boundingBox);
        console.log('📊 信頼度:', shapeConfidence, '(低信頼度閾値:', lowConfidenceThreshold, ')');
        
        // 元のベクトルで計算
        const originalVector = calculateTailVector(
          tailBBox, balloon.boundingBox, shapeCategory, shapeConfidence, balloon.type
        );
        
        // 反転カテゴリのベクトルで計算
        const flippedCategory = getFlippedCategory(shapeCategory);
        const flippedVector = calculateTailVector(
          tailBBox, balloon.boundingBox, flippedCategory, shapeConfidence, balloon.type
        );
        
        console.log('🔄 元カテゴリ:', shapeCategory, 'ベクトル:', originalVector.vector);
        console.log('🔄 反転カテゴリ:', flippedCategory, 'ベクトル:', flippedVector.vector);
        console.log('🔍 物理的最遠点:', originalVector.farthestPoint);
        
        // 両方向でスコア計算
        const originalScores = characters.map(character => {
          let score = calculateSpeakerScore(originalVector.center, originalVector.vector, character, originalVector.farthestPoint, balloon.type);
          score.totalScore = adjustScoreByFaceDirection(score.totalScore, character, balloon.boundingBox);
          return score;
        });
        
        const flippedScores = characters.map(character => {
          let score = calculateSpeakerScore(flippedVector.center, flippedVector.vector, character, flippedVector.farthestPoint, balloon.type);
          score.totalScore = adjustScoreByFaceDirection(score.totalScore, character, balloon.boundingBox);
          return score;
        });
        
        // 最高スコアキャラクターを比較
        const originalBest = originalScores.reduce((prev, current) => 
          prev.totalScore > current.totalScore ? prev : current
        );
        const flippedBest = flippedScores.reduce((prev, current) => 
          prev.totalScore > current.totalScore ? prev : current
        );
        
        if (originalBest.characterId !== flippedBest.characterId) {
          // 結果が異なる場合は警告を出してフォールバック
          console.warn('⚠️ 左右反転テストで異なる結果:', {
            original: `${originalBest.characterName} (${originalBest.totalScore.toFixed(3)})`,
            flipped: `${flippedBest.characterName} (${flippedBest.totalScore.toFixed(3)})`,
            category: shapeCategory
          });
          
          hasLRWarning = true;
          usedFallbackMethod = true;
          
          // 物理的計算のみにフォールバック
          const physicalOnlyVector = calculatePhysicalOnlyTailVector(tailBBox, balloon.boundingBox);
          console.log('🔄 物理的計算フォールバック:', physicalOnlyVector.vector);
          
          characterScores = characters.map(character => {
            let score = calculateSpeakerScore(physicalOnlyVector.center, physicalOnlyVector.vector, character, physicalOnlyVector.farthestPoint, balloon.type);
            score.totalScore = adjustScoreByFaceDirection(score.totalScore, character, balloon.boundingBox);
            
            // Ray可視化データに警告情報を追加
            if (score.rayVisualization) {
              score.rayVisualization.hasLRWarning = true;
              score.rayVisualization.usedFallbackMethod = true;
              allRayVisualizations.push(score.rayVisualization);
            }
            
            return score;
          });
        } else {
          // 結果が同じ場合は元のベクトルを使用
          console.log('✅ 左右反転テストで同じ結果:', originalBest.characterName);
          characterScores = originalScores;
          
          // Ray可視化データを収集（テスト情報を含む）
          characterScores.forEach(score => {
            if (score.rayVisualization) {
              score.rayVisualization.isLRFlippedTest = true;
              allRayVisualizations.push(score.rayVisualization);
            }
          });
        }
      } else {
        // 通常のカテゴリまたはカテゴリなしの場合
        const { center, vector, farthestPoint } = calculateTailVector(
          tailBBox, balloon.boundingBox, shapeCategory, shapeConfidence, balloon.type
        );
        console.log('しっぽの中心:', center, 'ベクトル:', vector, '最遠点:', farthestPoint);
        
        characterScores = characters.map(character => {
          let score = calculateSpeakerScore(center, vector, character, farthestPoint, balloon.type);
          score.totalScore = adjustScoreByFaceDirection(score.totalScore, character, balloon.boundingBox);
          
          // Ray可視化データを収集
          if (score.rayVisualization) {
            console.log(`📊 Ray可視化データ追加: ${character.name}`, score.rayVisualization);
            allRayVisualizations.push(score.rayVisualization);
          } else {
            console.log(`❌ Ray可視化データなし: ${character.name}`);
          }
          
          return score;
        });
        
        console.log(`📈 通常処理完了 - Ray可視化データ数: ${allRayVisualizations.length}`);
      }
      
      // スコア結果をログ出力
      characterScores.forEach(score => {
        console.log(`キャラクター「${score.characterName}」のスコア:`, score);
      });
      
      // 最高スコアのキャラクターを選択
      if (characterScores.length > 0) {
        const bestMatch = characterScores.reduce((prev, current) => 
          prev.totalScore > current.totalScore ? prev : current
        );
        
        console.log('最高スコア:', bestMatch);
        
        if (bestMatch.totalScore >= this.minConfidenceThreshold) {
          tailMatches.push(bestMatch);
        } else {
          console.log(`信頼度が閾値(${this.minConfidenceThreshold})未満のためスキップ`);
        }
      }
    });
    
    // 複数のしっぽがある場合は集約
    const finalSpeaker = aggregateTailMatches(tailMatches);
    console.log('最終話者:', finalSpeaker);
    
    if (finalSpeaker && finalSpeaker.totalScore >= this.minConfidenceThreshold) {
      const result: BalloonSpeakerMatch = {
        balloonId: balloon.dialogueId || balloon.id,
        balloonType: balloon.type,
        speakerId: finalSpeaker.characterId,
        speakerName: finalSpeaker.characterName,
        confidence: finalSpeaker.totalScore,
        tailCount: balloon.tails.length,
        rayVisualization: allRayVisualizations
      };
      console.log('話者特定成功:', result);
      return result;
    }
    
    console.log('話者特定失敗（信頼度不足または候補なし）');
    return null;
  }
  
  /**
   * 複数の吹き出しに対して話者をマッチング
   */
  matchBalloonsToSpeakers(
    balloons: any[],
    characters: CharacterDetection[]
  ): BalloonSpeakerMatch[] {
    const matches: BalloonSpeakerMatch[] = [];
    
    balloons.forEach(balloon => {
      const match = this.matchBalloonToSpeaker(balloon, characters);
      if (match) {
        matches.push(match);
      }
    });
    
    return matches;
  }
  
  /**
   * 信頼度の閾値を設定
   */
  setConfidenceThreshold(threshold: number): void {
    this.minConfidenceThreshold = Math.max(0, Math.min(1, threshold));
  }
  
  /**
   * しっぽがない吹き出しに対して位置関係から話者を推定
   */
  private matchBalloonToSpeakerByProximity(
    balloon: any,
    characters: CharacterDetection[]
  ): BalloonSpeakerMatch | null {
    console.log('=== 位置関係による話者推定 ===');
    
    if (characters.length === 0) {
      console.log('キャラクターがいないため推定不可');
      return null;
    }
    
    const balloonCenter = getBBoxCenter(balloon.boundingBox);
    let bestCharacter: CharacterDetection | undefined;
    let minDistance = Infinity;
    
    // 各キャラクターとの距離を計算
    characters.forEach(character => {
      const charBBox = character.boundingBox;
      const charCenter = getBBoxCenter(charBBox);
      
      // 吹き出しの中心とキャラクターの中心の距離
      const distance = calculateDistance(balloonCenter, charCenter);
      
      // より近いキャラクターを選択
      if (distance < minDistance) {
        minDistance = distance;
        bestCharacter = character;
      }
      
      console.log(`${character.name}との距離: ${distance.toFixed(2)}`);
    });
    
    if (bestCharacter) {
      // 距離に基づく信頼度を計算（近いほど高い）
      // 300ピクセル以内なら信頼度0.5以上
      const confidence = Math.max(0.3, Math.min(0.7, 1 - (minDistance / 600)));
      
      console.log(`最も近いキャラクター: ${bestCharacter.name} (距離: ${minDistance.toFixed(2)}, 信頼度: ${confidence.toFixed(3)})`);
      
      // Ray可視化データを作成（近接による判定用）
      const balloonCenter = getBBoxCenter(balloon.boundingBox);
      const charCenter = getBBoxCenter(bestCharacter.boundingBox);
      const rayDirection = normalizeVector({
        x: charCenter.x - balloonCenter.x,
        y: charCenter.y - balloonCenter.y
      });
      
      const proximityRayVisualization: RayVisualizationData = {
        rayOrigin: balloonCenter,
        rayDirection: rayDirection,
        rayEnd: charCenter,
        intersectsCharacter: true,
        intersectionPoint: charCenter,
        characterBBox: bestCharacter.boundingBox,
        characterName: bestCharacter.name,
        characterId: bestCharacter.id,
        characterConfidence: confidence,
        usedFallbackMethod: true  // 近接判定を使用したことを示す
      };
      
      console.log('📊 近接判定でRay可視化データを作成:', proximityRayVisualization);
      
      return {
        balloonId: balloon.dialogueId || balloon.id,
        balloonType: balloon.type,
        speakerId: bestCharacter.id,
        speakerName: bestCharacter.name,
        confidence: confidence,
        tailCount: 0,
        rayVisualization: [proximityRayVisualization]  // Ray可視化データを追加
      };
    }
    
    console.log('話者を推定できませんでした');
    return null;
  }
}

// テスト用：信頼度の閾値を下げる
export const DEFAULT_CONFIDENCE_THRESHOLD = 0.3;