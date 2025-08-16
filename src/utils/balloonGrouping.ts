/**
 * 吹き出しのグループ化とソートのユーティリティ
 */

import { BalloonDetection } from './balloonDetection';

/**
 * 2つのバウンディングボックスの重なり率を計算
 */
function calculateOverlapRatio(box1: any, box2: any): number {
  const x1 = Math.max(box1.x1, box2.x1);
  const y1 = Math.max(box1.y1, box2.y1);
  const x2 = Math.min(box1.x2, box2.x2);
  const y2 = Math.min(box1.y2, box2.y2);
  
  // 重なりがない場合
  if (x1 >= x2 || y1 >= y2) {
    return 0;
  }
  
  // 重なり面積
  const overlapArea = (x2 - x1) * (y2 - y1);
  
  // 各ボックスの面積
  const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
  
  // 小さい方のボックスに対する重なり率
  const minArea = Math.min(area1, area2);
  return overlapArea / minArea;
}

/**
 * 座標が重なる吹き出しをグループ化
 * @param balloons 吹き出し検出結果
 * @param overlapThreshold 重なり率の閾値（デフォルト: 0.8 = 80%）
 */
export function groupOverlappingBalloons(
  balloons: BalloonDetection[],
  overlapThreshold: number = 0.8
): BalloonDetection[][] {
  const groups: BalloonDetection[][] = [];
  const processed = new Set<number>();
  
  balloons.forEach((balloon, index) => {
    if (processed.has(index)) return;
    
    const group: BalloonDetection[] = [balloon];
    processed.add(index);
    
    // 他の吹き出しとの重なりをチェック
    balloons.forEach((other, otherIndex) => {
      if (index === otherIndex || processed.has(otherIndex)) return;
      
      const overlapRatio = calculateOverlapRatio(
        balloon.boundingBox,
        other.boundingBox
      );
      
      if (overlapRatio >= overlapThreshold) {
        group.push(other);
        processed.add(otherIndex);
      }
    });
    
    groups.push(group);
  });
  
  return groups;
}

/**
 * グループ内で最も適切な吹き出しを選択
 * 優先順位: 1. confidence が高い, 2. 面積が大きい
 */
export function selectBestFromGroup(group: BalloonDetection[]): BalloonDetection {
  return group.reduce((best, current) => {
    // confidence で比較
    if (current.confidence > best.confidence) {
      return current;
    } else if (current.confidence === best.confidence) {
      // 面積で比較
      const currentArea = (current.boundingBox.x2 - current.boundingBox.x1) * 
                         (current.boundingBox.y2 - current.boundingBox.y1);
      const bestArea = (best.boundingBox.x2 - best.boundingBox.x1) * 
                      (best.boundingBox.y2 - best.boundingBox.y1);
      return currentArea > bestArea ? current : best;
    }
    return best;
  });
}

/**
 * 吹き出しを右から左へソート（右側が若い番号）
 */
export function sortBalloonsRightToLeft(balloons: BalloonDetection[]): BalloonDetection[] {
  return [...balloons].sort((a, b) => {
    // x座標の中心で比較（大きい方が先 = 右側が先）
    const centerA = (a.boundingBox.x1 + a.boundingBox.x2) / 2;
    const centerB = (b.boundingBox.x1 + b.boundingBox.x2) / 2;
    
    // 同じx座標の場合は、y座標で比較（上が先）
    if (Math.abs(centerA - centerB) < 10) {
      const centerYA = (a.boundingBox.y1 + a.boundingBox.y2) / 2;
      const centerYB = (b.boundingBox.y1 + b.boundingBox.y2) / 2;
      return centerYA - centerYB;
    }
    
    return centerB - centerA; // 右側（大きいx）が先
  });
}

/**
 * 吹き出しをグループ化し、右から左へソートして新しいIDを割り当て
 */
export function processAndReorderBalloons(
  balloons: BalloonDetection[],
  panelPrefix: string = ''
): BalloonDetection[] {
  console.log('=== 吹き出しのグループ化と並び替え開始 ===');
  console.log(`入力吹き出し数: ${balloons.length}`);
  
  // 1. 重なる吹き出しをグループ化
  const groups = groupOverlappingBalloons(balloons);
  console.log(`グループ数: ${groups.length}`);
  
  // 2. 各グループから最適な吹き出しを選択
  const selectedBalloons = groups.map(group => {
    const best = selectBestFromGroup(group);
    console.log(`グループ(${group.length}個) -> 選択: ${best.type} (confidence: ${best.confidence}, speaker: ${best.speakerCharacterId || 'なし'})`);
    return best;
  });
  
  // 3. 右から左へソート
  const sortedBalloons = sortBalloonsRightToLeft(selectedBalloons);
  
  // 4. 新しいIDを割り当て（readingOrderIndexも更新）
  const reorderedBalloons = sortedBalloons.map((balloon, index) => ({
    ...balloon,
    dialogueId: `${panelPrefix}balloon_${index + 1}`,
    readingOrderIndex: index + 1,
    // 話者情報を保持
    speakerCharacterId: balloon.speakerCharacterId
  }));
  
  console.log(`最終吹き出し数: ${reorderedBalloons.length}`);
  console.log('並び順（右から）:', reorderedBalloons.map(b => `${b.dialogueId} (speaker: ${b.speakerCharacterId || 'なし'})`));
  
  return reorderedBalloons;
}