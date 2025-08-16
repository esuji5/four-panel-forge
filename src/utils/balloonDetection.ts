/**
 * 吹き出し検出の統合ユーティリティ
 */

import { Serif } from '../types/app';
import { processAndReorderBalloons } from './balloonGrouping';

/**
 * 旧形式の吹き出しタイプを新形式に変換
 */
export function convertOldBalloonType(oldType: string): string {
  const typeMap: { [key: string]: string } = {
    'speechBubble': 'speech_bubble',
    'outsideBubble': 'offserif_bubble',
    'narration': 'narration_box',
  };
  return typeMap[oldType] || oldType;
}

export interface BalloonDetection {
  dialogueId: string;
  type: string;
  boundingBox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    width: number;
    height: number;
  };
  coordinate: [number, number];
  confidence: number;
  classId: number;
  readingOrderIndex: number;
  speakerCharacterId?: string | null;
  tails?: Array<{
    globalPosition?: [number, number];
    shape_category?: string;
    shape_confidence?: number;
  }>;
}

/**
 * 吹き出し検出結果をSerifデータに変換
 * @param detection 検出結果
 * @param panelPrefix パネル識別用のプレフィックス（例: "p1_", "p2_"）
 */
export function convertBalloonDetectionToSerif(detection: BalloonDetection, panelPrefix?: string): Serif {
  // 話者の設定
  let speakerId = detection.speakerCharacterId || null;
  console.log(`🎯 convertBalloonDetectionToSerif: dialogueId=${detection.dialogueId}, speakerCharacterId=${detection.speakerCharacterId}, type=${detection.type}`);
  
  // キャラクター専用吹き出しから話者を推定（既存の話者設定を上書き）
  if (detection.type.startsWith('chractor_bubble_')) {
    const character = detection.type.replace('chractor_bubble_', '');
    const characterMap: { [key: string]: string } = {
      'yuzuko': 'char_A',
      'yukari': 'char_B',
      'yui': 'char_C',
      'yoriko': 'char_D',
      'chiho': 'char_E',
      'kei': 'char_F',
      'fumi': 'char_G',
    };
    speakerId = characterMap[character] || null;
    console.log(`🎯 キャラクター専用吹き出し検出: ${detection.type} -> ${character} -> ${speakerId}`);
  }
  
  // dialogueIdをそのまま使用（processAndReorderBalloonsで既にプレフィックスが付与されている）
  const dialogueId = detection.dialogueId;
  
  return {
    dialogueId: dialogueId,
    text: '', // テキストは空で初期化
    type: detection.type,
    speakerCharacterId: speakerId,
    boundingBox: detection.boundingBox,
    readingOrderIndex: detection.readingOrderIndex,
    coordinate: detection.coordinate,
  };
}

/**
 * 既存のセリフデータと吹き出し検出結果をマージ
 * @param existingSerifs 既存のセリフデータ
 * @param balloonDetections 吹き出し検出結果
 * @param panelPrefix パネル識別用のプレフィックス（例: "p1_", "p2_"）
 */
export function mergeSerifsWithBalloonDetections(
  existingSerifs: Serif[],
  balloonDetections: BalloonDetection[],
  panelPrefix?: string
): Serif[] {
  console.log('=== mergeSerifsWithBalloonDetections開始 ===');
  console.log('既存セリフ数:', existingSerifs.length);
  console.log('検出された吹き出し数:', balloonDetections.length);
  console.log('🎯 入力吹き出しの話者情報:');
  balloonDetections.forEach((balloon, index) => {
    console.log(`  ${index}: ${balloon.dialogueId} -> speaker: ${balloon.speakerCharacterId || 'なし'}, type: ${balloon.type}`);
  });
  
  // 吹き出しをグループ化して右から左へ並び替え
  const processedDetections = processAndReorderBalloons(balloonDetections, panelPrefix || '');
  console.log('グループ化・並び替え後の吹き出し数:', processedDetections.length);
  
  // デバッグ: 既存セリフのIDを確認
  if (existingSerifs.length > 0) {
    console.log('既存セリフのID一覧:', existingSerifs.map(s => s.dialogueId));
  }
  if (processedDetections.length > 0) {
    console.log('処理後の吹き出しのID一覧:', processedDetections.map(b => b.dialogueId));
  }
  
  // 既存セリフがない場合は、処理済み検出結果をそのまま返す
  if (!existingSerifs || existingSerifs.length === 0) {
    console.log('既存セリフがないため、処理済み検出結果をそのまま使用');
    const newSerifs = processedDetections.map(detection => {
      const serif = convertBalloonDetectionToSerif(detection);
      console.log(`📝 新規セリフ作成: ${serif.dialogueId}, speaker: ${serif.speakerCharacterId || 'なし'}, detectionSpeaker: ${detection.speakerCharacterId || 'なし'}`);
      return serif;
    });
    // すでにソート済みなので、再ソートは不要
    return newSerifs;
  }
  
  // 既存セリフが古い形式の場合の判定を改善
  // p1_d001のような古いIDを持つセリフがある場合は、検出結果で置き換える
  const hasOldFormatIds = existingSerifs.some(serif => 
    serif.dialogueId && (serif.dialogueId.startsWith('p1_d') || serif.dialogueId.startsWith('p2_d') || 
    serif.dialogueId.startsWith('p3_d') || serif.dialogueId.startsWith('p4_d'))
  );
  
  if (hasOldFormatIds) {
    console.log('既存セリフが古い形式のため、処理済み検出結果で置き換え');
    const newSerifs = processedDetections.map(detection => convertBalloonDetectionToSerif(detection));
    
    // 既存のテキストを保持する（古いセリフから新しいセリフへ）
    existingSerifs.forEach((oldSerif, index) => {
      if (index < newSerifs.length && oldSerif.text) {
        newSerifs[index].text = oldSerif.text;
        // 既存の話者も保持（もしあれば）
        if (oldSerif.speakerCharacterId) {
          newSerifs[index].speakerCharacterId = oldSerif.speakerCharacterId;
        }
      }
    });
    
    // すでにソート済みなので、再ソートは不要
    return newSerifs;
  }
  
  // 既存のセリフをdialogueIdでマップ化
  const existingSerifMap = new Map<string, Serif>();
  const processedIds = new Set<string>();
  
  existingSerifs.forEach(serif => {
    existingSerifMap.set(serif.dialogueId, serif);
  });

  // 処理済み検出結果をマップ化
  const detectionMap = new Map<string, BalloonDetection>();
  processedDetections.forEach(detection => {
    detectionMap.set(detection.dialogueId, detection);
  });
  
  console.log('既存セリフのID:', Array.from(existingSerifMap.keys()));
  console.log('検出された吹き出しのID:', Array.from(detectionMap.keys()));

  // 1. 既存のセリフを更新（検出結果がある場合のみ）
  const updatedExistingSerifs = existingSerifs.map(existingSerif => {
    const detection = detectionMap.get(existingSerif.dialogueId);
    processedIds.add(existingSerif.dialogueId);
    
    if (detection) {
      console.log(`既存セリフ更新: ${existingSerif.dialogueId}`);
      // 検出結果がある場合は更新
      const updatedSerif: Serif = {
        ...existingSerif,
        // タイプと座標は常に更新
        type: detection.type,
        coordinate: detection.coordinate,
        readingOrderIndex: detection.readingOrderIndex,
        // boundingBoxも更新
        boundingBox: detection.boundingBox,
      };
      
      // 話者の設定（キャラクター専用吹き出しは最優先）
      if (detection.type.startsWith('chractor_bubble_')) {
        // キャラクター専用吹き出しの場合は、既存の話者設定を上書き
        const character = detection.type.replace('chractor_bubble_', '');
        const characterMap: { [key: string]: string } = {
          'yuzuko': 'char_A',
          'yukari': 'char_B',
          'yui': 'char_C',
          'yoriko': 'char_D',
          'chiho': 'char_E',
          'kei': 'char_F',
          'fumi': 'char_G',
        };
        updatedSerif.speakerCharacterId = characterMap[character] || null;
        console.log(`🎯 キャラクター専用吹き出し強制設定: ${existingSerif.dialogueId} -> ${detection.type} -> ${updatedSerif.speakerCharacterId}`);
      } else if (detection.speakerCharacterId) {
        // 検出結果に話者が含まれている場合は、それを使用
        console.log(`✅ 話者情報更新: ${existingSerif.dialogueId} -> ${detection.speakerCharacterId}`);
        updatedSerif.speakerCharacterId = detection.speakerCharacterId;
      } else {
        // その他の場合は既存の話者を保持
        updatedSerif.speakerCharacterId = existingSerif.speakerCharacterId;
      }
      
      return updatedSerif;
    } else {
      console.log(`既存セリフ保持（検出されず）: ${existingSerif.dialogueId}`);
      // 検出されなかった既存セリフもそのまま保持（ただし旧形式は変換）
      return {
        ...existingSerif,
        type: convertOldBalloonType(existingSerif.type),
      };
    }
  });

  // 2. 新規検出された吹き出しを追加
  const newSerifs = processedDetections
    .filter(detection => !processedIds.has(detection.dialogueId))
    .map(detection => {
      console.log(`📝 新規セリフ追加: ${detection.dialogueId}, speaker: ${detection.speakerCharacterId || 'なし'}`);
      const serif = convertBalloonDetectionToSerif(detection);
      console.log(`📝 変換後のセリフ: ${serif.dialogueId}, speaker: ${serif.speakerCharacterId || 'なし'}`);
      return serif;
    });

  // 3. 既存と新規を結合
  const mergedSerifs = [...updatedExistingSerifs, ...newSerifs];
  
  console.log('更新されたセリフ数:', updatedExistingSerifs.length);
  console.log('新規セリフ数:', newSerifs.length);
  console.log('マージ後の合計:', mergedSerifs.length);

  // readingOrderIndexでソート（右から左の順序が保たれる）
  mergedSerifs.sort((a, b) => a.readingOrderIndex - b.readingOrderIndex);

  // 最終結果の話者情報を確認
  console.log('🎯 最終結果の話者情報:');
  mergedSerifs.forEach((serif, index) => {
    console.log(`  ${index}: ${serif.dialogueId} -> speaker: ${serif.speakerCharacterId || 'なし'}`);
  });

  return mergedSerifs;
}

/**
 * 吹き出しタイプの日本語表示名を取得
 */
export function getBalloonTypeDisplayName(type: string): string {
  const typeMap: { [key: string]: string } = {
    'speech_bubble': '吹き出し',
    'thought_bubble': '思考',
    'exclamation_bubble': '感嘆',
    'combined_bubble': '結合',
    'offserif_bubble': 'オフセリフ',
    'inner_voice_bubble': '内なる声',
    'narration_box': 'ナレーション',
    'chractor_bubble_yuzuko': 'ゆずこ専用',
    'chractor_bubble_yukari': 'ゆかり専用',
    'chractor_bubble_yui': '唯専用',
    'chractor_bubble_yoriko': 'よりこ専用',
    'chractor_bubble_chiho': '千穂専用',
    'chractor_bubble_kei': '恵専用',
    'chractor_bubble_fumi': '史専用',
  };

  return typeMap[type] || type;
}

/**
 * 吹き出しタイプから話者を推定
 */
export function inferSpeakerFromBalloonType(type: string): string | null {
  if (type.startsWith('chractor_bubble_')) {
    const character = type.replace('chractor_bubble_', '');
    const characterMap: { [key: string]: string } = {
      'yuzuko': '野々原ゆずこ',
      'yukari': '日向縁',
      'yui': '櫟井唯',
      'yoriko': '松本頼子',
      'chiho': '相川千穂',
      'kei': '岡野佳',
      'fumi': '長谷川ふみ',
    };
    return characterMap[character] || character;
  }
  return null;
}