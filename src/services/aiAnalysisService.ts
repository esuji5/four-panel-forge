/**
 * AI分析処理のサービス層
 */
import axios from 'axios';
import { ImageData, ImagePathList, Character, CSVRow } from '../types/app';
import { 
  BalloonDetection, 
  mergeSerifsWithBalloonDetections 
} from '../utils/balloonDetection';
import { processAndReorderBalloons } from '../utils/balloonGrouping';
import { 
  BalloonSpeakerMatcher, 
  CharacterDetection 
} from '../utils/speakerIdentification';
import { getImagePath } from '../config';
import { 
  combineFourPanelImages,
  loadGroundTruthImage
} from '../utils/imageProcessing';
import { getCharacterIdMap, getEnglishToCharIdMap } from '../utils/dataHelpers';

interface PanelDetectionResult {
  imageKey: string;
  characters: any[];
  balloons: any[];
}

export class AIAnalysisService {
  // デバッグ用保存処理
  static async saveDetectionDebugInfo(
    timestamp: string,
    imagePathList: ImagePathList,
    detectionResults: PanelDetectionResult[],
    enhancedPrompt: string,
    step: string
  ): Promise<void> {
    try {
      const debugDir = `tmp/ai_detection_debug/detection_${timestamp}`;
      
      // バックエンドに保存リクエストを送信
      await axios.post("http://localhost:8000/api/save-detection-debug", {
        debugDir,
        step,
        imagePathList,
        detectionResults,
        enhancedPrompt,
        timestamp,
        promptLength: enhancedPrompt.length
      });
      
      console.log(`💾 AI検出デバッグ情報保存: ${debugDir}/${step}.json`);
    } catch (error) {
      console.warn("デバッグ情報保存エラー:", error);
    }
  }

  // 事前AI検出処理
  static async executePreDetection(imagePathList: ImagePathList): Promise<PanelDetectionResult[]> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '_').replace('T', '_').substring(0, 19);
    console.log("🔍 executePreDetection開始", { imagePathList });
    
    const promises = [1, 2, 3, 4].map(async (num) => {
      const imageKey = `image${num}`;
      
      try {
        console.log(`🎯 パネル${num}の検出開始:`, imagePathList[imageKey as keyof ImagePathList]);
        
        // キャラクター検出
        const charResponse = await axios.post(
          "http://localhost:8000/api/detect-characters-yolo-dinov2",
          {
            komaPath: imagePathList[imageKey as keyof ImagePathList],
            mode: "single",
            detectionThreshold: 0.25,
            classificationThreshold: 0.5
          }
        );
        
        console.log(`👥 パネル${num}キャラクター検出結果:`, charResponse.data);
        
        // 吹き出し検出
        const balloonResponse = await axios.post(
          "http://localhost:8000/api/detect-balloons",
          {
            imagePath: imagePathList[imageKey as keyof ImagePathList],
            confidenceThreshold: 0.15,
            maxDet: 300
          }
        );
        
        console.log(`💭 パネル${num}吹き出し検出結果:`, balloonResponse.data);
        
        const result = {
          imageKey,
          characters: charResponse.data.characters || [],
          balloons: balloonResponse.data.detections || []
        };
        
        console.log(`✅ パネル${num}統合結果:`, result);
        return result;
      } catch (error) {
        console.warn(`❌ パネル${num}の事前検出エラー:`, error);
        return {
          imageKey,
          characters: [],
          balloons: []
        };
      }
    });
    
    const results = await Promise.all(promises);
    console.log("🏁 executePreDetection完了:", results);
    
    // 検出結果をデバッグ保存
    await this.saveDetectionDebugInfo(
      timestamp,
      imagePathList,
      results,
      "", // この時点ではプロンプトなし
      "1_detection_results"
    );
    
    return results;
  }

  // AI検出結果を含むプロンプト生成（既存形式準拠）
  static async generateEnhancedPrompt(
    detectionResults: PanelDetectionResult[], 
    timestamp: string, 
    imagePathList: ImagePathList
  ): Promise<string> {
    console.log("📝 generateEnhancedPrompt開始", { detectionResults });
    
    let prompt = `あなたは4コマ漫画『ゆゆ式』の画像を分析するAIアシスタントです。
事前のAI検出結果を活用し、指定されたJSONスキーマに沿って高精度な情報抽出を行ってください。
日本の4コマ漫画は右から左に読みます。与えられる座標は画像の左上を原点とした相対座標(x,y)です。

タスク:
提供された4コマ漫画画像とAI検出結果を基に、詳細な情報を抽出してください。

## AI検出結果の活用方法:
プロンプト末尾に以下の形式でAI検出結果が提供されます：
\`\`\`json
{
  "detected_characters": [
    {
      "characterId": "野々原ゆずこ",   // 正式名称で提供される（注意: characterIdフィールド）
      "size": "0.20, 0.25",           // 幅, 高さの数値形式で提供される（注意: sizeフィールド）
      "coordinate": [0.3, 0.5],
      "confidence": 0.92              // 検出信頼度（数値）
    }
  ],
  "detected_bubbles": [
    {
      "coordinate": [0.8, 0.3],
      "confidence": 0.75,
      "type": "speech_bubble",        // 吹き出しタイプ（speech_bubble、thought_bubble、offserif、chractor_bubble_yuzuko等）
      "size": "0.39, 0.26",           // 吹き出しのサイズ（幅, 高さ）正規化された値（0.0〜1.0）
      "speakerId": "char_A"           // キャラクター専用吹き出しの場合のみ
    }
  ]
}
\`\`\`

**重要**: AI検出結果のcharacterIdとsizeの値を活用してください。
- characterIdが「野々原ゆずこ」→ characterフィールドに「野々原ゆずこ」を使用
- sizeが「0.20, 0.25」→ characterSizeフィールドに「0.20, 0.25」を使用

## キャラクター特定の優先順位:
1. AI検出結果の detected_charactersを最優先
2. セリフ内容と役割の一致度による補正
3. 文脈と前後関係による最終判断

## 話者特定の具体的計算手順:

**重要: 以下の優先順位で話者を判定してください**

### 優先度1: キャラクター専用吹き出しタイプ（最優先）
detected_bubblesのtypeが以下の場合は、対応するキャラクターを話者とする：
- \`chractor_bubble_yuzuko\` → 野々原ゆずこ（char_A）
- \`chractor_bubble_yukari\` → 日向縁（char_B）  
- \`chractor_bubble_yui\` → 櫟井唯（char_C）
- その他のchractor_bubble_xxx → 対応するキャラクター

### 優先度2: 距離計算による判定（通常の吹き出しの場合）
1. **各吹き出しについて、全ての人物との距離を計算**
   - detected_bubblesのcoordinateとdetected_charactersのcoordinateを使用
   - 距離 = sqrt((吹き出しX - 人物X)^2 + (吹き出しY - 人物Y)^2)
   - 例: 吹き出し[0.7, 0.3]と人物[0.8, 0.4]の距離 = sqrt((0.7-0.8)^2 + (0.3-0.4)^2) = 0.141

2. **距離による話者判定ルール**
   - 距離が0.15未満: その人物が話者である可能性が非常に高い（最優先）
   - 距離が0.15-0.25: 話者の候補として考慮
   - 距離が0.25-0.35: 低確率の候補
   - 距離が0.35以上: 基本的に話者ではない（オフセリフを除く）

3. **同距離の場合の優先順位**
   - 右側の人物を優先（X座標が大きい方）
   - Y座標の差が0.1未満の場合は同じ高さとみなす

### 優先度3: セリフ内容による補正
距離が近い候補が複数いる場合のみ適用：
- 「わからないー！」「いとをかしー！」→ 野々原ゆずこの可能性大
- 「そうな」「あーそうか」→ 櫟井唯の可能性大
- 「おー」「〜ですね」→ 日向縁の可能性大

### 優先度4: 特殊ケースの処理
- **オフセリフ（offserif）**: 最も遠い人物、または画面外の人物
- **思考吹き出し（thought_bubble）**: 表情や視線から判断
- **叫び吹き出し（exclamation_bubble）**: 驚いた表情の人物を優先

**重要**: detected_charactersのcharacterIdフィールドは正式名称（「野々原ゆずこ」等）ですが、
出力のspeakerCharacterIdはchar_X形式（char_A等）で指定してください。

## 分析項目（簡潔版）:
各コマについて以下を抽出してください：

### キャラクター情報:
- AI検出された人物の確認と修正（必要な場合のみ）
- 表情の詳細（AI検出では不可能な部分）
- 服装の詳細（制服種別、私服の説明など）
- ショットタイプとキャラクターサイズ

### セリフ情報:
- AI検出された吹き出しと話者の確認
- セリフ内容のOCR結果

### シーン情報:
- 場所の特定（教室、廊下、屋外など）
- 背景効果（集中線、流線、汗マークなど）
- カメラアングルとフレーミング

## キャラクター名の記載方法:
**重要**: characterフィールドには必ず以下の正式名称を使用してください（char_A形式ではなく）:
- 野々原ゆずこ（ピンク髪ボブショート、ボケ役）
- 日向縁（黒髪ストレート、天然）
- 櫟井唯（金髪おさげ、ツッコミ役）
- 松本頼子（茶髪、先生）
- 相川千穂（茶髪ロング、委員長）
- 岡野佳（黒髪セミロング、クール）
- 長谷川ふみ（黒髪ショート、情報通）

## characterSizeの記載方法:
AI検出結果に含まれるcharacterSizeの値（"幅, 高さ"形式）をそのまま使用してください。
例: "0.20, 0.25"
AI検出結果がない場合のみ、以下の目安で推定してください：
- 全身ショット: "0.15, 0.30" 〜 "0.20, 0.40"
- バストショット: "0.18, 0.22" 〜 "0.25, 0.30"
- クローズアップ: "0.30, 0.40" 〜 "0.50, 0.60"

## 4コマ分析の注意事項:
- 各コマを image1, image2, image3, image4 として個別に分析
- 右から左の読み順序を厳守
- キャラクターの一貫性を保持（同じキャラクターには同じIDを使用）

## 出力JSON形式:
{
    "image1": {
        "charactersNum": 0,  // AI検出結果から
        "serifsNum": 0,      // AI検出結果から
        "detectionConfidence": {  // AI検出の信頼度
            "characters": 0.95,
            "bubbles": 0.88
        },
        "serifs": [
            {
                "dialogueId": "d001_p001",
                "text": "OCRで抽出されたセリフ",
                "type": "speech_bubble",
                "speakerCharacterId": "char_C",  // AI検出結果を優先（char_X形式で）
                "detectedSpeakerId": "char_C",    // AI検出の結果（char_X形式）
                "speakerConfidence": 0.85,        // 話者推定の信頼度（メタデータ）
                "boundingBox": null,              // 既存形式のため追加
                "coordinate": [0.1, 0.1],         // AI検出座標
                "readingOrderIndex": 0
            }
        ],
        "characters": [
            {
                "character": "野々原ゆずこ",      // 正式名称を使用（必須）
                "coordinate": [0.3, 0.5],        // AI検出座標
                "position": "0.92",              // 検出信頼度を文字列で
                "faceDirection": "正面",
                "shotType": "バストショット",
                "characterSize": "0.20, 0.25",   // 幅, 高さの数値形式（必須）
                "expression": "笑顔",            
                "clothing": "制服(冬服)",
                "isVisible": true
            }
        ],
        "sceneData": {
            "scene": "教室での会話",
            "location": "教室",
            "backgroundEffects": ["集中線"],
            "cameraAngle": "アイレベル",
            "framing": "中央"
        }
    },
    "image2": { /* image1と同じ形式 */ },
    "image3": { /* image1と同じ形式 */ },
    "image4": { /* image1と同じ形式 */ }
}

## 絶対に守るべきルール:
1. **character**フィールド: 必ず「野々原ゆずこ」「日向縁」「櫟井唯」など正式名称を使用（char_A形式は使用禁止）
2. **characterSize**フィールド: AI検出結果の「幅, 高さ」形式をそのまま使用（例: "0.20, 0.25"）
3. **AI検出結果の優先**: detectionResultsに含まれるcharacter、characterSize、coordinate、positionの値を変更せずにそのまま使用
4. **speakerCharacterId**フィールド: char_A〜char_G形式を使用（セリフの話者IDのみ）
5. **追加分析項目**: AI検出結果にない項目（expression、clothing、faceDirection等）のみを分析して追加
6. **話者特定**: 櫟井唯がツッコミ役をしているときは語調や吹き出しの形（角ばっている事が多い）でわかりやすいのでここから埋めていくとよいです

必ず上記の4パネル形式のJSONとルールに従って回答してください。

## AI検出結果:
`;

    // AI検出結果を既存形式に変換
    console.log("🔄 検出結果変換前:", detectionResults);
    const detectionResultsFormatted = this.convertToLegacyDetectionFormat(detectionResults);
    console.log("🔄 検出結果変換後:", detectionResultsFormatted);
    
    const detectionResultsJson = JSON.stringify(detectionResultsFormatted, null, 2);
    console.log("📋 追加される検出結果JSON:", detectionResultsJson);
    
    prompt += detectionResultsJson;
    console.log("📄 最終プロンプト長さ:", prompt.length, "文字");
    
    // プロンプト生成完了をデバッグ保存
    await this.saveDetectionDebugInfo(
      timestamp,
      imagePathList,
      detectionResults,
      prompt,
      "2_enhanced_prompt"
    );
    
    return prompt;
  }

  // AI検出結果を既存形式に変換
  static convertToLegacyDetectionFormat(detectionResults: PanelDetectionResult[]): any {
    const formatted: any = {};
    
    detectionResults.forEach((result, index) => {
      const panelKey = `panel${index + 1}`;
      
      // キャラクター検出結果を変換
      const detected_characters = result.characters.map((char: any) => {
        // バウンディングボックスからサイズを計算
        let size = "0.20, 0.25"; // デフォルト値
        if (char.bbox && Array.isArray(char.bbox) && char.bbox.length === 4) {
          const [x1, y1, x2, y2] = char.bbox;
          // 正規化されたサイズを計算（幅, 高さ）
          const width = Math.abs(x2 - x1);
          const height = Math.abs(y2 - y1);
          size = `${width.toFixed(2)}, ${height.toFixed(2)}`;
        } else if (char.size) {
          size = char.size;
        }
        
        return {
          characterId: char.character || "不明",
          coordinate: char.coordinate || [0.5, 0.5],
          confidence: char.confidence || char.classificationConfidence || 0.5,
          size: size
        };
      });
      
      // 吹き出し検出結果を変換
      const detected_bubbles = result.balloons.map((balloon: any, balloonIndex: number) => {
        // バウンディングボックスからサイズを計算
        let size = "0.30, 0.40"; // デフォルト値
        if (balloon.bbox && Array.isArray(balloon.bbox) && balloon.bbox.length === 4) {
          const [x1, y1, x2, y2] = balloon.bbox;
          // 正規化されたサイズを計算（幅, 高さ）
          const width = Math.abs(x2 - x1);
          const height = Math.abs(y2 - y1);
          size = `${width.toFixed(2)}, ${height.toFixed(2)}`;
        } else if (balloon.size) {
          size = balloon.size;
        }
        
        return {
          coordinate: balloon.coordinate || [0.5, 0.5],
          confidence: balloon.confidence || 0.5,
          type: balloon.type || "speech_bubble",
          size: size,
          speakerId: balloon.speakerId || balloon.speaker_character || null,
          dialogueId: balloon.dialogueId || `p${index + 1}_balloon_${balloonIndex + 1}`
        };
      });
      
      formatted[panelKey] = {
        detected_characters,
        detected_bubbles
      };
    });
    
    return formatted;
  }

  // AI検出結果を含む改良版プロンプト生成（既存形式準拠・超詳細版）
  static async generateImprovedPrompt(
    detectionResults: PanelDetectionResult[], 
    timestamp: string, 
    imagePathList: ImagePathList
  ): Promise<string> {
    let prompt = `あなたは4コマ漫画『ゆゆ式』の画像を分析する高精度AIアシスタントです。
事前のAI検出結果を最大限活用し、指定されたJSONスキーマに沿って極めて詳細で高精度な情報抽出を行ってください。
日本の4コマ漫画は右から左に読みます。与えられる座標は画像の左上を原点とした相対座標(x,y)です。

タスク:
提供された4コマ漫画画像と高精度AI検出結果を基に、極めて詳細な情報を抽出してください。

## 高精度AI検出結果の活用方法:
プロンプト末尾に以下の形式で高精度AI検出結果が提供されます：
\`\`\`json
{
  "detected_characters": [
    {
      "characterId": "野々原ゆずこ",   // 高精度検出による正式名称（注意: characterIdフィールド）
      "size": "0.20, 0.25",           // 高精度計算による幅, 高さの数値形式（注意: sizeフィールド）
      "coordinate": [0.3, 0.5],       // 高精度バウンディングボックス中心座標
      "confidence": 0.92              // 高精度検出信頼度（数値）
    }
  ],
  "detected_bubbles": [
    {
      "coordinate": [0.8, 0.3],       // 高精度吹き出し中心座標
      "confidence": 0.75,             // 高精度検出信頼度
      "type": "speech_bubble",        // 高精度分類による吹き出しタイプ
      "size": "0.39, 0.26",           // 高精度計算による吹き出しサイズ（幅, 高さ）
      "speakerId": "char_A"           // 高精度話者推定結果（キャラクター専用吹き出しの場合）
    }
  ]
}
\`\`\`

**超重要**: 高精度AI検出結果のcharacterIdとsizeの値を絶対的に優先してください。
- characterIdが「野々原ゆずこ」→ characterフィールドに「野々原ゆずこ」を厳格に使用
- sizeが「0.20, 0.25」→ characterSizeフィールドに「0.20, 0.25」を厳格に使用

## キャラクター特定の優先順位（高精度版）:
1. **最優先**: 高精度AI検出結果の detected_charactersを絶対的に優先
2. **第2優先**: セリフ内容と役割の一致度による微調整
3. **第3優先**: 文脈と前後関係による最終補正

## 話者特定の高精度計算手順:

**超重要: 以下の高精度優先順位で話者を厳密に判定してください**

### 【最優先度1】: キャラクター専用吹き出しタイプ（絶対優先）
detected_bubblesのtypeが以下の場合は、対応するキャラクターを話者として絶対確定：
- \`chractor_bubble_yuzuko\` → 野々原ゆずこ（char_A）【絶対確定】
- \`chractor_bubble_yukari\` → 日向縁（char_B）【絶対確定】
- \`chractor_bubble_yui\` → 櫟井唯（char_C）【絶対確定】
- その他のchractor_bubble_xxx → 対応するキャラクター【絶対確定】

### 【優先度2】: 高精度距離計算による判定（通常の吹き出しの場合）
1. **高精度距離計算プロセス**
   - detected_bubblesのcoordinateとdetected_charactersのcoordinateを使用
   - 距離 = sqrt((吹き出しX - 人物X)^2 + (吹き出しY - 人物Y)^2)
   - **計算例**: 吹き出し[0.7, 0.3]と人物[0.8, 0.4]の距離 = sqrt((0.7-0.8)^2 + (0.3-0.4)^2) = 0.141

2. **高精度距離による話者判定ルール**
   - **距離 < 0.12**: その人物が話者である可能性【極めて高い】（最高優先）
   - **距離 0.12-0.18**: その人物が話者である可能性【非常に高い】（高優先）
   - **距離 0.18-0.28**: 話者の有力候補として厳重考慮
   - **距離 0.28-0.40**: 低確率候補として検討
   - **距離 > 0.40**: 基本的に話者ではない（オフセリフを除く）

3. **同距離の場合の高精度優先順位**
   - 右側の人物を優先（X座標が大きい方）【日本語読み順準拠】
   - Y座標の差が0.08未満の場合は同じ高さとみなす【高精度閾値】

### 【優先度3】: セリフ内容による高精度補正
距離が近い候補が複数いる場合のみ適用（高精度言語解析）：
- **「わからないー！」「いとをかしー！」「えーっと」** → 野々原ゆずこの可能性【極大】
- **「そうな」「あーそうか」「なるほど」** → 櫟井唯の可能性【極大】
- **「おー」「〜ですね」「そうですね」** → 日向縁の可能性【極大】

### 【優先度4】: 特殊ケースの高精度処理
- **オフセリフ（offserif_bubble）**: 最も遠い人物、または画面外の人物【高精度推定】
- **思考吹き出し（thought_bubble）**: 表情や視線から高精度判断
- **叫び吹き出し（exclamation_bubble）**: 驚いた表情の人物を最優先

### 【話者判定の高精度計算例】:
\`\`\`
detected_bubbles[0]: coordinate=[0.7, 0.3], type="speech_bubble"
detected_characters:
  - characterId="野々原ゆずこ", coordinate=[0.8, 0.4]
  - characterId="日向縁", coordinate=[0.2, 0.3]  
  - characterId="櫟井唯", coordinate=[0.5, 0.7]

高精度距離計算:
- 野々原ゆずこ: sqrt((0.7-0.8)² + (0.3-0.4)²) = 0.141 → 最も近い（0.18未満）
- 日向縁: sqrt((0.7-0.2)² + (0.3-0.3)²) = 0.500 → 遠い
- 櫟井唯: sqrt((0.7-0.5)² + (0.3-0.7)²) = 0.447 → 遠い

高精度判定結果: speakerCharacterId="char_A"（野々原ゆずこ）【確信度: 極高】
\`\`\`

**超重要**: detected_charactersのcharacterIdフィールドは正式名称（「野々原ゆずこ」等）ですが、
出力のspeakerCharacterIdは厳格にchar_X形式（char_A等）で指定してください。

## 高精度分析項目（詳細版）:
各コマについて以下を極めて詳細に抽出してください：

### 【キャラクター情報】（高精度版）:
- 高精度AI検出された人物の確認と微調整（必要最小限）
- **表情の超詳細分析**（AI検出では不可能な部分）
- **服装の詳細記述**（制服種別、私服の詳細説明など）
- **ショットタイプとキャラクターサイズ**（高精度計算値使用）
- **姿勢・体の向き・手の位置**（詳細観察）
- **感情状態・心理的表現**（微細な表情変化から推定）

### 【セリフ情報】（高精度版）:
- 高精度AI検出された吹き出しと話者の確認
- **セリフ内容の高精度OCR結果**
- **セリフの調子・感情・音量レベル**
- **語尾の特徴・キャラクター特有の話し方**

### 【シーン情報】（高精度版）:
- **場所の詳細特定**（教室、廊下、屋外の具体的位置）
- **背景効果の詳細**（集中線、流線、汗マーク、効果音等）
- **カメラアングルとフレーミング**（詳細分析）
- **シーンの雰囲気・照明・時間帯**

## キャラクター名の厳格記載方法:
**絶対ルール**: characterフィールドには必ず以下の正式名称を厳格に使用（char_A形式は絶対禁止）:
- **野々原ゆずこ**（ピンク髪ボブショート、天然ボケ役、食いしん坊）
- **日向縁**（黒髪ストレート、天然マイペース、読書好き）
- **櫟井唯**（金髪おさげ、ツッコミ役、しっかり者）
- **松本頼子**（茶髪、担任の先生、優しい）
- **相川千穂**（茶髪ロング、委員長、真面目）
- **岡野佳**（黒髪セミロング、クール、内向的）
- **長谷川ふみ**（黒髪ショート、情報通、活発）

## characterSizeの厳格記載方法:
高精度AI検出結果に含まれるcharacterSizeの値（"幅, 高さ"形式）を絶対にそのまま使用してください。
例: "0.20, 0.25"
高精度AI検出結果がない場合のみ、以下の高精度目安で推定：
- **全身ショット**: "0.15, 0.30" 〜 "0.20, 0.40"
- **バストショット**: "0.18, 0.22" 〜 "0.25, 0.30"
- **クローズアップ**: "0.30, 0.40" 〜 "0.50, 0.60"

## 4コマ高精度分析の注意事項:
- 各コマを **image1, image2, image3, image4** として個別に高精度分析
- **右から左の読み順序を厳格に遵守**
- **キャラクターの一貫性を絶対保持**（同じキャラクターには同じIDを使用）
- **4コマ全体の文脈・ストーリー展開を考慮**

## 高精度出力JSON形式:
{
    "image1": {
        "charactersNum": 0,  // 高精度AI検出結果から
        "serifsNum": 0,      // 高精度AI検出結果から
        "detectionConfidence": {  // 高精度AI検出の信頼度
            "characters": 0.95,
            "bubbles": 0.88
        },
        "serifs": [
            {
                "dialogueId": "d001_p001",
                "text": "高精度OCRで抽出されたセリフ",
                "type": "speech_bubble",
                "speakerCharacterId": "char_C",  // 高精度AI検出結果を絶対優先（char_X形式で）
                "detectedSpeakerId": "char_C",    // 高精度AI検出の結果（char_X形式）
                "speakerConfidence": 0.85,        // 高精度話者推定の信頼度（メタデータ）
                "boundingBox": null,              // 既存形式のため追加
                "coordinate": [0.1, 0.1],         // 高精度AI検出座標
                "readingOrderIndex": 0,
                "tone": "明るい",                 // セリフの調子
                "volume": "普通",                 // 音量レベル
                "characteristicSpeech": "ゆずこ特有の語尾"
            }
        ],
        "characters": [
            {
                "character": "野々原ゆずこ",      // 正式名称を厳格使用（必須）
                "coordinate": [0.3, 0.5],        // 高精度AI検出座標
                "position": "0.92",              // 高精度検出信頼度を文字列で
                "faceDirection": "正面やや右",
                "shotType": "バストショット",
                "characterSize": "0.20, 0.25",   // 高精度計算による幅, 高さ（必須）
                "expression": "困惑した笑顔",     // 詳細表情分析            
                "clothing": "冬服制服（紺ブレザー、白シャツ、赤リボン）",
                "pose": "右手を頬に当てている",
                "emotion": "戸惑い",
                "isVisible": true
            }
        ],
        "sceneData": {
            "scene": "教室での日常会話、放課後の穏やかな雰囲気",
            "location": "高校の教室（窓側の席）",
            "backgroundEffects": ["集中線", "汗マーク"],
            "cameraAngle": "アイレベル",
            "framing": "中央やや右寄り",
            "mood": "穏やか",
            "lighting": "自然光（夕方）",
            "timeOfDay": "放課後"
        }
    },
    "image2": { /* image1と同じ高精度形式 */ },
    "image3": { /* image1と同じ高精度形式 */ },
    "image4": { /* image1と同じ高精度形式 */ }
}

## 絶対に守るべき高精度ルール:
1. **character**フィールド: 必ず「野々原ゆずこ」「日向縁」「櫟井唯」など正式名称を厳格使用（char_A形式は絶対禁止）
2. **characterSize**フィールド: 高精度AI検出結果の「幅, 高さ」形式を絶対にそのまま使用（例: "0.20, 0.25"）
3. **高精度AI検出結果の絶対優先**: detectionResultsに含まれるcharacter、characterSize、coordinate、positionの値を一切変更せず絶対にそのまま使用
4. **speakerCharacterId**フィールド: char_A〜char_G形式を厳格使用（セリフの話者IDのみ）
5. **追加分析項目**: 高精度AI検出結果にない項目（expression、clothing、faceDirection等）のみを詳細分析して追加
6. **話者特定**: 櫟井唯がツッコミ役をしているときは語調や吹き出しの形（角ばっている事が多い）でわかりやすいのでここから優先的に埋めていくとよいです
7. **4コマ全体の一貫性**: ストーリー展開、キャラクター配置、時系列を考慮した高精度分析

必ず上記の4パネル高精度形式のJSONとルールに厳格に従って回答してください。

## 高精度AI検出結果:
`;

    // AI検出結果を既存形式に変換
    const detectionResultsFormatted = this.convertToLegacyDetectionFormat(detectionResults);
    prompt += JSON.stringify(detectionResultsFormatted, null, 2);

    // 改良版プロンプト生成完了をデバッグ保存
    await this.saveDetectionDebugInfo(
      timestamp,
      imagePathList,
      detectionResults,
      prompt,
      "3_improved_prompt"
    );

    return prompt;
  }

  // 4パネル YOLO+DINOv2 検出処理
  static async handleFourPanelYoloDinov2Detection(
    imagePathList: ImagePathList,
    imageData: ImageData,
    setImageData: (updater: (prev: ImageData) => ImageData) => void,
    setIsYoloDetecting: (detecting: boolean) => void,
    saveToJSONWithData?: (data: ImageData) => Promise<void>
  ): Promise<void> {
    setIsYoloDetecting(true);
    try {
      console.log("4コマ一括YOLO+DINOv2検出開始");
      
      // 4つのパネルを並列で検出（キャラクターと吹き出し両方）
      const promises = [1, 2, 3, 4].map(async (num) => {
        const imageKey = `image${num}`;
        
        // キャラクター検出
        const charResponse = await axios.post(
          "http://localhost:8000/api/detect-characters-yolo-dinov2",
          {
            komaPath: imagePathList[imageKey as keyof ImagePathList],
            mode: "single",
            detectionThreshold: 0.25,
            classificationThreshold: 0.5,
            visualize: true
          }
        );
        
        // 吹き出し検出
        let balloonResponse: any = null;
        try {
          balloonResponse = await axios.post(
            "http://localhost:8000/api/detect-balloons",
            {
              imagePath: imagePathList[imageKey as keyof ImagePathList],
              confidenceThreshold: 0.15,
              maxDet: 300
            }
          );
        } catch (error) {
          console.warn(`パネル${num}の吹き出し検出エラー:`, error);
        }
        
        return {
          imageKey,
          characters: charResponse.data.characters || [],
          balloons: balloonResponse?.data?.detections || []
        };
      });
      
      const results = await Promise.all(promises);
      console.log("4コマ検出結果:", results);
      
      // 結果をimageDataに統合
      const newImageData = { ...imageData };
      const englishToCharIdMap = getEnglishToCharIdMap();
      
      for (const result of results) {
        const existingData = newImageData[result.imageKey as keyof ImageData];
        
        // 話者判定処理
        if (result.balloons.length > 0 && result.characters.length > 0) {
          const detectedCharacters: CharacterDetection[] = result.characters.map((char: any) => ({
            id: char.character || "",
            name: char.character || "",
            boundingBox: char.bbox ? {
              x1: char.bbox[0],
              y1: char.bbox[1],
              x2: char.bbox[2],
              y2: char.bbox[3]
            } : { x1: 0, y1: 0, x2: 0, y2: 0 },
            faceDirection: char.faceDirection || undefined
          }));
          
          const panelNum = result.imageKey.replace('image', '');
          const panelPrefix = `p${panelNum}_`;
          
          // 話者特定を実行
          if (result.balloons.length > 0 && detectedCharacters.length > 0) {
            console.log(`=== パネル${panelNum}の話者特定処理を開始 ===`);
            
            const speakerMatcher = new BalloonSpeakerMatcher();
            speakerMatcher.setImageUrl(getImagePath(imagePathList[result.imageKey as keyof ImagePathList]));
            
            const processedDetections = processAndReorderBalloons(result.balloons, panelPrefix);
            
            const speakerMatches = speakerMatcher.matchBalloonsToSpeakers(
              processedDetections,
              detectedCharacters
            );
            
            console.log(`パネル${panelNum}話者特定結果:`, speakerMatches);
            
            // 話者情報を吹き出しに反映
            const processedDetectionsWithSpeakers = processedDetections.map(detection => {
              const match = speakerMatches.find(m => m.balloonId === detection.dialogueId);
              if (match && match.confidence > 0.3) {
                const charId = englishToCharIdMap[match.speakerName] || '';
                console.log(`✅ パネル${panelNum}話者設定: ${detection.dialogueId} -> ${match.speakerName} -> ${charId}`);
                
                return {
                  ...detection,
                  speakerCharacterId: charId
                };
              }
              return detection;
            });
            
            // セリフデータをマージして更新
            const mergedSerifs = mergeSerifsWithBalloonDetections(
              existingData.serifs || [],
              processedDetectionsWithSpeakers,
              panelPrefix
            );
            
            newImageData[result.imageKey as keyof ImageData] = {
              ...existingData,
              serifs: mergedSerifs,
              serifsNum: mergedSerifs.length,
            };
          }
        }
        
        // キャラクター情報の更新
        if (result.characters.length > 0) {
          const updatedCharacters = existingData.characters.map((existingChar, index) => {
            const matchingDetection = result.characters.find((detected: any) => {
              // 位置の近似マッチング
              const coordSimilar = existingChar.coordinate && detected.coordinate &&
                Math.abs(existingChar.coordinate[0] - detected.coordinate[0]) < 50 &&
                Math.abs(existingChar.coordinate[1] - detected.coordinate[1]) < 50;
              return coordSimilar || index < result.characters.length;
            });
            
            if (matchingDetection) {
              return {
                ...existingChar,
                character: matchingDetection.character || existingChar.character,
                coordinate: matchingDetection.coordinate || existingChar.coordinate,
                position: matchingDetection.classificationConfidence ? 
                  matchingDetection.classificationConfidence.toFixed(3) : 
                  existingChar.position
              };
            }
            return existingChar;
          });
          
          newImageData[result.imageKey as keyof ImageData] = {
            ...newImageData[result.imageKey as keyof ImageData],
            characters: updatedCharacters,
            charactersNum: updatedCharacters.length,
          };
        }
      }
      
      setImageData(() => newImageData);
      console.log("4コマ一括検出完了:", newImageData);
      
      // 自動保存
      if (saveToJSONWithData) {
        console.log("🔄 自動保存を実行します... (handleFourPanelYoloDinov2Detection)");
        setTimeout(async () => {
          try {
            await saveToJSONWithData(newImageData);
            console.log("✅ 自動保存完了 (handleFourPanelYoloDinov2Detection)");
          } catch (error) {
            console.error("❌ 自動保存エラー (handleFourPanelYoloDinov2Detection):", error);
          }
        }, 500);
      } else {
        console.warn("⚠️ saveToJSONWithDataが未定義のため自動保存をスキップ (handleFourPanelYoloDinov2Detection)");
      }
      
    } catch (error) {
      console.error("❌ 4コマ一括検出エラー:", error);
      alert("4コマ一括検出エラー: " + (error as any).message);
    } finally {
      setIsYoloDetecting(false);
    }
  }

  // 単一パネル分析処理
  static async handleSubmit(
    imageKey: string,
    imagePathList: ImagePathList,
    imageData: ImageData,
    setImageData: (updater: (prev: ImageData) => ImageData) => void,
    saveToJSONWithData: (data: ImageData) => Promise<void>
  ): Promise<void> {
    const formData = new FormData();
    
    console.log(`単一パネル分析開始: ${imageKey}`);
    console.log("送信する画像パス:", imagePathList[imageKey as keyof ImagePathList]);

    try {
      const response = await axios.post(
        "http://localhost:8000/api/analyze-image/",
        {
          komaPath: imagePathList[imageKey as keyof ImagePathList],
          mode: "single-panel"
        }
      );
      
      const contentData = response.data.content_data;
      
      // 画像パスを正規化
      const normalizedContentData = {
        ...contentData,
        imagePath: imagePathList[imageKey as keyof ImagePathList]
      };
      
      console.log("正規化されたレスポンス:", normalizedContentData);

      setImageData((prevData) => {
        const newData = { ...prevData };
        
        // キャラクター情報の更新
        if (normalizedContentData.characters && normalizedContentData.characters.length > 0) {
          console.log("キャラクター情報を更新:", normalizedContentData.characters);
          
          const currentPanel = newData[imageKey as keyof ImageData];
          const existingCharacters = currentPanel.characters || [];
          
          // 既存の空のキャラクターを除去
          const chars = normalizedContentData.characters.filter(
            (char: Character) => char.character && char.character.trim() !== ""
          );
          
          // 既存のキャラクターを更新または追加
          const updatedCharacters = [...existingCharacters];
          
          chars.forEach((newChar: Character, index: number) => {
            if (updatedCharacters[index]) {
              updatedCharacters[index] = { ...updatedCharacters[index], ...newChar };
            } else {
              updatedCharacters.push(newChar);
            }
          });
          
          newData[imageKey as keyof ImageData] = {
            ...currentPanel,
            characters: updatedCharacters,
            charactersNum: updatedCharacters.length,
          };
        }
        
        // 自動保存
        setTimeout(() => {
          saveToJSONWithData(newData);
        }, 500);
        
        return newData;
      });
      
    } catch (error) {
      console.error("単一パネル分析エラー:", error);
      alert("分析エラーが発生しました: " + (error as any).message);
    }
  }

  // 4コマ分析API処理
  static async handleFourPanelAnalyze(
    imagePathList: ImagePathList,
    imageData: ImageData,
    setImageData: (updater: (prev: ImageData) => ImageData) => void,
    setIsFourPanelAnalyzing: (analyzing: boolean) => void,
    fourPanelPromptType: string,
    saveToJSONWithData?: (data: ImageData) => Promise<void>
  ): Promise<void> {
    setIsFourPanelAnalyzing(true);
    try {
      console.log("4コマ分析開始:", fourPanelPromptType);
      
      // 事前に各パネルのAI検出を実行
      console.log("🎯 事前AI検出を実行中...");
      const detectionResults = await this.executePreDetection(imagePathList);
      
      // 4パネルの画像を結合
      const combinedImageBase64 = await combineFourPanelImages(imagePathList);
      const combinedImageUrl = `data:image/jpeg;base64,${combinedImageBase64}`;
      
      // AI検出結果を含むプロンプトを生成
      const timestamp = new Date().toISOString().replace(/[:.]/g, '_').replace('T', '_').substring(0, 19);
      const enhancedPrompt = await this.generateEnhancedPrompt(detectionResults, timestamp, imagePathList);
      
      const response = await axios.post("http://localhost:8000/api/analyze-image/", {
        komaPath: combinedImageUrl,
        mode: "combined-four-panel",
        forceAPI: true,
        imagePathList: imagePathList,
        prompt: enhancedPrompt
      });
      
      const contentData = response.data.content_data;
      console.log("4コマ分析結果:", contentData);
      
      // 結果をimageDataに統合
      const newData = { ...imageData };
      
      // 各パネルのデータを更新（順序を保証）
      ['image1', 'image2', 'image3', 'image4'].forEach((key) => {
        if (contentData[key]) {
          const panelData = contentData[key];
          newData[key as keyof ImageData] = {
            ...newData[key as keyof ImageData],
            ...panelData
          };
        }
      });
      
      setImageData(() => newData);
      
      // 自動保存
      if (saveToJSONWithData) {
        console.log("🔄 自動保存を実行します... (handleFourPanelAnalyze)");
        setTimeout(async () => {
          try {
            await saveToJSONWithData(newData);
            console.log("✅ 自動保存完了 (handleFourPanelAnalyze)");
          } catch (error) {
            console.error("❌ 自動保存エラー (handleFourPanelAnalyze):", error);
          }
        }, 500);
      } else {
        console.warn("⚠️ saveToJSONWithDataが未定義のため自動保存をスキップ (handleFourPanelAnalyze)");
      }
      
    } catch (error) {
      console.error("4コマ分析エラー:", error);
      alert("4コマ分析エラー: " + (error as any).message);
    } finally {
      setIsFourPanelAnalyzing(false);
    }
  }

  // 4コマ改良分析API処理
  static async handleFourPanelAnalyzeImproved(
    imagePathList: ImagePathList,
    imageData: ImageData,
    setImageData: (updater: (prev: ImageData) => ImageData) => void,
    setIsFourPanelAnalyzing: (analyzing: boolean) => void,
    fourPanelPromptType: string,
    saveToJSONWithData?: (data: ImageData) => Promise<void>
  ): Promise<void> {
    setIsFourPanelAnalyzing(true);
    try {
      console.log("4コマ改良分析開始:", fourPanelPromptType);
      
      // 事前に各パネルのAI検出を実行
      console.log("🎯 事前AI検出を実行中...");
      const detectionResults = await this.executePreDetection(imagePathList);
      
      // 4パネルの画像を結合
      const combinedImageBase64 = await combineFourPanelImages(imagePathList);
      const combinedImageUrl = `data:image/jpeg;base64,${combinedImageBase64}`;
      
      // AI検出結果を含むプロンプトを生成（改良版）
      const timestamp = new Date().toISOString().replace(/[:.]/g, '_').replace('T', '_').substring(0, 19);
      const enhancedPrompt = await this.generateImprovedPrompt(detectionResults, timestamp, imagePathList);
      
      const response = await axios.post("http://localhost:8000/api/analyze-image-improved/", {
        komaPath: combinedImageUrl,
        mode: "combined-four-panel-improved",
        forceAPI: true,
        imagePathList: imagePathList,
        prompt: enhancedPrompt
      });
      
      const contentData = response.data.content_data;
      console.log("4コマ改良分析結果:", contentData);
      
      // 結果をimageDataに統合
      const newData = { ...imageData };
      
      // 各パネルのデータを更新（順序を保証）
      ['image1', 'image2', 'image3', 'image4'].forEach((key) => {
        if (contentData[key]) {
          const panelData = contentData[key];
          newData[key as keyof ImageData] = {
            ...newData[key as keyof ImageData],
            ...panelData
          };
        }
      });
      
      setImageData(() => newData);
      
      // 自動保存
      if (saveToJSONWithData) {
        console.log("🔄 自動保存を実行します... (handleFourPanelAnalyzeImproved)");
        setTimeout(async () => {
          try {
            await saveToJSONWithData(newData);
            console.log("✅ 自動保存完了 (handleFourPanelAnalyzeImproved)");
          } catch (error) {
            console.error("❌ 自動保存エラー (handleFourPanelAnalyzeImproved):", error);
          }
        }, 500);
      } else {
        console.warn("⚠️ saveToJSONWithDataが未定義のため自動保存をスキップ (handleFourPanelAnalyzeImproved)");
      }
      
    } catch (error) {
      console.error("4コマ改良分析エラー:", error);
      alert("4コマ改良分析エラー: " + (error as any).message);
    } finally {
      setIsFourPanelAnalyzing(false);
    }
  }
}