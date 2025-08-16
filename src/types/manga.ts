// 漫画分析用の共通型定義

// 尻尾形状分類結果
export interface TailShapeClassification {
  predicted_category: string;
  confidence: number;
  top3_predictions: Array<{
    category: string;
    confidence: number;
  }>;
  error?: string;
}

export interface Serif {
  dialogueId: string; // コマ内でのセリフのID
  text: string; // OCRで抽出されたセリフの内容
  type: "speechBubble" | "outsideBubble" | "narration"; // セリフの種類
  speakerCharacterId: string | null; // このセリフを話しているキャラクターのID（推定）
  boundingBox: [number, number, number, number] | null; // セリフテキストの画像内の座標
  readingOrderIndex: number; // コマ内でのセリフの読順（0から始まる）
  coordinate: [number, number]; // 座標を(x,y)で画像に対する比率
  tail_shape_classification?: TailShapeClassification; // 尻尾形状分類結果
}

export interface Character {
  character: string;
  faceDirection: string;
  position: string;
  shotType?: string; // ショットタイプ（映り方）
  characterSize?: string; // キャラクターサイズ
  expression: string;
  serif: string;
  clothing: string;
  isVisible: boolean;
}

export interface SceneData {
  scene: string;
  location: string; // このコマの舞台となっている場所
  backgroundEffects: string; // 背景やキャラクター周辺の視覚効果
  cameraAngle: string; // "アイレベル"、"俯瞰"、"あおり"、"斜め"
  framing: string; // "中央"、"左寄り"、"右寄り"、"上寄り"、"下寄り"
}

export interface PanelData {
  charactersNum: number;
  serifsNum: number;
  serifs: Serif[];
  characters: Character[];
  sceneData: SceneData;
}

export interface FourPanelData {
  panel1: PanelData;
  panel2: PanelData;
  panel3: PanelData;
  panel4: PanelData;
}

export interface TestResult {
  charactersNum?: number;
  serifsNum?: number;
  serifs?: Serif[];
  characters?: Character[];
  sceneData?: SceneData;
  error?: string;
  rawResponse?: string;
  provider?: string;
  timestamp?: string;
}

export interface ProviderTestResults {
  [provider: string]: TestResult;
}

export type LLMProvider = "gemini" | "gemini-pro" | "claude" | "openai";
export type PromptType = "combined" | "four-panel";
export type ProcessingMode = "single" | "four-panel";