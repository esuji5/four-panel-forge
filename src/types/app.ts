// アプリケーション全体の型定義

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
  dialogueId: string;
  text: string;
  type: string;
  speakerCharacterId?: string | null;
  boundingBox?: any;
  readingOrderIndex: number;
  coordinate: [number, number];
  tail_shape_classification?: TailShapeClassification; // 尻尾形状分類結果
}

export interface Character {
  character: string;
  faceDirection: string;
  position: string;
  shotType: string;
  characterSize: string;
  expression: string;
  clothing: string;
  isVisible: boolean | number;
  coordinate: [number, number];
}

export interface SceneData {
  scene: string;
  location: string;
  backgroundEffects: string;
  cameraAngle?: string;
  framing?: string;
}

export interface PanelData {
  komaPath: string;
  characters: Character[];
  sceneData: SceneData;
  serifs?: Serif[];
  charactersNum?: number;
  serifsNum?: number;
}

export interface ImageData {
  image1: PanelData;
  image2: PanelData;
  image3: PanelData;
  image4: PanelData;
}

export interface ImagePathList {
  [key: string]: string;
}

export interface CSVRow {
  koma_path: string;
  [key: string]: any;
}

export interface ChatItem {
  question: string;
  answer: string;
}

// Human in the Loop関連の型定義
export interface AIProposal {
  image_path: string;
  timestamp: string;
  model: string;
  proposal: any;
  confidence_scores?: { [key: string]: number };
  processing_time?: number;
  api_cost?: number;
}

export interface RevisionEntry {
  timestamp: string;
  revision_type: string;
  changes: DiffChange[];
  editor: string;
  confidence?: number;
  notes?: string;
}

export interface RevisionHistory {
  image_path: string;
  created_at: string;
  last_updated: string;
  total_revisions: number;
  ai_proposals: AIProposal[];
  revisions: RevisionEntry[];
  current_data: any;
}

export interface DiffChange {
  field_path: string;
  old_value: any;
  new_value: any;
  change_type: "added" | "modified" | "deleted";
  similarity?: number;
}