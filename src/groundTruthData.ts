// Ground Truth データ（public/saved_json/から転記）

// 型定義
export interface Character {
  character: string;
  faceDirection: string;
  position: string;
  expression: string;
  serif: string;
  clothing: string;
  isVisible: number | boolean;
}

export interface SceneData {
  scene: string;
  location: string;
  backgroundEffects: string;
}

export interface PanelData {
  komaPath: string;
  characters: Character[];
  sceneData: SceneData;
}

export interface DatasetData {
  image1: PanelData;
  image2: PanelData;
  image3: PanelData;
  image4: PanelData;
}

export interface GroundTruthData {
  [key: string]: DatasetData;
}

export interface AvailableDataset {
  value: string;
  label: string;
}

export const groundTruthData: GroundTruthData = {
  yuyu10_112: {
    "image1": {
      "komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-069-1-pad-shaved.jpg",
      "characters": [
        {"character": "野々原ゆずこ", "faceDirection": "左", "position": "左から0.20, 0.68", "expression": "普通", "serif": "包丁ってどこの家にもあるけど結構強い武器じゃない?", "clothing": "制服", "isVisible": 1},
        {"character": "日向縁", "faceDirection": "前", "position": "左から0.50, 0.58", "expression": "普通", "serif": "なし", "clothing": "制服", "isVisible": 1},
        {"character": "櫟井唯", "faceDirection": "前", "position": "左から0.75, 0.64", "expression": "普通", "serif": "あーそうか そうな", "clothing": "制服", "isVisible": 1}
      ],
      "sceneData": {"scene": "教室での会話", "location": "不明", "backgroundEffects": "なし"}
    },
    "image2": {
      "komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-069-2-pad-shaved.jpg",
      "characters": [
        {"character": "日向縁", "faceDirection": "左", "position": "右から0.2, 0.4", "expression": "無表情", "serif": "女の武器は涙ーとかあるよ", "clothing": "制服", "isVisible": 1},
        {"character": "櫟井唯", "faceDirection": "正面", "position": "左から0.2, 0.4", "expression": "困惑", "serif": "あーそうか持ってた 使わんけど…", "clothing": "制服", "isVisible": 1}
      ],
      "sceneData": {"scene": "会話", "location": "教室", "backgroundEffects": "なし"}
    },
    "image3": {
      "komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-069-3-pad-shaved.jpg",
      "characters": [
        {"character": "日向縁", "faceDirection": "左", "position": "右側(画像の80%位置)", "expression": "考え込んでいる顔", "serif": "どっち強いー？ ", "clothing": "制服", "isVisible": 1},
        {"character": "櫟井唯", "faceDirection": "上", "position": "中央(画像の50%位置)", "expression": "考えているような顔", "serif": "んー 包丁かなあ", "clothing": "制服", "isVisible": 1},
        {"character": "野々原ゆずこ", "faceDirection": "上", "position": "左側(画像の40%位置)", "expression": "無表情", "serif": "なし", "clothing": "制服", "isVisible": 1}
      ],
      "sceneData": {"scene": "会話", "location": "教室", "backgroundEffects": "なし"}
    },
    "image4": {
      "komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-069-4-pad-shaved.jpg",
      "characters": [
        {"character": "野々原ゆずこ", "faceDirection": "正面", "position": "右側上部", "expression": "笑顔", "serif": "泣きながら包丁持ってる女は?", "clothing": "不明", "isVisible": 1},
        {"character": "日向縁", "faceDirection": "いない", "position": "いない", "expression": "いない", "serif": "つよいっ", "clothing": "いない", "isVisible": 0},
        {"character": "櫟井唯", "faceDirection": "", "position": "", "expression": "", "serif": "つよっ", "clothing": "", "isVisible": false}
      ],
      "sceneData": {"scene": "会話", "location": "不明", "backgroundEffects": "なし"}
    }
  },
  yuyu10_113: {
    "image1": {
      "komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-069-5-pad-shaved.jpg",
      "characters": [
        {"character": "櫟井唯", "faceDirection": "右", "position": "右側(約0.75, 0.5)", "expression": "普通", "serif": "二刀流 ある意味", "clothing": "不明", "isVisible": 1},
        {"character": "野々原ゆずこ", "faceDirection": "左", "position": "左側上部(約0.3, 0.6)", "expression": "普通", "serif": "やばい つよい。", "clothing": "セーラー服", "isVisible": 1},
        {"character": "日向縁", "faceDirection": "右", "position": "左側下部(約0.3, 0.4)", "expression": "驚いた", "serif": "なし", "clothing": "セーラー服", "isVisible": false}
      ],
      "sceneData": {"scene": "会話", "location": "不明", "backgroundEffects": "なし"}
    },
    "image2": {
      "komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-069-6-pad-shaved.jpg",
      "characters": [
        {"character": "野々原ゆずこ", "faceDirection": "正面", "position": "右側 (約0.75, 0.4)", "expression": "目を大きく見開いて少し口を開けている、やや真剣そうな表情", "serif": "ちょっと… 五輪書かく？ あたらしい", "clothing": "セーラー服", "isVisible": 1},
        {"character": "櫟井唯", "faceDirection": "左", "position": "左側 (約0.2, 0.5)", "expression": "不明 （後ろ姿のため）", "serif": "何書くの？", "clothing": "不明 （詳細が描かれていない）", "isVisible": 1},
        {"character": "", "faceDirection": "いない", "position": "いない", "expression": "いない", "serif": "なし", "clothing": "いない", "isVisible": 0}
      ],
      "sceneData": {"scene": "ゆずこが何かを書こうとしている様子を唯が聞いている場面", "location": "不明", "backgroundEffects": "なし"}
    },
    "image3": {
      "komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-069-7-pad-shaved.jpg",
      "characters": [
        {"character": "日向縁", "faceDirection": "右", "position": "右", "expression": "笑顔", "serif": "なんで泣いてるんだろーね？ 女の人", "clothing": "セーラー服", "isVisible": 1},
        {"character": "野々原ゆずこ", "faceDirection": "左", "position": "中央", "expression": "驚き", "serif": "まあ… 浮気されたとか不倫の末とかかな", "clothing": "セーラー服", "isVisible": 1},
        {"character": "櫟井唯", "faceDirection": "右", "position": "左", "expression": "ぼんやり・驚き", "serif": "あー…", "clothing": "セーラー服", "isVisible": 1}
      ],
      "sceneData": {"scene": "登場人物3人が会話しながら場面を想像している", "location": "不明", "backgroundEffects": "円形が散らばる効果描写"}
    },
    "image4": {
      "komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-069-8-pad-shaved.jpg",
      "characters": [
        {"character": "日向縁", "faceDirection": "左", "position": "右側", "expression": "無表情", "serif": "あ 不倫の書かく？", "clothing": "セーラー服", "isVisible": 1},
        {"character": "野々原ゆずこ", "faceDirection": "後ろ向き", "position": "中央", "expression": "後ろ姿のため不明", "serif": "んふっ", "clothing": "セーラー服", "isVisible": 1},
        {"character": "櫟井唯", "faceDirection": "右", "position": "左側", "expression": "困惑", "serif": "読みたくないわ", "clothing": "セーラー服", "isVisible": 1}
      ],
      "sceneData": {"scene": "ゆずこの発言に縁と唯が反応している場面", "location": "不明", "backgroundEffects": "網掛けトーン"}
    }
  },
  yuyu10_114: {
    "image1": {"komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-070-1-pad-shaved.jpg", "characters": [], "sceneData": {"scene": "", "location": "", "backgroundEffects": ""}},
    "image2": {"komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-070-2-pad-shaved.jpg", "characters": [], "sceneData": {"scene": "", "location": "", "backgroundEffects": ""}},
    "image3": {"komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-070-3-pad-shaved.jpg", "characters": [], "sceneData": {"scene": "", "location": "", "backgroundEffects": ""}},
    "image4": {"komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-070-4-pad-shaved.jpg", "characters": [], "sceneData": {"scene": "", "location": "", "backgroundEffects": ""}}
  },
  yuyu10_115: {
    "image1": {"komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-070-5-pad-shaved.jpg", "characters": [], "sceneData": {"scene": "", "location": "", "backgroundEffects": ""}},
    "image2": {"komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-070-6-pad-shaved.jpg", "characters": [], "sceneData": {"scene": "", "location": "", "backgroundEffects": ""}},
    "image3": {"komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-070-7-pad-shaved.jpg", "characters": [], "sceneData": {"scene": "", "location": "", "backgroundEffects": ""}},
    "image4": {"komaPath": "/yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/image-070-8-pad-shaved.jpg", "characters": [], "sceneData": {"scene": "", "location": "", "backgroundEffects": ""}}
  }
};

// データセット名の一覧
export const availableDatasets: AvailableDataset[] = [
  { value: 'yuyu10_112', label: 'yuyu10_112 - 包丁と涙の話' },
  { value: 'yuyu10_113', label: 'yuyu10_113 - 二刀流と五輪書' },
  { value: 'yuyu10_114', label: 'yuyu10_114' },
  { value: 'yuyu10_115', label: 'yuyu10_115' }
];