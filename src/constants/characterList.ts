// ゆゆ式のキャラクター候補リスト
// プロンプトの順番に基づいて定義
export const CHARACTER_OPTIONS = [
  "野々原ゆずこ",
  "日向縁",
  "櫟井唯",
  "松本頼子",
  "相川千穂",
  "岡野佳",
  "長谷川ふみ",
  "お母さん",
  "先生",
  "生徒A",
  "生徒B",
  "その他",
] as const;

// 全角文字を半角に変換する関数
export const toHalfWidth = (str: string): string => {
  return str.replace(/[０-９]/g, (match) => {
    return String.fromCharCode(match.charCodeAt(0) - 0xFEE0);
  }).replace(/[ａ-ｚＡ-Ｚ]/g, (match) => {
    return String.fromCharCode(match.charCodeAt(0) - 0xFEE0);
  });
};

// ショートカットキーマッピング（半角・全角両対応）
export const CHARACTER_SHORTCUTS: Record<string, string> = {
  // 半角数字
  "1": "野々原ゆずこ",
  "2": "日向縁", 
  "3": "櫟井唯",
  "4": "松本頼子",
  "5": "相川千穂",
  "6": "岡野佳",
  "7": "長谷川ふみ",
  "8": "お母さん",
  "9": "先生",
  "0": "生徒A",
  "q": "生徒B",
  "w": "その他",
  // 全角数字
  "１": "野々原ゆずこ",
  "２": "日向縁",
  "３": "櫟井唯", 
  "４": "松本頼子",
  "５": "相川千穂",
  "６": "岡野佳",
  "７": "長谷川ふみ",
  "８": "お母さん",
  "９": "先生",
  "０": "生徒A",
  "ｑ": "生徒B",
  "ｗ": "その他",
};

// ショートカットキーの逆引き
export const getShortcutForCharacter = (character: string): string | null => {
  const entry = Object.entries(CHARACTER_SHORTCUTS).find(([_, char]) => char === character);
  return entry ? entry[0] : null;
};