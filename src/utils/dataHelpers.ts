/**
 * データ変換とヘルパー関数のユーティリティ
 */
import { Character, PanelData, ImageData } from '../types/app';

// 空のキャラクターデータを作成するヘルパー関数
export const createEmptyCharacter = (): Character => ({
  character: "",
  faceDirection: "",
  position: "",
  shotType: "",
  characterSize: "",
  expression: "",
  clothing: "",
  isVisible: true,
  coordinate: [0, 0],
});

// キャラクターデータが実質的に空かどうかを判定するヘルパー関数
export const isEmptyCharacterData = (char: Character): boolean => {
  return (
    (!char.character || char.character.trim() === "") &&
    (!char.faceDirection || char.faceDirection.trim() === "") &&
    (!char.position || char.position.trim() === "") &&
    (!char.shotType || char.shotType.trim() === "") &&
    (!char.characterSize || char.characterSize.trim() === "") &&
    (!char.expression || char.expression.trim() === "") &&
    (!char.clothing || char.clothing.trim() === "")
  );
};

// 空のパネルデータを作成するヘルパー関数
export const createEmptyPanelData = (komaPath: string): PanelData => ({
  komaPath,
  characters: [createEmptyCharacter()],
  sceneData: {
    scene: "",
    location: "",
    backgroundEffects: "",
    cameraAngle: "",
    framing: "",
  },
  serifs: [],
  charactersNum: 0,
  serifsNum: 0,
});

// 旧データを新形式に変換するヘルパー関数
export const convertOldDataToNew = (data: any): ImageData => {
  // 名前をchar_X形式に変換するマッピング
  const nameToCharIdMap: { [key: string]: string } = {
    '野々原ゆずこ': 'char_A',
    '日向縁': 'char_B',
    '櫟井唯': 'char_C',
    '松本頼子': 'char_D',
    '相川千穂': 'char_E',
    '岡野佳': 'char_F',
    '長谷川ふみ': 'char_G',
  };
  
  const convertPanel = (panel: any): PanelData => {
    // パネルデータの変換処理
    if (!panel) {
      return createEmptyPanelData("");
    }

    // charactersの処理
    const characters = Array.isArray(panel.characters) 
      ? panel.characters.map((char: any) => ({
          ...createEmptyCharacter(),
          ...char,
          coordinate: char.coordinate || [0, 0],
        }))
      : [createEmptyCharacter()];

    // serifsの処理
    const serifs = Array.isArray(panel.serifs)
      ? panel.serifs.map((serif: any) => {
          // speakerCharacterIdの変換処理
          let convertedSpeakerId = serif.speakerCharacterId;
          if (typeof convertedSpeakerId === 'string') {
            // 日本語名の場合はchar_X形式に変換
            if (nameToCharIdMap[convertedSpeakerId]) {
              convertedSpeakerId = nameToCharIdMap[convertedSpeakerId];
            }
          }

          return {
            ...serif,
            speakerCharacterId: convertedSpeakerId,
            coordinate: serif.coordinate || [0, 0],
            readingOrderIndex: serif.readingOrderIndex || 0,
          };
        })
      : [];

    // sceneDataの処理
    const sceneData = panel.sceneData || panel.scene || {
      scene: "",
      location: "",
      backgroundEffects: "",
      cameraAngle: "",
      framing: "",
    };

    // description → scene への変換
    if (sceneData.description && !sceneData.scene) {
      sceneData.scene = sceneData.description;
    }

    return {
      komaPath: panel.komaPath || panel.imagePath || "",
      characters,
      sceneData,
      serifs,
      charactersNum: characters.length,
      serifsNum: serifs.length,
    };
  };

  // データ変換実行
  return {
    image1: convertPanel(data.image1 || data.panel1),
    image2: convertPanel(data.image2 || data.panel2),
    image3: convertPanel(data.image3 || data.panel3),
    image4: convertPanel(data.image4 || data.panel4),
  };
};

// キャラクター名からchar_ID形式への変換マップ
export const getCharacterIdMap = (): { [key: string]: string } => ({
  '野々原ゆずこ': 'char_A',
  '日向縁': 'char_B', 
  '櫟井唯': 'char_C',
  '松本頼子': 'char_D',
  '相川千穂': 'char_E',
  '岡野佳': 'char_F',
  '長谷川ふみ': 'char_G',
});

// char_ID形式からキャラクター名への変換マップ
export const getCharacterNameMap = (): { [key: string]: string } => ({
  'char_A': '野々原ゆずこ',
  'char_B': '日向縁',
  'char_C': '櫟井唯',
  'char_D': '松本頼子',
  'char_E': '相川千穂',
  'char_F': '岡野佳',
  'char_G': '長谷川ふみ',
});

// 英語名からchar_ID形式への変換マップ
export const getEnglishToCharIdMap = (): { [key: string]: string } => ({
  'yuzuko': 'char_A',
  'yukari': 'char_B',
  'yui': 'char_C',
  'yoriko': 'char_D',
  'chiho': 'char_E',
  'kei': 'char_F',
  'fumi': 'char_G'
});

// 英語名から日本語名へのマッピング
export const getEnglishToJapaneseMap = (): { [key: string]: string } => ({
  'yuzuko': '野々原ゆずこ',
  'yukari': '日向縁',
  'yui': '櫟井唯',
  'yoriko': '松本頼子',
  'chiho': '相川千穂',
  'kei': '岡野佳',
  'fumi': '長谷川ふみ',
  'unknown': '不明'
});

// データが空かどうかをチェック
export const isEmptyImageData = (imageData: ImageData): boolean => {
  return Object.values(imageData).every(panel => 
    (!panel.characters || panel.characters.every((char: Character) => isEmptyCharacterData(char))) &&
    (!panel.serifs || panel.serifs.length === 0) &&
    (!panel.sceneData || (!panel.sceneData.scene && !panel.sceneData.location))
  );
};