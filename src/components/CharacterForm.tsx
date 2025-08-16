import React, { useEffect, useState } from "react";
import { Character, ImageData } from "../types/app";
import { CHARACTER_OPTIONS, CHARACTER_SHORTCUTS, getShortcutForCharacter, toHalfWidth } from "../constants/characterList";

// キャラクターの色定義（パステル調の明るい色合い）- SerifFormと同じ
const CHARACTER_COLORS: Record<string, string> = {
  '野々原ゆずこ': '#FFD1DC',  // yuzuko - パステルピンク（薄い桜色）
  '日向縁': '#E6E6FA',        // yukari - パステル紫（ラベンダー）
  '櫟井唯': '#FFFACD',        // yui - パステル黄色（レモンシフォン）
  '松本頼子': '#FFDAB9',      // yoriko - パステルオレンジ（ピーチパフ）
  '相川千穂': '#F0FFF0',      // chiho - パステル緑（ハニーデュー）
  '岡野佳': '#F5DEB3',        // kei - パステル茶色（ウィート）
  '長谷川ふみ': '#F5F5F5',    // fumi - パステルグレー（ホワイトスモーク）
  'お母さん': '#FFF0F5',      // パステルローズ（ラベンダーブラッシュ）
  '先生': '#F0F8FF',          // パステルブルー（アリスブルー）
  '生徒A': '#F8F8FF',        // パステルラベンダー（ゴーストホワイト）
  '生徒B': '#F5FFFA',        // パステルミント（ミントクリーム）
  'その他': '#FDF5E6'         // パステルベージュ（オールドレース）
};

interface CharacterFormProps {
  characters: Character[];
  imageKey: string;
  onCharacterChange: (imageKey: string, index: number, field: keyof Character, value: string | boolean | number | [number, number]) => void;
  onCharacterSwap: (imageKey: string, index1: number, index2: number) => void;
  onRemoveCharacter: (imageKey: string, index: number) => void;
  onAddCharacter: (imageKey: string) => void;
}

const CharacterForm: React.FC<CharacterFormProps> = ({
  characters,
  imageKey,
  onCharacterChange,
  onCharacterSwap,
  onRemoveCharacter,
  onAddCharacter,
}) => {
  const [focusedInput, setFocusedInput] = useState<string | null>(null);
  const [showDropdown, setShowDropdown] = useState<Record<string, boolean>>({});

  // キャラクター名に基づいて背景色を取得する関数
  const getCharacterBackgroundColor = (characterName: string): string => {
    const trimmedName = characterName.trim();
    return CHARACTER_COLORS[trimmedName] || '#FFFFFF';
  };

  // テキストの可読性のために文字色を決める関数（パステル調なので全て黒文字）
  const getTextColor = (backgroundColor: string): string => {
    // パステル調の明るい色合いなので、全て黒文字で統一
    return '#000000';
  };

  // キーボードショートカットの処理
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (focusedInput) {
        const key = e.key;
        const halfWidthKey = toHalfWidth(key);
        const charName = CHARACTER_SHORTCUTS[key] || CHARACTER_SHORTCUTS[halfWidthKey];
        
        if (charName) {
          e.preventDefault();
          const [imageKey, indexStr] = focusedInput.split('-');
          const index = parseInt(indexStr);
          onCharacterChange(imageKey, index, "character", charName);
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [focusedInput, onCharacterChange]);
  return (
    <div className="character-grid">
      <button className="btn btn-sm" onClick={() => onAddCharacter(imageKey)}>
        キャラクター追加
      </button>
      {characters.slice().reverse().map((character, reversedIndex) => {
        const index = characters.length - 1 - reversedIndex;
        return (
        <div key={index} className="character-card">
          <h4>人物{index + 1}</h4>
          <div className="character-row">
            <div style={{ position: 'relative' }}>
              <input
                className="form-control input-sm"
                name="character"
                value={character.character}
                onChange={(e) =>
                  onCharacterChange(imageKey, index, "character", e.target.value)
                }
                onFocus={() => {
                  setFocusedInput(`${imageKey}-${index}`);
                  setShowDropdown({ ...showDropdown, [`${imageKey}-${index}`]: true });
                }}
                onBlur={() => {
                  setTimeout(() => {
                    setFocusedInput(null);
                    setShowDropdown({ ...showDropdown, [`${imageKey}-${index}`]: false });
                  }, 200);
                }}
                placeholder="名前 (数字キーで選択)"
                style={{
                  backgroundColor: getCharacterBackgroundColor(character.character),
                  color: getTextColor(getCharacterBackgroundColor(character.character)),
                  fontWeight: character.character && character.character.trim() !== '' ? 'bold' : 'normal'
                }}
              />
              {showDropdown[`${imageKey}-${index}`] && (
                <div style={{
                  position: 'absolute',
                  top: '100%',
                  left: 0,
                  right: 0,
                  backgroundColor: 'white',
                  border: '1px solid #ccc',
                  borderRadius: '3px',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                  zIndex: 1000,
                  maxHeight: '200px',
                  overflowY: 'auto',
                  fontSize: '11px'
                }}>
                  {CHARACTER_OPTIONS.map((option) => {
                    const shortcut = getShortcutForCharacter(option);
                    const optionBgColor = getCharacterBackgroundColor(option);
                    const optionTextColor = getTextColor(optionBgColor);
                    const isSelected = character.character === option;
                    return (
                      <div
                        key={option}
                        style={{
                          padding: '4px 8px',
                          cursor: 'pointer',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          backgroundColor: isSelected ? optionBgColor : 'white',
                          color: isSelected ? optionTextColor : '#000000',
                          fontWeight: isSelected ? 'bold' : 'normal',
                          border: isSelected ? '2px solid #333' : 'none'
                        }}
                        onMouseDown={(e) => {
                          e.preventDefault();
                          onCharacterChange(imageKey, index, "character", option);
                        }}
                        onMouseEnter={(e) => {
                          if (!isSelected) {
                            e.currentTarget.style.backgroundColor = '#e8e8e8';
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (!isSelected) {
                            e.currentTarget.style.backgroundColor = 'white';
                          }
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center' }}>
                          <div 
                            style={{
                              width: '12px',
                              height: '12px',
                              backgroundColor: optionBgColor,
                              marginRight: '6px',
                              border: '1px solid #ccc',
                              borderRadius: '2px'
                            }}
                          />
                          <span>{option}</span>
                        </div>
                        {shortcut && (
                          <span style={{
                            fontSize: '10px',
                            color: '#666',
                            marginLeft: '8px',
                            fontWeight: 'bold'
                          }}>({shortcut})</span>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
            <input
              className="form-control input-sm"
              name="faceDirection"
              value={character.faceDirection}
              onChange={(e) =>
                onCharacterChange(imageKey, index, "faceDirection", e.target.value)
              }
              placeholder="顔の向き"
            />
            <input
              className="form-control input-sm"
              name="coordinate"
              value={character.coordinate ? 
                `${character.coordinate[0].toFixed(2)}, ${character.coordinate[1].toFixed(2)}` : 
                "0.00, 0.00"}
              onChange={(e) => {
                const coords = e.target.value.split(",").map(v => parseFloat(v.trim()) || 0);
                onCharacterChange(imageKey, index, "coordinate", [coords[0] || 0, coords[1] || 0]);
              }}
              placeholder="座標 (x, y)"
            />
          </div>
          <div className="character-row">
            <input
              className="form-control input-sm"
              name="shotType"
              value={character.shotType}
              onChange={(e) =>
                onCharacterChange(imageKey, index, "shotType", e.target.value)
              }
              placeholder="ショットタイプ"
            />
            <input
              className="form-control input-sm"
              name="characterSize"
              value={character.characterSize}
              onChange={(e) =>
                onCharacterChange(imageKey, index, "characterSize", e.target.value)
              }
              placeholder="キャラクターサイズ"
            />
            <input
              className="form-control input-sm"
              name="expression"
              value={character.expression}
              onChange={(e) =>
                onCharacterChange(imageKey, index, "expression", e.target.value)
              }
              placeholder="表情"
            />
          </div>
          <div className="character-row">
            <input
              className="form-control input-sm"
              name="clothing"
              value={character.clothing}
              onChange={(e) =>
                onCharacterChange(imageKey, index, "clothing", e.target.value)
              }
              placeholder="服装"
            />
            <div className="checkbox-label">
              <input
                type="checkbox"
                name="isVisible"
                checked={!!character.isVisible}
                onChange={(e) =>
                  onCharacterChange(imageKey, index, "isVisible", e.target.checked)
                }
              />
              表示
            </div>
            <input
              className="form-control input-sm"
              name="position"
              value={character.position}
              onChange={(e) =>
                onCharacterChange(imageKey, index, "position", e.target.value)
              }
              placeholder="検出信頼度"
            />
          </div>
          <div className="button-row">
            {index < characters.length - 1 && (
              <button
                className="btn btn-sm"
                onClick={() => onCharacterSwap(imageKey, index, index + 1)}
              >
                ←
              </button>
            )}
            <button className="btn btn-sm" onClick={() => onRemoveCharacter(imageKey, index)}>
              削除
            </button>
            {index > 0 && (
              <button
                className="btn btn-sm"
                onClick={() => onCharacterSwap(imageKey, index, index - 1)}
              >
                →
              </button>
            )}
          </div>
        </div>
        );
      })}
    </div>
  );
};

export default CharacterForm;