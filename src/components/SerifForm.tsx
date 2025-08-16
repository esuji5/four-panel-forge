import React, { useEffect, useState } from "react";
import { Serif, TailShapeClassification } from "../types/app";
import { CHARACTER_OPTIONS, CHARACTER_SHORTCUTS, getShortcutForCharacter, toHalfWidth } from "../constants/characterList";
import { getBalloonTypeDisplayName, inferSpeakerFromBalloonType } from "../utils/balloonDetection";

// キャラクターの色定義（パステル調の明るい色合い）- CharacterFormと同じ
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

interface SerifFormProps {
  serifs: Serif[];
  imageKey: string;
  onSerifChange: (imageKey: string, serifIndex: number, field: string, value: string | number | [number, number]) => void;
  onAddSerif: (imageKey: string) => void;
  onRemoveSerif: (imageKey: string, serifIndex: number) => void;
  onSerifSwap?: (imageKey: string, index1: number, index2: number) => void;
}

const SerifForm: React.FC<SerifFormProps> = ({
  serifs,
  imageKey,
  onSerifChange,
  onAddSerif,
  onRemoveSerif,
  onSerifSwap,
}) => {
  const [focusedInput, setFocusedInput] = useState<string | null>(null);
  const [showDropdown, setShowDropdown] = useState<Record<string, boolean>>({});

  // セリフデータの監視とオフセリフの話者クリア（AI検出時のみ）
  // Note: この自動クリア機能はAI検出時に必要な場合のみ有効化
  // 通常の手動編集では無効化

  // キャラクター名に基づいて背景色を取得する関数
  const getCharacterBackgroundColor = (characterName: string): string => {
    const trimmedName = characterName.trim();
    return CHARACTER_COLORS[trimmedName] || '#FFFFFF';
  };

  // キーボードショートカットの処理（全角・半角両対応）
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!focusedInput) return;
      
      // 入力されたキーを確認（全角・半角両方をチェック）
      const key = e.key;
      const halfWidthKey = toHalfWidth(key);
      
      // どちらかでマッチするかチェック
      const charName = CHARACTER_SHORTCUTS[key] || CHARACTER_SHORTCUTS[halfWidthKey];
      
      if (charName) {
        e.preventDefault();
        const [imageKey, serifIndexStr] = focusedInput.split('-serif-');
        const serifIndex = parseInt(serifIndexStr);
        
        // char_A, char_B などの形式に変換
        let charId = '';
        if (charName === "野々原ゆずこ") charId = 'char_A';
        else if (charName === "日向縁") charId = 'char_B';
        else if (charName === "櫟井唯") charId = 'char_C';
        else if (charName === "松本頼子") charId = 'char_D';
        else if (charName === "相川千穂") charId = 'char_E';
        else if (charName === "岡野佳") charId = 'char_F';
        else if (charName === "長谷川ふみ") charId = 'char_G';
        
        if (charId) {
          onSerifChange(imageKey, serifIndex, 'speakerCharacterId', charId);
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [focusedInput, onSerifChange]);

  // charIdからキャラクター名を取得
  const getCharacterName = (charId: string): string => {
    const charMap: Record<string, string> = {
      'char_A': '野々原ゆずこ',
      'char_B': '日向縁',
      'char_C': '櫟井唯',
      'char_D': '松本頼子',
      'char_E': '相川千穂',
      'char_F': '岡野佳',
      'char_G': '長谷川ふみ',
    };
    return charMap[charId] || '';
  };
  return (
    <div className="serifs-section mt-3">
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        marginBottom: '8px' 
      }}>
        <h4 style={{ margin: 0, fontSize: '14px' }}>セリフ一覧</h4>
        <button className="btn btn-sm" onClick={() => onAddSerif(imageKey)}>追加</button>
      </div>
      {serifs.map((serif, serifIndex) => (
        <div key={serifIndex} className="serif-edit-card" style={{ 
          border: '1px solid #ddd', 
          borderRadius: '4px', 
          padding: '8px', 
          marginBottom: '8px',
          backgroundColor: '#f9f9f9'
        }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            marginBottom: '4px' 
          }}>
            <span style={{ fontSize: '12px', fontWeight: 'bold' }}>#{serifIndex + 1}</span>
            <div style={{ display: 'flex', gap: '4px' }}>
              {onSerifSwap && serifIndex > 0 && (
                <button 
                  className="btn btn-sm" 
                  onClick={() => onSerifSwap(imageKey, serifIndex, serifIndex - 1)}
                  style={{ padding: '2px 6px', fontSize: '11px' }}
                  title="上へ移動"
                >
                  ↑
                </button>
              )}
              {onSerifSwap && serifIndex < serifs.length - 1 && (
                <button 
                  className="btn btn-sm" 
                  onClick={() => onSerifSwap(imageKey, serifIndex, serifIndex + 1)}
                  style={{ padding: '2px 6px', fontSize: '11px' }}
                  title="下へ移動"
                >
                  ↓
                </button>
              )}
              <button 
                className="btn btn-sm" 
                onClick={() => onRemoveSerif(imageKey, serifIndex)}
                style={{ marginLeft: 'auto', padding: '2px 6px' }}
              >
                削除
              </button>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '4px', marginBottom: '4px' }}>
            <textarea
              className="form-control input-sm"
              value={serif.text}
              onChange={(e) => onSerifChange(imageKey, serifIndex, 'text', e.target.value)}
              placeholder="セリフテキスト"
              style={{ fontSize: '12px', minHeight: '24px', resize: 'vertical', lineHeight: '1.2' }}
              rows={1}
            />
            <div style={{ position: 'relative' }}>
              <input
                className="form-control input-sm"
                value={getCharacterName(serif.speakerCharacterId || '')}
                onChange={(e) => {
                  // 手動入力は無効化
                }}
                onFocus={() => {
                  setFocusedInput(`${imageKey}-serif-${serifIndex}`);
                  setShowDropdown({ ...showDropdown, [`${imageKey}-serif-${serifIndex}`]: true });
                }}
                onBlur={() => {
                  setTimeout(() => {
                    setFocusedInput(null);
                    setShowDropdown({ ...showDropdown, [`${imageKey}-serif-${serifIndex}`]: false });
                  }, 200);
                }}
                placeholder="話者 (数字キーで選択)"
                readOnly
                style={{ 
                  fontSize: '12px', 
                  cursor: 'pointer',
                  backgroundColor: getCharacterBackgroundColor(getCharacterName(serif.speakerCharacterId || '')),
                  color: '#000000',
                  fontWeight: getCharacterName(serif.speakerCharacterId || '').trim() !== '' ? 'bold' : 'normal'
                }}
              />
              {showDropdown[`${imageKey}-serif-${serifIndex}`] && (
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
                  <div
                    style={{
                      padding: '4px 8px',
                      cursor: 'pointer',
                      backgroundColor: !serif.speakerCharacterId ? '#f0f0f0' : 'white'
                    }}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      onSerifChange(imageKey, serifIndex, 'speakerCharacterId', '');
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = '#e8e8e8';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = !serif.speakerCharacterId ? '#f0f0f0' : 'white';
                    }}
                  >
                    <span>話者不明</span>
                  </div>
                  {CHARACTER_OPTIONS.slice(0, 7).map((option, idx) => {
                    const charIds = ['char_A', 'char_B', 'char_C', 'char_D', 'char_E', 'char_F', 'char_G'];
                    const charId = charIds[idx];
                    const shortcut = getShortcutForCharacter(option);
                    const optionBgColor = getCharacterBackgroundColor(option);
                    const isSelected = serif.speakerCharacterId === charId;
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
                          color: '#000000',
                          fontWeight: isSelected ? 'bold' : 'normal',
                          border: isSelected ? '2px solid #333' : 'none'
                        }}
                        onMouseDown={(e) => {
                          e.preventDefault();
                          onSerifChange(imageKey, serifIndex, 'speakerCharacterId', charId);
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
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '4px', marginBottom: '4px' }}>
            <select
              className="form-control input-sm"
              value={serif.type}
              onChange={(e) => {
                const newType = e.target.value;
                onSerifChange(imageKey, serifIndex, 'type', newType);
                
                // 吹き出しタイプに応じて話者を自動設定またはクリア
                const inferredSpeaker = inferSpeakerFromBalloonType(newType);
                if (inferredSpeaker) {
                  // キャラクター専用吹き出しの場合、話者を自動設定
                  const charMap: Record<string, string> = {
                    '野々原ゆずこ': 'char_A',
                    '日向縁': 'char_B',
                    '櫟井唯': 'char_C',
                    '松本頼子': 'char_D',
                    '相川千穂': 'char_E',
                    '岡野佳': 'char_F',
                    '長谷川ふみ': 'char_G',
                  };
                  const charId = charMap[inferredSpeaker];
                  if (charId) {
                    onSerifChange(imageKey, serifIndex, 'speakerCharacterId', charId);
                  }
                } 
                // オフセリフの場合も話者は保持する（クリアしない）
              }}
              style={{ fontSize: '12px' }}
              title={getBalloonTypeDisplayName(serif.type)}
            >
              <option value="speech_bubble">吹き出し</option>
              <option value="thought_bubble">思考</option>
              <option value="exclamation_bubble">感嘆</option>
              <option value="combined_bubble">結合</option>
              <option value="offserif_bubble">オフセリフ</option>
              <option value="inner_voice_bubble">内なる声</option>
              <option value="narration_box">ナレーション</option>
              <option value="chractor_bubble_yuzuko">ゆずこ専用</option>
              <option value="chractor_bubble_yukari">ゆかり専用</option>
              <option value="chractor_bubble_yui">唯専用</option>
              <option value="chractor_bubble_yoriko">よりこ専用</option>
              <option value="chractor_bubble_chiho">千穂専用</option>
              <option value="chractor_bubble_kei">恵専用</option>
              <option value="chractor_bubble_fumi">史専用</option>
              <option value="speechBubble">吹き出し（旧）</option>
              <option value="outsideBubble">吹き出し外（旧）</option>
              <option value="narration">ナレーション（旧）</option>
            </select>
            <input
              className="form-control input-sm"
              value={serif.coordinate ? `${serif.coordinate[0].toFixed(2)}, ${serif.coordinate[1].toFixed(2)}` : '0.00, 0.00'}
              onChange={(e) => {
                const coords = e.target.value.split(',').map(v => parseFloat(v.trim()) || 0);
                onSerifChange(imageKey, serifIndex, 'coordinate', [coords[0] || 0, coords[1] || 0] as [number, number]);
              }}
              placeholder="座標 (x, y)"
              style={{ fontSize: '12px' }}
            />
            <input
              className="form-control input-sm"
              type="number"
              value={serif.readingOrderIndex}
              onChange={(e) => onSerifChange(imageKey, serifIndex, 'readingOrderIndex', parseInt(e.target.value) || 0)}
              placeholder="読順"
              style={{ fontSize: '12px' }}
            />
          </div>
          
          {/* 尻尾形状分類結果の表示 */}
          {serif.tail_shape_classification && (
            <div style={{ 
              marginTop: '8px', 
              padding: '10px', 
              backgroundColor: '#e8f5e8', 
              borderRadius: '6px', 
              border: '2px solid #28a745' 
            }}>
              <div style={{ 
                fontSize: '12px', 
                fontWeight: 'bold', 
                color: '#155724', 
                marginBottom: '6px',
                display: 'flex',
                alignItems: 'center'
              }}>
                🎯 尻尾形状分類結果
              </div>
              {serif.tail_shape_classification.error ? (
                <div style={{ 
                  fontSize: '11px', 
                  color: '#dc3545',
                  backgroundColor: '#f8d7da',
                  padding: '4px',
                  borderRadius: '3px'
                }}>
                  エラー: {serif.tail_shape_classification.error}
                </div>
              ) : (
                <>
                  <div style={{ 
                    fontSize: '14px', 
                    fontWeight: 'bold', 
                    color: '#155724',
                    marginBottom: '6px',
                    backgroundColor: 'white',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    border: '1px solid #c3e6cb'
                  }}>
                    {serif.tail_shape_classification.predicted_category} 
                    <span style={{ 
                      fontSize: '12px', 
                      fontWeight: 'normal', 
                      color: '#6c757d',
                      marginLeft: '8px'
                    }}>
                      ({(serif.tail_shape_classification.confidence * 100).toFixed(1)}%)
                    </span>
                  </div>
                  {serif.tail_shape_classification.top3_predictions && (
                    <div style={{ 
                      fontSize: '11px', 
                      color: '#495057',
                      lineHeight: '1.4'
                    }}>
                      <strong>上位3位:</strong><br/>
                      {serif.tail_shape_classification.top3_predictions.map((pred, i) => 
                        `${i + 1}位: ${pred.category} (${(pred.confidence * 100).toFixed(1)}%)`
                      ).join(' / ')}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default SerifForm;