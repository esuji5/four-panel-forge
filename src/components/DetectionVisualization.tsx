import React from 'react';
import { BalloonDetection } from '../utils/balloonDetection';

interface DetectionVisualizationProps {
  balloonDetections?: BalloonDetection[];
  characterDetections?: any[];
  imageWidth: number;
  imageHeight: number;
  naturalImageWidth?: number;
  naturalImageHeight?: number;
  className?: string;
}

const DetectionVisualization: React.FC<DetectionVisualizationProps> = ({
  balloonDetections = [],
  characterDetections = [],
  imageWidth,
  imageHeight,
  naturalImageWidth,
  naturalImageHeight,
  className = ''
}) => {
  if (balloonDetections.length === 0 && characterDetections.length === 0) {
    return null;
  }

  // 座標変換関数（元画像サイズから表示サイズへ）
  const scaleX = naturalImageWidth && naturalImageWidth > 0 ? imageWidth / naturalImageWidth : 1;
  const scaleY = naturalImageHeight && naturalImageHeight > 0 ? imageHeight / naturalImageHeight : 1;

  const transformX = (x: number) => x * scaleX;
  const transformY = (y: number) => y * scaleY;

  // 吹き出し検出結果の色分け
  const getBalloonColor = (type?: string) => {
    switch(type) {
      case 'speechBubble': return '#4CAF50';  // 緑
      case 'thoughtBubble': return '#2196F3'; // 青
      case 'exclamationBubble': return '#FF9800'; // オレンジ
      case 'yuzuko_bubble': return '#E91E63'; // ピンク
      default: return '#9E9E9E'; // グレー
    }
  };

  // キャラクター検出結果の色分け
  const getCharacterColor = (characterName?: string) => {
    const colorMap: {[key: string]: string} = {
      'yuzuko': '#FF6B6B',    // 赤
      'yukari': '#4ECDC4',    // 青緑
      'yui': '#45B7D1',       // 青
      'yoriko': '#96CEB4',    // 緑
      'chiho': '#FFEAA7',     // 黄
      'kei': '#DDA0DD',       // 紫
      'fumi': '#FFB6C1'       // ピンク
    };
    return colorMap[characterName || ''] || '#FFA726';
  };

  return (
    <svg
      className={`detection-visualization ${className}`}
      width={imageWidth}
      height={imageHeight}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        pointerEvents: 'none',
        zIndex: 5
      }}
    >
      {/* 吹き出し検出結果 */}
      {balloonDetections.map((balloon, index) => (
        <g key={`balloon-${index}`} className="balloon-detection">
          {/* 吹き出しバウンディングボックス */}
          <rect
            x={transformX(balloon.boundingBox?.x1 || 0)}
            y={transformY(balloon.boundingBox?.y1 || 0)}
            width={transformX((balloon.boundingBox?.x2 || 0) - (balloon.boundingBox?.x1 || 0))}
            height={transformY((balloon.boundingBox?.y2 || 0) - (balloon.boundingBox?.y1 || 0))}
            fill="none"
            stroke={getBalloonColor(balloon.type)}
            strokeWidth="2"
            strokeDasharray="4,4"
            opacity="0.8"
          />
          
          {/* 吹き出しタイプラベル */}
          <text
            x={transformX(balloon.boundingBox?.x1 || 0) + 2}
            y={transformY(balloon.boundingBox?.y1 || 0) - 2}
            fontSize="10"
            fill={getBalloonColor(balloon.type)}
            fontWeight="bold"
            style={{ textShadow: '1px 1px 2px rgba(0,0,0,0.7)' }}
          >
            {balloon.type || 'balloon'} ({(balloon.confidence * 100).toFixed(0)}%)
          </text>

          {/* しっぽ検出結果 */}
          {balloon.tails && balloon.tails.length > 0 && balloon.tails.map((tail, tailIndex) => {
            const globalPosX = tail.globalPosition?.[0];
            const globalPosY = tail.globalPosition?.[1];
            if (globalPosX === undefined || globalPosY === undefined) return null;
            
            return (
              <g key={`tail-${tailIndex}`} className="tail-detection">
                <circle
                  cx={transformX(globalPosX * (naturalImageWidth || imageWidth))}
                  cy={transformY(globalPosY * (naturalImageHeight || imageHeight))}
                  r="3"
                  fill={getBalloonColor(balloon.type)}
                  stroke="white"
                  strokeWidth="1"
                  opacity="0.9"
                />
                <text
                  x={transformX(globalPosX * (naturalImageWidth || imageWidth)) + 5}
                  y={transformY(globalPosY * (naturalImageHeight || imageHeight)) - 5}
                  fontSize="8"
                  fill={getBalloonColor(balloon.type)}
                  fontWeight="bold"
                  style={{ textShadow: '1px 1px 2px rgba(0,0,0,0.7)' }}
                >
                  {tail.shape_category || 'tail'} ({((tail.shape_confidence || 0) * 100).toFixed(0)}%)
                </text>
              </g>
            );
          })}
        </g>
      ))}

      {/* 人物検出結果 */}
      {characterDetections.map((character, index) => (
        <g key={`character-${index}`} className="character-detection">
          {/* 人物バウンディングボックス */}
          <rect
            x={transformX(character.bbox?.[0] || 0)}
            y={transformY(character.bbox?.[1] || 0)}
            width={transformX((character.bbox?.[2] || 0) - (character.bbox?.[0] || 0))}
            height={transformY((character.bbox?.[3] || 0) - (character.bbox?.[1] || 0))}
            fill="none"
            stroke={getCharacterColor(character.characterName)}
            strokeWidth="3"
            opacity="0.8"
          />
          
          {/* キャラクター名ラベル */}
          <text
            x={transformX(character.bbox?.[0] || 0) + 2}
            y={transformY(character.bbox?.[1] || 0) - 2}
            fontSize="12"
            fill={getCharacterColor(character.characterName)}
            fontWeight="bold"
            style={{ textShadow: '1px 1px 2px rgba(0,0,0,0.8)' }}
          >
            {character.characterName} ({((character.confidence || 0) * 100).toFixed(0)}%)
          </text>
        </g>
      ))}
    </svg>
  );
};

export default DetectionVisualization;