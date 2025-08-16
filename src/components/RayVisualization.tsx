import React from 'react';
import { RayVisualizationData } from '../utils/speakerIdentification';

interface RayVisualizationProps {
  rayData: RayVisualizationData[];
  imageWidth: number;
  imageHeight: number;
  className?: string;
}

const RayVisualization: React.FC<RayVisualizationProps> = ({
  rayData,
  imageWidth,
  imageHeight,
  className = ''
}) => {
  if (!rayData || rayData.length === 0) {
    return null;
  }

  return (
    <svg
      className={`ray-visualization ${className}`}
      width={imageWidth}
      height={imageHeight}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        pointerEvents: 'none',
        zIndex: 10
      }}
    >
      {rayData.map((ray, index) => (
        <g key={index} className="ray-group">
          {/* キャラクターのバウンディングボックス */}
          {ray.characterBBox && (
            <rect
              x={ray.characterBBox.x1}
              y={ray.characterBBox.y1}
              width={ray.characterBBox.x2 - ray.characterBBox.x1}
              height={ray.characterBBox.y2 - ray.characterBBox.y1}
              fill="none"
              stroke={ray.intersectsCharacter ? "#00ff00" : "#ff6b6b"}
              strokeWidth="2"
              strokeDasharray="5,5"
              opacity="0.7"
            />
          )}
          
          {/* Ray線 */}
          <line
            x1={ray.rayOrigin.x}
            y1={ray.rayOrigin.y}
            x2={ray.rayEnd.x}
            y2={ray.rayEnd.y}
            stroke={ray.intersectsCharacter ? "#00ff00" : "#ff6b6b"}
            strokeWidth="2"
            opacity="0.8"
            markerEnd="url(#arrowhead)"
          />
          
          {/* Ray起点（しっぽ中心） */}
          <circle
            cx={ray.rayOrigin.x}
            cy={ray.rayOrigin.y}
            r="4"
            fill="#ffd93d"
            stroke="#333"
            strokeWidth="1"
          />
          
          {/* 最遠点 */}
          {ray.farthestPoint && (
            <circle
              cx={ray.farthestPoint.x}
              cy={ray.farthestPoint.y}
              r="3"
              fill="#ff6b6b"
              stroke="#333"
              strokeWidth="1"
            />
          )}
          
          {/* 交差点 */}
          {ray.intersectionPoint && (
            <>
              <circle
                cx={ray.intersectionPoint.x}
                cy={ray.intersectionPoint.y}
                r="5"
                fill="#00ff00"
                stroke="#333"
                strokeWidth="2"
              />
              <circle
                cx={ray.intersectionPoint.x}
                cy={ray.intersectionPoint.y}
                r="8"
                fill="none"
                stroke="#00ff00"
                strokeWidth="2"
                opacity="0.6"
              />
            </>
          )}
          
          {/* Ray方向ベクトルの小さな矢印 */}
          <line
            x1={ray.rayOrigin.x}
            y1={ray.rayOrigin.y}
            x2={ray.rayOrigin.x + ray.rayDirection.x * 50}
            y2={ray.rayOrigin.y + ray.rayDirection.y * 50}
            stroke="#ffd93d"
            strokeWidth="3"
            opacity="0.9"
            markerEnd="url(#smallArrowhead)"
          />
        </g>
      ))}
      
      {/* 矢印マーカー定義 */}
      <defs>
        <marker
          id="arrowhead"
          markerWidth="10"
          markerHeight="7"
          refX="9"
          refY="3.5"
          orient="auto"
        >
          <polygon
            points="0 0, 10 3.5, 0 7"
            fill="#666"
            opacity="0.8"
          />
        </marker>
        
        <marker
          id="smallArrowhead"
          markerWidth="8"
          markerHeight="6"
          refX="7"
          refY="3"
          orient="auto"
        >
          <polygon
            points="0 0, 8 3, 0 6"
            fill="#ffd93d"
            opacity="0.9"
          />
        </marker>
      </defs>
    </svg>
  );
};

export default RayVisualization;