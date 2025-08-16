import React, { useState, useRef, useEffect } from "react";
import { PanelData, Serif, Character } from "../types/app";
import CharacterForm from "./CharacterForm";
import SerifForm from "./SerifForm";
import SceneForm from "./SceneForm";
import { getImagePath } from "../config";
import RayVisualization from "./RayVisualization";
import DetectionVisualization from "./DetectionVisualization";
import { RayVisualizationData } from "../utils/speakerIdentification";
import { BalloonDetection } from "../utils/balloonDetection";

interface PanelEditorProps {
  panelData: PanelData;
  panelNumber: number;
  imageKey: string;
  imagePath: string;
  isBlurred?: boolean;
  isCurrentTitlePage?: boolean;
  currentPageNumber?: string;
  currentImage?: string;
  isAnalyzing?: boolean;
  isYoloDetecting?: boolean;
  rayVisualizationData?: RayVisualizationData[];
  balloonDetectionData?: BalloonDetection[];
  characterDetectionData?: any[];
  showDetectionVisualization?: boolean;
  onCharacterChange: (imageKey: string, index: number, field: keyof Character, value: string | boolean | number | [number, number]) => void;
  onCharacterSwap: (imageKey: string, index1: number, index2: number) => void;
  onRemoveCharacter: (imageKey: string, index: number) => void;
  onAddCharacter: (imageKey: string) => void;
  onSerifChange: (imageKey: string, serifIndex: number, field: string, value: string | number | [number, number]) => void;
  onAddSerif: (imageKey: string) => void;
  onRemoveSerif: (imageKey: string, serifIndex: number) => void;
  onSerifSwap?: (imageKey: string, index1: number, index2: number) => void;
  onSceneChange: (imageKey: string, field: string, value: string) => void;
  onYoloDetect?: (imageKey: string, visualize?: boolean) => void;
}

const PanelEditor: React.FC<PanelEditorProps> = ({
  panelData,
  panelNumber,
  imageKey,
  imagePath,
  isBlurred,
  isCurrentTitlePage,
  currentPageNumber,
  currentImage,
  isAnalyzing,
  isYoloDetecting,
  rayVisualizationData,
  balloonDetectionData,
  characterDetectionData,
  showDetectionVisualization,
  onCharacterChange,
  onCharacterSwap,
  onRemoveCharacter,
  onAddCharacter,
  onSerifChange,
  onAddSerif,
  onRemoveSerif,
  onSerifSwap,
  onSceneChange,
  onYoloDetect,
}) => {
  const imageRef = useRef<HTMLImageElement>(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [naturalImageSize, setNaturalImageSize] = useState({ width: 0, height: 0 });

  // 画像サイズを取得（表示サイズと自然サイズの両方）
  useEffect(() => {
    const img = imageRef.current;
    if (img && img.complete) {
      setImageSize({ width: img.offsetWidth, height: img.offsetHeight });
      setNaturalImageSize({ width: img.naturalWidth, height: img.naturalHeight });
    }
  }, [imagePath]);

  const handleImageLoad = () => {
    const img = imageRef.current;
    if (img) {
      setImageSize({ width: img.offsetWidth, height: img.offsetHeight });
      setNaturalImageSize({ width: img.naturalWidth, height: img.naturalHeight });
    }
  };
  return (
    <div className="koma-container">
      <div className="koma-image-section">
        <div className="koma-image-wrapper" style={{ position: 'relative' }}>
          <img
            ref={imageRef}
            src={getImagePath(imagePath)}
            alt={`コマ${panelNumber}`}
            className={isBlurred ? "blur" : ""}
            onLoad={handleImageLoad}
            onError={(e) => {
              console.error(`画像読み込みエラー: ${imagePath}`);
              e.currentTarget.style.display = 'none';
            }}
          />
          {/* 検出結果可視化オーバーレイ */}
          {showDetectionVisualization && imageSize.width > 0 && imageSize.height > 0 && naturalImageSize.width > 0 && naturalImageSize.height > 0 && (
            <DetectionVisualization
              balloonDetections={balloonDetectionData}
              characterDetections={characterDetectionData}
              imageWidth={imageSize.width}
              imageHeight={imageSize.height}
              naturalImageWidth={naturalImageSize.width}
              naturalImageHeight={naturalImageSize.height}
              className="panel-detection-visualization"
            />
          )}
          {/* Ray可視化オーバーレイ */}
          {(() => {
            // デバッグログ
            if (showDetectionVisualization) {
              console.log(`📊 Panel ${panelNumber} Ray表示条件:`, {
                showDetectionVisualization,
                rayVisualizationData,
                rayDataLength: rayVisualizationData?.length || 0,
                imageSize,
                shouldRender: showDetectionVisualization && rayVisualizationData && imageSize.width > 0 && imageSize.height > 0
              });
            }
            
            return showDetectionVisualization && rayVisualizationData && imageSize.width > 0 && imageSize.height > 0 ? (
              <RayVisualization
                rayData={rayVisualizationData}
                imageWidth={imageSize.width}
                imageHeight={imageSize.height}
                className="panel-ray-visualization"
              />
            ) : null;
          })()}
          <div className="koma-number">
            {(() => {
              if (isCurrentTitlePage) {
                return `${panelNumber}コマ目 (${currentPageNumber}-${[3,4,7,8][panelNumber-1]})`;
              } else {
                const isPage1 = currentImage?.includes('-1');
                const komaNumbers = isPage1 ? [1, 2, 3, 4] : [5, 6, 7, 8];
                return `${panelNumber}コマ目 (${currentPageNumber}-${komaNumbers[panelNumber-1]})`;
              }
            })()}
          </div>
        </div>
        <div className="panel-buttons">
          {onYoloDetect && (
            <>
              <button 
                className="btn btn-sm btn-yolo" 
                onClick={() => onYoloDetect(imageKey)}
                disabled={isYoloDetecting}
                style={{ marginLeft: '5px', backgroundColor: '#4CAF50', color: 'white' }}
              >
                {isYoloDetecting ? "検出中..." : "🎯 AI検出"}
              </button>
              <button 
                className="btn btn-sm btn-yolo-vis" 
                onClick={() => onYoloDetect(imageKey, true)}
                disabled={isYoloDetecting}
                style={{ marginLeft: '5px', backgroundColor: '#2196F3', color: 'white' }}
                title="検出結果を画像で表示"
              >
                {isYoloDetecting ? "検出中..." : "📸 結果表示"}
              </button>
            </>
          )}
        </div>
      </div>
      <div className="koma-data-form">
        <CharacterForm
          characters={panelData.characters}
          imageKey={imageKey}
          onCharacterChange={onCharacterChange}
          onCharacterSwap={onCharacterSwap}
          onRemoveCharacter={onRemoveCharacter}
          onAddCharacter={onAddCharacter}
        />
        
        <SceneForm
          sceneData={panelData.sceneData}
          imageKey={imageKey}
          onSceneChange={onSceneChange}
        />
        
        <SerifForm
          serifs={panelData.serifs || []}
          imageKey={imageKey}
          onSerifChange={onSerifChange}
          onAddSerif={onAddSerif}
          onRemoveSerif={onRemoveSerif}
          onSerifSwap={onSerifSwap}
        />
      </div>
    </div>
  );
};

export default PanelEditor;