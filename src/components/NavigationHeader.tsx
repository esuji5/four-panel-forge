import React from "react";
import type { PromptType } from "../types/manga";

interface NavigationHeaderProps {
  currentIndex: number;
  totalImages: number;
  currentImage: string;
  isCurrentTitlePage: boolean;
  fourPanelPromptType: PromptType;
  isFourPanelAnalyzing: boolean;
  isYoloDetecting?: boolean;
  useGroundTruth: boolean;
  groundTruthIndex: number | null;
  showDetectionVisualization?: boolean;
  onPrevious: () => void;
  onNext: () => void;
  onIndexChange?: (index: number) => void;
  onPromptTypeChange: (type: PromptType) => void;
  onFourPanelAnalyzeAPI?: () => Promise<void>;
  onFourPanelAnalyzeImprovedAPI?: () => Promise<void>;
  onFourPanelYoloDetect?: () => Promise<void>;
  onBalloonCropClassify?: () => Promise<void>;
  isBalloonProcessing?: boolean;
  onGroundTruthToggle: (checked: boolean) => void;
  onGroundTruthIndexChange: (index: number | null) => void;
  onToggleDetectionVisualization?: (show: boolean) => void;
  onClearFourPanelData?: () => void;
}

const NavigationHeader: React.FC<NavigationHeaderProps> = ({
  currentIndex,
  totalImages,
  currentImage,
  isCurrentTitlePage,
  fourPanelPromptType,
  isFourPanelAnalyzing,
  isYoloDetecting,
  useGroundTruth,
  groundTruthIndex,
  showDetectionVisualization,
  onPrevious,
  onNext,
  onIndexChange,
  onPromptTypeChange,
  onFourPanelAnalyzeAPI,
  onFourPanelAnalyzeImprovedAPI,
  onFourPanelYoloDetect,
  onGroundTruthToggle,
  onGroundTruthIndexChange,
  onToggleDetectionVisualization,
  onClearFourPanelData,
}) => {
  return (
    <div className="header">
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
        <button 
          className="btn btn-outline" 
          onClick={onPrevious} 
          disabled={currentIndex === 0}
        >
          ◀前
        </button>
        <span style={{ fontSize: '12px', whiteSpace: 'nowrap' }}>
          {currentIndex + 1}/{totalImages}: {currentImage} {isCurrentTitlePage ? '(扉絵)' : ''}
        </span>
        <button 
          className="btn btn-outline" 
          onClick={onNext} 
          disabled={currentIndex === totalImages - 1}
        >
          次▶
        </button>
        
        {onIndexChange && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <input
              type="number"
              min="0"
              max={totalImages - 1}
              value={currentIndex}
              onChange={(e) => {
                const value = parseInt(e.target.value);
                if (!isNaN(value)) {
                  onIndexChange(value);
                }
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.currentTarget.blur();
                }
              }}
              style={{ 
                width: '60px', 
                fontSize: '12px', 
                padding: '2px 4px' 
              }}
              title="インデックスを入力してジャンプ"
            />
            <span style={{ fontSize: '11px', color: '#666' }}>へ移動</span>
          </div>
        )}

        <select
          value={fourPanelPromptType}
          onChange={(e) => onPromptTypeChange(e.target.value as PromptType)}
          style={{ fontSize: '11px', padding: '2px' }}
        >
          <option value="combined">結合分析</option>
          <option value="four-panel">4コマ分析</option>
        </select>


        {onFourPanelAnalyzeAPI && (
          <button
            className="btn btn-outline"
            onClick={onFourPanelAnalyzeAPI}
            disabled={isFourPanelAnalyzing}
            style={{ 
              fontSize: '11px', 
              padding: '2px 6px',
              backgroundColor: '#1976D2',
              color: 'white',
              border: '1px solid #1976D2'
            }}
            title="Gemini API直接呼び出し"
          >
            {isFourPanelAnalyzing ? "分析中..." : "4コマ一括取得 (API)"}
          </button>
        )}

        {onFourPanelYoloDetect && (
          <button
            className="btn btn-outline btn-yolo"
            onClick={onFourPanelYoloDetect}
            disabled={isYoloDetecting}
            style={{ 
              fontSize: '11px', 
              padding: '2px 6px',
              backgroundColor: '#4CAF50',
              color: 'white',
              border: '1px solid #4CAF50'
            }}
          >
            {isYoloDetecting ? "検出中..." : "🎯 4コマAI検出"}
          </button>
        )}


        {onFourPanelAnalyzeImprovedAPI && (
          <button
            className="btn btn-outline"
            onClick={onFourPanelAnalyzeImprovedAPI}
            disabled={isFourPanelAnalyzing}
            style={{ 
              fontSize: '11px', 
              padding: '2px 6px',
              backgroundColor: '#E91E63',
              color: 'white',
              border: '1px solid #E91E63'
            }}
            title="改善版プロンプトで分析（API）"
          >
            {isFourPanelAnalyzing ? "分析中..." : "🚀 改善版分析 (API)"}
          </button>
        )}

        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <input
            type="checkbox"
            id="groundTruthCheckbox"
            checked={useGroundTruth}
            onChange={(e) => onGroundTruthToggle(e.target.checked)}
            style={{ margin: 0 }}
          />
          <label htmlFor="groundTruthCheckbox" style={{ fontSize: '11px', margin: 0 }}>
            例使用
          </label>
          {useGroundTruth && (
            <input
              type="number"
              value={groundTruthIndex ?? ''}
              onChange={(e) => {
                const value = e.target.value;
                onGroundTruthIndexChange(value === '' ? null : parseInt(value, 10));
              }}
              placeholder="Index"
              min={0}
              max={totalImages - 1}
              style={{ width: '60px', fontSize: '11px', padding: '2px' }}
            />
          )}
        </div>

        {onToggleDetectionVisualization && (
          <button
            className="btn btn-outline"
            onClick={() => onToggleDetectionVisualization(!showDetectionVisualization)}
            style={{ 
              fontSize: '11px', 
              padding: '2px 6px',
              backgroundColor: showDetectionVisualization ? '#FF5722' : '#9E9E9E',
              color: 'white',
              border: `1px solid ${showDetectionVisualization ? '#FF5722' : '#9E9E9E'}`
            }}
            title="検出結果可視化の表示/非表示"
          >
            {showDetectionVisualization ? '🚫 可視化OFF' : '👁️ 可視化ON'}
          </button>
        )}

        {onClearFourPanelData && (
          <button
            className="btn btn-outline btn-danger"
            onClick={onClearFourPanelData}
            style={{ 
              fontSize: '11px', 
              padding: '2px 6px',
              backgroundColor: '#dc3545',
              color: 'white',
              border: '1px solid #dc3545',
              marginLeft: '8px'
            }}
            title="4コマ分のデータをクリア"
          >
            🗑️ データクリア
          </button>
        )}
      </div>
    </div>
  );
};

export default NavigationHeader;