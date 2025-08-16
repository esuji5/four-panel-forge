import React from "react";

interface HumanInTheLoopControlsProps {
  onSaveToCSV: () => void;
  onViewHistory: () => void;
  onExportLearningData: () => void;
  isSavingData: boolean;
}

const HumanInTheLoopControls: React.FC<HumanInTheLoopControlsProps> = ({
  onSaveToCSV,
  onViewHistory,
  onExportLearningData,
  isSavingData,
}) => {
  return (
    <div className="human-in-the-loop-controls" style={{ marginTop: '20px', padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '5px' }}>
      <h3 style={{ fontSize: '18px', marginBottom: '15px' }}>データ管理とHuman in the Loop</h3>
      <div style={{ display: 'flex', gap: '10px', marginBottom: '15px', flexWrap: 'wrap' }}>
        <button
          className="btn btn-primary"
          onClick={onSaveToCSV}
          disabled={isSavingData}
          style={{ padding: '8px 16px' }}
        >
          {isSavingData ? "保存中..." : "CSVに書き出す"}
        </button>

        <button
          className="btn btn-outline"
          onClick={onViewHistory}
          style={{ padding: '8px 16px' }}
        >
          修正履歴を表示
        </button>

        <button
          className="btn btn-outline"
          onClick={onExportLearningData}
          style={{ padding: '8px 16px' }}
        >
          学習データをエクスポート
        </button>
      </div>
      
      <div style={{ 
        padding: '10px', 
        backgroundColor: '#e8f4f8', 
        borderRadius: '4px',
        fontSize: '12px',
        lineHeight: '1.5'
      }}>
        <p style={{ fontWeight: 'bold', marginBottom: '5px' }}>Human in the Loop機能について：</p>
        <ul style={{ margin: '0', paddingLeft: '20px' }}>
          <li><strong>修正履歴</strong>: AI提案と人間による修正の履歴を確認できます</li>
          <li><strong>学習データ</strong>: 人間のフィードバックをAI改善用データとしてエクスポートできます</li>
          <li><strong>信頼度スコア</strong>: AIの各認識結果に対する信頼度が自動計算されます</li>
        </ul>
      </div>
    </div>
  );
};

export default HumanInTheLoopControls;