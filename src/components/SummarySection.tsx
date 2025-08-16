import React from 'react';

interface SummarySectionProps {
  summary: string;
  isFetchingFeedback: boolean;
  onSummaryChange: (value: string) => void;
  onFetchFeedback: () => void;
}

const SummarySection: React.FC<SummarySectionProps> = ({
  summary,
  isFetchingFeedback,
  onSummaryChange,
  onFetchFeedback,
}) => {
  return (
    <div className="summary-section">
      <h3>4コマまとめ</h3>
      <button 
        className="btn btn-outline" 
        onClick={onFetchFeedback}
        disabled={isFetchingFeedback}
      >
        {isFetchingFeedback ? "取得中..." : "4コマ全体の感想を取得"}
      </button>
      <br />
      <textarea
        className="form-control input-sm"
        value={summary}
        onChange={(e) => onSummaryChange(e.target.value)}
        style={{ marginTop: '10px', minHeight: '100px', width: '100%' }}
      />
    </div>
  );
};

export default SummarySection;