import React from "react";

interface RevisionEntry {
  timestamp: string;
  revision_type: string;
  changes: any[];
  editor: string;
  confidence?: number;
  notes?: string;
}

interface AIProposal {
  image_path: string;
  timestamp: string;
  model: string;
  proposal: any;
  confidence_scores?: { [key: string]: number };
  processing_time?: number;
  api_cost?: number;
}

interface RevisionHistory {
  image_path: string;
  created_at: string;
  last_updated: string;
  total_revisions: number;
  ai_proposals: AIProposal[];
  revisions: RevisionEntry[];
  current_data: any;
}

interface RevisionHistoryDialogProps {
  isOpen: boolean;
  history: RevisionHistory | null;
  onClose: () => void;
  onViewDiff: (revision: RevisionEntry) => void;
}

const RevisionHistoryDialog: React.FC<RevisionHistoryDialogProps> = ({
  isOpen,
  history,
  onClose,
  onViewDiff,
}) => {
  if (!isOpen || !history) return null;

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString("ja-JP", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getRevisionTypeLabel = (type: string) => {
    switch (type) {
      case "ai_proposal":
        return "🤖 AI提案";
      case "human_edit":
        return "✏️ 人間による修正";
      case "auto_save":
        return "💾 自動保存";
      default:
        return type;
    }
  };

  const getConfidenceColor = (confidence: number | null | undefined) => {
    if (!confidence) return "";
    if (confidence >= 0.8) return "has-text-success";
    if (confidence >= 0.6) return "has-text-warning";
    return "has-text-danger";
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    }}>
      <div style={{
        backgroundColor: 'white',
        borderRadius: '8px',
        width: '90%',
        maxWidth: '800px',
        maxHeight: '90vh',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)'
      }}>
        <header style={{
          padding: '20px',
          borderBottom: '1px solid #e0e0e0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <h2 style={{ margin: 0, fontSize: '20px' }}>修正履歴</h2>
          <button 
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '24px',
              cursor: 'pointer',
              padding: '0',
              width: '30px',
              height: '30px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            ×
          </button>
        </header>
        <section style={{
          padding: '20px',
          overflowY: 'auto',
          flex: 1
        }}>
          <div className="content">
            <div className="box has-background-light">
              <p>
                <strong>画像パス:</strong> {history.image_path}
              </p>
              <p>
                <strong>作成日時:</strong> {formatTimestamp(history.created_at)}
              </p>
              <p>
                <strong>最終更新:</strong> {formatTimestamp(history.last_updated)}
              </p>
              <p>
                <strong>総修正回数:</strong> {history.total_revisions}回
              </p>
            </div>

            <h3 className="title is-5">AI提案履歴</h3>
            {history.ai_proposals.length > 0 ? (
              <div className="table-container">
                <table className="table is-fullwidth is-striped">
                  <thead>
                    <tr>
                      <th>日時</th>
                      <th>モデル</th>
                      <th>平均信頼度</th>
                      <th>処理時間</th>
                      <th>コスト</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.ai_proposals.map((proposal, index) => {
                      const avgConfidence = proposal.confidence_scores
                        ? Object.values(proposal.confidence_scores).reduce((a, b) => a + b, 0) /
                          Object.values(proposal.confidence_scores).length
                        : 0;
                      return (
                        <tr key={index}>
                          <td>{formatTimestamp(proposal.timestamp)}</td>
                          <td>{proposal.model}</td>
                          <td className={getConfidenceColor(avgConfidence)}>
                            {avgConfidence ? (avgConfidence * 100).toFixed(1) : "-"}%
                          </td>
                          <td>{proposal.processing_time ? `${proposal.processing_time.toFixed(2)}秒` : "-"}</td>
                          <td>{proposal.api_cost ? `¥${proposal.api_cost.toFixed(2)}` : "-"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="has-text-grey">AI提案はまだありません。</p>
            )}

            <h3 className="title is-5 mt-5">修正履歴タイムライン</h3>
            {history.revisions.length > 0 ? (
              <div className="timeline">
                {history.revisions.slice().reverse().map((revision, index) => (
                  <div key={index} className="box mb-3">
                    <div className="level">
                      <div className="level-left">
                        <div>
                          <p className="heading">{formatTimestamp(revision.timestamp)}</p>
                          <p className="title is-6">{getRevisionTypeLabel(revision.revision_type)}</p>
                          {revision.notes && (
                            <p className="subtitle is-7 has-text-grey">{revision.notes}</p>
                          )}
                        </div>
                      </div>
                      <div className="level-right">
                        <div className="level-item">
                          {revision.changes.length > 0 && (
                            <div>
                              <span className="tag is-primary">{revision.changes.length} 変更</span>
                              {revision.confidence && (
                                <span className={`tag ml-2 ${getConfidenceColor(revision.confidence)}`}>
                                  信頼度: {(revision.confidence * 100).toFixed(1)}%
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                        <div className="level-item">
                          {revision.changes.length > 0 && (
                            <button
                              className="button is-small is-info"
                              onClick={() => onViewDiff(revision)}
                            >
                              差分を表示
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="has-text-grey">修正履歴はまだありません。</p>
            )}
          </div>
        </section>
        <footer className="modal-card-foot">
          <button className="button" onClick={onClose}>
            閉じる
          </button>
        </footer>
      </div>
    </div>
  );
};

export default RevisionHistoryDialog;