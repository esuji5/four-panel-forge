import React from "react";

interface DiffChange {
  field_path: string;
  old_value: any;
  new_value: any;
  change_type: "added" | "modified" | "deleted";
  similarity?: number;
}

interface DiffViewerProps {
  isOpen: boolean;
  changes: DiffChange[];
  onClose: () => void;
}

const DiffViewer: React.FC<DiffViewerProps> = ({ isOpen, changes, onClose }) => {
  if (!isOpen || !changes) return null;

  const formatValue = (value: any): string => {
    if (value === null || value === undefined) return "（なし）";
    if (typeof value === "boolean") return value ? "はい" : "いいえ";
    if (typeof value === "object") return JSON.stringify(value, null, 2);
    return String(value);
  };

  const getChangeTypeLabel = (type: string) => {
    switch (type) {
      case "added":
        return { label: "追加", className: "has-text-success" };
      case "modified":
        return { label: "変更", className: "has-text-warning" };
      case "deleted":
        return { label: "削除", className: "has-text-danger" };
      default:
        return { label: type, className: "" };
    }
  };

  const getFieldLabel = (path: string): string => {
    const parts = path.split(".");
    const labels: { [key: string]: string } = {
      characters: "キャラクター",
      character: "名前",
      faceDirection: "顔の向き",
      position: "位置",
      expression: "表情",
      clothing: "服装",
      isVisible: "表示",
      serif: "セリフ",
      sceneData: "シーンデータ",
      scene: "シーン説明",
      location: "場所",
      backgroundEffects: "背景効果",
      cameraAngle: "カメラアングル",
      framing: "フレーミング",
    };

    return parts
      .map((part) => {
        // 数字の場合はインデックスとして扱う
        if (/^\d+$/.test(part)) {
          return `[${parseInt(part) + 1}]`;
        }
        return labels[part] || part;
      })
      .join(" → ");
  };

  const getSimilarityColor = (similarity?: number) => {
    if (!similarity) return "";
    if (similarity >= 0.8) return "has-text-success";
    if (similarity >= 0.5) return "has-text-warning";
    return "has-text-danger";
  };

  // 重要な変更（類似度が低い）を上位に表示
  const sortedChanges = [...changes].sort((a, b) => {
    const simA = a.similarity || 0;
    const simB = b.similarity || 0;
    return simA - simB;
  });

  return (
    <div className="modal is-active">
      <div className="modal-background" onClick={onClose}></div>
      <div className="modal-card" style={{ width: "90%", maxWidth: "900px" }}>
        <header className="modal-card-head">
          <p className="modal-card-title">変更内容の詳細</p>
          <button className="delete" aria-label="close" onClick={onClose}></button>
        </header>
        <section className="modal-card-body">
          <div className="content">
            <p className="has-text-grey">
              変更数: {changes.length}件
              {changes.filter((c) => (c.similarity || 0) < 0.5).length > 0 && (
                <span className="has-text-danger ml-3">
                  （重要な変更: {changes.filter((c) => (c.similarity || 0) < 0.5).length}件）
                </span>
              )}
            </p>

            <div className="table-container">
              <table className="table is-fullwidth is-striped">
                <thead>
                  <tr>
                    <th style={{ width: "25%" }}>項目</th>
                    <th style={{ width: "30%" }}>変更前</th>
                    <th style={{ width: "30%" }}>変更後</th>
                    <th style={{ width: "10%" }}>種類</th>
                    <th style={{ width: "5%" }}>類似度</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedChanges.map((change, index) => {
                    const typeInfo = getChangeTypeLabel(change.change_type);
                    return (
                      <tr key={index}>
                        <td>
                          <strong>{getFieldLabel(change.field_path)}</strong>
                        </td>
                        <td>
                          <pre className="has-background-light p-2" style={{ fontSize: "0.85rem" }}>
                            {formatValue(change.old_value)}
                          </pre>
                        </td>
                        <td>
                          <pre className="has-background-light p-2" style={{ fontSize: "0.85rem" }}>
                            {formatValue(change.new_value)}
                          </pre>
                        </td>
                        <td>
                          <span className={`tag ${typeInfo.className}`}>{typeInfo.label}</span>
                        </td>
                        <td>
                          {change.similarity !== undefined && (
                            <span className={`tag ${getSimilarityColor(change.similarity)}`}>
                              {(change.similarity * 100).toFixed(0)}%
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {changes.filter((c) => (c.similarity || 0) < 0.5).length > 0 && (
              <article className="message is-warning mt-5">
                <div className="message-header">
                  <p>重要な変更の要約</p>
                </div>
                <div className="message-body">
                  <ul>
                    {changes
                      .filter((c) => (c.similarity || 0) < 0.5)
                      .map((change, index) => (
                        <li key={index}>
                          <strong>{getFieldLabel(change.field_path)}</strong>:
                          「{formatValue(change.old_value)}」→「{formatValue(change.new_value)}」
                        </li>
                      ))}
                  </ul>
                </div>
              </article>
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

export default DiffViewer;