import React from 'react';

interface LoadingOverlayProps {
  isLoading: boolean;
  isAnalyzing?: boolean;
  inline?: boolean; // インライン表示フラグ
}

const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ isLoading, isAnalyzing, inline = false }) => {
  if (!isLoading) return null;

  // インライン表示（非ブロッキング）
  if (inline) {
    return (
      <div
        style={{
          position: "fixed",
          top: "60px", // タブナビゲーションの下
          right: "20px",
          backgroundColor: "#f0f8ff",
          border: "1px solid #add8e6",
          borderRadius: "8px",
          padding: "12px 16px",
          display: "flex",
          alignItems: "center",
          gap: "8px",
          fontSize: "14px",
          color: "#2c5aa0",
          zIndex: 1000, // 他の要素より前面に表示
          boxShadow: "0 2px 8px rgba(0, 0, 0, 0.1)",
          maxWidth: "300px"
        }}
      >
        <div
          style={{
            width: "16px",
            height: "16px",
            border: "2px solid #add8e6",
            borderTop: "2px solid #2c5aa0",
            borderRadius: "50%",
            animation: "spin 1s linear infinite"
          }}
        />
        <div>
          {isAnalyzing
            ? "🤖 AI分析中... (この間も画面操作できます)"
            : "データを読み込み中..."}
        </div>
        <style>
          {`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}
        </style>
      </div>
    );
  }

  // 従来のモーダル表示（フルスクリーンブロッキング）
  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.5)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 9999,
      }}
    >
      <div
        style={{
          backgroundColor: "white",
          padding: "20px",
          borderRadius: "8px",
          textAlign: "center",
        }}
      >
        <div style={{ fontSize: "20px", marginBottom: "10px" }}>
          {isAnalyzing
            ? "🤖 AI分析中..."
            : "データを読み込み中..."}
        </div>
        <div style={{ fontSize: "14px", color: "#666" }}>
          {isAnalyzing
            ? "4コマの画像を解析しています"
            : "しばらくお待ちください"}
        </div>
      </div>
    </div>
  );
};

export default LoadingOverlay;