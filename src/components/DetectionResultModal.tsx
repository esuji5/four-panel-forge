import React from 'react';

interface DetectionResultModalProps {
  isOpen: boolean;
  onClose: () => void;
  imageUrl: string;
  title: string;
  canvasData?: string; // Base64エンコードされたcanvasデータ
}

const DetectionResultModal: React.FC<DetectionResultModalProps> = ({
  isOpen,
  onClose,
  imageUrl,
  title,
  canvasData
}) => {
  if (!isOpen) return null;

  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div 
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 1000,
        padding: '20px'
      }}
      onClick={handleBackdropClick}
    >
      <div
        style={{
          backgroundColor: 'white',
          borderRadius: '8px',
          padding: '20px',
          maxWidth: '90vw',
          maxHeight: '90vh',
          overflow: 'auto',
          position: 'relative',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)'
        }}
      >
        {/* 閉じるボタン */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '10px',
            right: '15px',
            background: 'none',
            border: 'none',
            fontSize: '24px',
            cursor: 'pointer',
            color: '#666',
            padding: '5px',
            borderRadius: '50%',
            width: '40px',
            height: '40px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = '#f0f0f0';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
          }}
        >
          ×
        </button>

        {/* タイトル */}
        <h2 style={{ 
          textAlign: 'center', 
          marginBottom: '20px',
          color: '#333',
          paddingRight: '40px' // 閉じるボタンとの重複を避ける
        }}>
          {title}
        </h2>

        {/* 画像表示 */}
        <div style={{ textAlign: 'center' }}>
          <img
            src={canvasData || imageUrl}
            alt="検出結果"
            style={{
              maxWidth: '100%',
              maxHeight: '70vh',
              border: '2px solid #ccc',
              borderRadius: '4px',
              boxShadow: '0 4px 8px rgba(0,0,0,0.2)'
            }}
          />
        </div>

        {/* フッターボタン */}
        <div style={{ 
          textAlign: 'center', 
          marginTop: '20px',
          paddingTop: '20px',
          borderTop: '1px solid #eee'
        }}>
          <button
            onClick={onClose}
            style={{
              padding: '10px 20px',
              backgroundColor: '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '16px'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#0056b3';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#007bff';
            }}
          >
            閉じる
          </button>
        </div>
      </div>
    </div>
  );
};

export default DetectionResultModal;