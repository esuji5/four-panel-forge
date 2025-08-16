import React from 'react';
import type { ChatItem } from '../types/app';

interface DiscussionSectionProps {
  isDiscussing: boolean;
  chatHistory: ChatItem[];
  onDiscussionClick: (question: string) => void;
}

const DiscussionSection: React.FC<DiscussionSectionProps> = ({
  isDiscussing,
  chatHistory,
  onDiscussionClick,
}) => {
  const questions = [
    { text: "どう感じた？", label: "どう感じた？" },
    { text: "かわいいのはだれ？", label: "かわいいのはだれ？" },
    { text: "次はどうなると思う？", label: "次はどうなると思う？" },
    { text: "特に気になる表現は？", label: "特に気になる表現は？" },
    { text: "今までになかったような表現や新しい情報はありますか？", label: "今までになかったような表現や新しい情報はありますか？" },
  ];

  return (
    <div className="discussion-section">
      <h3>ディスカッション</h3>
      <div style={{ marginBottom: '20px' }}>
        {questions.map((question, index) => (
          <button
            key={index}
            className="btn btn-outline"
            onClick={() => onDiscussionClick(question.text)}
            disabled={isDiscussing}
            style={{ marginRight: '10px', marginBottom: '10px' }}
          >
            {isDiscussing ? "回答中..." : question.label}
          </button>
        ))}
      </div>
      
      <div className="chat-section">
        {chatHistory.map((chat, index) => (
          <div key={index} style={{ marginBottom: '15px', padding: '10px', backgroundColor: '#f9f9f9', borderRadius: '5px' }}>
            <strong>質問:</strong> {chat.question}
            <br />
            <strong>回答:</strong> {chat.answer}
          </div>
        ))}
      </div>
    </div>
  );
};

export default DiscussionSection;