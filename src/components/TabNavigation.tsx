import React from 'react';

interface TabNavigationProps {
  currentTab: string;
  onTabChange: (tab: string) => void;
}

const TabNavigation: React.FC<TabNavigationProps> = ({ currentTab, onTabChange }) => {
  const tabs = [
    { id: "annotator", label: "アノテーター" },
  ];

  return (
    <div className="tabs is-centered" style={{ marginBottom: "20px" }}>
      <ul>
        {tabs.map((tab) => (
          <li key={tab.id} className={currentTab === tab.id ? "is-active" : ""}>
            <a
              onClick={() => onTabChange(tab.id)}
              style={{ cursor: "pointer" }}
            >
              <span>{tab.label}</span>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default TabNavigation;