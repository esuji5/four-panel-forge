import React from "react";
import { SceneData } from "../types/app";

interface SceneFormProps {
  sceneData: SceneData;
  imageKey: string;
  onSceneChange: (imageKey: string, field: string, value: string) => void;
}

const SceneForm: React.FC<SceneFormProps> = ({
  sceneData,
  imageKey,
  onSceneChange,
}) => {
  return (
    <div className="scene-section">
      <h4>状況</h4>
      <div className="scene-row">
        <input
          className="form-control input-sm"
          name="scene"
          value={sceneData.scene}
          onChange={(e) => onSceneChange(imageKey, "scene", e.target.value)}
          placeholder="シーン説明"
        />
        <input
          className="form-control input-sm"
          name="location"
          value={sceneData.location}
          onChange={(e) => onSceneChange(imageKey, "location", e.target.value)}
          placeholder="場所"
        />
        <input
          className="form-control input-sm"
          name="backgroundEffects"
          value={sceneData.backgroundEffects}
          onChange={(e) => onSceneChange(imageKey, "backgroundEffects", e.target.value)}
          placeholder="背景効果"
        />
      </div>
    </div>
  );
};

export default SceneForm;