import { useState } from 'react';
import axios from 'axios';
import { ImageData, ImagePathList } from '../types/app';
import { 
  BalloonDetection, 
  mergeSerifsWithBalloonDetections 
} from '../utils/balloonDetection';
import { processAndReorderBalloons } from '../utils/balloonGrouping';
import { 
  BalloonSpeakerMatcher, 
  CharacterDetection, 
  RayVisualizationData 
} from '../utils/speakerIdentification';
import { getImagePath } from '../config';
import { createRayVisualizationSVG } from '../utils/rayVisualization';
import { getEnglishToJapaneseMap } from '../utils/dataHelpers';

export const useAIDetection = () => {
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [isYoloDetecting, setIsYoloDetecting] = useState<boolean>(false);
  const [rayVisualizationData, setRayVisualizationData] = useState<{[key: string]: RayVisualizationData[]}>({});
  const [balloonDetectionData, setBalloonDetectionData] = useState<{[key: string]: BalloonDetection[]}>({});
  const [characterDetectionData, setCharacterDetectionData] = useState<{[key: string]: any[]}>({});

  // Ray可視化データを保存するためのグローバル変数
  let currentRayData: RayVisualizationData[] = [];

  // 英語名からchar_A形式への変換マップ
  const charNameToId: { [key: string]: string } = {
    'yuzuko': 'char_A',
    'yukari': 'char_B',
    'yui': 'char_C',
    'yoriko': 'char_D',
    'chiho': 'char_E',
    'kei': 'char_F',
    'fumi': 'char_G'
  };

  // AI検出（尻尾形状分析付き）
  const handleAIDetectionWithTailShape = async (
    imageKey: string, 
    imagePathList: ImagePathList,
    setImageData: (updater: (prev: ImageData) => ImageData) => void,
    visualize: boolean = false,
    detectionMode?: "face_recognition" | "multiclass"
  ): Promise<void> => {
    setIsYoloDetecting(true);
    try {
      console.log("🎯 AI検出開始:", imageKey, imagePathList[imageKey as keyof ImagePathList]);
      
      // detectionModeが指定されている場合は検出モードを切り替え
      if (detectionMode) {
        try {
          const modeResponse = await axios.post("http://localhost:8000/api/detection-mode", {
            mode: detectionMode
          });
          console.log("検出モード切り替え結果:", modeResponse.data);
        } catch (error) {
          console.warn("検出モード切り替えエラー:", error);
        }
      }
      
      // AI検出エンドポイントを呼び出し
      const response = await axios.post(
        "http://localhost:8000/api/analyze-image/",
        {
          komaPath: imagePathList[imageKey as keyof ImagePathList],
          mode: "single-panel",
          enableBalloonDetection: true,
          enableVisualization: visualize,
          detectOnly: true,
          detectionMode: detectionMode // 検出モード情報を追加
        }
      );
      
      console.log("✅ AI検出結果:", response.data);
      
      // 可視化画像がある場合は表示（モーダル）
      if (visualize && response.data.visualization_image) {
        await showDetectionModal(
          response.data.visualization_image, 
          imageKey, 
          imagePathList
        );
      }
      
      // 話者判定処理
      if (response.data && response.data.detection_results) {
        const detectionResults = response.data.detection_results;
        console.log('🔍 検出結果の詳細構造:', JSON.stringify(detectionResults, null, 2));
        
        // キャラクター検出結果をCharacterDetection形式に変換
        const detectedCharacters: CharacterDetection[] = [];
        const characterConfidences: { [key: string]: number } = {}; // 信頼度を別途管理
        
        if (detectionResults.character_detections) {
          console.log('🧑 character_detections:', detectionResults.character_detections);
          detectionResults.character_detections.forEach((char: any, index: number) => {
            if (char.boundingBox && char.characterName) {
              const charKey = `${char.characterName}_${index}`;
              characterConfidences[charKey] = char.classificationConfidence || char.confidence || 0;
              
              detectedCharacters.push({
                id: char.characterName,
                name: char.characterName,
                boundingBox: {
                  x1: char.boundingBox.x1,
                  y1: char.boundingBox.y1,
                  x2: char.boundingBox.x2,
                  y2: char.boundingBox.y2
                },
                faceDirection: char.faceDirection || undefined
              });
            }
          });
        } else {
          console.log('⚠️ character_detections が見つかりません');
          console.log('🔍 利用可能なキー:', Object.keys(detectionResults));
        }
        
        console.log('変換後のキャラクター情報:', detectedCharacters);
        
        // キャラクター情報を人物フォームに反映
        if (detectedCharacters.length > 0) {
          const englishToJapaneseMap = getEnglishToJapaneseMap();
          
          // 右から左の順番（X座標の降順）でキャラクターをソート
          const sortedCharacters = [...detectedCharacters].sort((a, b) => {
            const centerXA = (a.boundingBox.x1 + a.boundingBox.x2) / 2;
            const centerXB = (b.boundingBox.x1 + b.boundingBox.x2) / 2;
            return centerXB - centerXA; // 降順（右から左）
          });
          
          console.log('ソート前キャラクター:', detectedCharacters.map(c => `${c.name}: ${(c.boundingBox.x1 + c.boundingBox.x2) / 2}`));
          console.log('ソート後キャラクター:', sortedCharacters.map(c => `${c.name}: ${(c.boundingBox.x1 + c.boundingBox.x2) / 2}`));
          
          setImageData((prevData) => {
            const currentPanel = prevData[imageKey as keyof ImageData];
            const existingCharacters = currentPanel.characters || [];
            
            // ソートされたキャラクターを人物フォームに反映
            const updatedCharacters: any[] = [];
            
            sortedCharacters.forEach((detected: CharacterDetection, index: number) => {
              if (detected.name && detected.name.trim() !== "") {
                // 英語名を日本語名に変換
                const japaneseName = englishToJapaneseMap[detected.name] || detected.name;
                const charKey = `${detected.name}_${index}`;
                const confidence = characterConfidences[charKey] || 0;
                
                // 新しいキャラクターを右から左の順番で追加
                updatedCharacters.push({
                  character: japaneseName,
                  faceDirection: detected.faceDirection || "",
                  position: confidence > 0 ? confidence.toFixed(3) : "",
                  shotType: "",
                  characterSize: "",
                  expression: "",
                  clothing: "",
                  isVisible: true,
                  coordinate: [
                    (detected.boundingBox.x1 + detected.boundingBox.x2) / 2,
                    (detected.boundingBox.y1 + detected.boundingBox.y2) / 2
                  ]
                });
              }
            });
            
            // 残りの空フォームを追加（必要に応じて）
            while (updatedCharacters.length < Math.max(existingCharacters.length, 1)) {
              updatedCharacters.push({
                character: "",
                faceDirection: "",
                position: "",
                shotType: "",
                characterSize: "",
                expression: "",
                clothing: "",
                isVisible: true,
                coordinate: [0, 0],
              });
            }
            
            return {
              ...prevData,
              [imageKey]: {
                ...currentPanel,
                characters: updatedCharacters,
                charactersNum: updatedCharacters.length,
              }
            };
          });
        }
        
        // 吹き出し検出結果を取得
        const balloonDetections = detectionResults.balloon_detections || [];
        console.log('吹き出し検出結果:', balloonDetections);
        
        // 人物検出結果を取得
        const characterDetections = detectionResults.character_detections || [];
        console.log('人物検出結果:', characterDetections);
        
        // 検出結果を状態に保存
        setBalloonDetectionData(prev => ({
          ...prev,
          [imageKey]: balloonDetections
        }));
        setCharacterDetectionData(prev => ({
          ...prev,
          [imageKey]: characterDetections
        }));
        
        // パネル番号からプレフィックスを生成（例: image1 → p1_）
        const panelNum = imageKey.replace('image', '');
        const panelPrefix = `p${panelNum}_`;
        
        // 話者特定を実行
        if (balloonDetections.length > 0 && detectedCharacters.length > 0) {
          console.log('=== 話者特定処理を開始 ===');
          
          const speakerMatcher = new BalloonSpeakerMatcher();
          speakerMatcher.setImageUrl(getImagePath(imagePathList[imageKey as keyof ImagePathList]));
          
          // 吹き出しをグループ化・ソート
          const processedDetections = processAndReorderBalloons(balloonDetections, panelPrefix);
          console.log('グループ化・ソート後の吹き出し:', processedDetections);
          
          const speakerMatches = speakerMatcher.matchBalloonsToSpeakers(
            processedDetections,
            detectedCharacters
          );
          
          console.log("話者特定結果:", speakerMatches);
          
          // Ray可視化データを収集
          const rayData: RayVisualizationData[] = [];
          speakerMatches.forEach(match => {
            if (match.rayVisualization) {
              rayData.push(...match.rayVisualization);
            }
          });
          console.log("Ray可視化データ:", rayData);
          
          // モーダル表示用にRay可視化データを保存
          currentRayData = rayData;
          
          // Ray可視化データを状態に保存（空でも保存して古いデータをクリア）
          console.log('💾 Ray可視化データを保存中:', { 
            imageKey, 
            rayDataLength: rayData.length,
            rayData: rayData 
          });
          setRayVisualizationData(prev => {
            const newData = {
              ...prev,
              [imageKey]: rayData
            };
            console.log('💾 保存後のrayVisualizationData:', {
              imageKey,
              rayDataLength: rayData.length,
              allKeys: Object.keys(newData),
              fullData: newData
            });
            return newData;
          });
          
          // 話者情報を吹き出しに反映
          console.log('🎯 話者マッチング結果:', speakerMatches);
          const processedDetectionsWithSpeakers = processedDetections.map(detection => {
            const match = speakerMatches.find(m => m.balloonId === detection.dialogueId);
            console.log(`🔍 ${detection.dialogueId}: match=${match ? `${match.speakerName} (${match.confidence})` : 'なし'}`);
            if (match && match.confidence > 0.3) {
              const charId = charNameToId[match.speakerName] || '';
              console.log(`✅ 話者設定: ${detection.dialogueId} -> ${match.speakerName} -> ${charId}`);
              
              return {
                ...detection,
                speakerCharacterId: charId
              };
            }
            return detection;
          });
          
          // セリフデータをマージして更新
          console.log('🔗 話者付き検出結果をmergeSerifsWithBalloonDetectionsに渡します');
          processedDetectionsWithSpeakers.forEach((det, i) => {
            console.log(`  ${i}: ${det.dialogueId} -> speaker: ${det.speakerCharacterId || 'なし'}`);
          });
          
          setImageData((prevData) => {
            const existingData = prevData[imageKey as keyof ImageData];
            const mergedSerifs = mergeSerifsWithBalloonDetections(
              existingData.serifs || [],
              processedDetectionsWithSpeakers,
              panelPrefix
            );
            
            return {
              ...prevData,
              [imageKey]: {
                ...existingData,
                serifs: mergedSerifs,
                serifsNum: mergedSerifs.length,
              },
            };
          });
        }
      }
      
    } catch (error) {
      console.error("❌ AI検出エラー:", error);
      alert("AI検出エラー: " + (error as any).message);
    } finally {
      setIsYoloDetecting(false);
    }
  };

  // YOLO+DINOv2キャラクター検出
  const handleYoloDinov2Detection = async (
    imageKey: string,
    imagePathList: ImagePathList,
    setImageData: (updater: (prev: ImageData) => ImageData) => void,
    visualize: boolean = true,
    detectionMode?: "face_recognition" | "multiclass"
  ): Promise<void> => {
    setIsYoloDetecting(true);
    try {
      console.log("YOLO+DINOv2検出開始:", imageKey, imagePathList[imageKey as keyof ImagePathList]);
      
      // detectionModeが指定されている場合は検出モードを切り替え
      if (detectionMode) {
        try {
          const modeResponse = await axios.post("http://localhost:8000/api/detection-mode", {
            mode: detectionMode
          });
          console.log("検出モード切り替え結果:", modeResponse.data);
        } catch (error) {
          console.warn("検出モード切り替えエラー:", error);
        }
      }
      
      // キャラクター検出
      const response = await axios.post(
        "http://localhost:8000/api/detect-characters-yolo-dinov2",
        {
          komaPath: imagePathList[imageKey as keyof ImagePathList],
          mode: "single",
          detectionThreshold: 0.2,
          classificationThreshold: 0.5,
          visualize: visualize,
          detectionMode: detectionMode // 検出モード情報を追加
        }
      );
      
      // 吹き出し検出も実行
      let balloonResponse: any = null;
      try {
        balloonResponse = await axios.post(
          "http://localhost:8000/api/detect-balloons",
          {
            imagePath: imagePathList[imageKey as keyof ImagePathList],
            confidenceThreshold: 0.15,
            maxDet: 300
          }
        );
        console.log("吹き出し検出結果:", balloonResponse.data);
      } catch (error) {
        console.warn("吹き出し検出エラー:", error);
      }
      
      console.log("YOLO+DINOv2検出結果:", response.data);
      
      // 可視化画像がある場合は表示（モーダル）
      if (visualize && response.data.visualization) {
        await showDetectionModal(
          response.data.visualization, 
          imageKey, 
          imagePathList
        );
      }
      
      // 検出結果をフォームに反映
      if (response.data && response.data.characters) {
        // キャラクター情報を人物フォームに反映
        const englishToJapaneseMap = getEnglishToJapaneseMap();
        
        // 右から左の順番（X座標の降順）でキャラクターをソート
        const sortedCharacters = [...response.data.characters].sort((a: any, b: any) => {
          const centerXA = a.coordinate ? a.coordinate[0] : (a.bbox ? (a.bbox[0] + a.bbox[2]) / 2 : 0);
          const centerXB = b.coordinate ? b.coordinate[0] : (b.bbox ? (b.bbox[0] + b.bbox[2]) / 2 : 0);
          return centerXB - centerXA; // 降順（右から左）
        });
        
        console.log('YOLO+DINOv2ソート前キャラクター:', response.data.characters.map((c: any) => `${c.character}: ${c.coordinate ? c.coordinate[0] : 'no coord'}`));
        console.log('YOLO+DINOv2ソート後キャラクター:', sortedCharacters.map((c: any) => `${c.character}: ${c.coordinate ? c.coordinate[0] : 'no coord'}`));
        
        setImageData((prevData) => {
          const currentPanel = prevData[imageKey as keyof ImageData];
          const existingCharacters = currentPanel.characters || [];
          
          // ソートされたキャラクターを人物フォームに反映
          const updatedCharacters: any[] = [];
          
          sortedCharacters.forEach((detected: any, index: number) => {
            if (detected.character && detected.character.trim() !== "") {
              // 英語名を日本語名に変換
              const japaneseName = englishToJapaneseMap[detected.character] || detected.character;
              
              // 新しいキャラクターを右から左の順番で追加
              updatedCharacters.push({
                character: japaneseName,
                faceDirection: "",
                position: detected.classificationConfidence ? detected.classificationConfidence.toFixed(3) : "",
                shotType: "",
                characterSize: detected.characterSize || "",
                expression: "",
                clothing: "",
                isVisible: true,
                coordinate: detected.coordinate || [0, 0]
              });
            }
          });
          
          // 残りの空フォームを追加（必要に応じて）
          while (updatedCharacters.length < Math.max(existingCharacters.length, 1)) {
            updatedCharacters.push({
              character: "",
              faceDirection: "",
              position: "",
              shotType: "",
              characterSize: "",
              expression: "",
              clothing: "",
              isVisible: true,
              coordinate: [0, 0],
            });
          }
          
          return {
            ...prevData,
            [imageKey]: {
              ...currentPanel,
              characters: updatedCharacters,
              charactersNum: updatedCharacters.length,
            }
          };
        });
        
        // 話者判定処理を実行
        await processSpeakerIdentification(
          imageKey,
          imagePathList,
          response.data.characters,
          balloonResponse?.data?.detections || [],
          setImageData
        );
      }
      
    } catch (error) {
      console.error("❌ YOLO+DINOv2検出エラー:", error);
      alert("YOLO+DINOv2検出エラー: " + (error as any).message);
    } finally {
      setIsYoloDetecting(false);
    }
  };

  // 話者判定処理のヘルパー関数
  const processSpeakerIdentification = async (
    imageKey: string,
    imagePathList: ImagePathList,
    characters: any[],
    balloonDetections: BalloonDetection[],
    setImageData: (updater: (prev: ImageData) => ImageData) => void
  ) => {
    // キャラクター情報をCharacterDetection形式に変換
    const detectedCharacters: CharacterDetection[] = characters.map((char: any) => ({
      id: char.character || "",
      name: char.character || "",
      boundingBox: char.bbox ? {
        x1: char.bbox[0],
        y1: char.bbox[1],
        x2: char.bbox[2],
        y2: char.bbox[3]
      } : { x1: 0, y1: 0, x2: 0, y2: 0 },
      faceDirection: char.faceDirection || undefined
    }));
    
    // パネル番号からプレフィックスを生成
    const panelNum = imageKey.replace('image', '');
    const panelPrefix = `p${panelNum}_`;
    
    // 話者特定を実行
    if (balloonDetections.length > 0 && detectedCharacters.length > 0) {
      console.log('=== 話者特定処理を開始 ===');
      
      const speakerMatcher = new BalloonSpeakerMatcher();
      speakerMatcher.setImageUrl(getImagePath(imagePathList[imageKey as keyof ImagePathList]));
      
      const processedDetections = processAndReorderBalloons(balloonDetections, panelPrefix);
      console.log('グループ化・ソート後の吹き出し:', processedDetections);
      
      const speakerMatches = speakerMatcher.matchBalloonsToSpeakers(
        processedDetections,
        detectedCharacters
      );
      
      console.log("話者特定結果:", speakerMatches);
      
      // Ray可視化データを収集
      const rayData: RayVisualizationData[] = [];
      speakerMatches.forEach(match => {
        if (match.rayVisualization) {
          rayData.push(...match.rayVisualization);
        }
      });
      
      // Ray可視化データを状態に保存（空でも保存して古いデータをクリア）
      console.log('💾 Ray可視化データを保存中(YOLO):', { 
        imageKey, 
        rayDataLength: rayData.length,
        rayData: rayData 
      });
      setRayVisualizationData(prev => {
        const newData = {
          ...prev,
          [imageKey]: rayData
        };
        console.log('💾 保存後のrayVisualizationData(YOLO):', {
          imageKey,
          rayDataLength: rayData.length,
          allKeys: Object.keys(newData),
          fullData: newData
        });
        return newData;
      });
      
      // 話者情報を吹き出しに反映
      const processedDetectionsWithSpeakers = processedDetections.map(detection => {
        const match = speakerMatches.find(m => m.balloonId === detection.dialogueId);
        if (match && match.confidence > 0.3) {
          const charId = charNameToId[match.speakerName] || '';
          
          return {
            ...detection,
            speakerCharacterId: charId
          };
        }
        return detection;
      });
      
      // セリフデータをマージして更新
      setImageData((prevData) => {
        const existingData = prevData[imageKey as keyof ImageData];
        const mergedSerifs = mergeSerifsWithBalloonDetections(
          existingData.serifs || [],
          processedDetectionsWithSpeakers,
          panelPrefix
        );
        
        return {
          ...prevData,
          [imageKey]: {
            ...existingData,
            serifs: mergedSerifs,
            serifsNum: mergedSerifs.length,
          },
        };
      });
    }
  };

  // モーダル表示のヘルパー関数
  const showDetectionModal = async (
    visualizationImage: string,
    imageKey: string,
    imagePathList: ImagePathList
  ): Promise<void> => {
    // 既存のモーダルを削除（重複防止）
    const existingModals = document.querySelectorAll('[data-modal-type="ai-detection-result"]');
    existingModals.forEach(modal => modal.remove());
    
    // モーダル表示を作成
    const modal = document.createElement('div');
    modal.setAttribute('data-modal-type', 'ai-detection-result');
    modal.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.8);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 10000;
      cursor: pointer;
      padding: 20px;
      box-sizing: border-box;
    `;
    
    // タイトル要素
    const title = document.createElement('h2');
    title.textContent = `🎯 尻尾形状分類結果 - ${imageKey}`;
    title.style.cssText = `
      color: white;
      margin-bottom: 20px;
      text-align: center;
      font-size: 24px;
      text-shadow: 0 2px 4px rgba(0, 0, 0, 0.8);
    `;
    
    // 画像コンテナ（Ray可視化用）
    const imageContainer = document.createElement('div');
    imageContainer.style.cssText = `
      position: relative;
      max-width: 90%;
      max-height: 70%;
      display: flex;
      align-items: center;
      justify-content: center;
    `;
    
    // 画像要素
    const resultImg = document.createElement('img');
    resultImg.src = visualizationImage;
    resultImg.style.cssText = `
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
      border-radius: 4px;
      cursor: default;
    `;
    
    // 画像読み込み完了後にRay可視化を追加
    resultImg.onload = () => {
      console.log('🖼️ モーダル画像読み込み完了');
      console.log('📊 直接渡されたRayデータ:', currentRayData);
      console.log('🔑 imageKey:', imageKey);
      console.log('🎯 Rayデータ長:', currentRayData.length);
      
      if (currentRayData && currentRayData.length > 0) {
        console.log('✅ Ray可視化データが存在します。SVG生成開始...');
        
        // SVG要素を生成
        const svgElement = createRayVisualizationSVG(currentRayData, resultImg.width, resultImg.height);
        
        // SVGを画像コンテナに追加
        if (svgElement) {
          svgElement.style.position = 'absolute';
          svgElement.style.top = '0';
          svgElement.style.left = '0';
          svgElement.style.pointerEvents = 'none';
          svgElement.style.zIndex = '10';
          
          imageContainer.appendChild(svgElement);
          console.log('✅ Ray可視化SVGを追加しました');
        } else {
          console.error('❌ SVG要素の生成に失敗しました');
        }
      } else {
        console.log('❌ Ray可視化データが存在しません');
      }
    };
    
    imageContainer.appendChild(resultImg);
    
    // 閉じるボタン
    const closeButton = document.createElement('button');
    closeButton.textContent = '閉じる';
    closeButton.style.cssText = `
      padding: 10px 20px;
      font-size: 16px;
      background: #4CAF50;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      margin-top: 20px;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    `;
    
    modal.appendChild(title);
    modal.appendChild(imageContainer);
    modal.appendChild(closeButton);
    
    // クリックイベントでモーダルを閉じる
    const closeModal = () => {
      document.body.removeChild(modal);
      document.removeEventListener('keydown', handleEscKey);
    };
    
    // ESCキーでモーダルを閉じる
    const handleEscKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        closeModal();
      }
    };
    document.addEventListener('keydown', handleEscKey);
    
    modal.onclick = (e) => {
      if (e.target === modal) closeModal();
    };
    closeButton.onclick = closeModal;
    
    // 画像のクリックでは閉じない
    resultImg.onclick = (e) => e.stopPropagation();
    imageContainer.onclick = (e) => e.stopPropagation();
    title.onclick = (e) => e.stopPropagation();
    
    document.body.appendChild(modal);
  };

  return {
    // State
    isAnalyzing,
    setIsAnalyzing,
    isYoloDetecting,
    setIsYoloDetecting,
    rayVisualizationData,
    setRayVisualizationData,
    balloonDetectionData,
    setBalloonDetectionData,
    characterDetectionData,
    setCharacterDetectionData,
    
    // Functions
    handleAIDetectionWithTailShape,
    handleYoloDinov2Detection,
    processSpeakerIdentification,
    
    // Global variables
    currentRayData,
  };
};