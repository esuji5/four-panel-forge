import React, {useState, useEffect} from "react";
import axios from "axios";
import Papa from "papaparse";
import {
  getImagePath,
  IMAGE_SOURCE,
  WORKERS_API_URL,
  R2_PUBLIC_URL,
} from "./config";
import { normalizeImageDataPaths } from "./utils/imagePath";
import NavigationHeader from "./components/NavigationHeader";
import PanelEditor from "./components/PanelEditor";
import RayVisualization from "./components/RayVisualization";
import {
  FOUR_PANEL_PROMPT_DETAILED,
  DEFAULT_PROMPT_COMBINED,
  SYSTEM_INSTRUCTION,
  COMMON_USER_INSTRUCTION,
  SINGLE_PANEL_JSON_SCHEMA,
} from "./prompts";
import type {PromptType} from "./types/manga";
import type {
  Serif,
  Character,
  SceneData,
  PanelData,
  ImageData,
  ImagePathList,
  CSVRow,
  ChatItem,
} from "./types/app";
import { combinedKomaImages } from "./constants/imageList";
import { 
  combineFourPanelImages,
  loadGroundTruthImage
} from "./utils/imageProcessing";
import { getBalloonTypeDisplayName } from "./utils/balloonDetection";
import { getTailDirectionDisplayName } from "./utils/tailDirection";
import { RayVisualizationData } from "./utils/speakerIdentification";
import LoadingOverlay from "./components/LoadingOverlay";
import TabNavigation from "./components/TabNavigation";
import SummarySection from "./components/SummarySection";
import DiscussionSection from "./components/DiscussionSection";
import RevisionHistoryDialog from "./components/RevisionHistoryDialog";
import DiffViewer from "./components/DiffViewer";
import HumanInTheLoopControls from "./components/HumanInTheLoopControls";

// 新しいカスタムフックをimport
import { useImageData } from "./hooks/useImageData";
import { useAIDetection } from "./hooks/useAIDetection";
import { createRayVisualizationSVG } from "./utils/rayVisualization";
import { 
  createEmptyCharacter, 
  isEmptyCharacterData, 
  createEmptyPanelData, 
  convertOldDataToNew,
  getCharacterIdMap,
  getEnglishToCharIdMap,
  isEmptyImageData
} from "./utils/dataHelpers";
import { ApiService } from "./services/apiService";
import { AIAnalysisService } from "./services/aiAnalysisService";

import "./App.css";


const App: React.FC = () => {
  // カスタムフックを使用してアプリケーション状態を管理
  const imageDataHook = useImageData();
  const aiDetectionHook = useAIDetection();
  const [currentTab, setCurrentTab] = useState<string>("annotator");

  // 分割されたhookから必要な値を取得
  const {
    imageData,
    setImageData,
    summary,
    setSummary,
    rows,
    currentIndex,
    setCurrentIndex,
    currentImage,
    imagePathList,
    isLoadingData,
    setIsLoadingData,
    isSavingData,
    isCurrentTitlePage,
    currentPageNumber,
    loadImageData,
    saveToJSONWithData,
    saveToCSV,
    handleClearFourPanelData,
    handleSerifChange,
    handleAddSerif,
    handleRemoveSerif,
    handleSerifSwap,
    handlePanelSwap,
    createEmptyImageData,
  } = imageDataHook;

  const {
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
    handleAIDetectionWithTailShape,
    handleYoloDinov2Detection,
  } = aiDetectionHook;

  // 可視化表示制御
  const [showDetectionVisualization, setShowDetectionVisualization] = useState<boolean>(true);

  // 旧式のupdateCurrentIndex関数を維持（後方互換性のため）
  const updateCurrentIndex = (newIndex: number) => {
    setCurrentIndex(newIndex);
  };
  // saveCurrentIndexToStorage は useCurrentIndex フックで自動処理


  useEffect(() => {
    // currentIndexが変更されたときに関連データをリセット
    setSummary(""); // summaryを空文字にリセット
    setDiscussionResult(""); // discussionResultを
    setChatHistory([]);
  }, [currentIndex]); // currentIndexが変更され

  useEffect(() => {
    // 重複削除：useImageDataフックで処理される

    const fetchSummary = async () => {
      try {
        let response;
        const url =
          IMAGE_SOURCE === "r2"
            ? `${WORKERS_API_URL}/api/data/saved_json/feedback_yuyu10_${currentIndex}.json`
            : `saved_json/feedback_yuyu10_${currentIndex}.json`;

        response = await fetch(url);

        if (response.ok) {
          // Content-Typeをチェックしてからパース
          const contentType = response.headers.get('content-type');
          if (contentType && contentType.includes('application/json')) {
            const data = await response.json();
            console.log(`Summary loaded from ${IMAGE_SOURCE}:`, data);

            // レスポンス形式に応じて適切にデータを取得
            if (data && typeof data === "object") {
              if (
                data.content &&
                Array.isArray(data.content) &&
                data.content[0]?.text
              ) {
                setSummary(data.content[0].text);
              } else if (data.summary && typeof data.summary === "string") {
                setSummary(data.summary);
              } else if (typeof data === "string") {
                setSummary(data);
              } else {
                console.warn("Unexpected summary format:", data);
              }
            }
          } else {
            console.log(`Summary response is not JSON (${contentType}):`, url);
          }
        } else {
          console.log(`Summary not found (${response.status}):`, url);
        }
      } catch (error) {
        console.log("Error loading summary:", (error as Error).message);
      }
    };
    fetchSummary();

    const loadChatHistory = async () => {
      try {
        let response;
        const url =
          IMAGE_SOURCE === "r2"
            ? `${WORKERS_API_URL}/api/data/saved_json/discussion_yuyu10_${currentIndex}.json`
            : `saved_json/discussion_yuyu10_${currentIndex}.json`;

        response = await fetch(url);

        if (response.ok) {
          // Content-Typeをチェックしてからパース
          const contentType = response.headers.get('content-type');
          if (contentType && contentType.includes('application/json')) {
            const jsonData = await response.json();
            console.log(`Chat history loaded from ${IMAGE_SOURCE}:`, jsonData);

            let messages: any[] = [];
            if (jsonData && typeof jsonData === "object") {
            if (jsonData.chatHistory && Array.isArray(jsonData.chatHistory)) {
              messages = jsonData.chatHistory;
            } else if (Array.isArray(jsonData)) {
              // 配列の場合は各要素を処理
              messages = jsonData.map((item: any) => {
                try {
                  return typeof item === "string" ? JSON.parse(item) : item;
                } catch (parseError) {
                  console.warn("Failed to parse chat item:", item, parseError);
                  return item;
                }
              });
            }

            // messagesをchatHistory形式に変換
            const formattedHistory: ChatItem[] = messages.map((data: any) => ({
              question: data.question || "質問なし",
              answer:
                data.content?.[0]?.text || data.answer || data || "回答なし",
            }));

            setChatHistory(formattedHistory);
            }
          } else {
            console.log(`Chat history response is not JSON (${contentType}):`, url);
          }
        } else {
          console.log(`Chat history not found (${response.status}):`, url);
        }
      } catch (error) {
        console.log("Error loading chat history:", (error as Error).message);
      }
    };
    loadChatHistory();
  }, [currentIndex]);


  // currentIndexの範囲チェックと自動修正（combinedKomaImages定義後）
  useEffect(() => {
    if (combinedKomaImages.length > 0) {
      if (currentIndex >= combinedKomaImages.length) {
        console.warn(
          `🚀 currentIndex(${currentIndex})が範囲外です。最大値(${
            combinedKomaImages.length - 1
          })に修正します。`
        );
        const correctedIndex = combinedKomaImages.length - 1;
        setCurrentIndex(correctedIndex); // 直接設定（無限ループ回避）
        // saveCurrentIndexToStorage は useCurrentIndex フックで自動処理される
      } else if (currentIndex < 0) {
        console.warn(
          `🚀 currentIndex(${currentIndex})が範囲外です。0に修正します。`
        );
        setCurrentIndex(0); // 直接設定（無限ループ回避）
        // saveCurrentIndexToStorage は useCurrentIndex フックで自動処理される
      }
    }
  }, [currentIndex]); // saveCurrentIndexToStorageは依存配列から除外

  // ナビゲーション関数
  const handleNext = () => {
    const newIndex = (currentIndex + 1) % combinedKomaImages.length;
    updateCurrentIndex(newIndex);
  };

  const handlePrev = () => {
    const newIndex =
      (currentIndex - 1 + combinedKomaImages.length) %
      combinedKomaImages.length;
    updateCurrentIndex(newIndex);
  };

  const [characters, setCharacters] = useState<Character[]>([]);

  // updateKomaPathInImageData と saveToJSONWithData は useImageDataフックで提供





  // CSV保存機能は useImageData フックで提供。追加機能は別関数で処理
  const saveToCSVWithExtras = async (): Promise<void> => {
    try {
      // Human revisionの保存（必要な場合）
      if (isManualSave) {
        await ApiService.saveHumanRevision(imageData, currentImage, currentIndex, "ユーザーがCSVエクスポートボタンをクリック");
        setIsManualSave(false);
      }
      
      // フックのsaveToCSVを呼び出し
      await saveToCSV();
    } catch (error) {
      console.error("CSV保存エラー:", error);
    }
  };

  const handleCharacterChange = (
    imageKey: string,
    index: number,
    field: keyof Character,
    value: string | boolean | number | [number, number]
  ): void => {
    setImageData((prevData) => {
      const updatedCharacters = [
        ...prevData[imageKey as keyof ImageData].characters,
      ];
      (updatedCharacters[index] as any)[field] = value;
      return {
        ...prevData,
        [imageKey]: {
          ...prevData[imageKey as keyof ImageData],
          characters: updatedCharacters,
        },
      };
    });
  };

  const addCharacter = (imageKey: string): void => {
    const newCharacter = createEmptyCharacter();
    setCharacters([...characters, newCharacter]);
    setImageData((prevData) => ({
      ...prevData,
      [imageKey]: {
        ...prevData[imageKey as keyof ImageData],
        characters: [
          ...prevData[imageKey as keyof ImageData].characters,
          newCharacter,
        ],
      },
    }));
  };

  // セリフ編集用ヘルパー関数は useImageData フックで提供

  // addSerif は useImageData フックで提供（handleAddSerif）

  // removeSerif は useImageData フックで提供（handleRemoveSerif）

  // handleSerifSwap は useImageData フックで提供

  const removeCharacter = (imageKey: string, index: number): void => {
    setCharacters(characters.filter((_, i) => i !== index));
    setImageData((prevData) => ({
      ...prevData,
      [imageKey]: {
        ...prevData[imageKey as keyof ImageData],
        characters: prevData[imageKey as keyof ImageData].characters.filter(
          (_, i) => i !== index
        ),
      },
    }));
  };

  const handleSceneChange = (
    imageKey: string,
    field: string,
    value: string
  ): void => {
    setImageData((prevData) => ({
      ...prevData,
      [imageKey]: {
        ...prevData[imageKey as keyof ImageData],
        sceneData: {
          ...prevData[imageKey as keyof ImageData].sceneData,
          [field]: value,
        },
      },
    }));
  };

  const [isBlurred, setIsBlurred] = useState<boolean>(false);

  const toggleBlur = () => {
    setIsBlurred(!isBlurred); // ブラー状態を切り替え
  };

  const [results, setResults] = useState<any>(null);

  // ローディング状態（フック以外で管理するもののみ）
  const [isFetchingFeedback, setIsFetchingFeedback] = useState<boolean>(false);
  const [isDiscussing, setIsDiscussing] = useState<boolean>(false);
  const [isFourPanelAnalyzing, setIsFourPanelAnalyzing] =
    useState<boolean>(false);
  const [fourPanelPromptType, setFourPanelPromptType] =
    useState<PromptType>("combined");
  
  // Human in the Loop用の状態
  const [showRevisionHistory, setShowRevisionHistory] = useState<boolean>(false);
  const [currentRevisionHistory, setCurrentRevisionHistory] = useState<any>(null);
  const [showDiffViewer, setShowDiffViewer] = useState<boolean>(false);
  const [currentDiff, setCurrentDiff] = useState<any>(null);
  const [lastAIProposal, setLastAIProposal] = useState<ImageData | null>(null);
  const [isManualSave, setIsManualSave] = useState<boolean>(false);
  const [hasAnalysisRun, setHasAnalysisRun] = useState<boolean>(false);

  // グラウンドトゥルース機能用のstate
  const [useGroundTruth, setUseGroundTruth] = useState<boolean>(false);
  const [groundTruthIndex, setGroundTruthIndex] = useState<number | null>(null);
  
  // 検出モード切り替え用のstate
  const [detectionMode, setDetectionMode] = useState<"face_recognition" | "multiclass">("multiclass");
  const [isChangingMode, setIsChangingMode] = useState<boolean>(false);
  const [groundTruthData, setGroundTruthData] = useState<ImageData | null>(
    null
  );

  // Ray可視化機能用のstateは useAIDetection フックで管理

  // グラウンドトゥルースデータの読み込み
  useEffect(() => {
    if (
      useGroundTruth &&
      groundTruthIndex !== null &&
      groundTruthIndex !== currentIndex
    ) {
      const loadGroundTruthData = async () => {
        try {
          const url =
            IMAGE_SOURCE === "r2"
              ? `${WORKERS_API_URL}/api/data/saved_json/imageData_yuyu10_${groundTruthIndex}.json`
              : `saved_json/imageData_yuyu10_${groundTruthIndex}.json`;

          const response = await fetch(url);

          if (response.ok) {
            const data = await response.json();
            setGroundTruthData(data);
            console.log(
              `🎯 グラウンドトゥルースデータ読み込み成功: Index ${groundTruthIndex}`
            );
          } else {
            console.warn(
              `🎯 グラウンドトゥルースデータが存在しません: Index ${groundTruthIndex}`
            );
            setGroundTruthData(null);
          }
        } catch (error) {
          console.error("🎯 グラウンドトゥルースデータ読み込みエラー:", error);
          setGroundTruthData(null);
        }
      };
      loadGroundTruthData();
    } else if (!useGroundTruth) {
      setGroundTruthData(null);
    }
  }, [useGroundTruth, groundTruthIndex, currentIndex]);

  // 4コマ分のデータをクリアする関数（useImageDataフック + 追加機能）
  const handleClearFourPanelDataWithExtras = async (): Promise<void> => {
    if (window.confirm("現在の4コマ分のデータをすべてクリアしますか？この操作は取り消せません。")) {
      try {
        // フックの機能でメインデータをクリア
        await handleClearFourPanelData();
        
        // 追加機能：キャラクター、チャット、AI提案のクリア
        setCharacters([createEmptyCharacter()]);
        setChatHistory([]);
        setLastAIProposal(null);
        
        // JSONファイルも削除
        const response = await fetch("http://localhost:8000/api/delete-json", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            dataName: "yuyu10",
            currentIndex: currentIndex
          }),
        });
        
        if (response.ok) {
          console.log("JSONファイルを削除しました");
        } else {
          console.error("JSONファイルの削除に失敗しました");
        }
        
        console.log("4コマ分のデータをクリアしました");
      } catch (error) {
        console.error("JSONファイル削除エラー:", error);
      }
    }
  };

  // 検出モード切り替え関数
  const handleDetectionModeChange = async (newMode: "face_recognition" | "multiclass") => {
    if (isChangingMode) return;
    
    setIsChangingMode(true);
    try {
      const response = await axios.post("http://localhost:8000/api/detection-mode", {
        mode: newMode
      });
      
      if (response.data.success) {
        setDetectionMode(newMode);
        console.log("検出モード切り替え成功:", response.data.message);
      } else {
        console.error("検出モード切り替え失敗:", response.data.message);
      }
    } catch (error) {
      console.error("検出モード切り替えエラー:", error);
    } finally {
      setIsChangingMode(false);
    }
  };

  // 現在の検出モードを取得する関数
  const fetchCurrentDetectionMode = async () => {
    try {
      const response = await axios.get("http://localhost:8000/api/detection-mode");
      if (response.data.success) {
        setDetectionMode(response.data.current_mode);
      }
    } catch (error) {
      console.error("現在の検出モード取得エラー:", error);
    }
  };

  // 初期化時に現在のモードを取得
  useEffect(() => {
    fetchCurrentDetectionMode();
  }, []);

  const handleSubmit = async (imageKey: string): Promise<void> => {
    const formData = new FormData();
    // updateKomaPathInImageData は不要（imagePathList は常に最新の状態で管理されている）
    formData.append(
      "komaPath",
      JSON.stringify({komaPath: imagePathList[imageKey]})
    );

    setIsAnalyzing(true);
    try {
      const response = await axios.post(
        "http://localhost:8000/api/analyze-image-legacy/",
        {komaPath: imagePathList[imageKey]}
      );
      setResults(response);
      const contentData = response.data.content_data;
      // APIレスポンスのパスも正規化
      const normalizedContentData = normalizeImageDataPaths(contentData);
      setImageData((prevData) => ({
        ...prevData,
        [imageKey]: {
          ...prevData[imageKey as keyof ImageData],
          characters: (() => {
            let processedChars = normalizedContentData.characters
              .map((char: any) => ({
                character: char.character || "",
                faceDirection: char.faceDirection || "",
                position: char.position || "",
                shotType: char.shotType || "",
                characterSize: char.characterSize || "",
                expression: char.expression || "",
                clothing: char.clothing || "",
                isVisible: char.isVisible !== undefined ? char.isVisible : true,
                coordinate: char.coordinate || [0, 0],
              }))
              // 実質的に空のキャラクターデータをフィルターで除外（但し、最初はすべて保持）
              .filter((char: Character, index: number) => {
                // 最初のキャラクターは空でも保持（編集可能にするため）
                if (index === 0) return true;
                // 2番目以降は実質的に空でないもののみ保持
                return !isEmptyCharacterData(char);
              });
            // キャラクターが0の場合は最低1つの空フォームを確保
            if (processedChars.length === 0) {
              processedChars.push(createEmptyCharacter());
            }
            console.log(
              `個別分析処理後のキャラクター数:`,
              processedChars.length
            );
            return processedChars;
          })(),
          sceneData: {
            scene: contentData.sceneData?.scene || "",
            location: contentData.sceneData?.location || "",
            backgroundEffects: contentData.sceneData?.backgroundEffects || "",
            cameraAngle: contentData.sceneData?.cameraAngle || "",
            framing: contentData.sceneData?.framing || "",
          },
          serifs: contentData.serifs || [],
          charactersNum: (() => {
            const chars = normalizedContentData.characters.filter(
              (char: any) => !isEmptyCharacterData(char)
            );
            return contentData.charactersNum || chars.length;
          })(),
          serifsNum:
            contentData.serifsNum ||
            (contentData.serifs ? contentData.serifs.length : 0),
        },
      }));
    } catch (error) {
      console.error("Error uploading file:", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // AI検出機能は useAIDetection フックで提供済み。重複削除。

  // YOLO+DINOv2キャラクター検出機能は useAIDetection フックで提供されています。
  // 以下の重複関数は削除しました。




  const handleCharacterSwap = (
    imageKey: string,
    index1: number,
    index2: number
  ): void => {
    setImageData((prevData) => {
      const updatedCharacters = [
        ...prevData[imageKey as keyof ImageData].characters,
      ];
      const temp = updatedCharacters[index1];
      updatedCharacters[index1] = updatedCharacters[index2];
      updatedCharacters[index2] = temp;
      return {
        ...prevData,
        [imageKey]: {
          ...prevData[imageKey as keyof ImageData],
          characters: updatedCharacters,
        },
      };
    });
  };

  const fetchOverallFeedback = async (): Promise<void> => {
    setIsFetchingFeedback(true);
    try {
      const response = await axios.post("http://localhost:8000/api/feedback", {
        dataName: "yuyu10",
        currentIndex: currentIndex,
        imageData,
        imagePathList,
      });
      // console.log("Feedback:", response.data.feedback); // フィードバックをコンソールに表示
      // レスポンスデータの処理（Geminiの場合はcontent[0].textを取得）
      if (response.data.content && Array.isArray(response.data.content)) {
        setSummary(response.data.content[0].text);
      } else {
        setSummary(response.data);
      }

      // R2に自動保存（IMAGE_SOURCEがr2の場合）
      if (IMAGE_SOURCE === "r2" && response.data) {
        const feedbackPath = `saved_json/feedback_yuyu10_${currentIndex}.json`;
        await fetch(`${WORKERS_API_URL}/api/data/save`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            dataName: "yuyu10",
            currentIndex: currentIndex,
            path: feedbackPath,
            data: response.data,
          }),
        });
      }
    } catch (error) {
      console.error("Error fetching feedback:", error);
    } finally {
      setIsFetchingFeedback(false);
    }
  };
  const [chatHistory, setChatHistory] = useState<ChatItem[]>([]);
  const [discussionResult, setDiscussionResult] = useState<any>([]);

  const handleDiscussionButtonClick = async (
    question: string
  ): Promise<void> => {
    const requestData = {
      dataName: "yuyu10",
      currentIndex: currentIndex,
      imagePathList,
      imageData,
      summary,
      question,
    };

    setIsDiscussing(true);
    try {
      const response = await axios.post(
        "http://localhost:8000/api/discussion",
        requestData
      ); // LLMにリクエストを送信
      console.log(response);
      setDiscussionResult(response.data);

      // チャット履歴を更新
      const answer = response.data.content?.[0]?.text || response.data;
      setChatHistory((prev) => [...prev, {question, answer}]);

      // R2に自動保存（IMAGE_SOURCEがr2の場合）
      if (IMAGE_SOURCE === "r2") {
        const discussionPath = `saved_json/discussion_yuyu10_${currentIndex}.json`;
        const updatedHistory = [...chatHistory, {question, answer}];
        await fetch(`${WORKERS_API_URL}/api/data/save`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            dataName: "yuyu10",
            currentIndex: currentIndex,
            path: discussionPath,
            data: {chatHistory: updatedHistory},
          }),
        });
      }
    } catch (error) {
      console.error("Error fetching chat response:", error);
    } finally {
      setIsDiscussing(false);
    }
  };

  return (
    <div className="container">
      {/* データロード中はモーダル表示、AI分析中はインライン表示 */}
      <LoadingOverlay
        isLoading={isLoadingData}
        isAnalyzing={false}
        inline={false}
      />
      
      {/* AI分析中はインライン表示（非ブロッキング） */}
      <LoadingOverlay
        isLoading={isFourPanelAnalyzing || isYoloDetecting}
        isAnalyzing={true}
        inline={true}
      />

      {/* タブナビゲーション */}
      <TabNavigation
        currentTab={currentTab}
        onTabChange={setCurrentTab}
      />

      {/* タブコンテンツ */}
      {currentTab === "annotator" ? (
        <div className="annotator-content">
          <NavigationHeader
            currentIndex={currentIndex}
            totalImages={combinedKomaImages.length}
            currentImage={currentImage}
            isCurrentTitlePage={isCurrentTitlePage}
            fourPanelPromptType={fourPanelPromptType}
            isFourPanelAnalyzing={isFourPanelAnalyzing}
            isYoloDetecting={isYoloDetecting}
            useGroundTruth={useGroundTruth}
            groundTruthIndex={groundTruthIndex}
            showDetectionVisualization={showDetectionVisualization}
            onPrevious={handlePrev}
            onNext={handleNext}
            onIndexChange={(index) => {
              if (index >= 0 && index < combinedKomaImages.length) {
                setCurrentIndex(index);
                // useCurrentIndexフックで自動保存されるため、手動保存は不要
              }
            }}
            onPromptTypeChange={setFourPanelPromptType}
            onFourPanelAnalyzeAPI={() => AIAnalysisService.handleFourPanelAnalyze(imagePathList, imageData, setImageData, setIsFourPanelAnalyzing, fourPanelPromptType, saveToJSONWithData)}
            onFourPanelAnalyzeImprovedAPI={() => AIAnalysisService.handleFourPanelAnalyzeImproved(imagePathList, imageData, setImageData, setIsFourPanelAnalyzing, fourPanelPromptType, saveToJSONWithData)}
            onFourPanelYoloDetect={() => AIAnalysisService.handleFourPanelYoloDinov2Detection(imagePathList, imageData, setImageData, setIsYoloDetecting, saveToJSONWithData)}
            onGroundTruthToggle={(checked) => {
              setUseGroundTruth(checked);
              if (checked && groundTruthIndex === null) {
                const defaultIndex = currentIndex > 0 ? currentIndex - 1 : 0;
                setGroundTruthIndex(defaultIndex);
              }
            }}
            onGroundTruthIndexChange={(index) => {
              if (
                index === null ||
                (index >= 0 &&
                  index < combinedKomaImages.length &&
                  index !== currentIndex)
              ) {
                setGroundTruthIndex(index);
              }
            }}
            onToggleDetectionVisualization={setShowDetectionVisualization}
            onClearFourPanelData={handleClearFourPanelData}
          />

          {/* 検出モード切り替えパネル */}
          <div className="detection-mode-panel" style={{ 
            margin: '10px 0', 
            padding: '15px', 
            backgroundColor: '#f8f9fa', 
            borderRadius: '8px', 
            border: '1px solid #dee2e6' 
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '15px', flexWrap: 'wrap' }}>
              <div style={{ fontWeight: 'bold', color: '#495057' }}>
                検出モード:
              </div>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  className={`mode-toggle-btn ${detectionMode === 'face_recognition' ? 'active' : ''}`}
                  onClick={() => handleDetectionModeChange('face_recognition')}
                  disabled={isChangingMode}
                  style={{
                    padding: '8px 16px',
                    border: 'none',
                    borderRadius: '5px',
                    backgroundColor: detectionMode === 'face_recognition' ? '#007bff' : '#e9ecef',
                    color: detectionMode === 'face_recognition' ? 'white' : '#495057',
                    cursor: isChangingMode ? 'not-allowed' : 'pointer',
                    fontWeight: '500',
                    transition: 'all 0.2s ease'
                  }}
                >
                  顔検出+認識
                </button>
                <button
                  className={`mode-toggle-btn ${detectionMode === 'multiclass' ? 'active' : ''}`}
                  onClick={() => handleDetectionModeChange('multiclass')}
                  disabled={isChangingMode}
                  style={{
                    padding: '8px 16px',
                    border: 'none',
                    borderRadius: '5px',
                    backgroundColor: detectionMode === 'multiclass' ? '#28a745' : '#e9ecef',
                    color: detectionMode === 'multiclass' ? 'white' : '#495057',
                    cursor: isChangingMode ? 'not-allowed' : 'pointer',
                    fontWeight: '500',
                    transition: 'all 0.2s ease'
                  }}
                >
                  マルチクラス検出
                </button>
              </div>
              {isChangingMode && (
                <div style={{ color: '#6c757d', fontSize: '14px' }}>
                  切り替え中...
                </div>
              )}
              <div style={{ fontSize: '12px', color: '#6c757d', marginLeft: 'auto' }}>
                {detectionMode === 'face_recognition' 
                  ? '顔検出→DINOv2認識の2段階' 
                  : 'YOLO直接8クラス検出'}
              </div>
            </div>
          </div>

          <div className="main-content">
            <div className="koma-list">
              {[1, 2, 3, 4].map((num, index) => {
                const imageKey = `image${num}`;
                const panelData = imageData[imageKey as keyof ImageData];
                const imagePath = imagePathList[imageKey];

                // panelDataがundefinedの場合はスキップ（初期化中）
                if (!panelData) {
                  return (
                    <div key={num} className={`panel-${num} koma-item`}>
                      <div style={{ padding: '20px', textAlign: 'center' }}>
                        読み込み中...
                      </div>
                    </div>
                  );
                }

                // 入れ替えボタンを表示するかどうかの判定（統一）
                const shouldShowSwapButton = () => {
                  // 扉絵ページも通常ページも同じ：最後のコマ以外の後にボタンを表示
                  return num < 4;
                };

                return (
                  <React.Fragment key={num}>
                    <PanelEditor
                      panelData={panelData}
                      panelNumber={num}
                      imageKey={imageKey}
                      imagePath={imagePath}
                      isBlurred={isBlurred}
                      isCurrentTitlePage={isCurrentTitlePage}
                      currentPageNumber={currentPageNumber}
                      currentImage={currentImage}
                      isAnalyzing={isAnalyzing}
                      isYoloDetecting={isYoloDetecting}
                      rayVisualizationData={rayVisualizationData[imageKey]}
                      balloonDetectionData={balloonDetectionData[imageKey]}
                      characterDetectionData={characterDetectionData[imageKey]}
                      showDetectionVisualization={showDetectionVisualization}
                      onCharacterChange={handleCharacterChange}
                      onCharacterSwap={handleCharacterSwap}
                      onRemoveCharacter={removeCharacter}
                      onAddCharacter={addCharacter}
                      onSerifChange={handleSerifChange}
                      onAddSerif={handleAddSerif}
                      onRemoveSerif={handleRemoveSerif}
                      onSerifSwap={handleSerifSwap}
                      onYoloDetect={(imageKey, visualize) => handleAIDetectionWithTailShape(imageKey, imagePathList, setImageData, visualize, detectionMode)}
                      onSceneChange={(imageKey, field, value) => {
                        setImageData((prevData) => ({
                          ...prevData,
                          [imageKey]: {
                            ...prevData[imageKey as keyof ImageData],
                            sceneData: {
                              ...prevData[imageKey as keyof ImageData].sceneData,
                              [field]: value,
                            },
                          },
                        }));
                      }}
                    />
                    
                    {/* コマ間入れ替えボタン */}
                    {shouldShowSwapButton() && (
                      <div 
                        className="panel-swap-button-container" 
                        style={{
                          display: 'flex',
                          justifyContent: 'center',
                          alignItems: 'center',
                          padding: '8px 0',
                          marginTop: '8px',
                          marginBottom: '8px'
                        }}
                      >
                        <button
                          className="panel-swap-button"
                          onClick={() => {
                            const currentKey = `image${num}`;
                            const nextKey = `image${num + 1}`;
                            handlePanelSwap(currentKey, nextKey);
                          }}
                          title={`${num}コマ目と${num + 1}コマ目のデータを入れ替えます`}
                          style={{
                            padding: '6px 14px',
                            backgroundColor: '#6c757d',
                            color: 'white',
                            border: 'none',
                            borderRadius: '5px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '5px',
                            fontSize: '13px',
                            fontWeight: '500',
                            transition: 'all 0.2s ease',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = '#5a6268';
                            e.currentTarget.style.transform = 'scale(1.05)';
                            e.currentTarget.style.boxShadow = '0 3px 6px rgba(0,0,0,0.15)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = '#6c757d';
                            e.currentTarget.style.transform = 'scale(1)';
                            e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
                          }}
                        >
                          <span style={{ fontSize: '16px' }}>↔️</span>
                          コマ {num}-{num + 1} 入れ替え
                        </button>
                      </div>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>
          <div>
            {/* <button onClick={handleSubmit}>Analyze Image</button> */}
            {results && <div>{JSON.stringify(results)}</div>}
          </div>
          <HumanInTheLoopControls
            onSaveToCSV={() => {
              setIsManualSave(true);
              saveToCSV();
            }}
            onViewHistory={() => ApiService.fetchRevisionHistory(currentImage, setCurrentRevisionHistory, setShowRevisionHistory)}
            onExportLearningData={() => ApiService.exportLearningData("all", 2)}
            isSavingData={isSavingData}
          />

          <SummarySection
            summary={summary}
            isFetchingFeedback={isFetchingFeedback}
            onSummaryChange={setSummary}
            onFetchFeedback={fetchOverallFeedback}
          />

          <DiscussionSection
            isDiscussing={isDiscussing}
            chatHistory={chatHistory}
            onDiscussionClick={handleDiscussionButtonClick}
          />

          {/* 修正履歴ダイアログ */}
          <RevisionHistoryDialog
            isOpen={showRevisionHistory}
            history={currentRevisionHistory}
            onClose={() => setShowRevisionHistory(false)}
            onViewDiff={(revision) => {
              setCurrentDiff(revision.changes);
              setShowDiffViewer(true);
            }}
          />

          {/* 差分ビューア */}
          <DiffViewer
            isOpen={showDiffViewer}
            changes={currentDiff}
            onClose={() => setShowDiffViewer(false)}
          />
        </div>
      )}
    </div>
  );
};

export default App;
