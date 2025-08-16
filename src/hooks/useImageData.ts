import { useState, useEffect } from 'react';
import axios from 'axios';
import Papa from 'papaparse';
import { 
  ImageData, 
  ImagePathList, 
  PanelData, 
  CSVRow,
  Serif,
  Character,
  SceneData 
} from '../types/app';
import { 
  generateImagePathList,
  getPageNumber,
  isTitlePage
} from '../utils/imageProcessing';
import { useCurrentIndex } from './useLocalStorage';
import { normalizeImageDataPaths } from '../utils/imagePath';
import { createEmptyPanelData } from '../utils/dataHelpers';
import { getImagePath } from '../config';

// 空の画像データを作成
const createEmptyImageData = (imagePathList: ImagePathList): ImageData => ({
  image1: createEmptyPanelData(imagePathList.image1 || ""),
  image2: createEmptyPanelData(imagePathList.image2 || ""),
  image3: createEmptyPanelData(imagePathList.image3 || ""),
  image4: createEmptyPanelData(imagePathList.image4 || ""),
});

export const useImageData = () => {
  const { currentIndex, setCurrentIndex, updateCurrentIndex, currentImage } = useCurrentIndex();
  
  // 現在の画像に基づく導出値
  const isCurrentTitlePage = isTitlePage(currentImage);
  const currentPageNumber = getPageNumber(currentImage);
  const imagePathList: ImagePathList = generateImagePathList(currentImage, isCurrentTitlePage, currentPageNumber);
  
  // 初期値を適切に設定
  const [imageData, setImageData] = useState<ImageData>(() => createEmptyImageData(imagePathList));
  const [summary, setSummary] = useState<string>("");
  const [rows, setRows] = useState<CSVRow[]>([]);
  const [isLoadingData, setIsLoadingData] = useState<boolean>(false);
  const [isSavingData, setIsSavingData] = useState<boolean>(false);

  // CSV読み込み
  const fetchCSV = async () => {
    try {
      const response = await fetch("/yuyu10_kanji.csv");
      const csvText = await response.text();
      Papa.parse(csvText, {
        header: true,
        complete: (results) => {
          setRows(results.data as CSVRow[]);
        },
      });
    } catch (error) {
      console.error("CSV読み込みエラー:", error);
    }
  };

  // 画像データ読み込み
  const loadImageData = async () => {
    setIsLoadingData(true);
    try {
      const response = await axios.get(
        `http://localhost:8000/api/load-json?dataName=yuyu10&currentIndex=${currentIndex}`
      );
      
      if (response.data) {
        // 新しいJSON形式では、response.data.imageDataが実際の画像データ
        const actualImageData = response.data.imageData || response.data;
        const normalizedData = normalizeImageDataPaths(actualImageData);
        setImageData(normalizedData);
        console.log("画像データ読み込み完了:", normalizedData);
      } else {
        console.log("保存済みデータが見つかりません。空のデータで初期化します。");
        setImageData(createEmptyImageData(imagePathList));
      }
    } catch (error) {
      console.error("画像データ読み込みエラー:", error);
      setImageData(createEmptyImageData(imagePathList));
    } finally {
      setIsLoadingData(false);
    }
  };

  // JSONデータ保存
  const saveToJSONWithData = async (dataToSave: ImageData, notes: string = ""): Promise<void> => {
    setIsSavingData(true);
    try {
      await axios.post("http://localhost:8000/api/save-json", {
        dataName: "yuyu10",
        currentIndex: currentIndex,
        komaPath: `yuyu10/pages_corrected/combined_koma/${currentImage}`,
        imageData: dataToSave,
        summary,
      });
      console.log(
        "自動JSON保存完了: imageData_yuyu10_" + currentIndex + ".json",
        "画像ファイル名:", currentImage
      );
      
      // 修正履歴も保存
      try {
        // 画像パスを適切に構築（currentImageが既にフルパスか部分パスかを判定）
        let imagePath: string;
        if (currentImage.includes('/')) {
          // 既にパスが含まれている場合はそのまま使用
          imagePath = currentImage;
        } else {
          // ファイル名のみの場合は標準パスを構築
          imagePath = `yuyu10/pages_corrected/combined_koma/${currentImage}`;
        }
        
        console.log(`修正履歴保存中 - imagePath: ${imagePath}, currentIndex: ${currentIndex}`);
        
        await axios.post("http://localhost:8000/api/save-human-revision", {
          imagePath,
          imageData: dataToSave,
          currentIndex,
          notes: notes || "手動保存",
          timestamp: new Date().toISOString()
        });
        console.log("修正履歴を保存しました");
      } catch (historyError) {
        console.warn("修正履歴の保存に失敗しました（メイン保存は成功）:", historyError);
        
        // エラーの型を安全に確認してから詳細情報を表示
        if (historyError && typeof historyError === 'object') {
          const errorObj = historyError as any;
          const errorDetail = errorObj.response?.data || errorObj.message || '詳細不明';
          console.warn("エラー詳細:", errorDetail);
        }
        
        // 履歴保存が失敗してもメイン保存は成功しているので続行
      }
    } catch (error) {
      console.error("JSON保存エラー:", error);
      throw error;
    } finally {
      setIsSavingData(false);
    }
  };

  // CSV保存
  const saveToCSV = async (): Promise<void> => {
    try {
      // データ名を現在の画像パスから抽出
      // currentImageが'four_panel_009.jpg'のような形式の場合
      let dataName = 'unknown';
      if (currentImage) {
        if (currentImage.includes('/')) {
          // パス形式の場合（例: '/yuyu10/...'）
          dataName = currentImage.split('/')[1];
        } else {
          // ファイル名のみの場合（例: 'four_panel_009.jpg'）
          // ファイル名から推測（four_panel_XXX.jpg -> yuyu10など）
          if (currentImage.startsWith('four_panel_')) {
            dataName = 'yuyu10'; // デフォルトデータセット名
          } else {
            dataName = 'unknown';
          }
        }
      }
      
      // デバッグ情報
      console.log('🔍 CSV保存デバッグ情報:');
      console.log('  currentImage:', currentImage);
      console.log('  dataName:', dataName);
      console.log('  currentIndex:', currentIndex);
      
      const requestData = {
        dataName,
        currentIndex,
        komaPath: currentImage || '',
        imageData,
        summary: '' // 空文字列で初期化
      };
      
      console.log('📤 送信データ:', requestData);
      
      const response = await axios.post("http://localhost:8000/api/save-csv", requestData);
      console.log("CSV保存完了:", response.data);
    } catch (error) {
      console.error("CSV保存エラー:", error);
      throw error;
    }
  };

  // データクリア
  const handleClearFourPanelData = async (): Promise<void> => {
    try {
      const emptyData = createEmptyImageData(imagePathList);
      setImageData(emptyData);
      await saveToJSONWithData(emptyData);
      console.log("4パネルデータクリア完了");
    } catch (error) {
      console.error("データクリアエラー:", error);
    }
  };

  // セリフ変更ハンドラ
  const handleSerifChange = (
    imageKey: string,
    serifIndex: number,
    field: string,
    value: string | number | [number, number]
  ) => {
    setImageData((prevData) => {
      const panel = prevData[imageKey as keyof ImageData];
      if (!panel || !panel.serifs) return prevData;

      const updatedSerifs = [...panel.serifs];
      if (updatedSerifs[serifIndex]) {
        (updatedSerifs[serifIndex] as any)[field] = value;
      }

      return {
        ...prevData,
        [imageKey]: {
          ...panel,
          serifs: updatedSerifs,
        },
      };
    });
  };

  // セリフ追加
  const handleAddSerif = (imageKey: string) => {
    setImageData((prevData) => {
      const panel = prevData[imageKey as keyof ImageData];
      if (!panel) return prevData;

      const currentSerifs = panel.serifs || [];
      const newSerif: Serif = {
        dialogueId: `${imageKey}_serif_${currentSerifs.length + 1}`,
        text: '',
        type: 'speech_bubble',
        speakerCharacterId: '',
        readingOrderIndex: currentSerifs.length + 1,
        coordinate: [0, 0],
      };

      return {
        ...prevData,
        [imageKey]: {
          ...panel,
          serifs: [...currentSerifs, newSerif],
          serifsNum: currentSerifs.length + 1,
        },
      };
    });
  };

  // セリフ削除
  const handleRemoveSerif = (imageKey: string, serifIndex: number) => {
    setImageData((prevData) => {
      const panel = prevData[imageKey as keyof ImageData];
      if (!panel || !panel.serifs) return prevData;

      const updatedSerifs = panel.serifs.filter((_, index) => index !== serifIndex);

      return {
        ...prevData,
        [imageKey]: {
          ...panel,
          serifs: updatedSerifs,
          serifsNum: updatedSerifs.length,
        },
      };
    });
  };

  // セリフ順序変更
  const handleSerifSwap = (imageKey: string, index1: number, index2: number) => {
    setImageData((prevData) => {
      const panel = prevData[imageKey as keyof ImageData];
      if (!panel || !panel.serifs) return prevData;

      const updatedSerifs = [...panel.serifs];
      [updatedSerifs[index1], updatedSerifs[index2]] = [updatedSerifs[index2], updatedSerifs[index1]];

      return {
        ...prevData,
        [imageKey]: {
          ...panel,
          serifs: updatedSerifs,
        },
      };
    });
  };

  // コマ間のデータ入れ替え
  const handlePanelSwap = (panelKey1: string, panelKey2: string) => {
    setImageData((prevData) => {
      const panel1 = prevData[panelKey1 as keyof ImageData];
      const panel2 = prevData[panelKey2 as keyof ImageData];
      
      if (!panel1 || !panel2) return prevData;

      // パネルデータを入れ替える（komaPathは保持）
      const panel1KomaPath = panel1.komaPath;
      const panel2KomaPath = panel2.komaPath;
      
      return {
        ...prevData,
        [panelKey1]: {
          ...panel2,
          komaPath: panel1KomaPath, // komaPathは元のまま
        },
        [panelKey2]: {
          ...panel1,
          komaPath: panel2KomaPath, // komaPathは元のまま
        },
      };
    });
  };

  // 初期化時の処理
  useEffect(() => {
    fetchCSV();
    loadImageData();
  }, [currentIndex]);

  return {
    // State
    imageData,
    setImageData,
    summary,
    setSummary,
    rows,
    currentIndex,
    setCurrentIndex: updateCurrentIndex,
    currentImage,
    imagePathList,
    isLoadingData,
    setIsLoadingData,
    isSavingData,
    
    // Computed
    isCurrentTitlePage,
    currentPageNumber,
    
    // Functions
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
  };
};