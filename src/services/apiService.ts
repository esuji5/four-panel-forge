/**
 * API呼び出し関数のサービス層
 */
import axios from 'axios';
import { ImageData } from '../types/app';

export class ApiService {
  // AI提案を保存する関数
  static async saveAIProposal(
    proposalData: ImageData, 
    currentImage: string,
    currentIndex: number,
    model: string = "gemini", 
    processingTime?: number
  ): Promise<void> {
    try {
      const imagePath = `yuyu10/pages_corrected/combined_koma/${currentImage}`;
      const response = await axios.post("http://localhost:8000/api/save-ai-proposal", {
        imagePath,
        currentIndex,
        proposalData,
        model,
        processingTime,
        timestamp: new Date().toISOString()
      });
      
      console.log(`AI提案を保存しました (${model}):`, response.data);
    } catch (error) {
      console.error("AI提案の保存に失敗しました:", error);
      throw error;
    }
  }

  // 人間による修正を保存する関数
  static async saveHumanRevision(
    dataToSave: ImageData, 
    currentImage: string,
    currentIndex: number,
    notes: string = ""
  ): Promise<void> {
    try {
      const imagePath = `yuyu10/pages_corrected/combined_koma/${currentImage}`;
      const response = await axios.post("http://localhost:8000/api/save-human-revision", {
        imagePath,
        currentIndex,
        revisionData: dataToSave,
        notes,
        timestamp: new Date().toISOString()
      });
      
      console.log("人間による修正を保存しました:", response.data);
    } catch (error) {
      console.error("修正の保存に失敗しました:", error);
      throw error;
    }
  }

  // 修正履歴を取得する関数
  static async fetchRevisionHistory(
    currentImage: string,
    setCurrentRevisionHistory: (history: any) => void,
    setShowRevisionHistory: (show: boolean) => void
  ): Promise<void> {
    try {
      const imagePath = `yuyu10/pages_corrected/combined_koma/${currentImage}`;
      const encodedPath = encodeURIComponent(imagePath);
      const response = await axios.get(`http://localhost:8000/api/get-revision-history/${encodedPath}`);
      
      setCurrentRevisionHistory(response.data);
      setShowRevisionHistory(true);
    } catch (error) {
      console.error("修正履歴の取得に失敗しました:", error);
      throw error;
    }
  }

  // 学習データをエクスポートする関数
  static async exportLearningData(
    exportType: string = "all", 
    minRevisions: number = 2
  ): Promise<void> {
    try {
      const response = await axios.post("http://localhost:8000/api/export-learning-data", {
        exportType,
        minRevisions,
        timestamp: new Date().toISOString()
      });
      
      console.log("学習データのエクスポートが完了しました:", response.data);
      
      // 成功メッセージを表示
      alert(`学習データのエクスポートが完了しました。\n${response.data.exported_files} ファイルが出力されました。`);
    } catch (error) {
      console.error("学習データのエクスポートに失敗しました:", error);
      alert("学習データのエクスポートに失敗しました。詳細はコンソールを確認してください。");
      throw error;
    }
  }

  // 全体フィードバックを取得する関数
  static async fetchOverallFeedback(
    imageData: ImageData,
    currentIndex: number,
    currentImage: string,
    setResults: (results: any) => void,
    setIsFetchingFeedback: (fetching: boolean) => void
  ): Promise<void> {
    setIsFetchingFeedback(true);
    try {
      const response = await axios.post("http://localhost:8000/api/feedback", {
        data: imageData,
        currentIndex,
        currentImage,
      });
      
      setResults(response.data);
      
      // フィードバックをファイルに保存
      try {
        const feedbackPath = `saved_json/feedback_yuyu10_${currentIndex}.json`;
        console.log(`フィードバックを保存中: ${feedbackPath}`);
        // TODO: ファイル保存処理の実装
      } catch (saveError) {
        console.warn("フィードバックの保存に失敗しましたが、処理を続行します:", saveError);
      }
    } catch (error) {
      console.error("フィードバックの取得に失敗しました:", error);
      alert("フィードバックの取得に失敗しました。サーバーとの接続を確認してください。");
      throw error;
    } finally {
      setIsFetchingFeedback(false);
    }
  }

  // ディスカッション（AIとの対話）処理
  static async handleDiscussion(
    question: string,
    imageData: ImageData,
    currentIndex: number,
    chatHistory: any[],
    setChatHistory: (history: any[]) => void,
    setDiscussionResult: (result: any) => void,
    setIsDiscussing: (discussing: boolean) => void
  ): Promise<void> {
    setIsDiscussing(true);
    const requestData = {
      question,
      imageData,
      currentIndex,
      chatHistory,
    };

    try {
      const response = await axios.post(
        "http://localhost:8000/api/discussion",
        requestData
      );
      
      const answer = response.data.content?.[0]?.text || response.data;
      
      try {
        // ディスカッション履歴をファイルに保存
        const discussionPath = `saved_json/discussion_yuyu10_${currentIndex}.json`;
        const updatedHistory = [...chatHistory, {question, answer}];
        console.log(`ディスカッション履歴を保存中: ${discussionPath}`);
        // TODO: ファイル保存処理の実装
        
        setChatHistory(updatedHistory);
      } catch (saveError) {
        console.warn("ディスカッション履歴の保存に失敗しましたが、処理を続行します:", saveError);
      }
      
      setDiscussionResult(answer);
    } catch (error) {
      console.error("ディスカッションの処理に失敗しました:", error);
      alert("ディスカッションの処理に失敗しました。サーバーとの接続を確認してください。");
      throw error;
    } finally {
      setIsDiscussing(false);
    }
  }

  // CSV保存（拡張版）
  static async saveToCSVWithExtras(
    imageData: ImageData,
    currentIndex: number,
    currentImage: string,
    rows: any[],
    isManualSave: boolean,
    setIsManualSave: (manual: boolean) => void
  ): Promise<void> {
    try {
      // データ名を現在の画像パスから抽出
      const dataName = currentImage ? currentImage.split('/')[1] : 'unknown';
      
      // 通常のCSV保存
      const response = await axios.post("http://localhost:8000/api/save-csv", {
        dataName,
        currentIndex,
        komaPath: currentImage || '',
        imageData,
        summary: '' // 空文字列で初期化
      });
      
      console.log("CSV保存完了:", response.data);

      // Human revisionの保存（必要な場合）
      if (isManualSave) {
        await ApiService.saveHumanRevision(
          imageData, 
          currentImage,
          currentIndex,
          "ユーザーがCSVエクスポートボタンをクリック"
        );
        setIsManualSave(false);
      }
    } catch (error) {
      console.error("CSV保存エラー:", error);
      throw error;
    }
  }

  // JSON削除処理
  static async deleteJSON(currentIndex: number): Promise<void> {
    try {
      const response = await fetch("http://localhost:8000/api/delete-json", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          dataName: "yuyu10",
          currentIndex: currentIndex,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        console.log("JSON削除完了:", result.message);
      } else {
        console.error("JSON削除に失敗しました:", response.statusText);
      }
    } catch (error) {
      console.error("JSON削除エラー:", error);
      throw error;
    }
  }
}