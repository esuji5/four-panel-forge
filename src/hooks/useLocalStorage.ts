import { useState, useCallback } from 'react';
import { combinedKomaImages } from '../constants/imageList';

export const useLocalStorage = <T>(key: string, initialValue: T): [T, (value: T) => void] => {
  // localStorageから値を読み込むヘルパー関数
  const readFromStorage = (): T => {
    try {
      const item = localStorage.getItem(key);
      if (item !== null) {
        const parsed = JSON.parse(item);
        console.log(`💾 localStorageから${key}を復元:`, parsed);
        return parsed;
      }
    } catch (error) {
      console.error(`💾 localStorageから${key}の読み込みエラー:`, error);
    }
    console.log(`💾 ${key}をデフォルト値で初期化`);
    return initialValue;
  };

  // 初期値を設定
  const [storedValue, setStoredValue] = useState<T>(() => readFromStorage());

  // localStorageに保存するヘルパー関数
  const setValue = useCallback((value: T) => {
    try {
      setStoredValue(value);
      localStorage.setItem(key, JSON.stringify(value));
      console.log(`💾 ${key}をlocalStorageに保存:`, value);
    } catch (error) {
      console.error(`💾 localStorageへの${key}保存エラー:`, error);
    }
  }, [key]);

  return [storedValue, setValue];
};

// currentIndex専用のフック
export const useCurrentIndex = () => {
  const loadCurrentIndexFromStorage = (): number => {
    try {
      const savedIndex = localStorage.getItem('fun_annotator_currentIndex');
      if (savedIndex !== null) {
        const parsedIndex = parseInt(savedIndex, 10);
        if (!isNaN(parsedIndex) && parsedIndex >= 0) {
          console.log('💾 localStorageからcurrentIndexを復元:', parsedIndex);
          return parsedIndex;
        } else {
          console.warn('💾 localStorageのcurrentIndexが無効な値:', savedIndex);
        }
      } else {
        console.log('💾 localStorageにcurrentIndexが未設定');
      }
    } catch (error) {
      console.error('💾 localStorageからcurrentIndexの読み込みエラー:', error);
    }
    console.log('💾 currentIndexをデフォルト値(0)で初期化');
    return 0;
  };

  const saveCurrentIndexToStorage = (index: number): void => {
    try {
      localStorage.setItem('fun_annotator_currentIndex', index.toString());
      console.log('💾 currentIndexをlocalStorageに保存:', index);
    } catch (error) {
      console.error('💾 localStorageへのcurrentIndex保存エラー:', error);
    }
  };

  const [currentIndex, setCurrentIndex] = useState<number>(() => loadCurrentIndexFromStorage());

  const updateCurrentIndex = useCallback((newIndex: number): void => {
    setCurrentIndex(newIndex);
    saveCurrentIndexToStorage(newIndex);
  }, []);

  const currentImage = combinedKomaImages[currentIndex] || "";

  return { currentIndex, updateCurrentIndex, setCurrentIndex, saveCurrentIndexToStorage, currentImage };
};