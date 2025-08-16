/**
 * 画像パスを正規化する
 * -pad-shaved を削除し、一貫性のあるパスにする
 */
export const normalizeImagePath = (path: string): string => {
  if (!path) return path;
  
  // -pad-shaved を削除
  return path.replace('-pad-shaved', '');
};

/**
 * 画像データ内のすべての画像パスを正規化する
 */
export const normalizeImageDataPaths = (data: any): any => {
  if (!data) return data;
  
  // 深いコピーを作成
  const normalized = JSON.parse(JSON.stringify(data));
  
  // komaPath を持つすべてのオブジェクトを処理
  const processObject = (obj: any): void => {
    if (!obj || typeof obj !== 'object') return;
    
    // komaPath プロパティがあれば正規化
    if (obj.komaPath && typeof obj.komaPath === 'string') {
      obj.komaPath = normalizeImagePath(obj.komaPath);
    }
    
    // 再帰的に処理
    Object.values(obj).forEach(value => {
      if (Array.isArray(value)) {
        value.forEach(item => processObject(item));
      } else if (typeof value === 'object') {
        processObject(value);
      }
    });
  };
  
  processObject(normalized);
  return normalized;
};

/**
 * Ground Truth データの画像パスを正規化する
 */
export const normalizeGroundTruthPaths = (groundTruthData: any): any => {
  const normalized: any = {};
  
  Object.entries(groundTruthData).forEach(([key, dataset]) => {
    normalized[key] = normalizeImageDataPaths(dataset);
  });
  
  return normalized;
};