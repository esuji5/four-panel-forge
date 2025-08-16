// アプリケーション設定

type ImageSource = 'local' | 'r2';

// 画像のベースURL設定
// ローカル開発時はpublicフォルダから、本番環境ではR2から読み込む
export const IMAGE_BASE_URL: string = process.env.REACT_APP_IMAGE_BASE_URL || '';

// R2のパブリックURL（本番環境用）
export const R2_PUBLIC_URL: string = process.env.REACT_APP_R2_PUBLIC_URL || 'https://pub-ff5f6cba29df4c968f5ea14b6e3f78e7.r2.dev';

// Workers APIのURL
export const WORKERS_API_URL: string = process.env.REACT_APP_WORKERS_API_URL || 'http://localhost:8787';

// 画像ソースの切り替え（local or r2）
export const IMAGE_SOURCE: ImageSource = (process.env.REACT_APP_IMAGE_SOURCE as ImageSource) || 'local';

// 画像パスを取得する関数
export const getImagePath = (path: string): string => {
  if (IMAGE_SOURCE === 'r2') {
    // R2のパブリックURLから直接画像を取得
    const cleanPath = path.startsWith('/') ? path.slice(1) : path;
    return `${R2_PUBLIC_URL}/${cleanPath}`;
  } else {
    // ローカルのpublicフォルダから取得
    const cleanPath = path.startsWith('/') ? path.slice(1) : path;
    // IMAGE_BASE_URLが空の場合は、先頭に / を付けて返す
    return IMAGE_BASE_URL ? `${IMAGE_BASE_URL}/${cleanPath}` : `/${cleanPath}`;
  }
};