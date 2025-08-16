import { getImagePath } from "../config";
import { normalizeImagePath } from "./imagePath";
import type { ImagePathList } from "../types/app";

// 扉絵判定関数（-1, -2がついていない画像は扉絵）
export const isTitlePage = (filename: string): boolean => {
  return !filename.includes("-1") && !filename.includes("-2");
};

// ページ番号を取得する関数
export const getPageNumber = (filename: string): string => {
  const match = filename.match(/four_panel_(\d+)/);
  return match ? match[1] : "";
};

// 画像パスリストを生成
export const generateImagePathList = (
  currentImage: string,
  isCurrentTitlePage: boolean,
  currentPageNumber: string
): ImagePathList => {
  if (isCurrentTitlePage) {
    // 扉絵の場合：同じページの3,4,7,8の個別画像を表示
    const individualImageDir =
      "yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/";
    return {
      image1: normalizeImagePath(
        getImagePath(
          `${individualImageDir}image-${currentPageNumber}-3-pad-shaved.jpg`
        )
      ),
      image2: normalizeImagePath(
        getImagePath(
          `${individualImageDir}image-${currentPageNumber}-4-pad-shaved.jpg`
        )
      ),
      image3: normalizeImagePath(
        getImagePath(
          `${individualImageDir}image-${currentPageNumber}-7-pad-shaved.jpg`
        )
      ),
      image4: normalizeImagePath(
        getImagePath(
          `${individualImageDir}image-${currentPageNumber}-8-pad-shaved.jpg`
        )
      ),
    };
  } else {
    // 通常ページの場合：-1なら1,2,3,4、-2なら5,6,7,8の個別画像を表示
    const individualImageDir =
      "yuyu10/pages_corrected/2_paint_out/0_koma/0_padding_shave/";
    const isPage1 = currentImage.includes("-1");
    const komaNumbers = isPage1 ? [1, 2, 3, 4] : [5, 6, 7, 8];

    return {
      image1: normalizeImagePath(
        getImagePath(
          `${individualImageDir}image-${currentPageNumber}-${komaNumbers[0]}-pad-shaved.jpg`
        )
      ),
      image2: normalizeImagePath(
        getImagePath(
          `${individualImageDir}image-${currentPageNumber}-${komaNumbers[1]}-pad-shaved.jpg`
        )
      ),
      image3: normalizeImagePath(
        getImagePath(
          `${individualImageDir}image-${currentPageNumber}-${komaNumbers[2]}-pad-shaved.jpg`
        )
      ),
      image4: normalizeImagePath(
        getImagePath(
          `${individualImageDir}image-${currentPageNumber}-${komaNumbers[3]}-pad-shaved.jpg`
        )
      ),
    };
  }
};

// 4コマ画像を縦に結合する関数
export const combineFourPanelImages = async (imagePathList: ImagePathList): Promise<string> => {
  const imagePaths = [
    imagePathList.image1,
    imagePathList.image2,
    imagePathList.image3,
    imagePathList.image4,
  ];

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Canvas context not available");
  }

  const images: HTMLImageElement[] = [];

  // 4つの画像を順番に読み込み
  for (let i = 0; i < 4; i++) {
    const img = new Image();
    img.crossOrigin = "anonymous";

    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () =>
        reject(new Error(`Failed to load image ${i + 1}: ${imagePaths[i]}`));

      // パスの正規化とエラーハンドリング
      const normalizedPath = normalizeImagePath(imagePaths[i]);
      const imagePath = normalizedPath.startsWith("/")
        ? normalizedPath.substring(1)
        : normalizedPath;
      const fullImageUrl = getImagePath(imagePath);

      console.log(`コマ${i + 1}画像読み込み: ${fullImageUrl}`);
      img.src = fullImageUrl;
    });

    images.push(img);
  }

  // キャンバスサイズを設定（最大幅、高さの合計）
  const maxWidth = Math.max(...images.map((img) => img.width));
  const totalHeight = images.reduce((sum, img) => sum + img.height, 0);

  canvas.width = maxWidth;
  canvas.height = totalHeight;

  // 白背景を設定
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // 画像を縦に並べて描画（中央寄せ）
  let currentY = 0;
  for (const img of images) {
    const x = (maxWidth - img.width) / 2; // 中央寄せ
    ctx.drawImage(img, x, currentY);
    currentY += img.height;
  }

  // Base64エンコードして返す
  return canvas.toDataURL("image/jpeg", 0.9).split(",")[1];
};

// グラウンドトゥルース画像を読み込む関数
export const loadGroundTruthImage = async (imageIndex: number, combinedKomaImages: string[]): Promise<string> => {
  try {
    const imageName = combinedKomaImages[imageIndex];
    const imagePath = `yuyu10/pages_corrected/combined_koma/${imageName}`;

    const img = new Image();
    img.crossOrigin = "anonymous";

    return new Promise<string>((resolve, reject) => {
      img.onload = () => {
        try {
          const canvas = document.createElement("canvas");
          const ctx = canvas.getContext("2d");
          if (!ctx) {
            reject(new Error("Canvas context not available"));
            return;
          }

          canvas.width = img.width;
          canvas.height = img.height;

          // 白背景を設定
          ctx.fillStyle = "white";
          ctx.fillRect(0, 0, canvas.width, canvas.height);

          // 画像を描画
          ctx.drawImage(img, 0, 0);

          // Base64エンコードして返す
          const base64 = canvas.toDataURL("image/jpeg", 0.9).split(",")[1];
          resolve(base64);
        } catch (error) {
          reject(error);
        }
      };

      img.onerror = () =>
        reject(new Error(`Failed to load ground truth image: ${imagePath}`));

      const fullImageUrl = getImagePath(imagePath);
      console.log(`グラウンドトゥルース画像読み込み: ${fullImageUrl}`);
      img.src = fullImageUrl;
    });
  } catch (error) {
    console.error("グラウンドトゥルース画像読み込みエラー:", error);
    throw error;
  }
};