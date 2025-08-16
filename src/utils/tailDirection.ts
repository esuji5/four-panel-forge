/**
 * しっぽの方向を日本語に変換するユーティリティ
 */

export const getTailDirectionDisplayName = (direction: string): string => {
  const directionMap: { [key: string]: string } = {
    'top': '上',
    'bottom': '下',
    'left': '左',
    'right': '右',
    'top-left': '左上',
    'top-right': '右上',
    'bottom-left': '左下',
    'bottom-right': '右下',
    'center': '中央'
  };
  
  return directionMap[direction] || direction;
};