/**
 * Ray可視化のためのSVG生成ユーティリティ
 */
import { RayVisualizationData } from './speakerIdentification';

/**
 * Ray可視化用のSVG要素を生成
 */
// ツールチップ管理用のヘルパー関数
let currentTooltip: HTMLElement | null = null;

const addTooltipEvents = (element: SVGElement, tooltipText: string) => {
  element.style.cursor = 'pointer';
  
  element.addEventListener('mouseenter', (e: MouseEvent) => {
    // 既存のツールチップを削除
    if (currentTooltip) {
      document.body.removeChild(currentTooltip);
    }
    
    // 新しいツールチップを作成
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = tooltipText;
    tooltip.style.position = 'fixed';
    tooltip.style.left = `${e.clientX}px`;
    tooltip.style.top = `${e.clientY - 40}px`;
    tooltip.style.background = 'rgba(0, 0, 0, 0.9)';
    tooltip.style.color = 'white';
    tooltip.style.padding = '8px 12px';
    tooltip.style.borderRadius = '6px';
    tooltip.style.fontSize = '12px';
    tooltip.style.fontFamily = '"Helvetica Neue", Arial, sans-serif';
    tooltip.style.whiteSpace = 'nowrap';
    tooltip.style.pointerEvents = 'none';
    tooltip.style.zIndex = '10000';
    tooltip.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.3)';
    tooltip.style.transform = 'translateX(-50%)';
    
    document.body.appendChild(tooltip);
    currentTooltip = tooltip;
  });
  
  element.addEventListener('mousemove', (e: MouseEvent) => {
    if (currentTooltip) {
      currentTooltip.style.left = `${e.clientX}px`;
      currentTooltip.style.top = `${e.clientY - 40}px`;
    }
  });
  
  element.addEventListener('mouseleave', () => {
    if (currentTooltip) {
      document.body.removeChild(currentTooltip);
      currentTooltip = null;
    }
  });
};

export const createRayVisualizationSVG = (rayData: RayVisualizationData[], imageWidth: number, imageHeight: number): SVGElement | null => {
  if (!rayData || rayData.length === 0) {
    return null;
  }

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'ray-visualization');
  svg.setAttribute('width', imageWidth.toString());
  svg.setAttribute('height', imageHeight.toString());
  svg.style.position = 'absolute';
  svg.style.top = '0';
  svg.style.left = '0';
  svg.style.pointerEvents = 'auto'; // ツールチップ用にイベントを有効化
  svg.style.zIndex = '1000';
  
  console.log(`🎨 SVG作成: サイズ=${imageWidth}x${imageHeight}, z-index=1000`);

  // ツールチップ用のスタイルを追加
  const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
  style.textContent = `
    .tooltip {
      position: absolute;
      background: rgba(0, 0, 0, 0.9);
      color: white;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-family: 'Helvetica Neue', Arial, sans-serif;
      white-space: nowrap;
      pointer-events: none;
      z-index: 10000;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
      transform: translateX(-50%);
    }
    .tooltip::before {
      content: '';
      position: absolute;
      top: 100%;
      left: 50%;
      transform: translateX(-50%);
      border: 6px solid transparent;
      border-top-color: rgba(0, 0, 0, 0.9);
    }
  `;
  svg.appendChild(style);

  // マーカー定義を追加
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  
  // 大きな矢印マーカー
  const arrowhead = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
  arrowhead.setAttribute('id', 'arrowhead');
  arrowhead.setAttribute('markerWidth', '10');
  arrowhead.setAttribute('markerHeight', '7');
  arrowhead.setAttribute('refX', '9');
  arrowhead.setAttribute('refY', '3.5');
  arrowhead.setAttribute('orient', 'auto');
  
  const arrowPolygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
  arrowPolygon.setAttribute('points', '0 0, 10 3.5, 0 7');
  arrowPolygon.setAttribute('fill', '#666');
  arrowPolygon.setAttribute('opacity', '0.8');
  arrowhead.appendChild(arrowPolygon);
  
  // 小さな矢印マーカー
  const smallArrowhead = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
  smallArrowhead.setAttribute('id', 'smallArrowhead');
  smallArrowhead.setAttribute('markerWidth', '8');
  smallArrowhead.setAttribute('markerHeight', '6');
  smallArrowhead.setAttribute('refX', '7');
  smallArrowhead.setAttribute('refY', '3');
  smallArrowhead.setAttribute('orient', 'auto');
  
  const smallArrowPolygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
  smallArrowPolygon.setAttribute('points', '0 0, 8 3, 0 6');
  smallArrowPolygon.setAttribute('fill', '#333');
  smallArrowPolygon.setAttribute('opacity', '0.6');
  smallArrowhead.appendChild(smallArrowPolygon);
  
  defs.appendChild(arrowhead);
  defs.appendChild(smallArrowhead);
  svg.appendChild(defs);

  // 各Rayを描画
  rayData.forEach((ray, index) => {
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('class', `ray-group-${index}`);

    console.log(`🎯 Ray${index + 1}描画開始:`, ray);
    console.log(`  起点: (${ray.rayOrigin.x}, ${ray.rayOrigin.y})`);
    console.log(`  終点: (${ray.rayEnd.x}, ${ray.rayEnd.y})`);
    console.log(`  方向: (${ray.rayDirection.x}, ${ray.rayDirection.y})`);
    console.log(`  ヒット: ${ray.intersectsCharacter}`);
    console.log(`  警告: ${ray.hasLRWarning}`);
    console.log(`  フォールバック: ${ray.usedFallbackMethod}`);

    // 画像境界内でクリップ
    let finalEndX = ray.rayEnd.x;
    let finalEndY = ray.rayEnd.y;
    
    // 画像境界を超えている場合は、境界との交点を計算
    if (finalEndX < 0 || finalEndX > imageWidth || finalEndY < 0 || finalEndY > imageHeight) {
      console.log(`🔧 Ray${index + 1}が画像境界を超えています。クリップします。`);
      
      // Ray方向ベクトル
      const dx = ray.rayDirection.x;
      const dy = ray.rayDirection.y;
      
      // 境界との交点を計算
      const intersections: Array<{x: number, y: number, t: number}> = [];
      
      if (dx !== 0) {
        // 左右境界
        const tLeft = (0 - ray.rayOrigin.x) / dx;
        const tRight = (imageWidth - ray.rayOrigin.x) / dx;
        if (tLeft > 0 && tLeft <= 1) {
          const y = ray.rayOrigin.y + dy * tLeft;
          if (y >= 0 && y <= imageHeight) intersections.push({x: 0, y, t: tLeft});
        }
        if (tRight > 0 && tRight <= 1) {
          const y = ray.rayOrigin.y + dy * tRight;
          if (y >= 0 && y <= imageHeight) intersections.push({x: imageWidth, y, t: tRight});
        }
      }
      
      if (dy !== 0) {
        // 上下境界
        const tTop = (0 - ray.rayOrigin.y) / dy;
        const tBottom = (imageHeight - ray.rayOrigin.y) / dy;
        if (tTop > 0 && tTop <= 1) {
          const x = ray.rayOrigin.x + dx * tTop;
          if (x >= 0 && x <= imageWidth) intersections.push({x, y: 0, t: tTop});
        }
        if (tBottom > 0 && tBottom <= 1) {
          const x = ray.rayOrigin.x + dx * tBottom;
          if (x >= 0 && x <= imageWidth) intersections.push({x, y: imageHeight, t: tBottom});
        }
      }
      
      // 最も遠い交点を選択（または画像中心方向に適切な長さを設定）
      if (intersections.length > 0) {
        const maxIntersection = intersections.reduce((max, curr) => curr.t > max.t ? curr : max);
        finalEndX = maxIntersection.x;
        finalEndY = maxIntersection.y;
      } else {
        // フォールバック: 短い線に調整
        const rayLength = Math.min(200, Math.max(imageWidth, imageHeight) * 0.3);
        finalEndX = ray.rayOrigin.x + ray.rayDirection.x * rayLength;
        finalEndY = ray.rayOrigin.y + ray.rayDirection.y * rayLength;
      }
    }
    
    console.log(`🎯 Ray${index + 1}クリップ前: (${ray.rayEnd.x}, ${ray.rayEnd.y}) → クリップ後: (${finalEndX}, ${finalEndY})`);
    
    // Ray線の色とスタイルを信頼度とヒット状況に応じて決定
    let rayColor = ray.intersectsCharacter ? '#00FF00' : '#FF4444';
    let strokeDashArray = '';
    let strokeWidth = ray.intersectsCharacter ? 10 : 6; // ヒット時はより太く
    let opacity = ray.intersectsCharacter ? 1.0 : 0.6; // ミス時は薄く
    let glowColor = rayColor;
    
    // 信頼度に応じた透明度調整（仮想的な信頼度として距離から計算）
    const distance = Math.sqrt(
      Math.pow(ray.rayEnd.x - ray.rayOrigin.x, 2) + 
      Math.pow(ray.rayEnd.y - ray.rayOrigin.y, 2)
    );
    const confidence = Math.max(0.3, Math.min(1.0, 1.0 - distance / 500));
    opacity *= confidence;
    
    if (ray.hasLRWarning) {
      rayColor = '#FF6600'; // 警告時はオレンジ色
      strokeDashArray = '10,5'; // 点線で警告を示す
      glowColor = '#FFB366';
    } else if (ray.usedFallbackMethod) {
      rayColor = '#9900FF'; // フォールバック使用時は紫色
      strokeDashArray = '15,10';
      glowColor = '#CC66FF';
    } else if (ray.isLRFlippedTest) {
      rayColor = ray.intersectsCharacter ? '#00CC00' : '#CC4444';
      glowColor = ray.intersectsCharacter ? '#66FF66' : '#FF8888';
    } else if (!ray.intersectsCharacter) {
      strokeDashArray = '8,4'; // ミス時は点線
    }
    
    // グロー効果フィルターを追加
    const glowFilter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
    glowFilter.setAttribute('id', `glow-${index}`);
    const feGaussianBlur = document.createElementNS('http://www.w3.org/2000/svg', 'feGaussianBlur');
    feGaussianBlur.setAttribute('stdDeviation', '3');
    feGaussianBlur.setAttribute('result', 'coloredBlur');
    glowFilter.appendChild(feGaussianBlur);
    
    const feMerge = document.createElementNS('http://www.w3.org/2000/svg', 'feMerge');
    const feMergeNode1 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
    feMergeNode1.setAttribute('in', 'coloredBlur');
    const feMergeNode2 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
    feMergeNode2.setAttribute('in', 'SourceGraphic');
    feMerge.appendChild(feMergeNode1);
    feMerge.appendChild(feMergeNode2);
    glowFilter.appendChild(feMerge);
    
    defs.appendChild(glowFilter);

    // Ray線（グロー効果）
    const rayLineGlow = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    rayLineGlow.setAttribute('x1', ray.rayOrigin.x.toString());
    rayLineGlow.setAttribute('y1', ray.rayOrigin.y.toString());
    rayLineGlow.setAttribute('x2', finalEndX.toString());
    rayLineGlow.setAttribute('y2', finalEndY.toString());
    rayLineGlow.setAttribute('stroke', glowColor);
    rayLineGlow.setAttribute('stroke-width', (strokeWidth + 6).toString());
    rayLineGlow.setAttribute('opacity', (opacity * 0.3).toString());
    rayLineGlow.setAttribute('filter', `url(#glow-${index})`);
    if (strokeDashArray) {
      rayLineGlow.setAttribute('stroke-dasharray', strokeDashArray);
    }
    group.appendChild(rayLineGlow);
    
    // Ray線（アウトライン）
    const rayLineOutline = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    rayLineOutline.setAttribute('x1', ray.rayOrigin.x.toString());
    rayLineOutline.setAttribute('y1', ray.rayOrigin.y.toString());
    rayLineOutline.setAttribute('x2', finalEndX.toString());
    rayLineOutline.setAttribute('y2', finalEndY.toString());
    rayLineOutline.setAttribute('stroke', '#000000');
    rayLineOutline.setAttribute('stroke-width', (strokeWidth + 4).toString());
    rayLineOutline.setAttribute('opacity', (opacity * 0.8).toString());
    if (strokeDashArray) {
      rayLineOutline.setAttribute('stroke-dasharray', strokeDashArray);
    }
    group.appendChild(rayLineOutline);
    
    // Ray線（メイン）
    const rayLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    rayLine.setAttribute('x1', ray.rayOrigin.x.toString());
    rayLine.setAttribute('y1', ray.rayOrigin.y.toString());
    rayLine.setAttribute('x2', finalEndX.toString());
    rayLine.setAttribute('y2', finalEndY.toString());
    rayLine.setAttribute('stroke', rayColor);
    rayLine.setAttribute('stroke-width', strokeWidth.toString());
    rayLine.setAttribute('opacity', opacity.toString());
    rayLine.setAttribute('marker-end', 'url(#arrowhead)');
    if (strokeDashArray) {
      rayLine.setAttribute('stroke-dasharray', strokeDashArray);
    }
    
    // ツールチップ情報を追加
    const tooltipInfo = `距離: ${Math.round(distance)}px | 信頼度: ${Math.round(confidence * 100)}% | ${ray.intersectsCharacter ? 'ヒット' : 'ミス'}`;
    rayLine.setAttribute('data-tooltip', tooltipInfo);
    
    // Ray線用ツールチップイベントを追加
    addTooltipEvents(rayLine, tooltipInfo);
    
    group.appendChild(rayLine);

    // 起点の円やアイコンを削除してしっぽ本体を見やすくする
    // （パルス効果、円、アイコンはすべて削除）

    // 最遠点の円も削除してよりクリーンに

    // ヒットした文字領域を表示
    if (ray.characterBBox && ray.characterName) {
      const char = {
        boundingBox: ray.characterBBox,
        name: ray.characterName
      };
      
      // 英語名から日本語名への変換
      const englishToJapaneseMap: { [key: string]: string } = {
        'yuzuko': '野々原ゆずこ',
        'yukari': '日向縁',
        'yui': '櫟井唯',
        'yoriko': '松本頼子',
        'chiho': '相川千穂',
        'kei': '岡野佳',
        'fumi': '長谷川ふみ',
        'unknown': '不明'
      };
      
      // キャラクター色定義（より濃い色）
      const characterColors: { [key: string]: string } = {
        '野々原ゆずこ': '#FF69B4',  // ホットピンク（より濃く）
        '日向縁': '#9370DB',        // ミディアムパープル（より濃く）
        '櫟井唯': '#FFD700',        // ゴールド（より濃く）
        '松本頼子': '#FF8C69',      // サーモン（より濃く）
        '相川千穂': '#98FB98',      // ペールグリーン（より濃く）
        '岡野佳': '#DEB887',        // バーリーウッド（より濃く）
        '長谷川ふみ': '#D3D3D3',    // ライトグレー（より濃く）
      };
      
      // 英語名を日本語名に変換してキャラクター色を取得
      const japaneseName = englishToJapaneseMap[char.name.toLowerCase()] || char.name;
      const characterColor = characterColors[japaneseName] || '#FFD1DC'; // デフォルトはピンク
      
      // 黒いアウトライン削除
      
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', char.boundingBox.x1.toString());
      rect.setAttribute('y', char.boundingBox.y1.toString());
      rect.setAttribute('width', (char.boundingBox.x2 - char.boundingBox.x1).toString());
      rect.setAttribute('height', (char.boundingBox.y2 - char.boundingBox.y1).toString());
      rect.setAttribute('fill', 'none');
      rect.setAttribute('stroke', characterColor);  // キャラクター毎の色
      rect.setAttribute('stroke-width', '3');
      rect.setAttribute('opacity', '1.0');
      group.appendChild(rect);

      // キャラクター名表示は削除（英語名表示を完全に除去）
    }

    // 方向ベクトル表示（起点から少し離れた位置）
    const directionLength = 30;
    const directionEndX = ray.rayOrigin.x + ray.rayDirection.x * directionLength;
    const directionEndY = ray.rayOrigin.y + ray.rayDirection.y * directionLength;
    
    const directionLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    directionLine.setAttribute('x1', ray.rayOrigin.x.toString());
    directionLine.setAttribute('y1', ray.rayOrigin.y.toString());
    directionLine.setAttribute('x2', directionEndX.toString());
    directionLine.setAttribute('y2', directionEndY.toString());
    directionLine.setAttribute('stroke', '#0000FF');
    directionLine.setAttribute('stroke-width', '4');
    directionLine.setAttribute('opacity', '1.0');
    directionLine.setAttribute('marker-end', 'url(#smallArrowhead)');
    group.appendChild(directionLine);

    svg.appendChild(group);
  });

  return svg;
};