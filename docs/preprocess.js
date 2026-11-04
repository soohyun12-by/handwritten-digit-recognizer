// 그린 그림을 MNIST 형식(28x28, 0~1 값, 784 길이 벡터)으로 변환
// Python 버전(digit_recognizer.py의 preprocess)과 같은 순서로 처리

const TARGET_SIZE = 28; // 최종 이미지 크기
const DIGIT_SIZE = 20; // 숫자가 들어갈 영역 크기 (MNIST와 동일)

// 캔버스에서 흑백 픽셀 배열(0~255)을 꺼내 변환
export function preprocessCanvas(canvas) {
  const { width, height } = canvas;
  const rgba = canvas.getContext("2d").getImageData(0, 0, width, height).data;
  const gray = new Float32Array(width * height);
  // 흰 선/검은 배경이므로 빨간 채널 값만 사용
  for (let i = 0; i < gray.length; i++) {
    gray[i] = rgba[i * 4];
  }
  return preprocessGray(gray, width, height);
}

// 흑백 배열을 받아 784 길이 Float32Array로 변환 (아무것도 없으면 null)
export function preprocessGray(gray, width, height) {
  // 1) 숫자가 그려진 영역(bounding box) 찾기
  let top = height, bottom = -1, left = width, right = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (gray[y * width + x] > 0) {
        if (y < top) top = y;
        if (y > bottom) bottom = y;
        if (x < left) left = x;
        if (x > right) right = x;
      }
    }
  }
  if (bottom < 0) return null;

  const cropW = right - left + 1;
  const cropH = bottom - top + 1;

  // 2) 비율을 유지하면서 긴 변이 20픽셀이 되도록 축소
  const scale = DIGIT_SIZE / Math.max(cropW, cropH);
  const newW = Math.max(1, Math.round(cropW * scale));
  const newH = Math.max(1, Math.round(cropH * scale));
  const resized = resizeArea(gray, width, left, top, cropW, cropH, newW, newH);

  // 3) 28x28 검은 배경의 중앙에 붙이기
  const digit = new Float32Array(TARGET_SIZE * TARGET_SIZE);
  const offX = Math.floor((TARGET_SIZE - newW) / 2);
  const offY = Math.floor((TARGET_SIZE - newH) / 2);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      digit[(y + offY) * TARGET_SIZE + (x + offX)] = resized[y * newW + x];
    }
  }

  // 4) 무게중심이 이미지 중앙(14, 14)에 오도록 이동
  let total = 0, sumX = 0, sumY = 0;
  for (let y = 0; y < TARGET_SIZE; y++) {
    for (let x = 0; x < TARGET_SIZE; x++) {
      const v = digit[y * TARGET_SIZE + x];
      total += v;
      sumX += x * v;
      sumY += y * v;
    }
  }
  const shiftX = Math.round(TARGET_SIZE / 2 - sumX / total);
  const shiftY = Math.round(TARGET_SIZE / 2 - sumY / total);
  const shifted = roll(digit, TARGET_SIZE, shiftX, shiftY);

  // 0~1 범위로 정규화
  for (let i = 0; i < shifted.length; i++) {
    shifted[i] /= 255;
  }
  return shifted;
}

// 면적 평균 방식 축소: 각 결과 픽셀이 덮는 원본 영역의 평균값을 계산
// (브라우저마다 다른 캔버스 축소 방식에 의존하지 않도록 직접 구현)
function resizeArea(src, srcWidth, left, top, cropW, cropH, newW, newH) {
  const out = new Float32Array(newW * newH);
  const sx = cropW / newW;
  const sy = cropH / newH;
  for (let oy = 0; oy < newH; oy++) {
    const y0 = oy * sy, y1 = y0 + sy;
    for (let ox = 0; ox < newW; ox++) {
      const x0 = ox * sx, x1 = x0 + sx;
      let sum = 0, area = 0;
      for (let y = Math.floor(y0); y < Math.ceil(y1); y++) {
        // 원본 픽셀이 결과 픽셀과 겹치는 세로 길이
        const wy = Math.min(y + 1, y1) - Math.max(y, y0);
        for (let x = Math.floor(x0); x < Math.ceil(x1); x++) {
          // 원본 픽셀이 결과 픽셀과 겹치는 가로 길이
          const wx = Math.min(x + 1, x1) - Math.max(x, x0);
          const w = wx * wy;
          sum += src[(top + y) * srcWidth + (left + x)] * w;
          area += w;
        }
      }
      out[oy * newW + ox] = sum / area;
    }
  }
  return out;
}

// numpy.roll과 같은 동작: 밖으로 밀려난 픽셀은 반대편으로 돌아옴
function roll(src, size, shiftX, shiftY) {
  const out = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const ny = (((y + shiftY) % size) + size) % size;
      const nx = (((x + shiftX) % size) + size) % size;
      out[ny * size + nx] = src[y * size + x];
    }
  }
  return out;
}
