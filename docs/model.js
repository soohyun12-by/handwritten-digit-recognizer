// model.json의 가중치로 신경망(MLP) 계산을 수행
// 계산식은 export_model.py의 forward()와 동일: 은닉층은 ReLU, 출력층은 Softmax

// JSON을 불러와 계산하기 쉬운 형태(1차원 Float32Array)로 변환
export async function loadModel(url = "model.json") {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load model: HTTP ${response.status}`);
  }
  const data = await response.json();
  if (data.format !== "mlp-v1") {
    throw new Error(`Unsupported model format: ${data.format}`);
  }

  const layers = data.layers.map(({ weights, bias }) => {
    const inSize = weights.length;
    const outSize = bias.length;
    // weights[i][j] (입력 i → 출력 j)를 1차원 배열로 펼침
    const flat = new Float32Array(inSize * outSize);
    for (let i = 0; i < inSize; i++) {
      flat.set(weights[i], i * outSize);
    }
    return { weights: flat, bias: Float32Array.from(bias), inSize, outSize };
  });

  return { layers, testAccuracy: data.testAccuracy };
}

// 입력 벡터(784)를 받아 0~9 각 숫자의 확률 배열(10)을 반환
export function predict(model, input) {
  let x = input;
  model.layers.forEach((layer, index) => {
    const { weights, bias, inSize, outSize } = layer;
    // 출력 = 입력 × 가중치 + 편향
    const out = Float32Array.from(bias);
    for (let i = 0; i < inSize; i++) {
      const xi = x[i];
      if (xi === 0) continue; // 0인 입력은 결과에 영향이 없으므로 건너뜀 (속도 향상)
      const row = i * outSize;
      for (let j = 0; j < outSize; j++) {
        out[j] += xi * weights[row + j];
      }
    }
    const isLast = index === model.layers.length - 1;
    if (!isLast) {
      // 은닉층: ReLU (음수는 0으로)
      for (let j = 0; j < outSize; j++) {
        if (out[j] < 0) out[j] = 0;
      }
    }
    x = out;
  });
  return softmax(x);
}

// Softmax: 출력값을 합이 1인 확률로 변환 (최댓값을 빼서 overflow 방지)
function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}
