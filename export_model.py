"""Export the trained scikit-learn MLP to JSON files for the web version."""

import json
import os

import joblib
import numpy as np
from sklearn.datasets import fetch_openml

from train_model import MODEL_PATH

# 출력 경로: GitHub Pages로 공개되는 docs 폴더
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
MODEL_JSON_PATH = os.path.join(DOCS_DIR, "model.json")
SAMPLES_JSON_PATH = os.path.join(DOCS_DIR, "test_samples.json")

# 파일 크기를 줄이기 위해 가중치를 반올림할 소수점 자릿수
DECIMALS = 5
# JS 검증용으로 저장할 테스트 샘플 개수
NUM_SAMPLES = 20


def forward(layers, x):
    """Run the MLP forward pass with plain numpy (same math as model.js)."""
    # 은닉층: 행렬 곱 + 편향 + ReLU
    for weights, bias in layers[:-1]:
        x = np.maximum(0.0, x @ weights + bias)
    # 출력층: 행렬 곱 + 편향 + Softmax
    weights, bias = layers[-1]
    logits = x @ weights + bias
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def main():
    model = joblib.load(MODEL_PATH)
    # 이 변환은 ReLU 은닉층 + Softmax 출력층 구조만 지원
    assert model.activation == "relu" and model.out_activation_ == "softmax"

    # 가중치와 편향을 반올림 (JSON 용량 절감)
    layers = [
        (np.round(w.astype(np.float64), DECIMALS), np.round(b.astype(np.float64), DECIMALS))
        for w, b in zip(model.coefs_, model.intercepts_)
    ]

    # MNIST 테스트 데이터 로드 (캐시 사용)
    x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    x_test = x[60000:].astype(np.float32) / 255.0
    y_test = y[60000:].astype(np.int64)

    # 반올림한 가중치로도 정확도가 유지되는지 확인
    original_acc = model.score(x_test, y_test)
    rounded_probs = forward(layers, x_test)
    rounded_acc = (rounded_probs.argmax(axis=1) == y_test).mean()
    print(f"Original accuracy: {original_acc * 100:.2f}%")
    print(f"Rounded accuracy:  {rounded_acc * 100:.2f}%")

    # model.json 저장: 각 층의 가중치는 [입력][출력] 형태의 2차원 배열
    os.makedirs(DOCS_DIR, exist_ok=True)
    model_data = {
        "format": "mlp-v1",
        "inputSize": 784,
        "activation": "relu",
        "outputActivation": "softmax",
        "testAccuracy": round(float(rounded_acc), 4),
        "layers": [{"weights": w.tolist(), "bias": b.tolist()} for w, b in layers],
    }
    with open(MODEL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(model_data, f, separators=(",", ":"))

    # test_samples.json 저장: JS 계산 결과가 Python과 같은지 비교하기 위한 데이터
    samples = []
    for i in range(NUM_SAMPLES):
        samples.append({
            "label": int(y_test[i]),
            # 픽셀 값은 0~255 정수로 저장해 용량 절감 (JS에서 255로 나눔)
            "pixels": (x[60000 + i]).astype(int).tolist(),
            "expectedProbs": np.round(rounded_probs[i], 6).tolist(),
        })
    with open(SAMPLES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, separators=(",", ":"))

    size_mb = os.path.getsize(MODEL_JSON_PATH) / 1024 / 1024
    print(f"Saved {MODEL_JSON_PATH} ({size_mb:.2f} MB)")
    print(f"Saved {SAMPLES_JSON_PATH} ({NUM_SAMPLES} samples)")


if __name__ == "__main__":
    main()
