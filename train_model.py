"""MNIST handwritten digit model training script."""

import os
import time

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

# 모델 저장 경로 (이 스크립트와 같은 폴더)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_model.joblib")


def load_mnist():
    """Download (or load cached) MNIST and split into train/test sets."""
    # OpenML에서 MNIST 70,000장 다운로드 (최초 1회만 다운로드, 이후 캐시 사용)
    print("Loading MNIST dataset...")
    x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    # 픽셀 값을 0~1 범위로 정규화
    x = x.astype(np.float32) / 255.0
    y = y.astype(np.int64)
    # 표준 분할: 앞 60,000장은 학습용, 뒤 10,000장은 테스트용
    return x[:60000], x[60000:], y[:60000], y[60000:]


def train_and_save():
    """Train an MLP classifier on MNIST and save it to disk."""
    x_train, x_test, y_train, y_test = load_mnist()

    # 은닉층 2개(256, 128)를 가진 다층 퍼셉트론 신경망
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=20,
        batch_size=256,
        early_stopping=True,  # 검증 성능이 더 이상 오르지 않으면 조기 종료
        random_state=42,
        verbose=True,
    )

    print("Training model...")
    start = time.time()
    model.fit(x_train, y_train)
    print(f"Training finished in {time.time() - start:.1f}s")

    # 테스트 데이터로 정확도 평가
    accuracy = model.score(x_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # 학습된 모델을 파일로 저장
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_and_save()
